"""qwen3_forward.py — Phase 0 oracle: Qwen3-0.6B forward pass in pure NumPy.

This file is the reference implementation the C engine will be validated
against (CLAUDE.md: "chain of oracles"). It grows one component at a time;
today it only loads the config and prints the architecture summary.

Rule: every dimension comes from config.json. Nothing is hardcoded.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class Qwen3Config:
    """The subset of config.json the forward pass actually needs.

    A frozen dataclass instead of a raw dict: a typo like cfg.head_dmi
    crashes immediately instead of silently returning None.
    """

    hidden_size: int            # width of the residual stream
    num_hidden_layers: int      # identical transformer blocks, applied in sequence
    num_attention_heads: int    # query heads
    num_key_value_heads: int    # KV heads; fewer than Q heads -> GQA
    head_dim: int               # explicit! NOT hidden_size / num_attention_heads
    intermediate_size: int      # SwiGLU MLP inner width
    vocab_size: int
    rms_norm_eps: float         # epsilon inside RMSNorm's sqrt
    rope_theta: float           # RoPE base frequency
    max_position_embeddings: int
    tie_word_embeddings: bool   # true -> LM head reuses the embedding matrix
    torch_dtype: str            # storage dtype of the weights (we compute in fp32)
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def from_json(cls, model_dir: Path) -> "Qwen3Config":
        raw = json.loads((model_dir / "config.json").read_text())
        assert raw["model_type"] == "qwen3", f"not a Qwen3 config: {raw['model_type']}"
        # Take only the fields we declared, so an unexpected config shape
        # fails loudly here rather than deep inside the forward pass.
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


def print_summary(cfg: Qwen3Config) -> None:
    """Architecture summary, printed at startup (CLAUDE.md principle #5)."""
    q_dim = cfg.num_attention_heads * cfg.head_dim   # width of Q projection output
    kv_dim = cfg.num_key_value_heads * cfg.head_dim  # width of K and V projections
    gqa_group = cfg.num_attention_heads // cfg.num_key_value_heads
    naive_head_dim = cfg.hidden_size // cfg.num_attention_heads

    print("=== Qwen3 architecture summary ===")
    print(f"layers                : {cfg.num_hidden_layers}")
    print(f"hidden_size           : {cfg.hidden_size}")
    print(f"vocab_size            : {cfg.vocab_size}")
    print(f"attention             : {cfg.num_attention_heads} Q heads, "
          f"{cfg.num_key_value_heads} KV heads (GQA: {gqa_group} Q per KV head)")
    print(f"head_dim              : {cfg.head_dim} (explicit; naive hidden/heads "
          f"would give {naive_head_dim} — the classic Qwen3 gotcha)")
    print(f"  Q proj              : {cfg.hidden_size} -> {q_dim}")
    print(f"  K/V proj            : {cfg.hidden_size} -> {kv_dim} each")
    print(f"  O proj              : {q_dim} -> {cfg.hidden_size}")
    print(f"MLP (SwiGLU)          : {cfg.hidden_size} -> {cfg.intermediate_size} -> "
          f"{cfg.hidden_size}")
    print(f"rope_theta            : {cfg.rope_theta:g}")
    print(f"max positions         : {cfg.max_position_embeddings}")
    print(f"rms_norm_eps          : {cfg.rms_norm_eps:g}")
    print(f"tied embeddings       : {cfg.tie_word_embeddings} "
          f"(no separate lm_head tensor in the weights)")
    print(f"weights dtype         : {cfg.torch_dtype} (oracle computes in fp32)")
    print(f"bos / eos             : {cfg.bos_token_id} / {cfg.eos_token_id}")


def load_weights(model_dir: Path) -> dict[str, np.ndarray]:
    """Load every tensor from model.safetensors as an fp32 numpy array.

    The weights are stored in bfloat16, which numpy cannot represent, so
    torch does the bf16 -> fp32 upcast. This is the ONLY place torch is
    allowed to touch the data: bf16 is the top half of fp32, so this
    conversion is exact (16 zero bits appended, nothing rounded).
    """
    import torch  # deliberately local: keeps torch out of the forward path

    weights = {}
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        for name in f.keys():
            weights[name] = f.get_tensor(name).to(torch.float32).numpy()
    return weights


def embed_tokens(weights: dict[str, np.ndarray], token_ids: list[int]) -> np.ndarray:
    """Token embedding is a row lookup, nothing more: matrix[token_id].

    Returns (seq_len, hidden_size) fp32 — the initial residual stream.
    """
    return weights["model.embed_tokens.weight"][token_ids]


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm: rescale each row to unit RMS, then apply the learned gain.

    LayerNorm minus mean-centering and bias — recentering turned out to be
    unnecessary, rescaling is what matters. eps lives INSIDE the sqrt; putting
    it outside changes low digits and breaks parity with the reference.
    """
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def rope_tables(positions: np.ndarray, head_dim: int, theta: float):
    """Precompute cos/sin for RoPE at the given positions.

    Pair i rotates with frequency theta**(-2i/head_dim): pair 0 spins ~1
    radian per token, the last pairs are nearly static — each pair watches
    the sequence at a different zoom level. Returns (seq, head_dim/2) each.
    """
    inv_freq = theta ** (-np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    angles = np.outer(positions.astype(np.float32), inv_freq)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate Q or K pairs. x: (..., seq, head_dim); applied to Q and K, never V.

    Convention gotcha: HF/Qwen pair dimension i with i + head_dim/2
    ("rotate_half"), NOT adjacent dims (i, i+1). The wrong pairing produces
    garbage logits with no error message — verify.py is what would catch it.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x2 * cos + x1 * sin], axis=-1)


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/Qwen3-0.6B")
    cfg = Qwen3Config.from_json(model_dir)
    print_summary(cfg)

    weights = load_weights(model_dir)
    # The checkpoint ships lm_head.weight even though the config says
    # tie_word_embeddings — measured: it is a byte-identical copy of the
    # embedding matrix. The forward pass will use embed_tokens for the head
    # (the config's contract); the copy may vanish in future checkpoints.
    dup = "lm_head.weight" in weights and np.array_equal(
        weights["lm_head.weight"], weights["model.embed_tokens.weight"])
    n_total = sum(w.size for w in weights.values())
    n_unique = n_total - (weights["lm_head.weight"].size if dup else 0)
    print("\n=== weights ===")
    print(f"tensors               : {len(weights)}")
    print(f"parameters            : {n_unique / 1e9:.3f} B unique"
          + (f" ({n_total / 1e9:.3f} B on disk: lm_head duplicates embed_tokens)"
             if dup else ""))

    emb = embed_tokens(weights, [cfg.bos_token_id])
    print(f"embed(bos) shape      : {emb.shape}, dtype {emb.dtype}, "
          f"first values {np.round(emb[0, :3], 4)}")

    # RMSNorm self-checks against its two defining properties.
    print("\n=== rms_norm checks ===")
    w0 = weights["model.layers.0.input_layernorm.weight"]
    y = rms_norm(emb, w0, cfg.rms_norm_eps)
    # 1. Unit RMS: before the gain w, every row must have RMS ~= 1.
    pre_gain_rms = np.sqrt(np.mean((y / w0) ** 2))
    print(f"pre-gain RMS          : {pre_gain_rms:.6f} (expected ~1)")
    # 2. Scale invariance: rms_norm(10x) == rms_norm(x), because the input's
    #    own magnitude is divided out. Only the *direction* of x survives.
    drift = np.max(np.abs(rms_norm(emb * 10.0, w0, cfg.rms_norm_eps) - y))
    print(f"scale invariance drift: {drift:.2e} (expected ~0)")

    # RoPE self-check: the defining property. Attention scores between
    # rotated q at position m and rotated k at position n must depend only
    # on m - n, so shifting both positions by 100 must not change the score.
    print("\n=== rope checks ===")
    rng = np.random.default_rng(0)
    q = rng.standard_normal(cfg.head_dim).astype(np.float32)
    k = rng.standard_normal(cfg.head_dim).astype(np.float32)

    def score(m: int, n: int) -> float:
        cos, sin = rope_tables(np.array([m, n]), cfg.head_dim, cfg.rope_theta)
        return float(apply_rope(q, cos[0], sin[0]) @ apply_rope(k, cos[1], sin[1]))

    s_near, s_far = score(3, 7), score(103, 107)
    print(f"score(3,7)            : {s_near:.6f}")
    print(f"score(103,107)        : {s_far:.6f} (must match: only m-n matters)")
    print(f"score(7,3)            : {score(7, 3):.6f} (need not match: order matters)")


if __name__ == "__main__":
    main()
