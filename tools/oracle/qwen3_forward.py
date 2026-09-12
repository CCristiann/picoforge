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


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable.

    Subtracting the row max first is free mathematically (numerator and
    denominator share the factor) but keeps every exponent <= 0, so exp()
    can never overflow fp32.
    """
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """Causal scaled-dot-product attention. q,k,v: (heads, seq, head_dim).

    The three moves: scores = scaled q.k, softmax into weights, weighted
    sum of values. Future positions get -inf BEFORE the softmax, which
    turns into exactly zero weight after it: each token sees only itself
    and its past — the property generation depends on.
    """
    scores = (q @ k.transpose(0, 2, 1)) * scale          # (heads, seq, seq)
    seq = scores.shape[-1]
    future = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    return softmax(scores) @ v                            # (heads, seq, head_dim)


def attention_block(x: np.ndarray, weights: dict, layer: int, cfg: Qwen3Config,
                    cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """One full Qwen3 attention sub-block, residual included.

    Order matters and is easy to get wrong: project -> split heads ->
    QK-norm (Qwen3's per-head RMSNorm on Q and K; skip it and logits are
    silently wrong) -> RoPE -> GQA share -> causal attention -> merge ->
    output projection -> add back to the residual stream.
    """
    p = f"model.layers.{layer}.self_attn"
    seq = x.shape[0]
    h = rms_norm(x, weights[f"model.layers.{layer}.input_layernorm.weight"],
                 cfg.rms_norm_eps)

    # PyTorch Linear stores (out_features, in_features), hence the .T
    q = h @ weights[f"{p}.q_proj.weight"].T   # (seq, 2048)
    k = h @ weights[f"{p}.k_proj.weight"].T   # (seq, 1024)
    v = h @ weights[f"{p}.v_proj.weight"].T   # (seq, 1024)

    # (seq, n_heads * head_dim) -> (n_heads, seq, head_dim)
    q = q.reshape(seq, cfg.num_attention_heads, cfg.head_dim).transpose(1, 0, 2)
    k = k.reshape(seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 0, 2)
    v = v.reshape(seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 0, 2)

    # QK-norm, then RoPE — never the other way around, and never on V.
    q = rms_norm(q, weights[f"{p}.q_norm.weight"], cfg.rms_norm_eps)
    k = rms_norm(k, weights[f"{p}.k_norm.weight"], cfg.rms_norm_eps)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    # GQA: each KV head serves num_heads/num_kv_heads consecutive Q heads.
    # The oracle duplicates for clarity; the C engine will index instead.
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    k = np.repeat(k, group, axis=0)
    v = np.repeat(v, group, axis=0)

    out = attention(q, k, v, 1.0 / np.sqrt(cfg.head_dim))
    out = out.transpose(1, 0, 2).reshape(seq, -1)         # merge heads
    return x + out @ weights[f"{p}.o_proj.weight"].T      # residual add


def silu(z: np.ndarray) -> np.ndarray:
    """silu(z) = z * sigmoid(z): a smooth ReLU (config's hidden_act)."""
    return z / (1.0 + np.exp(-z))


def mlp_block(x: np.ndarray, weights: dict, layer: int, cfg: Qwen3Config) -> np.ndarray:
    """SwiGLU MLP sub-block, residual included: x + down(silu(gate) * up).

    Per-token, no cross-token flow: attention moves information between
    positions, the MLP digests it in place. `up` carries content, silu(gate)
    is a learned per-channel valve deciding how much of it passes.
    """
    p = f"model.layers.{layer}.mlp"
    h = rms_norm(x, weights[f"model.layers.{layer}.post_attention_layernorm.weight"],
                 cfg.rms_norm_eps)
    gate = h @ weights[f"{p}.gate_proj.weight"].T   # (seq, 3072)
    up = h @ weights[f"{p}.up_proj.weight"].T       # (seq, 3072)
    return x + (silu(gate) * up) @ weights[f"{p}.down_proj.weight"].T


def forward(token_ids: list[int], weights: dict, cfg: Qwen3Config) -> np.ndarray:
    """The whole model: embeddings -> 28 identical layers -> norm -> logits.

    Returns (seq, vocab_size) fp32 logits. Row i is the model's belief
    about token i+1, computed — thanks to the causal mask — from tokens
    0..i only. The LM head is embed_tokens transposed (tied weights: the
    config's contract; the checkpoint's lm_head copy is ignored).
    """
    x = embed_tokens(weights, token_ids)
    cos, sin = rope_tables(np.arange(len(token_ids)), cfg.head_dim, cfg.rope_theta)
    for layer in range(cfg.num_hidden_layers):
        x = attention_block(x, weights, layer, cfg, cos, sin)
        x = mlp_block(x, weights, layer, cfg)
    x = rms_norm(x, weights["model.norm.weight"], cfg.rms_norm_eps)
    return x @ weights["model.embed_tokens.weight"].T


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

    # Attention self-check: causality, the defining property. Change what
    # token 5 offers (its K and V): outputs at positions 0..4 must not move
    # by a single bit, because the mask forbids them from ever seeing it.
    print("\n=== attention checks ===")
    seq, scale = 8, 1.0 / np.sqrt(cfg.head_dim)
    qs = rng.standard_normal((1, seq, cfg.head_dim)).astype(np.float32)
    ks = rng.standard_normal((1, seq, cfg.head_dim)).astype(np.float32)
    vs = rng.standard_normal((1, seq, cfg.head_dim)).astype(np.float32)
    out = attention(qs, ks, vs, scale)
    ks2, vs2 = ks.copy(), vs.copy()
    ks2[0, 5], vs2[0, 5] = 999.0, -999.0   # token 5 now offers garbage
    out2 = attention(qs, ks2, vs2, scale)
    past_drift = np.max(np.abs(out2[0, :5] - out[0, :5]))
    future_drift = np.max(np.abs(out2[0, 5:] - out[0, 5:]))
    print(f"positions 0-4 drift   : {past_drift:.1f} (must be exactly 0)")
    print(f"positions 5-7 drift   : {future_drift:.1f} (large: they DO see token 5)")
    print(f"pos 0 sees only itself: {np.allclose(out[0, 0], vs[0, 0])} "
          f"(row 0 of the weights is forced to [1, 0, 0, ...])")

    # Layer-0 attention block on real tokens: shapes and sanity only.
    # Numerical truth against transformers comes with verify.py.
    print("\n=== attention block (layer 0) ===")
    tokens = [cfg.bos_token_id, 9707, 11, 1879]           # arbitrary real ids
    x = embed_tokens(weights, tokens)
    cos, sin = rope_tables(np.arange(len(tokens)), cfg.head_dim, cfg.rope_theta)
    x1 = attention_block(x, weights, 0, cfg, cos, sin)
    delta = x1 - x                                        # the block's contribution
    print(f"output shape          : {x1.shape} (must equal input {x.shape})")
    print(f"all finite            : {np.isfinite(x1).all()}")
    print(f"residual delta RMS    : {np.sqrt(np.mean(delta ** 2)):.4f} "
          f"(nonzero and modest: the block added something, sanely)")

    # Full layer 0 = attention block + MLP block. Also demonstrate that the
    # MLP is strictly per-token: run it on just the first 2 rows and check
    # they come out identical to the 4-row run (no cross-token flow).
    print("\n=== full layer 0 (attention + MLP) ===")
    x2 = mlp_block(x1, weights, 0, cfg)
    print(f"output shape          : {x2.shape}, all finite: {np.isfinite(x2).all()}")
    print(f"layer delta RMS       : {np.sqrt(np.mean((x2 - x) ** 2)):.4f}")
    print(f"MLP per-token check   : {np.max(np.abs(mlp_block(x1[:2], weights, 0, cfg) - x2[:2])):.1f} "
          f"(must be exactly 0: token i's MLP ignores every other token)")

    # The moment of truth: the full forward pass on a real prompt. The
    # tokenizer is I/O, not math — the forward path above stays pure numpy.
    print("\n=== full forward pass ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt = "The capital of France is"
    ids = tok(prompt)["input_ids"]
    logits = forward(ids, weights, cfg)
    print(f"prompt                : {prompt!r} ({len(ids)} tokens)")
    print(f"logits shape          : {logits.shape} (seq, vocab)")
    probs = softmax(logits[-1:])[0]          # last row: what comes next?
    print("top-5 next tokens:")
    for t in np.argsort(probs)[::-1][:5]:
        print(f"  {probs[t]:6.1%}  {tok.decode([t])!r}")


if __name__ == "__main__":
    main()
