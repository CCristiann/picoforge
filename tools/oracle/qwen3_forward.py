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


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/Qwen3-0.6B")
    cfg = Qwen3Config.from_json(model_dir)
    print_summary(cfg)


if __name__ == "__main__":
    main()
