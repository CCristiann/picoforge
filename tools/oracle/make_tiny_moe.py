"""make_tiny_moe.py — a Qwen3-MoE small enough to verify against in seconds.

Phase 4 needs the MoE block in every link of the chain of oracles before the
61 GB checkpoint is worth downloading. A randomly initialised model of the
SAME architecture (transformers' Qwen3MoeForCausalLM) exercises the same code
paths: router, top-k, renormalised weights, per-expert SwiGLU, an untied LM
head -- in 3 layers instead of 48.

What it shares with Qwen3-30B-A3B, on purpose:
  - the on-disk layout. transformers 5 fuses experts in memory
    (gate_up_proj [E, 2I, H]) but save_pretrained writes one tensor per expert,
    model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight -- probed, and the
    same names as the real checkpoint's index.
  - bf16 storage, the vocabulary and the tokenizer (copied from Qwen3-0.6B).
  - norm_topk_prob, tie_word_embeddings=false.

What it adds that the 30B does not have: one dense layer (mlp_only_layers),
so the oracle's choice between the two MLP kinds is tested too.

Random init is not left at the defaults, because the defaults hide bugs:
RMSNorm gains of exactly 1.0 make a skipped norm weight invisible, and a
0.02 init spreads the logits so flat that argmax is decided by noise.

Run:  ./tools/venv/bin/python tools/oracle/make_tiny_moe.py build/tiny-qwen3-moe
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260914


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/tiny-qwen3-moe"
    torch.manual_seed(SEED)
    cfg = Qwen3MoeConfig(
        vocab_size=151936, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[1],
        tie_word_embeddings=False, rope_theta=1e6, rms_norm_eps=1e-6,
        max_position_embeddings=40960, bos_token_id=151643, eos_token_id=151645,
        initializer_range=0.2,
    )
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    model = model.to(torch.bfloat16)
    model.save_pretrained(out)

    # transformers 5 writes its own config dialect (num_local_experts, dtype,
    # rope_parameters). The real checkpoint was saved by 4.51 and says
    # num_experts, torch_dtype, rope_theta -- and that is what the engine must
    # read. Rewrite the file in the real checkpoint's words; main() checks
    # that transformers reads it back to the same numbers.
    path = out / "config.json"
    raw = json.loads(path.read_text())
    raw["num_experts"] = raw.pop("num_local_experts")
    raw["torch_dtype"] = raw.pop("dtype")
    raw["rope_theta"] = raw.pop("rope_parameters")["rope_theta"]
    raw.pop("pad_token_id", None)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    back = Qwen3MoeConfig.from_pretrained(out)
    for k in ["num_experts", "num_experts_per_tok", "moe_intermediate_size", "mlp_only_layers",
              "norm_topk_prob", "tie_word_embeddings", "head_dim"]:
        assert getattr(back, k) == getattr(cfg, k), f"{k}: {getattr(back, k)} != {getattr(cfg, k)}"
    assert back.rope_parameters["rope_theta"] == cfg.rope_parameters["rope_theta"]

    # The tokenizer and generation config are the 0.6B's: same vocabulary.
    src = ROOT / "models/Qwen3-0.6B"
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json"]:
        shutil.copy(src / f, out / f)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny Qwen3-MoE -> {out}  ({n / 1e6:.1f} M parameters, seed {SEED})")


if __name__ == "__main__":
    main()
