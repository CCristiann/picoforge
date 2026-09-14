"""make_moe_layer.py — one Qwen3-30B-A3B layer, at its real shapes, random weights.

The verify cost of a MoE is a question about shapes and routing, not about what
the weights say: how long the GPU takes to push n tokens through D distinct
experts of width 2048x768. Answering it needs the shapes on the machine, not
the 61 GB download. So: a checkpoint with ONE decoder layer of the 30B's
dimensions (read from the published config, reproduced below), 128 experts,
bf16, and a vocabulary cut to 1024 because the embedding is not the subject.

The numbers must stay the real model's. If the published config changes, this
file is wrong until it is updated:
  https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json

Run:  ./tools/venv/bin/python tools/synth/make_moe_layer.py build/synth-30b-layer
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260914

# Qwen3-30B-A3B, as published -- except the three marked lines.
CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768,
    "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
    "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": True,
    "decoder_sparse_step": 1, "mlp_only_layers": [],
    "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "max_position_embeddings": 40960,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu",
    "num_hidden_layers": 1,   # CHANGED: 48 in the real model
    "vocab_size": 1024,       # CHANGED: 151936 in the real model
}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/synth-30b-layer"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    c = CONFIG
    H, Ie, E = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"]
    q, kv, V = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"], c["vocab_size"]

    def w(*shape):
        return (0.02 * torch.randn(*shape)).to(torch.bfloat16)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n)).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": w(V, H), "lm_head.weight": w(V, H), "model.norm.weight": norm(H)}
    p = "model.layers.0"
    t |= {f"{p}.input_layernorm.weight": norm(H), f"{p}.post_attention_layernorm.weight": norm(H),
          f"{p}.self_attn.q_proj.weight": w(q, H), f"{p}.self_attn.k_proj.weight": w(kv, H),
          f"{p}.self_attn.v_proj.weight": w(kv, H), f"{p}.self_attn.o_proj.weight": w(H, q),
          f"{p}.self_attn.q_norm.weight": norm(c["head_dim"]),
          f"{p}.self_attn.k_norm.weight": norm(c["head_dim"]), f"{p}.mlp.gate.weight": w(E, H)}
    for e in range(E):
        t[f"{p}.mlp.experts.{e}.gate_proj.weight"] = w(Ie, H)
        t[f"{p}.mlp.experts.{e}.up_proj.weight"] = w(Ie, H)
        t[f"{p}.mlp.experts.{e}.down_proj.weight"] = w(H, Ie)
    save_file(t, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + "\n")
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]:
        shutil.copy(ROOT / "models/Qwen3-0.6B" / f, out / f)
    n = sum(x.numel() for x in t.values())
    print(f"one 30B-shaped layer -> {out}  ({n / 1e6:.0f} M parameters, {n * 2 / 1e9:.2f} GB bf16)")


if __name__ == "__main__":
    main()
