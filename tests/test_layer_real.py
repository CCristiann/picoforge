"""test_layer_real.py — the oracle's layers against transformers', on real weights.

verify.py holds the NumPy oracle to transformers on whole models, which works
for Qwen3-0.6B and for the tiny MoE. Qwen3-30B-A3B does not fit: 122 GB as
fp32. But a decoder layer is a pure function of its input and its own weights,
so the link can be tested one layer at a time: build transformers'
Qwen3MoeDecoderLayer, load that layer's real weights into it (experts fused the
way transformers 5 holds them, gate_up_proj = [gate; up]), and feed both
implementations the same input -- the prompt's real embeddings.

    ./tools/venv/bin/python tests/test_layer_real.py models/Qwen3-30B-A3B [LAYER ...]
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
from qwen3_forward import (Qwen3Config, attention_block, embed_tokens, load_weights,  # noqa: E402
                           mlp_block, moe_block, rope_tables)

MAX_REL_ERROR = 1e-4          # the logits budget of verify.py, applied to one layer
PROMPT = "The capital of France is Paris, and the capital of Italy is"


def load_layer(layer, weights, cfg, l: int) -> None:
    p = f"model.layers.{l}"
    t = lambda name: torch.from_numpy(np.ascontiguousarray(weights[name]))  # noqa: E731
    with torch.no_grad():
        for name in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, name).weight.copy_(t(f"{p}.{name}.weight"))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            getattr(layer.self_attn, name).weight.copy_(t(f"{p}.self_attn.{name}.weight"))
        if cfg.is_moe_layer(l):
            layer.mlp.gate.weight.copy_(t(f"{p}.mlp.gate.weight"))
            for e in range(cfg.num_experts):
                ex = f"{p}.mlp.experts.{e}"
                layer.mlp.experts.gate_up_proj[e].copy_(
                    torch.cat([t(f"{ex}.gate_proj.weight"), t(f"{ex}.up_proj.weight")], dim=0))
                layer.mlp.experts.down_proj[e].copy_(t(f"{ex}.down_proj.weight"))
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer.mlp, name).weight.copy_(t(f"{p}.mlp.{name}.weight"))


def main() -> None:
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (Qwen3MoeDecoderLayer,
                                                                  Qwen3MoeRotaryEmbedding)

    model_dir = Path(sys.argv[1])
    cfg = Qwen3Config.from_json(model_dir)
    layers = [int(x) for x in sys.argv[2:]] or [0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1]
    weights = load_weights(model_dir)
    hf_cfg = AutoConfig.from_pretrained(model_dir)
    hf_cfg._attn_implementation = "eager"

    ids = AutoTokenizer.from_pretrained(model_dir)(PROMPT)["input_ids"]
    n = len(ids)
    x = embed_tokens(weights, ids)
    cos, sin = rope_tables(np.arange(n), cfg.head_dim, cfg.rope_theta)
    rotary = Qwen3MoeRotaryEmbedding(hf_cfg)
    pos = torch.arange(n)[None]
    mask = torch.triu(torch.full((n, n), torch.finfo(torch.float32).min), diagonal=1)[None, None]

    ok = True
    for l in layers:
        layer = Qwen3MoeDecoderLayer(hf_cfg, l).float().eval()
        load_layer(layer, weights, cfg, l)
        xt = torch.from_numpy(x)[None]
        with torch.no_grad():
            ref = layer(xt, attention_mask=mask, position_ids=pos,
                        position_embeddings=rotary(xt, pos))[0].numpy()
        ours = attention_block(x, weights, l, cfg, cos, sin)
        ours = (moe_block if cfg.is_moe_layer(l) else mlp_block)(ours, weights, l, cfg)
        rel = float(np.abs(ours - ref).max() / np.abs(ref).max())
        good = rel < MAX_REL_ERROR
        ok &= good
        print(f"  layer {l:2d} ({'MoE' if cfg.is_moe_layer(l) else 'dense'}), {n} tokens: "
              f"relative error {rel:.2e} (budget {MAX_REL_ERROR:.0e})  {'PASS' if good else 'FAIL'}")
        del layer
    print("\n" + ("PASS: the oracle's layers are transformers' layers on these weights" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
