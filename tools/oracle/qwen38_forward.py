"""G1 graph oracle: the upstream layer ORDER over the component references.

Every operator comes from an existing, separately verified reference that
executes hash-pinned Transformers bodies: HC+GDN (C), QSA (D), PLE (E), MoE (F).
Weights are dequantized by the pinned DwarfStar decoder, never by picoforge.
The only new code is the composition, which mirrors, line by line:
  Qwen4ExpTextModel.forward        embed -> repeat(hc_count) -> layers -> mixer
  Qwen4ExpTextDecoderLayer.forward PLE add -> HC-attn -> GDN|QSA -> inject
                                   -> HC-ffn -> MoE -> inject
plus the untied LM head. Prefill only: each call recomputes the whole prefix,
so it never shares a cache algorithm with the engine. CPU, FP32 operators,
FP64 products for the projections we perform ourselves (as in C's oracle).
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_attention as att
import qwen38_moe as moe
import qwen38_ngram as ngram
import qwen38_packed as packed
import qwen38_recurrent as rec

SOURCE = ROOT / "build/qwen38-f/upstream/modeling_qwen4_exp.py"
REFERENCE = ROOT / "build/qwen38-b/reference/reference"
# C's composition slots (Q38CW_*), in the same order as q38c_layer_roles.
GDN_SLOTS = ("hc_attn_norm", "hc_attn_down", "hc_attn_up", "hc_attn_inject", "attn_qkv",
             "attn_gate", "ssm_alpha", "ssm_beta", "ssm_a", "ssm_dt.bias", "ssm_conv1d",
             "ssm_norm", "ssm_out")


class Weights:
    """GGUF tensors from the fixture manifest, decoded to HF [out, in] FP32."""

    def __init__(self, gguf, reference=REFERENCE):
        self.path, self.reference = Path(gguf), Path(reference)
        manifest = json.loads(self.path.with_suffix(".json").read_text())
        self.tensors = {t["name"]: t for t in manifest["tensors"]}
        self.meta, self.hf = manifest["metadata"], manifest["hf_text_config"]

    def physical(self, name):
        t = self.tensors[name]
        fd = os.open(self.path, os.O_RDONLY)
        try:
            raw = packed.pread_exact(fd, t["file_offset"], t["size_bytes"])
        finally:
            os.close(fd)
        shape = t["shape"]
        rows = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        w = packed.decode(self.reference, raw, t["type_id"], shape[0], rows)
        return w.reshape(*shape[:0:-1], shape[0])  # [experts?, out, in_physical]

    def matrix(self, name, k_logical=None):
        w = self.physical(name)
        return np.ascontiguousarray(w[..., :k_logical] if k_logical else w)

    def vector(self, name):
        return self.physical(name).reshape(-1).copy()


def linear(w, x):
    """FP64 product of the decoded FP32 weights, rounded once to FP32."""
    return torch.from_numpy(packed.matmul_reference(w, np.ascontiguousarray(x, np.float32), w.shape[1])[0]
                            .astype(np.float32))


class GdnAdapter:
    """C's composition_reference interface over one fixture layer."""

    def __init__(self, weights, layer):
        c = weights.hf
        self.w, self.prefix = weights, f"blk.{layer}."
        self.metadata = dict(eps=c["rms_norm_eps"], kernel=c["linear_conv_kernel_dim"],
                             hk=c["linear_num_key_heads"], hv=c["linear_num_value_heads"],
                             dk=c["linear_key_head_dim"], dv=c["linear_value_head_dim"])

    def name(self, slot):
        suffix = "" if GDN_SLOTS[slot] in ("ssm_a", "ssm_dt.bias") else ".weight"
        return self.prefix + GDN_SLOTS[slot] + suffix

    def constant(self, slot):
        return torch.from_numpy(self.w.matrix(self.name(slot)).copy())

    def linear(self, slot, x):
        return linear(self.w.matrix(self.name(slot)), x.detach().numpy().reshape(len(x), -1))


def hc(ref, w, prefix, residual, inject=True):
    t, h, e = residual.shape
    return rec.hc_reference(ref, residual, torch.from_numpy(w.vector(prefix + "_norm.weight").reshape(h, e)),
                            torch.from_numpy(w.matrix(prefix + "_down.weight")),
                            torch.from_numpy(w.matrix(prefix + "_up.weight")),
                            torch.from_numpy(w.matrix(prefix + "_inject.weight")) if inject else None,
                            eps=w.hf["rms_norm_eps"])


def qsa(hf_ref, w, p, x):
    c = w.hf
    cfg = att.Config(width=c["hidden_size"], heads=c["num_attention_heads"],
                     kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
                     index_heads=c["indexer_n_heads"], index_dim=c["indexer_head_dim"],
                     rotary_dim=int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"]),
                     ratio=c["indexer_compress_ratio"], budget=c["indexer_budget"],
                     eps=c["rms_norm_eps"], theta=float(c["rope_parameters"]["rope_theta"]))
    t, x = len(x), x.numpy()
    proj = lambda n: linear(w.matrix(p + n), x).numpy()
    inputs = att.Inputs(proj("attn_q.weight").reshape(t, cfg.heads, 2, cfg.head_dim),
        proj("attn_k.weight").reshape(t, cfg.kv_heads, cfg.head_dim),
        proj("attn_v.weight").reshape(t, cfg.kv_heads, cfg.head_dim),
        proj("indexer.q_proj.weight").reshape(t, cfg.index_heads, cfg.index_dim),
        proj("indexer.k_proj.weight").reshape(t, cfg.index_dim),
        *(w.vector(p + n).astype(np.float32) for n in ("attn_q_norm.weight", "attn_k_norm.weight",
                                                      "indexer.q_norm.weight", "indexer.k_norm.weight")))
    out = att.evaluate(hf_ref, cfg, inputs, 0)["gated_eager"].reshape(t, -1)
    return linear(w.matrix(p + "attn_output.weight"), out)


def ple(w, layer, tokens, residual):
    c, m, p = w.hf, w.meta, f"blk.{layer}."
    config = dict(ngram_size=c["ngram_size"], heads_per_ngram=c["heads_per_ngram"], vocab=c["vocab_size"],
                  eos=c["eos_token_id"], multipliers=m["qwen4exp.ple.layer_multipliers"],
                  sizes=m["qwen4exp.ple.head_vocab_sizes"], offsets=m["qwen4exp.ple.head_offsets"],
                  rows=m["qwen4exp.ple.row_count"], layer=layer)
    t = lambda n: torch.from_numpy(w.matrix(n))
    oracle = ngram.PleOracle(config, t("per_layer_token_embd.weight"), t(p + "ple_key.weight"),
                             t(p + "ple_value.weight"),
                             [torch.from_numpy(w.vector(p + f"ple_norm_{n}.weight")) for n in ("key", "query", "conv")],
                             t(p + "ple_conv1d.weight"), eps=c["rms_norm_eps"])
    return oracle.run(list(tokens), residual)[0]


def mlp(w, p, x):
    c = w.hf
    cfg = moe.FixtureConfig(hidden_size=c["hidden_size"], num_experts=c["num_experts"],
        num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
        shared_expert_intermediate_size=c["shared_expert_intermediate_size"])
    i = cfg.moe_intermediate_size
    weights = {"gate.weight": w.matrix(p + "ffn_gate_inp.weight"),
               "experts.gate_up_proj": np.concatenate([w.matrix(p + "ffn_gate_exps.weight"),
                                                       w.matrix(p + "ffn_up_exps.weight")], axis=1),
               "experts.down_proj": w.matrix(p + "ffn_down_exps.weight", i),
               "shared_expert.gate_proj.weight": w.matrix(p + "ffn_gate_shexp.weight"),
               "shared_expert.up_proj.weight": w.matrix(p + "ffn_up_shexp.weight"),
               "shared_expert.down_proj.weight": w.matrix(p + "ffn_down_shexp.weight"),
               "shared_expert_gate.weight": w.matrix(p + "ffn_gate_inp_shexp.weight").reshape(1, -1)}
    result = moe.upstream_forward(cfg, x.numpy(), weights)
    return torch.from_numpy(result["y"]), result["ids"]


def layer(refs, w, index, tokens, residual):
    """One Qwen4ExpTextDecoderLayer.forward on the whole prefix; returns trace."""
    c, p, out = w.hf, f"blk.{index}.", {"input": residual}
    if index + 1 in c["ple_layer_ids"]:                         # PLE add, before HC
        out["ple_delta"] = ple(w, index, tokens, residual)
        residual = residual + out["ple_delta"]
    if c["layer_types"][index] == "linear_attention":
        stages = rec.composition_reference(refs["c"], GdnAdapter(w, index), residual)
        out.update(attn_mixed=stages["mixed"], attn_inject=stages["injection"],
                   attn_contribution=stages["contribution"])
    else:
        h = hc(refs["c"], w, p + "hc_attn", residual)
        out.update(attn_mixed=h["mixed"], attn_inject=h["injection"],
                   attn_contribution=qsa(refs["d"], w, p, h["mixed"]))
    residual = residual + out["attn_inject"].unsqueeze(-1) * out["attn_contribution"].unsqueeze(1)
    out["attn_residual"] = residual
    h = hc(refs["c"], w, p + "hc_ffn", residual)
    out["ffn_mixed"], out["ffn_inject"] = h["mixed"], h["injection"]
    out["moe"], out["moe_ids"] = mlp(w, p, h["mixed"])
    out["output"] = residual + out["ffn_inject"].unsqueeze(-1) * out["moe"].unsqueeze(1)
    return out


def load_refs(source=SOURCE):
    return {"c": rec.load_upstream(source), "d": att.load_reference(Path(source))}


def forward(gguf, tokens, refs=None, layers=None):
    """Whole-prefix graph over the first `layers` blocks (all by default)."""
    refs, w = refs or load_refs(), Weights(gguf)
    c = w.hf
    layers = c["num_hidden_layers"] if layers is None else layers
    embed = w.matrix("token_embd.weight")[np.asarray(tokens)]
    residual = torch.from_numpy(embed).unsqueeze(1).repeat(1, c["hc_count"], 1)
    trace = {"embedding": torch.from_numpy(embed), "layers": []}
    for index in range(layers):
        trace["layers"].append(layer(refs, w, index, tokens, residual))
        residual = trace["layers"][-1]["output"]
    trace["final"] = hc(refs["c"], w, "output_hc", residual, inject=False)["mixed"]
    trace["logits"] = linear(w.matrix("output.weight"), trace["final"].numpy())
    return trace


def flatten(trace):
    flat = {k: v for k, v in trace.items() if k != "layers"}
    for i, layer_trace in enumerate(trace["layers"]):
        flat.update({f"layer{i}.{k}": v for k, v in layer_trace.items()})
    return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in flat.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with torch.no_grad():
        np.savez(args.output, tokens=np.asarray(args.tokens, np.int32), **flatten(forward(args.gguf, args.tokens, layers=args.layers)))
    print(args.output)
