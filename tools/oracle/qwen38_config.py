"""CPU schema oracle from the pinned HF text_config, without importing a model.

This checks the selected GGUF recipe, not arbitrary Qwen variants. HF is the
independent source for dimensions; PLE hash arrays receive structural checks
only, pending E's independent hash oracle. No tensor payload is read here.
"""
import argparse
import json
import math
from pathlib import Path
import struct

PREFIX = "qwen4exp."
FIELDS = {
    "embedding_length": "hidden_size", "vocab_size": "vocab_size",
    "context_length": "max_position_embeddings",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim", "attention.value_length": "head_dim",
    "attention.indexer.head_count": "indexer_n_heads",
    "attention.indexer.key_length": "indexer_head_dim",
    "attention.indexer.top_k": "indexer_budget",
    "expert_count": "num_experts", "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_key_head_dim", "ssm.group_count": "linear_num_key_heads",
    "ssm.time_step_rank": "linear_num_value_heads",
    "full_attention_interval": "full_attention_interval",
    "hyper_connection.count": "hc_count", "hyper_connection.low_rank": "hc_lowrank",
    "ple.ngram_size": "ngram_size", "ple.heads_per_ngram": "heads_per_ngram",
    "ple.conv_kernel": "ple_conv_kernel_size", "ple.eos_token_id": "eos_token_id",
    "ple.vocab_base": "ngram_vocab_size_base",
    "ple.vocab_divisor": "make_ngram_vocab_size_divisible_by",
}


def require(condition, subject):
    if not condition:
        raise ValueError(subject)


def positive(value, subject):
    require(type(value) is int and 0 < value < 2**32, subject)
    return value


def same_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def reference_metadata(hf):
    """Derive dimensions from HF, deliberately never searching nested keys."""
    require(hf["model_type"] == "qwen4_exp", "HF architecture")
    c = hf["text_config"]
    require(c["model_type"] == "qwen4_exp_text", "HF text architecture")
    values = {key: positive(c[field], field) for key, field in FIELDS.items()}
    layers = c["layer_types"]
    require(len(layers) == c["num_hidden_layers"], "HF layer_types length")
    require(all(x in ("linear_attention", "full_attention") for x in layers), "HF layer kind")
    mtp = c["mtp"]["layer_types"]
    require(len(mtp) == c["mtp_num_hidden_layers"] == c["mtp"]["num_hidden_layers"], "HF MTP length")
    require(all(x == "full_attention" for x in mtp), "HF MTP kind")
    heads = (c["ngram_size"] - 1) * c["heads_per_ngram"]
    require(heads > 0 and c["ple_embed_dim"] % heads == 0, "PLE embedding division")
    rotary = c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"]
    require(rotary == int(rotary) and rotary > 0 and int(rotary) % 2 == 0, "RoPE dimension")
    values.update({
        "block_count": len(layers) + len(mtp), "nextn_predict_layers": len(mtp),
        "ssm.inner_size": c["linear_num_value_heads"] * c["linear_value_head_dim"],
        "rope.dimension_count": int(rotary),
        "rope.freq_base": c["rope_parameters"]["rope_theta"],
        "attention.layer_norm_rms_epsilon": c["rms_norm_eps"],
        "attention.compress_ratios": [0 if x == "linear_attention" else c["indexer_compress_ratio"]
                                      for x in layers + mtp],
        "ple.layers": [i - 1 for i in c["ple_layer_ids"]],
        "embedding_length_per_layer_input": c["ple_embed_dim"] // heads,
        "ple.row_dimension": c["ple_embed_dim"] // heads,
    })
    return {PREFIX + k: v for k, v in values.items()}


def check_metadata(hf, metadata):
    expected = reference_metadata(hf)
    require(metadata["general.architecture"] == "qwen4exp", "GGUF architecture")
    floats = {PREFIX + "rope.freq_base", PREFIX + "attention.layer_norm_rms_epsilon"}
    for key, value in expected.items():
        actual = metadata[key]
        if key in floats:
            require(struct.pack("<f", actual) == struct.pack("<f", value), key)
        else:
            require(same_value(actual, value), key)
    c = hf["text_config"]
    offsets = metadata[PREFIX + "ple.head_offsets"]
    sizes = metadata[PREFIX + "ple.head_vocab_sizes"]
    heads = (c["ngram_size"] - 1) * c["heads_per_ngram"]
    require(len(offsets) == len(sizes) == heads, "PLE head array lengths")
    multipliers = metadata[PREFIX + "ple.layer_multipliers"]
    require(len(multipliers) == c["ngram_size"] and all(type(x) is int and 0 < x < 2**64
                                                     for x in multipliers), "PLE multipliers")
    end = 0
    for offset, size in zip(offsets, sizes):
        require(type(offset) is int and offset == end, "PLE head offset")
        end += positive(size, "PLE head vocabulary")
        require(end < 2**64, "PLE row overflow")
    divisor = c["make_ngram_vocab_size_divisible_by"]
    rows = positive(metadata[PREFIX + "ple.row_count"], "PLE row count")
    require(rows == (end + divisor - 1) // divisor * divisor, "PLE physical padding")
    logical = c["moe_intermediate_size"]
    require(metadata["ds4.qwen4.down.logical_input"] == logical, "down logical input")
    require(metadata["ds4.qwen4.down.physical_input"] == (logical + 255) // 256 * 256,
            "down physical padding")
    return {"metadata_fields": len(expected), "ordinary_layers": c["num_hidden_layers"],
            "mtp_layers": c["mtp_num_hidden_layers"], "ple_hash_rows": end,
            "ple_physical_rows": rows, "ple_padding_rows": rows - end}


# GGUF block layout, distinct from picoforge's q4_row/q8_row formats.
FORMATS = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q8_0": (8, 32, 34),
           "Q2_K": (10, 256, 84), "Q4_K": (12, 256, 144),
           "IQ2_XXS": (16, 256, 66), "BF16": (30, 1, 2), "MXFP4": (39, 32, 17)}


def reference_tensors(hf, rows):
    """Shapes from HF, names/packing from the pinned GGUF conversion schema.

    This is a binding contract, not an independent numerical graph oracle.
    rows is structurally checked metadata; E must verify the n-gram hashes.
    """
    c = hf["text_config"]
    e, v, h, r = (c[k] for k in ("hidden_size", "vocab_size", "hc_count", "hc_lowrank"))
    x, f, fs, ne = e*h, c["moe_intermediate_size"], c["shared_expert_intermediate_size"], c["num_experts"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    gv, gc = hv*dv, 2*hk*dk+hv*dv
    q, kv = c["num_attention_heads"]*c["head_dim"], c["num_key_value_heads"]*c["head_dim"]
    di, hi = c["indexer_head_dim"], c["indexer_n_heads"]
    ngram_heads = (c["ngram_size"]-1)*c["heads_per_ngram"]
    head_dim = c["ple_embed_dim"]//ngram_heads
    result = {}

    def add(prefix, name, shape, fmt, domain):
        result[prefix+name] = (shape, fmt, domain)

    def hc(prefix, name, domain, inject=False):
        add(prefix, name+"_norm.weight", [x], "F32", domain)
        add(prefix, name+"_down.weight", [x, r], "F16", domain)
        add(prefix, name+"_up.weight", [r, x], "F16", domain)
        if inject:
            add(prefix, name+"_inject.weight", [x, h], "F16", domain)

    add("", "token_embd.weight", [e, v], "BF16", "ordinary")
    add("", "output.weight", [e, v], "Q8_0", "ordinary")
    hc("", "output_hc", "ordinary")
    add("", "per_layer_token_embd.weight", [head_dim, rows], "BF16", "disk_ple")
    for layer, kind in enumerate(c["layer_types"]+c["mtp"]["layer_types"]):
        prefix = f"blk.{layer}."
        mtp = layer >= c["num_hidden_layers"]
        domain = "mtp" if mtp else "ordinary"
        hc(prefix, "hc_attn", domain, True)
        hc(prefix, "hc_ffn", domain, True)
        group = {
            "ffn_gate_inp.weight": ([e, ne], "F32"),
            "ffn_gate_inp_shexp.weight": ([e], "F32"),
            "ffn_down_shexp.weight": ([fs, e], "Q8_0"),
            "ffn_down_exps.weight": ([f if mtp else (f+255)//256*256, e, ne], "MXFP4" if mtp else "Q2_K"),
        }
        for part in ("gate", "up"):
            group[f"ffn_{part}_exps.weight"] = ([e, f, ne], "Q4_K" if mtp else "IQ2_XXS")
            group[f"ffn_{part}_shexp.weight"] = ([e, fs], "Q8_0")
        if kind == "linear_attention":
            group.update({"attn_qkv.weight": ([e, gc], "Q8_0"), "attn_gate.weight": ([e, gv], "Q8_0"),
                          "ssm_out.weight": ([gv, e], "Q8_0"), "ssm_a": ([hv], "F32"),
                          "ssm_dt.bias": ([hv], "F32"), "ssm_norm.weight": ([dv], "F32"),
                          "ssm_conv1d.weight": ([c["linear_conv_kernel_dim"], gc], "F32")})
            for name in ("alpha", "beta"):
                group[f"ssm_{name}.weight"] = ([e, hv], "F32")
        else:
            group.update({"attn_q.weight": ([e, 2*q], "Q8_0"), "attn_output.weight": ([q, e], "Q8_0"),
                          "indexer.q_proj.weight": ([e, hi*di], "Q8_0"),
                          "indexer.k_proj.weight": ([e, c["indexer_kv_heads"]*di], "Q8_0")})
            for name in ("k", "v"):
                group[f"attn_{name}.weight"] = ([e, kv], "Q8_0")
            for name in ("q", "k"):
                group[f"attn_{name}_norm.weight"] = ([c["head_dim"]], "F32")
                group[f"indexer.{name}_norm.weight"] = ([di], "F32")
        if layer+1 in c["ple_layer_ids"] and not mtp:
            group.update({"ple_key.weight": ([c["ple_embed_dim"], x], "Q8_0"),
                          "ple_value.weight": ([c["ple_embed_dim"], e], "Q8_0"),
                          "ple_conv1d.weight": ([c["ple_conv_kernel_size"], x], "F32")})
            for name in ("key", "query", "conv"):
                group[f"ple_norm_{name}.weight"] = ([x], "F32")
        if mtp:
            group.update({"nextn.eh_proj.weight": ([2*e, e], "Q8_0"),
                          "nextn.enorm.weight": ([e], "F32"), "nextn.hnorm.weight": ([x], "F32")})
            hc(prefix, "nextn.hc_head", domain)
        for name, (shape, fmt) in group.items():
            add(prefix, name, shape, fmt, domain)
    return result


def check_tensors(hf, metadata, tensors):
    expected = reference_tensors(hf, metadata[PREFIX+"ple.row_count"])
    actual = {t["name"]: t for t in tensors}
    require(len(actual) == len(tensors), "duplicate tensor name")
    require(actual.keys() == expected.keys(), "tensor name set: " + str(sorted(actual.keys() ^ expected.keys())[:4]))
    counts = {"ordinary": 0, "mtp": 0, "disk_ple": 0}
    for name, (shape, fmt, domain) in expected.items():
        t = actual[name]
        require(t["shape"] == shape and t["format"] == fmt, name+": shape/format")
        type_id, block, size = FORMATS[fmt]
        require(shape[0] % block == 0, name+": block alignment")
        require(t["type_id"] == type_id and t["size_bytes"] == math.prod(shape)//block*size, name+": storage")
        counts[domain] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("inventory", type=Path, help="metadata-only inventory JSON")
    args = parser.parse_args()
    hf, inv = (json.loads(p.read_text()) for p in (args.config, args.inventory))
    result = check_metadata(hf, inv["metadata"])
    result["tensor_domains"] = check_tensors(hf, inv["metadata"], inv["tensors"])
    print(json.dumps(result, indent=2, allow_nan=False))
