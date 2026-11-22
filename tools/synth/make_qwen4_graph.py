"""Tiny NON-degenerate qwen4exp GGUF for the G1 CPU graph.

A's fixture writes zero payloads to test descriptors only. This one keeps A's
schema (names, formats, shapes from the pinned HF config) and fills every
ordinary tensor with seeded, bounded values, so that every stage of the
48-layer recipe runs on finite, varied numbers: 3 GDN layers + 1 QSA layer
(the real 3,7,...,47 period), PLE on layer 1, MTP described but never run.

Packed formats are written directly as valid blocks: we pick the f16 scales
and fill the codes with random bytes. Dequantization stays B's (and the
DwarfStar oracle's) job; this script never decodes anything.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/oracle"))
sys.path.insert(0, str(ROOT / "tools/synth"))
import qwen4_config as ref
from make_qwen4_small import small_config, string

ALIGN = 16384


def graph_config():
    hf = small_config()
    hf["text_config"].update(num_hidden_layers=4, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    return hf


def variant_config():
    """The upstream defaults Flash-Next overrides, on the same tiny graph: no PLE
    (ple_layer_ids empty), the SiLU GDN gate (output_gate_type unset) and an
    explicit layer_types list with the QSA layer second, off the every-4th pattern."""
    hf = graph_config()
    c = hf["text_config"]
    c.update(ple_layer_ids=[], layer_types=["linear_attention", "full_attention"] + ["linear_attention"] * 2)
    c.pop("output_gate_type"); c.pop("full_attention_interval")
    return hf


def scale_for(name, shape):
    """Typical magnitude per role: norms near 1, products near unit variance."""
    if name.endswith("_norm.weight") or "norm_" in name:
        return None                       # gains are drawn in [0.75, 1.25]
    if name.endswith("token_embd.weight"):
        return 1.0
    if name.endswith(("conv1d.weight", "ssm_dt.bias")):
        return 0.5
    if "_up.weight" in name and name.startswith(("blk.", "output_hc")) and "hc_" in name:
        return 2.0 / math.sqrt(shape[0])  # spread the HC sigmoid gates
    if name.endswith("ffn_gate_inp.weight"):
        return 2.0 / math.sqrt(shape[0])  # spread router probabilities
    return 1.0 / math.sqrt(shape[0])      # GGUF dim 0 is the input width


def payload(rng, name, shape, fmt, domain, enable_mtp=False):
    count = math.prod(shape)
    type_id, block, size = ref.FORMATS[fmt]
    if domain == "mtp" and not enable_mtp:
        return bytes(count // block * size)  # described, never executed
    s = scale_for(name, shape)
    if name.endswith(("nextn.enorm.weight", "nextn.hnorm.weight")):
        s = None
    if name.endswith("ssm_a"):
        values = -np.exp(rng.uniform(-1.0, 0.5, count))  # already -exp(A_log)
    elif s is None:
        values = rng.uniform(0.75, 1.25, count)  # exact as 1+w for the HF oracle
    else:
        values = rng.standard_normal(count) * s
    if fmt == "F32":
        return values.astype("<f4").tobytes()
    if fmt == "F16":
        return values.astype("<f2").tobytes()
    if fmt == "BF16":
        return (values.astype("<f4").view("<u4") >> 16).astype("<u2").tobytes()
    blocks = count // block
    if fmt == "Q4_K":
        d = np.full((blocks, 1), s / 7, "<f2")
        dmin = np.full((blocks, 1), s * 7.5 / 7, "<f2")
        scales = np.tile(np.array([1]*8+[17]*4, dtype=np.uint8), (blocks, 1))
        codes = rng.integers(0, 256, (blocks, 128), dtype=np.uint8)
        return np.concatenate([d.view(np.uint8), dmin.view(np.uint8), scales, codes], 1).tobytes()
    if fmt == "MXFP4":
        exponent = np.full((blocks, 1), round(math.log2(s / 2.5)) + 127, dtype=np.uint8)
        codes = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
        return np.concatenate([exponent, codes], 1).tobytes()
    if fmt == "Q8_0":
        d = np.full((blocks, 1), s / 73.0, "<f2")  # std of uniform int8 is ~73
        q = rng.integers(-127, 128, (blocks, 32), dtype=np.int8)
        return np.concatenate([d.view(np.uint8), q.view(np.uint8)], 1).tobytes()
    if fmt == "Q2_K":
        codes = rng.integers(0, 256, (blocks, 80), dtype=np.uint8)
        d = np.full((blocks, 1), s / 11.0, "<f2")      # mean scale*q is ~11.25
        dmin = np.full((blocks, 1), 1.5 * s / 11.0, "<f2")  # cancels the mean
        return np.concatenate([codes, d.view(np.uint8), dmin.view(np.uint8)], 1).tobytes()
    if fmt == "IQ2_XXS":
        d = np.full((blocks, 1), s / 100.0, "<f2")     # grid*factor is ~100
        codes = rng.integers(0, 256, (blocks, 64), dtype=np.uint8)
        return np.concatenate([d.view(np.uint8), codes], 1).tobytes()
    raise ValueError(f"no generator for {fmt}")


def mix_experts(tensors, hf, hot, seed):
    """Picoforge-only mixed layout on every trunk layer: `hot` experts move to Q4 tensors
    (Q4_K gate/up, MXFP4 down at the logical input), the rest stay in the profile's tensors,
    and blk.L.ffn_exps_map.weight (F32 [E], exact integers) says where each one is:
    v >= 0 cold slot v, v < 0 hot slot -1-v. Returns tensors and {map name: values}."""
    c = hf["text_config"]
    ne, e, f = c["num_experts"], c["hidden_size"], c["moe_intermediate_size"]
    out, maps = dict(tensors), {}
    for name, (shape, fmt, domain) in tensors.items():
        if domain == "mtp" or not name.endswith("ffn_down_exps.weight"):
            continue
        layer = int(name.split(".")[1])
        chosen = set(np.random.default_rng(seed + 1000 + layer).permutation(ne)[:hot].tolist())
        cold_slot, hot_slot, values = 0, 0, []
        for x in range(ne):
            if x in chosen: values.append(-1 - hot_slot); hot_slot += 1
            else: values.append(cold_slot); cold_slot += 1
        p = f"blk.{layer}."
        for part in ("gate", "up", "down"):
            s0, f0, d0 = tensors[p + f"ffn_{part}_exps.weight"]
            out[p + f"ffn_{part}_exps.weight"] = (s0[:2] + [ne - hot], f0, d0)
        for part in ("gate", "up"):
            out[p + f"ffn_{part}_exps_hot.weight"] = ([e, f, hot], "Q4_K", domain)
        out[p + "ffn_down_exps_hot.weight"] = ([f, e, hot], "MXFP4", domain)
        out[p + "ffn_exps_map.weight"] = ([ne], "F32", domain)
        maps[p + "ffn_exps_map.weight"] = np.array(values, "<f4").tobytes()
    return out, maps


def write_graph(path, hf, seed, text_meta=None, enable_mtp=False, mixed_hot=0):
    meta = ref.reference_metadata(hf)
    ple = bool(hf["text_config"]["ple_layer_ids"])
    meta.update({"general.architecture": "qwen4exp", "general.alignment": ALIGN,
                 "ds4.qwen4.down.logical_input": 64, "ds4.qwen4.down.physical_input": 256})
    if ple:
        meta.update({ref.PREFIX + "ple.row_count": 128,
                     ref.PREFIX + "ple.layer_multipliers": [17, 31, 43],
                     ref.PREFIX + "ple.head_offsets": [0, 7, 14, 21],
                     ref.PREFIX + "ple.head_vocab_sizes": [7] * 4})
    ref.check_metadata(hf, meta)
    meta.update(text_meta or {})
    tensors, maps = ref.reference_tensors(hf, 128 if ple else 0), {}
    if mixed_hot:
        meta["picoforge.experts.mixed"] = True
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        if isinstance(value, str): kind, data = 8, string(value)
        elif isinstance(value, bool): kind, data = 7, struct.pack("<?", value)
        elif key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            kind, data = 9, struct.pack("<IQ", 8, len(value)) + b"".join(map(string, value))
        elif key == "tokenizer.ggml.token_type":
            kind, data = 9, struct.pack("<IQ", 5, len(value)) + struct.pack("<" + "i" * len(value), *value)
        elif isinstance(value, list):
            kind, data = 9, struct.pack("<IQ", 10, len(value)) + struct.pack("<" + "Q" * len(value), *value)
        elif key.endswith(("freq_base", "epsilon")): kind, data = 6, struct.pack("<f", value)
        else: kind, data = 4, struct.pack("<I", value)
        header += string(key) + struct.pack("<I", kind) + data
    # Every payload is drawn in the original order, so a mixed file shares all its other bytes
    # (and its cold experts) with the plain one; the hot experts get their own generator.
    rng = np.random.default_rng(seed)
    drawn = {name: payload(rng, name, shape, fmt, domain, enable_mtp) for name, (shape, fmt, domain) in tensors.items()}
    if mixed_hot:
        mixed, maps = mix_experts(tensors, hf, mixed_hot, seed)
        hot_rng = np.random.default_rng(seed + 7)
        for name, (shape, fmt, domain) in mixed.items():
            if name in maps:
                drawn[name] = maps[name]
            elif name.endswith("_exps_hot.weight"):
                drawn[name] = payload(hot_rng, name, shape, fmt, domain, enable_mtp)
            elif name.endswith("_exps.weight") and shape != tensors[name][0]:
                p, ne = name.rsplit(".ffn_", 1)[0] + ".", tensors[name][0][2]
                per = len(drawn[name]) // ne
                cold = [x for x, v in enumerate(np.frombuffer(maps[p + "ffn_exps_map.weight"], "<f4")) if v >= 0]
                drawn[name] = b"".join(drawn[name][x * per:(x + 1) * per] for x in cold)
        tensors = mixed
        header[8:16] = struct.pack("<Q", len(tensors))   # the count was written before the split
    offset, records, blobs = 0, [], []
    for name, (shape, fmt, domain) in tensors.items():
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        blob = drawn[name]
        header += string(name) + struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", ref.FORMATS[fmt][0], offset)
        records.append(dict(name=name, shape=shape, format=fmt, type_id=ref.FORMATS[fmt][0],
                            domain=domain, offset=offset, size_bytes=len(blob)))
        blobs.append(blob)
        offset += len(blob)
    start = (len(header) + ALIGN - 1) // ALIGN * ALIGN
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header + bytes(start - len(header)))
        for record, blob in zip(records, blobs):
            record["file_offset"] = start + record.pop("offset")
            f.seek(record["file_offset"])
            f.write(blob)
    manifest = dict(seed=seed, mtp_executable=enable_mtp, hf_text_config=hf["text_config"], metadata=meta, tensors=records)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/qwen4-g1/graph.gguf")
    parser.add_argument("--seed", type=int, default=3811)
    parser.add_argument("--variant", action="store_true", help="upstream defaults: no PLE, SiLU gate, QSA second")
    parser.add_argument("--mixed-hot", type=int, default=0, help="picoforge mixed experts: this many Q4 experts per trunk layer")
    args = parser.parse_args()
    records = write_graph(args.output, variant_config() if args.variant else graph_config(), args.seed,
                          mixed_hot=args.mixed_hot)
    print(f"{args.output}: {len(records)} tensors, {args.output.stat().st_size} bytes")
