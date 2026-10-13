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
import qwen38_config as ref
from make_qwen38_a import small_config, string

ALIGN = 16384


def graph_config():
    hf = small_config()
    hf["text_config"].update(num_hidden_layers=4, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
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


def payload(rng, name, shape, fmt, domain):
    count = math.prod(shape)
    type_id, block, size = ref.FORMATS[fmt]
    if domain == "mtp" or fmt in ("Q4_K", "MXFP4"):
        return bytes(count // block * size)  # described, never executed
    s = scale_for(name, shape)
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


def write_graph(path, hf, seed):
    meta = ref.reference_metadata(hf)
    meta.update({"general.architecture": "qwen4exp", "general.alignment": ALIGN,
                 ref.PREFIX + "ple.row_count": 128,
                 ref.PREFIX + "ple.layer_multipliers": [17, 31, 43],
                 ref.PREFIX + "ple.head_offsets": [0, 7, 14, 21],
                 ref.PREFIX + "ple.head_vocab_sizes": [7] * 4,
                 "ds4.qwen4.down.logical_input": 64, "ds4.qwen4.down.physical_input": 256})
    ref.check_metadata(hf, meta)
    tensors = ref.reference_tensors(hf, 128)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        if isinstance(value, str): kind, data = 8, string(value)
        elif isinstance(value, list):
            kind, data = 9, struct.pack("<IQ", 10, len(value)) + struct.pack("<" + "Q" * len(value), *value)
        elif key.endswith(("freq_base", "epsilon")): kind, data = 6, struct.pack("<f", value)
        else: kind, data = 4, struct.pack("<I", value)
        header += string(key) + struct.pack("<I", kind) + data
    rng, offset, records, blobs = np.random.default_rng(seed), 0, [], []
    for name, (shape, fmt, domain) in tensors.items():
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        blob = payload(rng, name, shape, fmt, domain)
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
    manifest = dict(seed=seed, hf_text_config=hf["text_config"], metadata=meta, tensors=records)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/qwen38-g1/graph.gguf")
    parser.add_argument("--seed", type=int, default=3811)
    args = parser.parse_args()
    records = write_graph(args.output, graph_config(), args.seed)
    print(f"{args.output}: {len(records)} tensors, {args.output.stat().st_size} bytes")
