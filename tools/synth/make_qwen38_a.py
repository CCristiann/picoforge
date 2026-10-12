"""Small GGUF schema fixture from the independent pinned HF-config oracle.

Writes sparse, zero payloads solely to exercise descriptors, never inference.
All generated files belong in the current worktree's build/qwen38-a directory.
"""
import copy
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_config as ref


def small_config():
    hf = json.loads((ROOT / "tests/fixtures/qwen38/a/config.json").read_text())
    c = hf["text_config"]
    c.update(hidden_size=256, vocab_size=512, hc_count=2, hc_lowrank=32,
             num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
             full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=32, indexer_n_heads=2, indexer_head_dim=32, indexer_budget=8,
             linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
             linear_num_value_heads=4, num_experts=4, num_experts_per_tok=2,
             moe_intermediate_size=64, shared_expert_intermediate_size=32,
             heads_per_ngram=2, ple_embed_dim=256, eos_token_id=511)
    c["rope_parameters"]["partial_rotary_factor"] = 0.5
    return hf


def string(value):
    b = value.encode("utf-8")
    return struct.pack("<Q", len(b))+b


def write_fixture(path, hf):
    meta = ref.reference_metadata(hf)
    meta.update({"general.architecture": "qwen4exp", "general.alignment": 16384,
                 ref.PREFIX+"ple.row_count": 128,
                 ref.PREFIX+"ple.layer_multipliers": [17, 31, 43],
                 ref.PREFIX+"ple.head_offsets": [0, 7, 14, 21],
                 ref.PREFIX+"ple.head_vocab_sizes": [7]*4,
                 "ds4.qwen4.down.logical_input": 64, "ds4.qwen4.down.physical_input": 256})
    tensors = ref.reference_tensors(hf, 128)
    header = bytearray(b"GGUF"+struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        if isinstance(value, str): kind, payload = 8, string(value)
        elif isinstance(value, list):
            kind, payload = 9, struct.pack("<IQ", 10, len(value))+struct.pack("<"+"Q"*len(value), *value)
        elif key.endswith(("freq_base", "epsilon")): kind, payload = 6, struct.pack("<f", value)
        else: kind, payload = 4, struct.pack("<I", value)
        header += string(key)+struct.pack("<I", kind)+payload
    offset = 0
    expected = []
    for name, (shape, fmt, domain) in tensors.items():
        type_id, block, size = ref.FORMATS[fmt]
        offset = (offset+16383)//16384*16384
        count = math.prod(shape)//block*size
        header += string(name)+struct.pack("<I", len(shape))+struct.pack("<"+"Q"*len(shape), *shape)
        header += struct.pack("<IQ", type_id, offset)
        expected.append(dict(name=name, shape=shape, type_id=type_id, format=fmt,
                             size_bytes=count, offset=offset, domain=domain))
        offset += count
    start = (len(header)+16383)//16384*16384
    for t in expected: t["file_offset"] = start+t.pop("offset")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.truncate(start+offset)
    path.with_suffix(".json").write_text(json.dumps(dict(metadata=meta, tensors=expected), indent=2)+"\n")
    return expected


if __name__ == "__main__":
    destination = ROOT / "build/qwen38-a/small.gguf"
    write_fixture(destination, small_config())
    print(destination)
