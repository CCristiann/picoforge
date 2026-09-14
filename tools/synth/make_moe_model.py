"""make_moe_model.py — a full-depth Qwen3-30B-A3B, random q8_row weights, for timing.

    ./tools/venv/bin/python tools/synth/make_moe_model.py build/synth-30b-q8 [LAYERS]

make_moe_layer.py answered what ONE MoE layer costs. End to end is a different
question -- 48 layers of attention over a real cache, a 151936-wide head, the
expert traffic of every layer at once, all inside the 41.75 GB buffer cap -- and
it only has an answer on the machine. This writes the real model's shapes
(config from the published file, reproduced in make_moe_layer.py) straight in
the engine's q8_row format, ~31 GB, streamed a tensor at a time.

The weights are noise, so the routing is noise: close to uniform, the worst case
for how many experts a verify step touches. Every timing taken on this file is
an upper bound for verify and exact for decode shapes; none of its outputs mean
anything. The C CPU path, the oracle for the GPU, is held to it on two tokens
before anything is timed.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_moe_layer import CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260914


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/synth-30b-q8"
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    cfg = dict(CONFIG, num_hidden_layers=layers, vocab_size=151936)
    H, Ie, E = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    hd, V = cfg["head_dim"], cfg["vocab_size"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd

    # (name, kind, shape): "q8" projections become .qweight + .scales.
    plan = [("model.embed_tokens.weight", "bf16", [V, H]), ("lm_head.weight", "bf16", [V, H]),
            ("model.norm.weight", "norm", [H])]
    for l in range(layers):
        p = f"model.layers.{l}"
        plan += [(f"{p}.input_layernorm.weight", "norm", [H]),
                 (f"{p}.post_attention_layernorm.weight", "norm", [H]),
                 (f"{p}.self_attn.q_norm.weight", "norm", [hd]),
                 (f"{p}.self_attn.k_norm.weight", "norm", [hd]),
                 (f"{p}.self_attn.q_proj", "q8", [q, H]), (f"{p}.self_attn.k_proj", "q8", [kv, H]),
                 (f"{p}.self_attn.v_proj", "q8", [kv, H]), (f"{p}.self_attn.o_proj", "q8", [H, q]),
                 (f"{p}.mlp.gate.weight", "bf16", [E, H])]
        for e in range(E):
            plan += [(f"{p}.mlp.experts.{e}.gate_proj", "q8", [Ie, H]),
                     (f"{p}.mlp.experts.{e}.up_proj", "q8", [Ie, H]),
                     (f"{p}.mlp.experts.{e}.down_proj", "q8", [H, Ie])]

    entries = []                                   # (name, dtype, shape, kind)
    for name, kind, shape in plan:
        if kind == "q8":
            entries += [(name + ".qweight", "I8", shape, "codes"),
                        (name + ".scales", "BF16", [shape[0], 1], "scales")]
        else:
            entries.append((name, "BF16", shape, kind))
    entries.sort()

    header, offset = {"__metadata__": {"picoforge.format": "q8_row", "synthetic": "random"}}, 0
    for name, dtype, shape, _ in entries:
        size = int(np.prod(shape)) * (1 if dtype == "I8" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    rng = np.random.default_rng(SEED)
    one = np.uint16(np.array([1.0], dtype=np.float32).view(np.uint32)[0] >> 16)
    out.mkdir(parents=True, exist_ok=True)

    def bf16(x: np.ndarray) -> bytes:
        return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()

    with open(out / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for name, dtype, shape, kind in entries:
            n = int(np.prod(shape))
            if kind == "codes":
                f.write(rng.integers(-127, 128, size=n, dtype=np.int8).tobytes())
            elif kind == "scales":
                # |w| up to 0.02 * sqrt(3): a uniform code times this scale has
                # the spread of the bf16 layer's 0.02 normal init.
                f.write(bf16(np.full(n, 0.02 * np.sqrt(3) / 127)))
            elif kind == "norm":
                f.write(np.full(n, one, dtype=np.uint16).tobytes())
            else:
                f.write(bf16(0.02 * rng.standard_normal(n, dtype=np.float32)))
        assert f.tell() == 8 + len(blob) + offset
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "generation_config.json"]:
        shutil.copy(ROOT / "models/Qwen3-0.6B" / fname, out / fname)
    print(f"{layers}-layer synthetic Qwen3-30B-A3B, q8_row -> {out} ({(8 + len(blob) + offset) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
