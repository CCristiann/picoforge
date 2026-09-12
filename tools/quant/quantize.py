"""quantize.py — write a quantised checkpoint the engine can mmap.

    tools/venv/bin/python tools/quant/quantize.py models/Qwen3-0.6B q4_g32 [--embed]

Output is models/<name>-<fmt>[-embed]/: the same config and tokenizer files,
and a model.safetensors in which every projection X.weight is replaced by

    X.qweight   I8 [N, K]      8-bit codes, or
                U8 [N, K/2]    4-bit codes, two per byte, LOW nibble first,
                               two's complement (what TensorOps reads, measured
                               by tools/probe/int4_probe)
    X.scales    BF16 [N, K/G]  one scale per block of G input columns

Staying inside safetensors keeps the engine's loader, its bounds checks and
the zero-copy Metal binding exactly as they are. Nothing needs to say which
format a file is in: I8 vs U8 gives the bits, K / scales.shape[1] the block.

The writer streams. The header is computed from shapes alone, then tensors are
quantised and written one at a time, so peak memory is one tensor, not one
model. On 0.6B that is a nicety; on a 61 GB MoE it is the only way it works.

lm_head.weight is dropped when the config ties it: it is a byte-identical copy
of the embedding the engine never reads, and 311 MB of it.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formats import parse, quantize  # noqa: E402

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def bf16_raw_to_f32(raw: np.ndarray) -> np.ndarray:
    """bf16 is the top half of fp32: widen by shifting, exactly as the engine does."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_raw(x: np.ndarray) -> np.ndarray:
    """Inverse of the above, for values already rounded to bf16 by formats.to_bf16.
    The low 16 bits must be zero; if not, something skipped the rounding."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("value is not representable in bf16")
    return (bits >> 16).astype(np.uint16)


def pack_int4(q: np.ndarray) -> np.ndarray:
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def plan(header: dict, fmt: str, embed: bool, tied: bool):
    """(out_name, dtype, shape, source_name, role) for every output tensor."""
    bits, _, group, _ = parse(fmt)
    out = []
    for name, meta in header.items():
        if name == "__metadata__" or (tied and name == "lm_head.weight"):
            continue
        n_shape = meta["shape"]
        quant = name.endswith("proj.weight") or (embed and name.endswith("embed_tokens.weight"))
        if not quant:
            out.append((name, meta["dtype"], n_shape, name, "copy"))
            continue
        n, k = n_shape
        g = group or k
        base = name[: -len(".weight")]
        out.append((base + ".qweight", "I8" if bits == 8 else "U8",
                    [n, k] if bits == 8 else [n, k // 2], name, "codes"))
        out.append((base + ".scales", "BF16", [n, k // g], name, "scales"))
    return out


def main() -> None:
    src_dir, fmt = Path(sys.argv[1]), sys.argv[2]
    embed = "--embed" in sys.argv
    if parse(fmt)[1] == "rowcol":
        sys.exit("rowcol is a sweep experiment, not a file format (it lost: see bench/)")
    dst_dir = src_dir.parent / f"{src_dir.name}-{fmt}{'-embed' if embed else ''}"
    dst_dir.mkdir(exist_ok=True)
    tied = json.loads((src_dir / "config.json").read_text()).get("tie_word_embeddings", False)

    src = src_dir / "model.safetensors"
    header, data_start = read_header(src)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    entries = plan(header, fmt, embed, tied)

    out_header, offset = {"__metadata__": {"picoforge.format": fmt + ("+embed" if embed else "")}}, 0
    for name, dtype, shape, _, _ in entries:
        size = int(np.prod(shape)) * DTYPE_BYTES[dtype]
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)                       # safetensors pads to 8

    with open(dst_dir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        cache = {}
        for name, dtype, shape, source, role in entries:
            meta = header[source]
            lo, hi = meta["data_offsets"]
            raw = mm[data_start + lo: data_start + hi]
            if role == "copy":
                f.write(raw.tobytes())
                continue
            if source not in cache:                        # codes then scales: one quantise
                w = bf16_raw_to_f32(raw.view(np.uint16)).reshape(meta["shape"])
                cache = {source: quantize(w, fmt)}
            parts = cache[source]
            if role == "codes":
                q = parts["q"]
                f.write((q if parse(fmt)[0] == 8 else pack_int4(q)).tobytes())
            else:
                f.write(f32_to_bf16_raw(parts["d"]).tobytes())
        assert f.tell() == 8 + len(blob) + offset, "wrote a different size than the header claims"

    for extra in src_dir.iterdir():
        if extra.is_file() and extra.name != "model.safetensors":
            shutil.copy2(extra, dst_dir / extra.name)
    print(f"{fmt}: {len(entries)} tensors, {(8 + len(blob) + offset) / 1e6:.1f} MB -> {dst_dir}")


if __name__ == "__main__":
    main()
