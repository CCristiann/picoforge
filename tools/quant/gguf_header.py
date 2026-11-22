"""gguf_header.py — read a GGUF header (metadata and tensor table) without touching the data.

Enough for the quantization tools: imatrix files and the published Q2 used as a
template. Values are decoded fully except arrays of numbers, which are kept as
NumPy arrays.
"""

import struct
from pathlib import Path

import numpy as np

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        return self.f.read(self.unpack("<Q")).decode("utf-8")

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.unpack("<I"), self.unpack("<Q")
            if et == 8:
                return [self.string() for _ in range(n)]
            if et == 9:
                return [self.value(9) for _ in range(n)]
            dt = np.dtype(_SCALAR[et])
            return np.frombuffer(self.f.read(dt.itemsize * n), dt).copy()
        return self.unpack(_SCALAR[t])


def read_header(path: Path) -> tuple[dict, dict]:
    """Returns (metadata, tensors); tensors: name -> (ggml type, dims, absolute data offset)."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.unpack("<I")
        if version != 3:
            raise ValueError(f"{path}: GGUF version {version}, expected 3")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        table = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            table.append((name, r.unpack("<I"), dims, r.unpack("<Q")))
        align = int(meta.get("general.alignment", 32))
        base = (f.tell() + align - 1) // align * align
    return meta, {name: (t, dims, base + off) for name, t, dims, off in table}
