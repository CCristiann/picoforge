"""Bounded, metadata-only inventory for the selected qwen4exp GGUF v3.

No tensor payload is mapped/read. --expected-size permits inspecting a download
prefix; it does NOT validate the absent payload or authenticate the checkpoint.
Layout: ggml-org/ggml docs/gguf.md; block sizes checked against ds4 fc80bd6.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct

FORMATS = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
           8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
           12: ("Q4_K", 256, 144), 16: ("IQ2_XXS", 256, 66),
           30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Header:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        if size < 0 or self.stream.tell() + size > 128 * 2**20:
            raise ValueError("header exceeds the 128 MiB inspection budget")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        self.digest.update(data)
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        length = self.number("Q")
        if length > 16 * 2**20:
            raise ValueError("oversized GGUF string")
        return self.read(length).decode("utf-8")

    def value(self, kind):
        if kind in SCALARS:
            return self.number(SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind != 9:
            raise ValueError(f"unsupported metadata type {kind}")
        element, count = self.number("I"), self.number("Q")
        if element not in (*SCALARS, 8) or count > 2_000_000:
            raise ValueError("invalid or oversized metadata array")
        # Retain small architectural arrays; tokenizer arrays are still checked
        # and hashed, but never materialized as a second vocabulary in memory.
        values = []
        for _ in range(count):
            value = self.value(element)
            if count <= 64:
                values.append(value)
        return values if count <= 64 else {"element_type": element, "count": count}


def inspect(path, expected_size=None):
    actual_size = path.stat().st_size
    file_size = actual_size if expected_size is None else expected_size
    if file_size < actual_size:
        raise ValueError("file exceeds expected size")
    with path.open("rb") as stream:
        h = Header(stream)
        if h.read(4) != b"GGUF" or h.number("I") != 3:
            raise ValueError("expected little-endian GGUF v3")
        count, kv_count = h.number("Q"), h.number("Q")
        if not 0 < count <= 100_000 or kv_count > 100_000:
            raise ValueError("invalid GGUF counts")
        metadata = {}
        for _ in range(kv_count):
            key = h.string()
            if key in metadata:
                raise ValueError("duplicate metadata key")
            metadata[key] = h.value(h.number("I"))
        if metadata.get("general.architecture") != "qwen4exp":
            raise ValueError("expected qwen4exp architecture")
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("invalid alignment")
        tensors, names = [], set()
        for _ in range(count):
            name, rank = h.string(), h.number("I")
            if name in names or not 1 <= rank <= 4:
                raise ValueError("duplicate tensor or invalid rank")
            names.add(name)
            dims = [h.number("Q") for _ in range(rank)]
            kind, offset = h.number("I"), h.number("Q")
            if kind not in FORMATS:
                raise ValueError(f"unsupported tensor type {kind}")
            fmt, block, size = FORMATS[kind]
            if not all(dims) or dims[0] % block or offset % alignment:
                raise ValueError(f"invalid layout: {name}")
            size = math.prod(dims) // block * size
            tensors.append(dict(name=name, shape=dims, type_id=kind,
                                format=fmt, offset=offset, size_bytes=size))
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        h.read(data_start - stream.tell())
        header_hash = h.digest.hexdigest()
    end, totals = data_start, collections.Counter()
    for tensor in sorted(tensors, key=lambda t: t["offset"]):
        start = data_start + tensor["offset"]
        if start < end or start + tensor["size_bytes"] > file_size:
            raise ValueError(f"overlapping/out-of-file tensor: {tensor['name']}")
        tensor["file_offset"] = start
        end = start + tensor["size_bytes"]
        totals[tensor["format"]] += tensor["size_bytes"]
    return dict(schema_version=1, actual_file_bytes=actual_size, expected_file_bytes=file_size,
                payload_verified=False, header_sha256=header_hash, data_offset=data_start,
                metadata=metadata, bytes_by_format=dict(totals), tensors=tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-size", type=int)
    args = parser.parse_args()
    print(json.dumps(inspect(args.path, args.expected_size), indent=2, allow_nan=False))
