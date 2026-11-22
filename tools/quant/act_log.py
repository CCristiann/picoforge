"""act_log.py — read the expert inputs captured by the engine with QW4_ACT_LOG.

The engine writes one record per full-trunk append (a prefill chunk or a
decode token): u32 magic "QW4A", version 1, position, rows t, layers n, hidden
H, top-k K; then per captured layer: u32 layer index, x f32[t,H] (the routed
experts' input, the same vector the router reads), ids u32[t,K] (chosen
experts), g f32[t,K] (their gate weights, as the reduction uses them).
"""

import struct
from pathlib import Path

import numpy as np

MAGIC, VERSION = 0x41345751, 1


def read_act_log(path: Path) -> list:
    data = Path(path).read_bytes()
    records, at = [], 0
    while at < len(data):
        magic, version, pos, t, n, h, k = struct.unpack_from("<7I", data, at)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{path}: bad record header at byte {at}")
        at += 28
        layers = {}
        for _ in range(n):
            (layer,) = struct.unpack_from("<I", data, at); at += 4
            x = np.frombuffer(data, "<f4", t * h, at).reshape(t, h); at += 4 * t * h
            ids = np.frombuffer(data, "<u4", t * k, at).reshape(t, k); at += 4 * t * k
            g = np.frombuffer(data, "<f4", t * k, at).reshape(t, k); at += 4 * t * k
            layers[layer] = (x, ids, g)
        records.append({"pos": pos, "rows": t, "layers": layers})
    return records


def layer_rows(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All captured rows of one layer, in append order: (x [N,H], ids [N,K], g [N,K])."""
    parts = [r["layers"][layer] for r in read_act_log(path) if layer in r["layers"]]
    if not parts:
        raise KeyError(f"{path}: layer {layer} was not captured")
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
