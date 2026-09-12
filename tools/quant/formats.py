"""formats.py — the quantisers, in one place, shared by the quality sweep and
by the converter. What the sweep measures is exactly what the engine will run.

Every format is the same idea at a different granularity:

    W[n, k]  ~=  d[n, k // G] * q[n, k]          q integer, d a bf16 scale

G is the block length along K, the INPUT dimension, because that is the axis
the matmul sums over: a block is a contiguous run of one weight row, and the
TensorOps kernel can take it as a K-slice. G = K is one scale per row (the
"per-row" format), G = 32 is GGUF's Q8_0 / Q4_0 layout.

Rank-1 ("rowcol") adds a per-column scale c[k] divided out BEFORE the row
quantiser: W ~= d[n] * q[n, k] * c[k]. Because c depends only on k it can be
moved onto the activations, (x * c) . q, so it costs the kernel nothing.

Scales are rounded to bf16 BEFORE q is computed. The engine stores bf16, and
rounding afterwards would quantise against a scale the engine never sees.
"""

import numpy as np
import torch

QMAX = {8: 127, 4: 7}


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to the nearest bfloat16, returned as fp32. Exact representation of
    what the file will hold: bf16 widens to fp32 with no loss."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) \
                .to(torch.bfloat16).to(torch.float32).numpy()


def block_quant(w: np.ndarray, bits: int, group: int):
    """Symmetric round-to-nearest over blocks of `group` along K.

    Q8: d = max|w| / 127, q in [-127, 127].
    Q4: llama.cpp's Q4_0 trick. The signed extreme m maps to exactly -8,
        d = m / -8, q in [-8, 7]. A symmetric max/7 would leave one of the 16
        codes unused; this spends it on the side that holds the extreme.
    Returns q (int8, [N, K]) and d (fp32 holding bf16 values, [N, K//G]).
    """
    n, k = w.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the block length {group}")
    b = w.reshape(n, k // group, group).astype(np.float64)

    if bits == 8:
        d = np.abs(b).max(-1) / 127.0
    else:
        idx = np.abs(b).argmax(-1)
        extreme = np.take_along_axis(b, idx[..., None], -1)[..., 0]
        d = extreme / -8.0
    d = to_bf16(d)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d[..., None] != 0, np.rint(b / d[..., None].astype(np.float64)), 0.0)
    lo = -127 if bits == 8 else -8
    q = np.clip(q, lo, QMAX[bits]).astype(np.int8).reshape(n, k)
    return q, d


def column_scale(w: np.ndarray) -> np.ndarray:
    """c[k] = sqrt(max_n |W[n, k]|), in bf16. A weight-only heuristic, not a
    fit: it halves (in log terms) the spread between loud and quiet input
    columns before the row quantiser sees them. A column of zeros gets 1."""
    c = np.sqrt(np.abs(w).max(0))
    return to_bf16(np.where(c > 0, c, 1.0))


def quantize(w: np.ndarray, fmt: str) -> dict[str, np.ndarray]:
    """fmt is 'q8_g32', 'q4_g64', 'q4_row', 'q4_rowcol', ... Returns the arrays
    the file will hold: q, d and, for rowcol, c."""
    bits = int(fmt[1])
    layout = fmt.split("_", 1)[1]
    k = w.shape[1]
    if layout == "rowcol":
        c = column_scale(w)
        q, d = block_quant(w / c, bits, k)
        return {"q": q, "d": d, "c": c}
    group = k if layout == "row" else int(layout[1:])
    q, d = block_quant(w, bits, group)
    return {"q": q, "d": d}


def dequantize(parts: dict[str, np.ndarray]) -> np.ndarray:
    """The exact fp32 value the engine's arithmetic stands for. Exact because a
    bf16 scale has 8 significant bits and q at most 8, so d*q needs 16 and fp32
    has 24. (In bf16 it would NOT be exact — the reason the reference is fp32.)"""
    q, d = parts["q"], parts["d"]
    n, k = q.shape
    g = k // d.shape[1]
    w = (q.reshape(n, d.shape[1], g).astype(np.float32) * d[..., None]).reshape(n, k)
    if "c" in parts:
        w = w * parts["c"][None, :]
    return w
