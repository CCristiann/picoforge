"""Fast GPU kernels against the oracle kernel and an FP64 reference (R1.1).

Cases: the real GGUF rows in the --slices folders (several layers/experts per
format, extracted once by qwen38-test-assets: the checkpoint is never opened)
plus seeded synthetic shapes. The CPU decoder's rows give the FP64 product r
and S = sum|w*x| per output. A fast kernel must meet both conditions:
  (a) |y - r| <= gamma(2K+8) * S + 2 ulp32(r)       (holds for any sum order)
  (b) |y_fast - y_oracle| / max(S, 2^-126) <= tol[kernel]
with tol from tests/qwen38_kernel_tolerances.json. Run through the GPU lock.
Formats in FAST must have a fast kernel (UNSUPPORTED fails); the Review Focus
tests (constraints.md 1-5) and the scale-decode test exercise them directly.
R1.5: the formats in INDEXED_FAST have an indexed (routed expert) fast kernel,
measured against q38_packed_linear_indexed by the *_indexed Review Focus
tests, test_expert_id_bounds and test_<format>_decode_matches_oracle.
R1.6a: q38c_rms_fast (RMS and the gated GDN norm, input check folded in)
against check + q38c_rms through q38c_gpu_encode, tolerance entry rms_fast.
R1.8: q38c_gdn_prepare_fast (L2 norms, value copy, gates, input check folded
in) against check + q38c_gdn_prepare, tolerance entry gdn_prepare_fast.
R2.1: split decode attention (q38d_attend_encode, family attention) against the
q38d_attend_gpu oracle and FP64, tolerance entry attend_fast.
R2.2: exact index scores, rank and compact: bit-identical to their oracles.
R2.4: q38c_gdn_scan_fast (family fuse) against check + q38c_gdn_scan and FP64,
tolerance entry gdn_scan_fast.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest

import numpy as np

import test_qwen38_packed as packed  # same oracle module and seeded fixtures

ROOT = Path(__file__).resolve().parents[1]
oracle = packed.oracle
NAMES = {0: "f32", 1: "f16", 8: "q8_0", 10: "q2_k", 12: "q4_k", 16: "iq2_xxs", 30: "bf16", 39: "mxfp4"}
SHAPES = [(1, 32, 1), (3, 640, 3), (33, 2560, 2), (512, 640, 1)]  # (N, K logical, M)
UNSUPPORTED, NONFINITE = 8, 16  # Q38_UNSUPPORTED (no fast kernel), Q38_NONFINITE (error bit 1)
FAST = {"q8_0", "f16", "f32"}  # formats whose fast kernel exists: UNSUPPORTED is a failure for them
KINDS = {name: kind for kind, name in NAMES.items() if name in FAST}
INDEXED_FAST = {"iq2_xxs", "q2_k", "q4_k", "mxfp4"}  # R1.5: formats with an indexed (routed expert) fast kernel
INDEXED_KINDS = {name: kind for kind, name in NAMES.items() if name in INDEXED_FAST}
GPU_ERROR = 20  # Q38_GPU_ERROR: error word without bit 1 (bounds, bit 2)
BANNER = re.compile(rb"(?m)^.*Metal API Validation Enabled\r?\n")
NATIVE, SLICES, TOLERANCES, CASES, ORACLE, RESULTS, CORPUS = None, [], {}, [], {}, {}, {}


def run_native(mode, kind, physical, logical, data, x, env=None, view=()):
    """(exit code, stdout, stderr without the validation banner, N); view = (xpitch, ypitch, offset)."""
    n = len(data) // (physical // oracle.FORMATS[kind][0] * oracle.FORMATS[kind][1])
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    result = subprocess.run([str(NATIVE), mode, str(kind), str(physical), str(logical), str(n), str(len(x)),
                             *map(str, view)], input=data + x.astype("<f4").tobytes(), capture_output=True,
                            timeout=60, pass_fds=(int(lock),) if lock else (), env=environment)
    return result.returncode, result.stdout, BANNER.sub(b"", result.stderr), n


def native(mode, kind, physical, logical, data, x, env=None, view=()):
    """(status, y [M,N], CPU-decoded rows [N,physical]); status 8 has no data."""
    code, stdout, stderr, n = run_native(mode, kind, physical, logical, data, x, env, view)
    if code == UNSUPPORTED and not stdout and not stderr:
        return UNSUPPORTED, None, None
    if code or stderr:
        raise AssertionError(f"{mode} kind {kind}: exit {code}, {stderr!r}")
    values = np.frombuffer(stdout, dtype="<f4")
    if values.size != len(x) * n + n * physical:
        raise AssertionError("invalid native transport")
    return 0, values[:len(x) * n].reshape(len(x), n), values[len(x) * n:].reshape(n, physical)


def cases():
    """(label, kind, physical, logical, packed bytes, x, saved decode or None)."""
    rng = np.random.default_rng(3813)
    for directory in SLICES:
        manifest, groups = json.loads((directory / "manifest.json").read_text()), {}
        for record in manifest["records"]:
            blobs = {name: (directory / info["file"]).read_bytes() for name, info in record["artifacts"].items()}
            if any(oracle.digest(blobs[name]) != info["sha256"] for name, info in record["artifacts"].items()):
                raise ValueError(f"slice checksum mismatch in {directory}")
            groups.setdefault((record["k_physical"], record["k_logical"]), []).append(blobs)
        for (physical, logical), rows in groups.items():
            # Rows of one shape form one matrix: every x row meets every weight
            # row, a superset of the saved products. The saved inputs (impulses,
            # sixteenths) often sum exactly, so 3 seeded dense rows are added.
            x = np.concatenate([np.frombuffer(b["input.f32"], "<f4").reshape(-1, logical) for b in rows] +
                               [rng.uniform(-1, 1, (3, logical)).astype(np.float32)])
            saved = np.stack([np.frombuffer(b["decoded.f32"], "<f4") for b in rows])
            yield (f"real:{NAMES[manifest['kind']]}:N{len(rows)}:K{logical}/{physical}:M{len(x)}",
                   manifest["kind"], physical, logical, b"".join(b["packed"] for b in rows), x, saved)
    for kind, name in NAMES.items():
        for n, logical, m in SHAPES:
            physical = -(-logical // oracle.FORMATS[kind][0]) * oracle.FORMATS[kind][0]
            x = rng.uniform(-1, 1, (m, logical)).astype(np.float32)
            yield (f"synthetic:{name}:N{n}:K{logical}/{physical}:M{m}", kind, physical, logical,
                   packed.packed_fixture(kind, physical, n, rng), x, None)


def reference(case):
    """Forced-oracle output, decoded rows, FP64 product and S (cached per case)."""
    label, kind, physical, logical, data, x, _ = case
    if label not in ORACLE:
        _, y, decoded = native("oracle", kind, physical, logical, data, x)
        ORACLE[label] = (y, decoded, *oracle.matmul_reference(decoded, x, logical))
    return ORACLE[label]


def against_fp64(y, ref, magnitude, logical):
    """Condition (a) and the relative error |y - r| / S of one case."""
    n, u = 2 * logical + 8, 2.0**-24
    bound = n * u / (1 - n * u) * magnitude + 2 * np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64)
    delta = np.abs(y.astype(np.float64) - ref)
    relative = delta / np.maximum(magnitude, 2.0**-126)
    return dict(outputs=int(y.size), finite=bool(np.isfinite(y).all()), condition_a=bool((delta <= bound).all()),
                max_abs=float(delta.max()), max_rel=float(relative.max()), p50_rel=float(np.percentile(relative, 50)),
                p99_rel=float(np.percentile(relative, 99)), max_bound_fraction=float((delta / bound).max()))


def fp64(decoded, x, logical):
    """FP64 product and S = sum|w*x| over the logical columns only (padding may be nonfinite)."""
    a, w = x.astype(np.float64), decoded[:, :logical].astype(np.float64)
    return a @ w.T, np.abs(a) @ np.abs(w.T)


def q8_0_rows(physical, n, rng):
    """Q8_0 rows [n*blocks, 34]: random codes, finite fp16 scales of both signs over 2^-14..2^-4."""
    raw = rng.integers(0, 256, (n * physical // 32, 34), dtype=np.uint8)
    scales = (rng.choice([-1, 1], len(raw)) * 2.0 ** rng.uniform(-14, -4, len(raw))).astype("<f2")
    raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    return raw


def q8_0_decode(raw, physical):
    """Independent Q8_0 decode [n, physical]: d*code, exact in FP32 (11 x 8 significant bits)."""
    blocks = raw.reshape(-1, 34)
    w = blocks[:, :2].copy().view("<f2").astype(np.float32) * blocks[:, 2:].view(np.int8).astype(np.float32)
    return w.reshape(-1, physical)


DENSE = {1: "<f2", 0: "<f4"}  # F16 and F32: one element per column, no blocks


def weight_rows(kind, physical, n, rng):
    """Packed rows [n, row bytes] of one fast format. F16/F32: values of both signs over
    2^-16..2^0 (F16 subnormals included), so products and sums exercise every exponent."""
    if kind == 8:
        return q8_0_rows(physical, n, rng).reshape(n, -1)
    values = rng.choice([-1, 1], (n, physical)) * 2.0 ** rng.uniform(-16, 0, (n, physical))
    return values.astype(DENSE[kind]).view(np.uint8).reshape(n, -1)


def decode_rows(kind, raw, physical):
    """Independent decode [n, physical] of weight_rows' layout (nonfinite padding allowed)."""
    if kind == 8:
        return q8_0_decode(raw, physical)
    return raw.copy().view(DENSE[kind]).astype(np.float32).reshape(-1, physical)


def poison(kind, raw, row, column, value):
    """Copy of raw with the weight that feeds (row, column) set to value: the Q8_0 block scale or the element."""
    bad = raw.copy()
    if kind == 8:
        bad[row, column // 32 * 34:column // 32 * 34 + 2] = np.array([value], "<f2").view(np.uint8)
    else:
        size = np.dtype(DENSE[kind]).itemsize
        bad[row, column * size:(column + 1) * size] = np.array([value], DENSE[kind]).view(np.uint8)
    return bad


def seeded(seed, kind):
    """Per-format generator; Q8_0 keeps its original seed so its corpus (and tol) is unchanged."""
    return np.random.default_rng(seed if kind == 8 else [seed, kind])


def scaled_difference(y, y_oracle, magnitude):
    """Condition (b)'s measured side: |y - y_oracle| / max(S, 2^-126)."""
    return np.abs(y.astype(np.float64) - y_oracle) / np.maximum(magnitude, 2.0**-126)


def against_oracle(y, y_oracle, magnitude):
    scaled = scaled_difference(y, y_oracle, magnitude)
    return dict(max=float(scaled.max()), p99=float(np.percentile(scaled, 99)))


MULTI_MAX, MULTI_NOUT = 8, (1, 4, 48, 128, 512, 640)  # Q38_MULTI_MAX; n_out of the multi test


def run_multi(projections, x, env=None, view=None):
    """One q38_packed_encode_multi call; projections [(kind, physical, n, packed)], view (xpitch, ypad, offset).
    (exit code, [y_i [M, n_i]] or None, stderr without the validation banner, dispatches encoded)."""
    (m, logical), lock = x.shape, os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    result = subprocess.run([str(NATIVE), "multi", str(m), *map(str, view or (logical, 0, 0)),
                             *(f"{kind}:{physical}:{logical}:{n}" for kind, physical, n, _ in projections)],
                            input=b"".join(p[3] for p in projections) + x.astype("<f4").tobytes(), capture_output=True,
                            timeout=120, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr:
        return result.returncode, None, stderr, None
    values, sizes = np.frombuffer(result.stdout[4:], "<f4"), [m * p[2] for p in projections]
    if values.size != sum(sizes):
        raise AssertionError("invalid native transport")
    dispatches = int(np.frombuffer(result.stdout[:4], "<u4")[0])
    return 0, [v.reshape(m, -1) for v in np.split(values, np.cumsum(sizes)[:-1])], b"", dispatches


def real_projections(kind, logical, names):
    """The real slice rows of one format and K as projections, one per GGUF tensor whose name
    contains one of `names` (3 rows each, as extracted by qwen38-test-assets)."""
    tensors = {}
    for directory in SLICES:
        manifest = json.loads((directory / "manifest.json").read_text())
        for record in manifest["records"]:
            if (manifest["kind"] == kind and record["k_logical"] == logical == record["k_physical"] and
                    any(name in record["tensor"] for name in names)):
                tensors.setdefault(record["tensor"], []).append((directory / record["artifacts"]["packed"]["file"]).read_bytes())
    return [(kind, logical, len(rows), b"".join(rows)) for rows in tensors.values()]


def run_indexed(mode, kind, physical, logical, n, experts, data, ids, x, divisor, env=None, view=()):
    """One q38_packed_encode_indexed call: (exit code, y [rows, n] or None, CPU-decoded rows
    [experts*n, physical] or None, stderr without the banner). y is also returned when the
    error word is set (exit 16/20): an element the kernel did not write is still NaN."""
    rows, lock = len(ids), os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    result = subprocess.run([str(NATIVE), "indexed", mode, str(kind), str(physical), str(logical), str(n), str(experts),
                             str(rows), str(divisor), *map(str, view)],
                            input=data + np.asarray(ids, "<u4").tobytes() + x.astype("<f4").tobytes(),
                            capture_output=True, timeout=120, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr, values = BANNER.sub(b"", result.stderr), np.frombuffer(result.stdout, "<f4")
    if result.returncode == UNSUPPORTED and not values.size:
        return UNSUPPORTED, None, None, stderr
    if values.size not in (rows * n, rows * n + experts * n * physical):
        raise AssertionError(f"{mode} kind {kind}: exit {result.returncode}, {stderr!r}")
    decoded = values[rows * n:].reshape(experts * n, physical) if values.size > rows * n else None
    return result.returncode, values[:rows * n].reshape(rows, n), decoded, stderr


def indexed_fp64(decoded, ids, x, divisor, logical, n):
    """FP64 product and S of the indexed contract: row p uses expert ids[p] and x[p // divisor]."""
    ids, rows = np.asarray(ids), np.arange(len(ids))
    a = x.astype(np.float64)[rows // divisor]
    ref, magnitude = np.zeros((len(ids), n)), np.zeros((len(ids), n))
    for expert in np.unique(ids):
        chosen, w = ids == expert, decoded[expert * n:(expert + 1) * n, :logical].astype(np.float64)
        ref[chosen], magnitude[chosen] = a[chosen] @ w.T, np.abs(a[chosen]) @ np.abs(w.T)
    return ref, magnitude


SCALES = {16: (0,), 10: (80, 82), 12: (0, 2)}  # byte offsets of the fp16 d (and dmin) in a block


def quant_rows(kind, physical, count, rng):
    """Packed rows [count, row bytes] of a block format: random codes, finite scales of both
    signs over 2^-14..2^-4 (fp16 d/dmin) or 2^-12..2^0 (MXFP4 exponent)."""
    block, size = oracle.FORMATS[kind]
    raw = rng.integers(0, 256, (count * physical // block, size), dtype=np.uint8)
    if kind == 39:
        raw[:, 0] = rng.integers(115, 128, len(raw))
    for at in SCALES.get(kind, ()):
        scales = (rng.choice([-1, 1], len(raw)) * 2.0 ** rng.uniform(-14, -4, len(raw))).astype("<f2")
        raw[:, at:at + 2] = scales.view(np.uint8).reshape(-1, 2)
    return raw.reshape(count, -1)


def poison_block(kind, raw, row, column, value):
    """Copy of raw with the scale of the block holding (row, column) set to a nonfinite value."""
    block, size = oracle.FORMATS[kind]
    bad, at = raw.copy(), column // block * size
    if kind == 39:
        bad[row, at] = 255  # the E8M0 escape: Inf or NaN
    else:
        offset = SCALES[kind][0]
        bad[row, at + offset:at + offset + 2] = np.array([value], "<f2").view(np.uint8)
    return bad



def run_gate_up(mode, kinds, physical, logical, n, experts, gate, up, ids, x, divisor, env=None, view=()):
    """One fused gate+up+SwiGLU call (fused) or the unfused sequence (sequence: indexed fast
    kernels; oracle: indexed oracle), kinds = gate kind or (gate, up): (exit code, act [rows, n]
    or None, CPU-decoded gate rows, up rows (or None), stderr). act is also returned when the
    error word is set (an element the kernel did not write is still NaN)."""
    kinds = kinds if isinstance(kinds, tuple) else (kinds, kinds)
    kind = str(kinds[0]) if kinds[0] == kinds[1] else f"{kinds[0]}/{kinds[1]}"
    rows, lock = len(ids), os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    result = subprocess.run([str(NATIVE), "gateup", mode, kind, str(physical), str(logical), str(n), str(experts),
                             str(rows), str(divisor), *map(str, view)],
                            input=gate + up + np.asarray(ids, "<u4").tobytes() + x.astype("<f4").tobytes(),
                            capture_output=True, timeout=120, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr, values = BANNER.sub(b"", result.stderr), np.frombuffer(result.stdout, "<f4")
    if result.returncode == UNSUPPORTED and not values.size:
        return UNSUPPORTED, None, None, None, stderr
    if values.size not in (rows * n, rows * n + 2 * experts * n * physical):
        raise AssertionError(f"gateup {mode} kind {kind}: exit {result.returncode}, {stderr!r}")
    decoded = values[rows * n:].reshape(2, experts * n, physical) if values.size > rows * n else (None, None)
    return result.returncode, values[:rows * n].reshape(rows, n), decoded[0], decoded[1], stderr


def gate_up_fp64(gate, up, ids, x, divisor, logical, n):
    """FP64 act = silu(g) * u; its scale S = |silu'(g)| S_g |u| + |silu(g)| S_u, the first-order
    effect of the gate and up dot errors (each bounded by its S), the S of conditions (a) and (b);
    and the GPU's flush below 2^-126: exp(g) for g below about -87 (silu off by up to |g| 2^-126),
    then the product: 2^-126 ((|g| + 1) |u| + 1)."""
    g, sg = indexed_fp64(gate, ids, x, divisor, logical, n)
    u, su = indexed_fp64(up, ids, x, divisor, logical, n)
    sigma = 1 / (1 + np.exp(np.minimum(-g, 700)))  # g < -700: sigma 1e-304, not an overflow
    silu, slope = g * sigma, sigma * (1 + g * (1 - sigma))
    return silu * u, np.abs(slope) * sg * np.abs(u) + np.abs(silu) * su, 2.0**-126 * ((np.abs(g) + 1) * np.abs(u) + 1)


def gate_up_against_fp64(act, ref, magnitude, flush, logical):
    """Condition (a) for act: gamma(2K+8) S + 8 ulp32(r) + flush. S covers the two dots, the
    8 ulp silu's exp and division and the product's rounding, flush the GPU's subnormal flush."""
    n, u = 2 * logical + 8, 2.0**-24
    bound = n * u / (1 - n * u) * magnitude + 8 * np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64)
    bound += flush
    delta = np.abs(act.astype(np.float64) - ref)
    return dict(outputs=int(act.size), finite=bool(np.isfinite(act).all()), condition_a=bool((delta <= bound).all()),
                max_rel=float((delta / np.maximum(magnitude, 2.0**-126)).max()),
                max_bound_fraction=float((delta / bound).max()))


def run_down_reduce(mode, kind, physical, logical, n, experts, data, ids, act, scores, shared, sigmoid, env=None, view=()):
    """One fused down+reduce call (fused) or the unfused sequence (sequence: indexed fast down, then
    q38f_reduce; oracle: indexed oracle down): (exit code, y [t, n] or None, routed [t, n] or None,
    CPU-decoded rows or None, stderr). y and routed are also returned when the error word is set
    (an element the kernel did not write is still NaN). Q38_REF_KERNELS is never inherited."""
    (tokens, slots), lock = scores.shape, os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    payload = [data, np.asarray(ids, "<u4").tobytes()] + [np.asarray(a, "<f4").tobytes() for a in (act, scores, shared, sigmoid)]
    result = subprocess.run([str(NATIVE), "downreduce", mode, str(kind), str(physical), str(logical), str(n), str(experts),
                             str(tokens), str(slots), *map(str, view)], input=b"".join(payload),
                            capture_output=True, timeout=120, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr, values = BANNER.sub(b"", result.stderr), np.frombuffer(result.stdout, "<f4")
    if result.returncode == UNSUPPORTED and not values.size:
        return UNSUPPORTED, None, None, None, stderr
    out = 2 * tokens * n
    if values.size not in (out, out + experts * n * physical):
        raise AssertionError(f"downreduce {mode} kind {kind}: exit {result.returncode}, {stderr!r}")
    decoded = values[out:].reshape(experts * n, physical) if values.size > out else None
    return result.returncode, values[:out // 2].reshape(tokens, n), values[out // 2:out].reshape(tokens, n), decoded, stderr


def down_reduce_fp64(decoded, ids, act, scores, shared, sigmoid, logical, n):
    """FP64 routed = sum_s w_s d_s and y = routed + sigma shared, d_s the down dot of pair t*k+s;
    their S (sum_s |w_s| S_s, plus |sigma shared| for y) and the per-dot ulp term
    2 sum_s |w_s| ulp32(d_s) of condition (a)."""
    tokens, slots = scores.shape
    d, sd = indexed_fp64(decoded, ids, act, 1, logical, n)
    w = scores.astype(np.float64)[..., None]
    d, sd = d.reshape(tokens, slots, n), sd.reshape(tokens, slots, n)
    routed, s_routed = (w * d).sum(1), (np.abs(w) * sd).sum(1)
    term = sigmoid.astype(np.float64)[:, None] * shared.astype(np.float64)
    ulps = 2 * (np.abs(w) * np.spacing(np.abs(d.astype(np.float32))).astype(np.float64)).sum(1)
    return (routed, s_routed, ulps), (routed + term, s_routed + np.abs(term), ulps)


def down_reduce_against_fp64(y, ref, magnitude, ulps, logical, slots):
    """Condition (a) for routed or y: each dot within gamma(2K+8) S_s + 2 ulp32(d_s) (B's bound),
    then k+1 products and sums: gamma(2K+8+2k+2) S + 2 ulp32(r) + the dots' ulp terms."""
    m, u = 2 * logical + 8 + 2 * slots + 2, 2.0**-24
    bound = m * u / (1 - m * u) * magnitude + 2 * np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64) + ulps
    delta = np.abs(y.astype(np.float64) - ref)
    return dict(outputs=int(y.size), finite=bool(np.isfinite(y).all()), condition_a=bool((delta <= bound).all()),
                max_rel=float((delta / np.maximum(magnitude, 2.0**-126)).max()),
                max_bound_fraction=float((delta / bound).max()))


def down_inputs(rng, tokens, slots, logical, n, scale=1.0):
    """act [t*k, logical], scores [t, k] (positive, normalized per token), shared [t, n], sigmoid [t]."""
    scores = rng.uniform(0.05, 1, (tokens, slots))
    return (rng.uniform(-scale, scale, (tokens * slots, logical)).astype(np.float32),
            (scores / scores.sum(1, keepdims=True)).astype(np.float32),
            rng.uniform(-1, 1, (tokens, n)).astype(np.float32), rng.uniform(0, 1, tokens).astype(np.float32))


def run_route(mode, logits, slots, norm, experts=None, env=None):
    """q38_moe_route_fast (fast) or q38f_route (oracle) on logits [tokens, pitch], the first `experts`
    (default: all) columns being the router row: (error word, ids [t, k], weights [t, k], probs [t, e]).
    Q38_REF_KERNELS is never inherited (set it through env)."""
    (tokens, pitch), lock = logits.shape, os.environ.get("PICOFORGE_GPU_LOCK_FD")
    experts = pitch if experts is None else experts
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    result = subprocess.run([str(NATIVE), "route", mode, str(experts), str(slots), str(tokens), str(norm), str(pitch)],
                            input=logits.astype("<f4").tobytes(), capture_output=True, timeout=120,
                            pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    pairs, out = tokens * slots, result.stdout
    if result.returncode or stderr or len(out) != 4 * (1 + 2 * pairs + tokens * experts):
        raise AssertionError(f"route {mode}: exit {result.returncode}, {stderr!r}")
    word = int(np.frombuffer(out[:4], "<u4")[0])
    ids = np.frombuffer(out[4:4 + 4 * pairs], "<u4").reshape(tokens, slots)
    weights = np.frombuffer(out[4 + 4 * pairs:4 + 8 * pairs], "<f4").reshape(tokens, slots)
    return word, ids, weights, np.frombuffer(out[4 + 8 * pairs:], "<f4").reshape(tokens, experts)


ROUTE_PAD = 5  # columns of NaN past the experts in a padded router row (pitch = experts + ROUTE_PAD)
ROUTE_KINDS = 11  # row shapes of route_logits


def route_logits(rng, tokens, experts, offset=0):
    """Router logits [tokens, experts] FP32: row i has shape (i + offset) % ROUTE_KINDS, the first ones plain
    normals, the rest built to make exact ties of p (the (larger p, smaller id) rule) common."""
    rows = []
    for i in range(tokens):
        kind, z = (i + offset) % ROUTE_KINDS, rng.normal(0, 1, experts)
        pick = rng.permutation(experts)
        if kind == 1:
            z = z * 10                                   # wide: exp underflows for most ids
        elif kind == 2:
            z = np.round(z * 2) / 2                      # a few distinct values: ties everywhere
        elif kind == 3:
            z = np.full(experts, 0.5)                    # all equal
        elif kind == 4:
            z = np.where(rng.random(experts) < 0.5, -0.0, 0.0)  # +0 / -0 mixed
        elif kind == 5:                                  # 9 distinct leaders, then 12 equal ones across the k-th place
            z = np.full(experts, -2.0)
            z[pick[:9]] = 3 + np.arange(len(pick[:9]))
            z[pick[9:21]] = 2.0
        elif kind == 6:
            z[pick[:5]] = z.max() + 1                    # the maximum duplicated across ids
        elif kind == 7:
            z = np.full(experts, -1e30)                  # p = 0 for all but 3: the rest are exact zero ties
            z[pick[:3]] = rng.normal(0, 1, min(3, experts))
        elif kind == 8:
            z = np.arange(experts) * 0.01                # ramp up (max last)
        elif kind == 9:
            z = -np.arange(experts) * 0.01               # ramp down (max first)
        elif kind == 10:                                 # neighbours in ULPs of 1.0
            z = np.float32(1.0) + rng.integers(-2, 3, experts) * np.float32(2 ** -23)
        rows.append(z.astype(np.float32))
    return np.stack(rows)


def route_reference_ids(probs, slots):
    """The oracle's rule on given probabilities: larger p first, equal p by smaller id."""
    return np.stack([np.lexsort((np.arange(len(p)), -p.astype(np.float64)))[:slots] for p in probs]).astype("<u4")


# ---- R1.6a parallel RMSNorm: q38c_rms_fast against check + q38c_rms, through q38c_gpu_encode ----
RMS_EPS = float(np.float32(1e-6))  # the harness's spec.eps and gdn.rms_eps (an FP32 value)
RMS_DISPATCHES = {"oracle": (2, 2), "fast": (1, 1)}  # (dispatches, barriers) of one encode


def run_rms(mode, x, gain, z=None, pitch=None, offset=0, inplace=False, initial=0, env=None):
    """RMS (z None) or GDN_NORM on x [t, groups, width], gain [width] or [groups, width]:
    (error word, dispatches, barriers, y [t, groups, width]); y is returned even with the word set."""
    t, groups, width = x.shape
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    data = b"".join(np.ascontiguousarray(a, "<f4").tobytes() for a in (x, gain, z) if a is not None)
    result = subprocess.run([str(NATIVE), "rms", mode, str(int(z is not None)), str(t), str(groups), str(width),
                             str(gain.ndim), str(pitch or width), str(offset), str(int(inplace)), str(initial)],
                            input=data, capture_output=True, timeout=60, pass_fds=(int(lock),) if lock else (),
                            env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr or len(result.stdout) != 12 + 4 * x.size:
        raise AssertionError(f"rms {mode}: exit {result.returncode}, {stderr!r}")
    word, dispatches, barriers = (int(v) for v in np.frombuffer(result.stdout[:12], "<u4"))
    return word, dispatches, barriers, np.frombuffer(result.stdout[12:], "<f4").reshape(x.shape)


def rms_fp64(x, gain, z=None):
    """FP64 r = x / sqrt(sum(x^2) / width + eps) * gain (* sigmoid(z)) and S = |r|."""
    a, g = x.astype(np.float64), gain.astype(np.float64)
    r = a / np.sqrt((a * a).sum(-1, keepdims=True) / a.shape[-1] + RMS_EPS) * g
    if z is not None:
        r = r / (1 + np.exp(-z.astype(np.float64)))
    return r, np.abs(r)


def rms_against_fp64(y, ref, magnitude, gated):
    """Condition (a) for RMS: |y - r| <= gamma(width + c) S + 2 ulp32(r), S = |r|. Any order of the width
    nonnegative squares and their sum is within gamma(width) of the exact sum; mean and eps add 2 roundings,
    sqrt halves the relative error, 1/sqrt and the two products add a few more: c = 16 covers them with
    margin (a few ulp of sqrt/division); gated (GDN), sigmoid(z) via exp (MSL precise: 4 ulp), 1 + e and
    the division add c = 16 more."""
    n, u = y.shape[-1] + (32 if gated else 16), 2.0**-24
    bound = n * u / (1 - n * u) * magnitude + 2 * np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64)
    delta = np.abs(y.astype(np.float64) - ref)
    return dict(outputs=int(y.size), finite=bool(np.isfinite(y).all()), condition_a=bool((delta <= bound).all()),
                max_rel=float((delta / np.maximum(magnitude, 2.0**-126)).max()),
                max_bound_fraction=float((delta / bound).max()))


RMS_ROW_SCALES = (1.0, 1e-10, 30.0, 1e-20, 1e3)  # plain, below eps, wide, squares that flush, large


def rms_inputs(rng, t, groups, width, gain_rank=1, gated=False, gain=None):
    """x [t, groups, width] (row r scaled by RMS_ROW_SCALES[r % 5], one outlier x100 in every third row),
    gain (the given real one, else N(1, 0.3)) and, gated, z ~ N(0, 3)."""
    rows = t * groups
    x = rng.normal(0, 1, (rows, width)) * np.array(RMS_ROW_SCALES)[np.arange(rows) % 5][:, None]
    x[::3, rng.integers(width)] *= 100
    if gain is None:
        gain = rng.normal(1, 0.3, width if gain_rank == 1 else (groups, width))
    z = rng.normal(0, 3, (t, groups, width)).astype(np.float32) if gated else None
    return x.reshape(t, groups, width).astype(np.float32), np.asarray(gain, np.float32), z


def real_gain(name):
    """A real F32 norm gain from the --slices folders (the decoded row of `name`)."""
    for directory in SLICES:
        for record in json.loads((directory / "manifest.json").read_text())["records"]:
            if record["tensor"] == name:
                return np.fromfile(directory / record["artifacts"]["decoded.f32"]["file"], "<f4")
    raise AssertionError(f"{name}: no slice (run qwen38-test-assets)")


# ---- R1.8 parallel GDN prepare: q38c_gdn_prepare_fast against check + q38c_gdn_prepare ----
L2_EPS = float(np.float32(1e-6))  # the harness's gdn.l2_eps
GDN_REAL = (16, 48, 128, 128)  # (hk, hv, dk, dv): ssm.group_count, ssm.time_step_rank, ssm.state_size (docs/specs/qwen38/C.md)


def run_prep(mode, inputs, dims, gap=0, offset=0, inplace=False, initial=0, env=None):
    """inputs = (qkv [t, C], alpha [t, hv], br [t, hv], A [hv], dt [hv]), dims = (hk, hv, dk, dv):
    (error word, dispatches, barriers, prepared, log_g, beta); outputs are returned even with the word set."""
    qkv = inputs[0]
    t, hv = qkv.shape[0], dims[1]
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    result = subprocess.run([str(NATIVE), "prep", mode, str(t), *map(str, dims), str(gap), str(offset),
                             str(int(inplace)), str(initial)],
                            input=b"".join(np.ascontiguousarray(a, "<f4").tobytes() for a in inputs),
                            capture_output=True, timeout=60, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr or len(result.stdout) != 12 + 4 * (qkv.size + 2 * t * hv):
        raise AssertionError(f"prep {mode}: exit {result.returncode}, {stderr!r}")
    word, dispatches, barriers = (int(v) for v in np.frombuffer(result.stdout[:12], "<u4"))
    out = np.frombuffer(result.stdout[12:], "<f4")
    return (word, dispatches, barriers, out[:qkv.size].reshape(qkv.shape),
            out[qkv.size:qkv.size + t * hv].reshape(t, hv), out[qkv.size + t * hv:].reshape(t, hv))


# ---- R2.1 split decode attention: q38d_attend_encode fast against the q38d_attend_gpu oracle ----
ATTEND_REAL = (24, 2, 256)  # (heads, kv heads, head dim): qwen4exp.attention.head_count(_kv), key_length


def run_attend(mode, q, gate, k, v, ids, env=None):
    """q, gate [h, d], k, v [n, j, d], ids [m]: (error word, dispatches, out [h, d])."""
    (h, d), (n, j, _) = q.shape, k.shape
    environment = {key: val for key, val in os.environ.items() if key != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    result = subprocess.run([str(NATIVE), "attend", mode, *map(str, (h, j, d, n, len(ids)))],
                            input=b"".join(np.ascontiguousarray(a, "<f4").tobytes() for a in (q, gate, k, v)) +
                            np.asarray(ids, "<u4").tobytes(), capture_output=True, timeout=60,
                            pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr or len(result.stdout) != 8 + 4 * h * d:
        raise AssertionError(f"attend {mode}: exit {result.returncode}, {stderr!r}")
    word, dispatches = (int(x) for x in np.frombuffer(result.stdout[:8], "<u4"))
    return word, dispatches, np.frombuffer(result.stdout[8:], "<f4").reshape(h, d)


def run_hctail(shape, inject=True, poison=False, env=None):
    """R2.6: the native hctail mode (fused HC tail and the fast sequence on the same seeded inputs):
    {status, equal, words, dispatches} parsed from its one line."""
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    result = subprocess.run([str(NATIVE), "hctail", *map(str, shape), str(int(inject)), str(int(poison))],
                            capture_output=True, timeout=60, pass_fds=(int(lock),) if lock else (), env=environment)
    fields = result.stdout.decode().split()
    if result.returncode or len(fields) != 10:
        raise AssertionError(f"hctail: exit {result.returncode}, {result.stderr!r}")
    return dict(status=int(fields[1]), equal=int(fields[3]), words=(int(fields[5]), int(fields[6])),
                dispatches=tuple(map(int, fields[-2:])))


def run_scan(mode, qkv, g, beta, state, dims, initial=0):
    """dims = (hk, hv, dk, dv): (error word, dispatches, out [t, hv, dv], state [hv, dv, dk])."""
    hk, hv, dk, dv = dims
    t = len(qkv)
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {key: val for key, val in os.environ.items() if key != "Q38_REF_KERNELS"}
    result = subprocess.run([str(NATIVE), "scan", mode, *map(str, (t, *dims, initial))],
                            input=b"".join(np.ascontiguousarray(a, "<f4").tobytes() for a in (qkv, g, beta, state)),
                            capture_output=True, timeout=60, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr or len(result.stdout) != 8 + 4 * (t * hv * dv + hv * dv * dk):
        raise AssertionError(f"scan {mode}: exit {result.returncode}, {stderr!r}")
    word, dispatches = (int(x) for x in np.frombuffer(result.stdout[:8], "<u4"))
    out = np.frombuffer(result.stdout[8:], "<f4")
    return word, dispatches, out[:t * hv * dv].reshape(t, hv, dv), out[t * hv * dv:].reshape(hv, dv, dk)


def scan_inputs(rng, t, dims):
    """As after q38c_gdn_prepare: q, k unit L2 rows (q / sqrt(dk)), v ~ N(0, 1), g = log decay in [-3, 0],
    beta in (0, 1), state ~ N(0, 0.5) with one x50 row."""
    hk, hv, dk, dv = dims
    qk = rng.normal(0, 1, (t, 2 * hk, dk))
    qk /= np.linalg.norm(qk, axis=-1, keepdims=True)
    qk[:, :hk] /= np.sqrt(dk)
    qkv = np.concatenate([qk.reshape(t, -1), rng.normal(0, 1, (t, hv * dv))], 1)
    state = rng.normal(0, 0.5, (hv, dv, dk))
    state[rng.integers(hv), rng.integers(dv)] *= 50
    return tuple(a.astype(np.float32) for a in (qkv, -rng.uniform(0, 3, (t, hv)), rng.uniform(0, 1, (t, hv)), state))


def scan_fp64(qkv, g, beta, state, dims):
    """One token (t = 1) in FP64 from the FP32 inputs: out r [hv, dv], state r_s [hv, dv, dk], magnitudes S
    (sum_j |s'_j q_j| for out, |s g| + |delta k_j| + |k_j| |beta| P per state element) and condition (a) bounds. Errors in any
    order: s1 = s*exp(g) within gamma(6) (exp 4 ulp + product); u = sum s1 k within gamma(dk+6) P, P =
    sum|s g k|; delta = (v - u) beta within E_d = |beta| (gamma(dk+6) P + gamma(2)(|v| + |u|)); s2 = s1 +
    delta k within e_j = gamma(6)|s g| + |k_j| E_d + gamma(2)(|s g| + |delta k_j|); o = sum s2 q within
    gamma(dk) sum|s2 q| + sum |q_j| e_j."""
    hk, hv, dk, dv = dims
    gam = lambda n: n * 2.0**-24 / (1 - n * 2.0**-24)
    x = qkv[0].astype(np.float64)
    q = x[:hk * dk].reshape(hk, dk)[np.arange(hv) % hk][:, None, :]           # [hv, 1, dk]
    k = x[hk * dk:2 * hk * dk].reshape(hk, dk)[np.arange(hv) % hk][:, None, :]
    v = x[2 * hk * dk:].reshape(hv, dv)
    s = state.astype(np.float64) * np.exp(g[0].astype(np.float64))[:, None, None]
    u = (s * k).sum(-1)
    P = np.abs(s * k).sum(-1)
    delta = (v - u) * beta[0].astype(np.float64)[:, None]
    s2 = s + delta[..., None] * k
    out = (s2 * q).sum(-1)
    e_d = np.abs(beta[0].astype(np.float64))[:, None] * (gam(dk + 6) * P + gam(2) * (np.abs(v) + np.abs(u)))
    e_s = gam(6) * np.abs(s) + np.abs(k) * e_d[..., None] + gam(2) * (np.abs(s) + np.abs(delta[..., None] * k))
    out_bound = gam(dk) * np.abs(s2 * q).sum(-1) + (np.abs(q) * e_s).sum(-1)
    # the state's S counts every magnitude its rounding scales with: |s g|, |delta k_j| and, through
    # delta, the prediction's sum |k_j| |beta| P (as sum|w x| for a matvec)
    mag_s = np.abs(s) + np.abs(delta[..., None] * k) + np.abs(k) * (np.abs(beta[0].astype(np.float64))[:, None] * P)[..., None]
    return out, s2, np.abs(s2 * q).sum(-1), mag_s, out_bound, e_s


def scan_magnitudes(qkv, g, beta, state, dims):
    """FP64 replay of t tokens: S of the last output (sum|s' q|) and of the final state, the running sum
    M <- M exp(g) + |delta k_j| + |k_j| |beta| P (P = sum|s exp(g) k|), M_0 = |state|: the absolute values
    every rounding of the state scales with, decayed as the state itself."""
    hk, hv, dk, dv = dims
    s, m = state.astype(np.float64), np.abs(state.astype(np.float64))
    for x, gt, bt in zip(qkv.astype(np.float64), g.astype(np.float64), beta.astype(np.float64)):
        q = x[:hk * dk].reshape(hk, dk)[np.arange(hv) % hk][:, None, :]
        k = x[hk * dk:2 * hk * dk].reshape(hk, dk)[np.arange(hv) % hk][:, None, :]
        decay = np.exp(gt)[:, None, None]
        s, m = s * decay, m * decay
        delta = (x[2 * hk * dk:].reshape(hv, dv) - (s * k).sum(-1)) * bt[:, None]
        m = m + np.abs(delta[..., None] * k) + np.abs(k) * (np.abs(bt)[:, None] * np.abs(s * k).sum(-1))[..., None]
        s = s + delta[..., None] * k
    return np.abs(s * q).sum(-1), m


def run_index(mode, iq, blocks, visible, ratio, budget):
    """(error word, scores [n], ids) of q38d SCORES then q38d_select_encode, oracle or exact fast kernels."""
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    environment = {key: val for key, val in os.environ.items() if key != "Q38_REF_KERNELS"}
    result = subprocess.run([str(NATIVE), "index", mode, *map(str, (*iq.shape, visible, ratio, budget))],
                            input=np.ascontiguousarray(iq, "<f4").tobytes() + np.ascontiguousarray(blocks, "<f4").tobytes(),
                            capture_output=True, timeout=60, pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode or stderr:
        raise AssertionError(f"index {mode}: exit {result.returncode}, {stderr!r}")
    n = len(blocks)
    return int(np.frombuffer(result.stdout[:4], "<u4")[0]), result.stdout[4:4 + 4 * n], result.stdout[4 + 4 * n:]


def attend_inputs(rng, dims, n, m, sharp=4.0):
    """q, k ~ N(0, sharp^2/sqrt(d)) so the scaled logits have std sharp^2/sqrt(d); v ~ N(0, 1) with one x100
    outlier row; gate = sigmoid outputs in (0, 1) with exact 0 and 1 present; ids a sorted random m-subset."""
    h, j, d = dims
    q = rng.normal(0, sharp / np.sqrt(d) ** 0.5, (h, d))
    k = rng.normal(0, sharp / np.sqrt(d) ** 0.5, (n, j, d))
    v = rng.normal(0, 1, (n, j, d))
    v[rng.integers(n)] *= 100
    gate = rng.uniform(0, 1, (h, d))
    gate.flat[:2] = (0.0, 1.0)
    ids = np.sort(rng.choice(n, m, replace=False)).astype(np.uint32)
    return tuple(a.astype(np.float32) for a in (q, gate, k, v)) + (ids,)


def attend_fp64(q, gate, k, v, ids):
    """FP64 r [h, d], S = gate * sum_i p_i |v_i| and the condition (a) factor F per head: |y - r| <= F S + 2 ulp(r).
    Logits in any order: |ds_i| <= gamma(d+1) * scale * sum|q k| =: E (one more rounding for the scale); s_i - max
    adds u|s_i - max|; exp (MSL precise, 4 ulp) adds 4u: every weight is off by a factor within exp(+-e), e =
    E + u R + 4u, R = max_i |s_i - max|, so the normalized weights are within expm1(2e) relative. The online
    rescales (one exp and one product per key at most, in either kernel), the m-term sums, the division and the
    gate add gamma(8m + 16) relative to sum_i p_i |v_i|."""
    h, d = q.shape
    group, u = h // k.shape[1], 2.0**-24
    kk, vv = k[ids].astype(np.float64), v[ids].astype(np.float64)  # [m, j, d]
    r, magnitude, factor = np.zeros((h, d)), np.zeros((h, d)), np.zeros((h, 1))
    for head in range(h):
        qh, kh, vh = q[head].astype(np.float64), kk[:, head // group], vv[:, head // group]
        s = kh @ qh / np.sqrt(d)
        p = np.exp(s - s.max()); p /= p.sum()
        e = (d + 1) * u / (1 - (d + 1) * u) * (np.abs(kh * qh).sum(1) / np.sqrt(d)).max() + u * (s.max() - s.min()) + 4 * u
        n = 8 * len(ids) + 16
        factor[head] = np.expm1(2 * e) + n * u / (1 - n * u)
        r[head], magnitude[head] = gate[head] * (p @ vh), gate[head] * (p @ np.abs(vh))
    return r, magnitude, factor


def attend_against_fp64(y, ref, magnitude, factor):
    bound = factor * magnitude + 2 * np.spacing(np.abs(ref.astype(np.float32))).astype(np.float64)
    delta = np.abs(y.astype(np.float64) - ref)
    return dict(outputs=int(y.size), finite=bool(np.isfinite(y).all()), condition_a=bool((delta <= bound).all()),
                max_rel=float((delta / np.maximum(magnitude, 2.0**-126)).max()),
                max_bound_fraction=float((delta / np.maximum(bound, 2.0**-149)).max()))


def prep_inputs(rng, t, dims):
    """qkv: head row r (q heads then k heads, all tokens) scaled by RMS_ROW_SCALES[r % 5], one outlier x100 in
    every third row; values, alpha, br ~ N(0, 3); A = -exp(N(0, 1)) (nonpositive, as ssm_a); dt ~ N(0, 1)."""
    hk, hv, dk, dv = dims
    heads = rng.normal(0, 1, (t * 2 * hk, dk)) * np.array(RMS_ROW_SCALES)[np.arange(t * 2 * hk) % 5][:, None]
    heads[::3, rng.integers(dk)] *= 100
    qkv = np.concatenate([heads.reshape(t, 2 * hk * dk), rng.normal(0, 3, (t, hv * dv))], 1)
    return tuple(a.astype(np.float32) for a in (qkv, rng.normal(0, 3, (t, hv)), rng.normal(0, 3, (t, hv)),
                                                -np.exp(rng.normal(0, 1, hv)), rng.normal(0, 1, hv)))


def prep_fp64(qkv, dims):
    """FP64 r = q / sqrt(sum q^2 + eps) / sqrt(dk) (k unscaled) over the 2 hk heads [t, 2 hk dk]; S = |r|."""
    hk, _, dk, _ = dims
    x = qkv[:, :2 * hk * dk].astype(np.float64).reshape(len(qkv), 2 * hk, dk)
    r = x / np.sqrt((x * x).sum(-1, keepdims=True) + L2_EPS)
    r[:, :hk] /= np.sqrt(dk)
    return r.reshape(len(qkv), -1), np.abs(r).reshape(len(qkv), -1)


class FastKernelTests(unittest.TestCase):
    def fast_case(self, label, kind, physical, logical, data, x, view=(), y_oracle=None, decoded=None):
        """Fast kernel on one case: supported, finite, (a) and (b); recorded in the corpus."""
        status, fast, cpu = native("fast", kind, physical, logical, data, x, view=view)
        decoded = cpu if decoded is None else decoded
        self.assertNotEqual(status, UNSUPPORTED, f"{label}: {NAMES[kind]} has no fast kernel")
        if y_oracle is None:
            _, y_oracle, _ = native("oracle", kind, physical, logical, data, x, view=view)
        ref, magnitude = fp64(decoded, x, logical)
        stats = against_fp64(fast, ref, magnitude, logical) | dict(vs_oracle=against_oracle(fast, y_oracle, magnitude))
        RESULTS.setdefault(label, {})["fast"] = stats
        CORPUS.setdefault(NAMES[kind], {})[label] = scaled_difference(fast, y_oracle, magnitude).ravel()
        tol = TOLERANCES["kernels"].get(NAMES[kind], {}).get("tol")
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertIsNotNone(tol, f"{NAMES[kind]}: fast kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return fast

    def assert_nonfinite(self, mode, physical, logical, data, x, label, kind=8):
        code, stdout, stderr, _ = run_native(mode, kind, physical, logical, data, x)
        self.assertEqual((code, stdout, stderr), (NONFINITE, b"", b"qwen38 Metal: error bits 1\n"), f"{mode} {label}")

    def test_selector_parsing(self):
        result = subprocess.run([str(NATIVE), "selector"], capture_output=True, timeout=30)
        self.assertEqual((result.returncode, result.stderr), (0, b""))

    def test_reference_against_itself_is_exact(self):
        """The forced oracle and the one Q38_REF_KERNELS=1 selects agree bit for bit."""
        for case in CASES:
            label, kind, physical, logical, data, x, saved = case
            with self.subTest(case=label):
                forced, decoded, ref, magnitude = reference(case)
                _, selected, _ = native("encode", kind, physical, logical, data, x, {"Q38_REF_KERNELS": "1"})
                np.testing.assert_array_equal(selected.view("<u4"), forced.view("<u4"))
                if saved is not None:  # the CPU decoder reproduces DwarfStar's saved decode
                    np.testing.assert_array_equal(decoded.view("<u4"), saved.view("<u4"))
                stats = against_fp64(forced, ref, magnitude, logical) | dict(vs_oracle=against_oracle(selected, forced, magnitude))
                RESULTS.setdefault(label, {})["oracle"] = stats
                self.assertTrue(stats["finite"] and stats["condition_a"], stats)
                self.assertEqual(stats["vs_oracle"], dict(max=0.0, p99=0.0))

    def test_fast_kernels_against_oracle_and_fp64(self):
        """A missing fast kernel is recorded, not failed; a present one meets (a) and (b)."""
        for case in CASES:
            label, kind, physical, logical, data, x, _ = case
            with self.subTest(case=label):
                y_oracle, _, ref, magnitude = reference(case)
                status, fast, _ = native("fast", kind, physical, logical, data, x)
                if status == UNSUPPORTED:
                    RESULTS.setdefault(label, {})["fast"] = "unsupported"
                    self.assertNotIn(NAMES[kind], FAST, f"{label}: fast kernel missing")
                    continue
                stats = against_fp64(fast, ref, magnitude, logical) | dict(vs_oracle=against_oracle(fast, y_oracle, magnitude))
                RESULTS.setdefault(label, {})["fast"] = stats
                CORPUS.setdefault(NAMES[kind], {})[label] = scaled_difference(fast, y_oracle, magnitude).ravel()
                tol = TOLERANCES["kernels"].get(NAMES[kind], {}).get("tol")
                self.assertTrue(stats["finite"] and stats["condition_a"], stats)
                self.assertIsNotNone(tol, f"{NAMES[kind]}: fast kernel without a measured tolerance")
                self.assertLessEqual(stats["vs_oracle"]["max"], tol)

    def real_within_tolerance(self, kind):
        """Every real slice group of one format runs fast within (a) and (b)."""
        real = [case for case in CASES if case[1] == kind and case[0].startswith("real:")]
        self.assertTrue(real, f"no real {NAMES[kind]} slices")
        for case in real:
            with self.subTest(case=case[0]):
                self.fast_case(*case[:6], y_oracle=reference(case)[0])

    def test_q8_0_real_tensors_within_tolerance(self):
        """Q8_0 slices: LM head, attn_qkv, ssm_out, shared experts, ..."""
        self.real_within_tolerance(8)

    def test_f16_real_tensors_within_tolerance(self):
        """F16 slices: the HC down/inject (K 10240) and up (K 320) tensors."""
        self.real_within_tolerance(1)

    def test_f32_real_tensors_within_tolerance(self):
        """F32 slices: router (ffn_gate_inp) and ssm_alpha (K 2560), norms."""
        self.real_within_tolerance(0)

    def test_shapes_not_multiple_of_tile(self):
        """Review Focus 1: n_out and K off every tile size, partial tails, one LM-head-sized n_out."""
        for name, kind in sorted(KINDS.items()):
            rng = seeded(3814, kind)
            # M rotates with n and K together, so every K meets every M in {1, 2, 5}.
            shapes = [(n, k, k, (1, 2, 5)[(i + j) % 3]) for i, n in enumerate((1, 3, 33, 512))
                      for j, k in enumerate((32, 640, 2560))]
            shapes += [(33, 64, 40, 3), (3, 2560, 2550, 2), (248320, 32, 32, 1)]  # (N, physical, logical, M)
            if kind != 8:  # no blocks: odd K (rows not 4-byte aligned), and HC inject's n_out 4 at K 10240
                shapes += [(33, 37, 37, 3), (7, 2561, 2561, 4), (4, 10240, 10240, 1), (4, 10240, 10240, 5)]
            for n, physical, logical, m in shapes:
                with self.subTest(kernel=name, n=n, physical=physical, logical=logical, m=m):
                    x = rng.uniform(-1, 1, (m, logical)).astype(np.float32)
                    self.fast_case(f"tile:{name}:N{n}:K{logical}/{physical}:M{m}", kind, physical, logical,
                                   weight_rows(kind, physical, n, rng).tobytes(), x)

    def test_logical_k_padding(self):
        """Review Focus 2: nonfinite padding never reaches y, and x is never read past k_logical."""
        for name, kind in sorted(KINDS.items()):
            rng = seeded(3815, kind)
            for n, physical, logical, m in ((33, 768, 640, 3), (5, 64, 40, 4), (3, 2560, 2550, 2)):
                with self.subTest(kernel=name, physical=physical, logical=logical):
                    raw = weight_rows(kind, physical, n, rng)
                    first = -(-logical // 32) * 32 if kind == 8 else logical  # first column wholly past k_logical
                    for column in range(first, physical, 32 if kind == 8 else 1):  # Q8_0: the block scale
                        raw = poison(kind, raw, slice(None), column, (np.nan, np.inf)[column // (32 if kind == 8 else 1) % 2])
                    x = rng.uniform(-1, 1, (m, logical)).astype(np.float32)
                    # Pitch logical+1: the NaN after each row and after x is what a read past k_logical
                    # meets. The CPU decoder rejects nonfinite padding, so the reference decodes here.
                    self.fast_case(f"padding:{name}:N{n}:K{logical}/{physical}:M{m}", kind, physical, logical,
                                   raw.tobytes(), x, view=(logical + 1, n, 0), decoded=decode_rows(kind, raw, physical))

    def test_nonfinite_sets_error_word(self):
        """Review Focus 3: NaN/Inf in x or in a used weight sets bit 1 (exit 16), oracle and fast alike."""
        for name, kind in sorted(KINDS.items()):
            rng = seeded(3816, kind)
            raw, x = weight_rows(kind, 640, 33, rng), rng.uniform(-1, 1, (3, 640)).astype(np.float32)
            for label, value in (("nan", np.nan), ("inf", np.inf)):
                poisoned = x.copy()
                poisoned[1, 100] = value
                bad = poison(kind, raw, 5, 96, value)
                for mode in ("oracle", "fast"):
                    with self.subTest(kernel=name, mode=mode, poison=label):
                        self.assert_nonfinite(mode, 640, 640, raw.tobytes(), poisoned, f"x {label}", kind)
                        self.assert_nonfinite(mode, 640, 640, bad.tobytes(), x, f"weight {label}", kind)

    def test_rows_bit_independent(self):
        """Review Focus 4: a row's bits depend neither on M, nor on its x position, nor on its n_out position."""
        for name, kind in sorted(KINDS.items()):
            rng = seeded(3817, kind)
            n, k, m = 37, 2560, 19
            raw, x = weight_rows(kind, k, n, rng), rng.uniform(-1, 1, (m, k)).astype(np.float32)
            with self.subTest(kernel=name):
                full = self.fast_case(f"rows:{name}:N{n}:K{k}:M{m}", kind, k, k, raw.tobytes(), x)
                for rows in ([0], [m - 1], list(range(3, 9)), list(range(m - 1, -1, -1)), [4] * 9):
                    with self.subTest(kernel=name, rows=rows[:4]):
                        _, y, _ = native("fast", kind, k, k, raw.tobytes(), x[rows])
                        np.testing.assert_array_equal(y.view("<u4"), full[rows].view("<u4"))
                _, shifted, _ = native("fast", kind, k, k, raw[1:].tobytes(), x)  # weight row j+1 becomes row j
                np.testing.assert_array_equal(shifted.view("<u4"), full[:, 1:].view("<u4"))

    def test_strided_offset_views(self):
        """Review Focus 5: pitches above K*4 and n*4 and nonzero offsets (an LM head on a slice) change no bit."""
        for name, kind in sorted(KINDS.items()):
            rng = seeded(3818, kind)
            n, k, m = 33, 640, 5
            raw, x = weight_rows(kind, k, n, rng).tobytes(), rng.uniform(-1, 1, (m, k)).astype(np.float32)
            with self.subTest(kernel=name):
                plain = self.fast_case(f"views:{name}:N{n}:K{k}:M{m}", kind, k, k, raw, x)
                for view in ((k + 12, n + 7, 37), (k, n, 5 * k), (k + 1, n + 1, 1)):
                    with self.subTest(kernel=name, view=view):
                        strided = self.fast_case(f"views:{name}:N{n}:K{k}:M{m}:{view}", kind, k, k, raw, x, view=view)
                        np.testing.assert_array_equal(strided.view("<u4"), plain.view("<u4"))

    def assert_multi_matches_single(self, label, projections, x, view=None, env=None, mode="fast"):
        """Every y_i of one multi call is bit-identical to the single `mode` call of projection i;
        one dispatch per fast format, one per projection without a fast kernel or under the oracle."""
        code, ys, stderr, dispatches = run_multi(projections, x, env, view)
        self.assertEqual((code, stderr), (0, b""), label)
        fast = {p[0] for p in projections if NAMES[p[0]] in FAST} if mode == "fast" else set()
        self.assertEqual(dispatches, len(fast) + sum(p[0] not in fast for p in projections), label)
        for i, ((kind, physical, n, data), y) in enumerate(zip(projections, ys, strict=True)):
            single = mode if NAMES[kind] in FAST else "oracle"  # no fast kernel: q38_packed_encode's oracle
            _, want, _ = native(single, kind, physical, x.shape[1], data, x)
            np.testing.assert_array_equal(y.view("<u4"), want.view("<u4"), f"{label}: projection {i}")
        return ys

    def test_multi_bit_identical_to_single(self):
        """R1.4: one multi call per format and mixed groups, against single calls, bit for bit."""
        rng = np.random.default_rng(3820)
        groups = []
        for name, kind in sorted(KINDS.items()):  # counts 1, 2, 5 (QSA), max; rows 1, 4, 33; K with partial tails
            for i, (count, physical, logical, m) in enumerate(((1, 640, 640, 1), (2, 2560, 2550, 4), (5, 2560, 2560, 33),
                                                               (MULTI_MAX, 2560, 2560, 4))):
                ns = [MULTI_NOUT[(i + j) % len(MULTI_NOUT)] for j in range(count)]
                groups.append((f"{name}:x{count}:K{logical}:M{m}", m, logical,
                               [(kind, physical, n, weight_rows(kind, physical, n, rng).tobytes()) for n in ns]))
        f16_hc = [(1, 10240, n, weight_rows(1, 10240, n, rng).tobytes()) for n in (320, 4)]  # HC down + inject, S = 8
        groups.append(("f16:hc:K10240:M4", 4, 10240, f16_hc))
        gdn = [(8, 2560, n, weight_rows(8, 2560, n, rng).tobytes()) for n in (640, 512)] + \
              [(0, 2560, 48, weight_rows(0, 2560, 48, rng).tobytes()) for _ in range(2)]
        groups += [(f"gdn:q8_0+f32:M{m}", m, 2560, gdn) for m in (1, 4, 33)]
        real = [("qsa", 8, 2560, ("attn_q.", "attn_k.", "attn_v.", "q_proj", "k_proj"), 5),
                ("gdn", 8, 2560, ("attn_qkv",), 1), ("gdn", 0, 2560, ("ssm_alpha",), 1),
                ("shared", 8, 2560, ("gate_shexp", "up_shexp"), 2), ("shared", 0, 2560, ("ffn_gate_inp",), 1),
                ("hc", 1, 10240, ("hc_down", "hc_attn_inject"), 2)]
        for group in dict.fromkeys(g[0] for g in real):  # the real call sites' tensors (slices of them)
            projections = [p for g in real if g[0] == group for p in real_projections(*g[1:4])]
            self.assertEqual(len(projections), sum(g[4] for g in real if g[0] == group), group)
            groups.append((f"real:{group}", 4, projections[0][1], projections))
        q2_k = (10, 2560, 4, packed.packed_fixture(10, 2560, 4, rng))  # no fast kernel: oracle, alone
        groups.append(("q8_0+q2_k+f32", 4, 2560, [gdn[0], q2_k, gdn[2]]))
        for label, m, logical, projections in groups:
            self.assertLessEqual(len(projections), MULTI_MAX, label)
            with self.subTest(group=label):
                x = rng.uniform(-1, 1, (m, logical)).astype(np.float32)
                ys = self.assert_multi_matches_single(label, projections, x)
                if label.startswith(("q8_0:x5", "gdn", "real", "f16:hc")):  # views change no bit
                    for view in ((logical + 12, 7, 37), (logical + 1, 1, 1)):
                        _, strided, _, _ = run_multi(projections, x, view=view)
                        for y, want in zip(strided, ys, strict=True):
                            np.testing.assert_array_equal(y.view("<u4"), want.view("<u4"), f"{label} {view}")
        with self.subTest(case="nonfinite"):  # one poisoned projection sets bit 1, whether weight or x
            for kind, n, value in ((8, 512, np.nan), (0, 48, np.inf)):
                bad = poison(kind, weight_rows(kind, 2560, n, rng), 7, 96, value)
                projections = [gdn[0], (kind, 2560, n, bad.tobytes()), gdn[3]]
                x = rng.uniform(-1, 1, (4, 2560)).astype(np.float32)
                self.assertEqual(run_multi(projections, x)[:3], (NONFINITE, None, b"qwen38 Metal: error bits 1\n"))
            x[2, 100] = np.nan
            self.assertEqual(run_multi(gdn, x)[:3], (NONFINITE, None, b"qwen38 Metal: error bits 1\n"))
        with self.subTest(case="reference"):  # Q38_REF_KERNELS=linear: the oracle per projection, bit for bit
            x = rng.uniform(-1, 1, (4, 2560)).astype(np.float32)
            oracle_ys = self.assert_multi_matches_single("ref", gdn, x, env={"Q38_REF_KERNELS": "linear"}, mode="oracle")
            fast_ys = self.assert_multi_matches_single("default", gdn, x)
            self.assertFalse(all(np.array_equal(a.view("<u4"), b.view("<u4")) for a, b in zip(oracle_ys, fast_ys)),
                             "case does not separate the paths")

    def separating_case(self, kind):
        """A dense [33x2560] case whose fast and oracle bits differ (else the selector tests prove nothing)."""
        rng = seeded(3819, kind)
        raw, x = weight_rows(kind, 2560, 33, rng).tobytes(), rng.uniform(-1, 1, (3, 2560)).astype(np.float32)
        _, fast, _ = native("fast", kind, 2560, 2560, raw, x)
        _, y_oracle, _ = native("oracle", kind, 2560, 2560, raw, x)
        self.assertFalse(np.array_equal(fast.view("<u4"), y_oracle.view("<u4")), "case does not separate the paths")
        return raw, x, fast, y_oracle

    def default_selects_fast(self, kind):
        """Unless `linear` is selected, q38_packed_encode runs the format's fast kernel, also on its real slices."""
        raw, x, fast, _ = self.separating_case(kind)
        for env in ({}, {"Q38_REF_KERNELS": "0"}, {"Q38_REF_KERNELS": ""}, {"Q38_REF_KERNELS": "indexed"}):
            with self.subTest(env=env):
                _, y, _ = native("encode", kind, 2560, 2560, raw, x, env)
                np.testing.assert_array_equal(y.view("<u4"), fast.view("<u4"))
        for case in [c for c in CASES if c[1] == kind and c[0].startswith("real:")]:
            label, _, physical, logical, data, xs, _ = case
            with self.subTest(case=label):
                _, y, _ = native("encode", kind, physical, logical, data, xs)
                _, want, _ = native("fast", kind, physical, logical, data, xs)
                np.testing.assert_array_equal(y.view("<u4"), want.view("<u4"))

    def test_default_selects_fast_q8_0(self):
        """R1.2b: Q8_0 runs the fast kernel by default; a format without one (Q2_K) keeps the oracle."""
        self.default_selects_fast(8)
        label, kind, physical, logical, data, xs, _ = [c for c in CASES if c[1] == 10][0]
        with self.subTest(case=label):
            _, y, _ = native("encode", kind, physical, logical, data, xs)
            _, want, _ = native("oracle", kind, physical, logical, data, xs)
            np.testing.assert_array_equal(y.view("<u4"), want.view("<u4"))

    def test_default_selects_fast_f16(self):
        """R1.3: the F16 HC tensors run the fast kernel by default."""
        self.default_selects_fast(1)

    def test_default_selects_fast_f32(self):
        """R1.3: the F32 router and ssm_alpha/beta run the fast kernel by default."""
        self.default_selects_fast(0)

    def test_reference_env_selects_oracle(self):
        """Q38_REF_KERNELS naming `linear` (or all, or a typo: ruling R18) restores the oracle bit for bit."""
        for name, kind in sorted(KINDS.items()):
            raw, x, _, y_oracle = self.separating_case(kind)
            for value in ("linear", "1", "all", "linear,indexed", "lineer"):
                with self.subTest(kernel=name, value=value):
                    _, y, _ = native("encode", kind, 2560, 2560, raw, x, {"Q38_REF_KERNELS": value})
                    np.testing.assert_array_equal(y.view("<u4"), y_oracle.view("<u4"))

    def test_q8_0_scale_decode_matches_oracle(self):
        """All 65536 fp16 scale patterns: finite ones decode exactly (one-hot x), nonfinite ones set bit 1."""
        patterns = np.arange(65536, dtype=np.uint16)
        finite = patterns[(patterns & 0x7C00) != 0x7C00]
        raw = np.zeros((len(finite), 34), np.uint8)
        raw[:, :2] = finite.view(np.uint8).reshape(-1, 2)
        raw[:, 2:5] = np.array([1, -128, 127], np.int8).view(np.uint8)
        x = np.eye(3, 32, dtype=np.float32)
        _, fast, _ = native("fast", 8, 32, 32, raw.tobytes(), x)
        self.assertIsNotNone(fast, "q8_0 has no fast kernel")
        _, y_oracle, _ = native("oracle", 8, 32, 32, raw.tobytes(), x)
        expected = finite.view("<f2").astype(np.float32)[None, :] * np.array([[1], [-128], [127]], np.float32)
        np.testing.assert_array_equal(fast, expected)  # exact value (+0 and -0 compare equal)
        np.testing.assert_array_equal(fast.view("<u4"), y_oracle.view("<u4"))  # and the oracle's bits
        nonfinite = patterns[(patterns & 0x7C00) == 0x7C00]
        for chosen in (nonfinite, *(np.array([p], np.uint16) for p in (0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFFFF))):
            bad = np.zeros((len(chosen), 34), np.uint8)
            bad[:, :2], bad[:, 2] = chosen.view(np.uint8).reshape(-1, 2), 1
            with self.subTest(patterns=len(chosen), first=hex(int(chosen[0]))):
                self.assert_nonfinite("fast", 32, 32, bad.tobytes(), x[:1], "scale pattern")


    def test_f16_decode_matches_oracle(self):
        """All 65536 fp16 weight patterns: finite ones decode exactly (one-hot x), nonfinite ones set bit 1."""
        patterns = np.arange(65536, dtype=np.uint16)
        finite = patterns[(patterns & 0x7C00) != 0x7C00]  # 63488 = 1984 rows of 32
        x = np.eye(32, dtype=np.float32)
        _, fast, _ = native("fast", 1, 32, 32, finite.tobytes(), x)
        self.assertIsNotNone(fast, "f16 has no fast kernel")
        _, y_oracle, _ = native("oracle", 1, 32, 32, finite.tobytes(), x)
        expected = finite.view("<f2").astype(np.float32).reshape(-1, 32).T
        np.testing.assert_array_equal(fast, expected)  # exact value (+0 and -0 compare equal)
        np.testing.assert_array_equal(fast.view("<u4"), y_oracle.view("<u4"))  # and the oracle's bits
        nonfinite = patterns[(patterns & 0x7C00) == 0x7C00]
        for chosen in (nonfinite, *(np.array([p], np.uint16) for p in (0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFFFF))):
            bad = np.zeros((len(chosen), 32), np.uint16)
            bad[:, 0] = chosen
            with self.subTest(patterns=len(chosen), first=hex(int(chosen[0]))):
                self.assert_nonfinite("fast", 32, 32, bad.tobytes(), x[:1], "weight pattern", 1)

    # ---- R1.5 indexed expert matvecs (q38_packed_encode_indexed) ----

    def indexed_case(self, label, kind, physical, logical, n, experts, data, ids, x, divisor, view=(), decoded=None):
        """Indexed fast kernel on one case: supported, finite, (a) and (b) against the indexed oracle."""
        code, fast, cpu, stderr = run_indexed("fast", kind, physical, logical, n, experts, data, ids, x, divisor, view=view)
        self.assertNotEqual(code, UNSUPPORTED, f"{label}: {NAMES[kind]} has no indexed fast kernel")
        self.assertEqual((code, stderr), (0, b""), label)
        code, y_oracle, _, stderr = run_indexed("oracle", kind, physical, logical, n, experts, data, ids, x, divisor, view=view)
        self.assertEqual((code, stderr), (0, b""), label)
        ref, magnitude = indexed_fp64(cpu if decoded is None else decoded, ids, x, divisor, logical, n)
        stats = against_fp64(fast, ref, magnitude, logical) | dict(vs_oracle=against_oracle(fast, y_oracle, magnitude))
        RESULTS.setdefault(label, {})["fast"] = stats
        RESULTS[label]["oracle"] = against_fp64(y_oracle, ref, magnitude, logical)  # the oracle's own distance
        CORPUS.setdefault(NAMES[kind], {})[label] = scaled_difference(fast, y_oracle, magnitude).ravel()
        tol = TOLERANCES["kernels"].get(NAMES[kind], {}).get("tol")
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertIsNotNone(tol, f"{NAMES[kind]}: fast kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return fast

    @staticmethod
    def indexed_shapes(kind):
        """(N, physical, logical): K off every chunk and tile, partial chunks, the largest N the
        transport allows (decoded rows cross stdout: 32 MiB)."""
        block = oracle.FORMATS[kind][0]
        ks = ((32, 32), (640, 640), (2560, 2560), (64, 40)) if block == 32 else \
             ((256, 256), (768, 640), (2560, 2560), (768, 600))
        shapes = [(n, *k) for i, n in enumerate((1, 3, 33, 512)) for k in ks[i % 2::2]]
        return shapes + [(248320 if block == 32 else 32768, block if block == 32 else 256, block if block == 32 else 256)]

    def test_shapes_not_multiple_of_tile_indexed(self):
        """Review Focus 1 (indexed): n_out and K off the tile and chunk, slots 1/2/10 as gate
        (x divisor = slots) and down (divisor 1), 3 experts with repeated ids."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3830, kind)
            for i, (n, physical, logical) in enumerate(self.indexed_shapes(kind)):
                slots, tokens, experts = (1, 2, 10)[i % 3], (1, 2, 3)[i % 2] if n < 1000 else 1, 3 if n < 1000 else 1
                divisor = slots if i % 2 else 1
                rows = slots * tokens
                with self.subTest(kernel=name, n=n, physical=physical, logical=logical, rows=rows, divisor=divisor):
                    ids = rng.integers(0, experts, rows)
                    x = rng.uniform(-1, 1, (rows // divisor, logical)).astype(np.float32)
                    self.indexed_case(f"indexed:{name}:N{n}:K{logical}/{physical}:R{rows}/{divisor}", kind, physical,
                                      logical, n, experts, quant_rows(kind, physical, experts * n, rng).tobytes(), ids, x, divisor)

    def test_logical_k_padding_indexed(self):
        """Review Focus 2 (indexed): the 640/768 down padding (random codes) and a wholly padded
        block with a nonfinite scale never reach y; x is never read past k_logical (NaN there)."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng, block = seeded(3831, kind), oracle.FORMATS[kind][0]
            for n, physical, logical, rows in ((33, 768, 640, 3), (5, 1024 if block == 256 else 768, 640, 4)):
                with self.subTest(kernel=name, physical=physical, logical=logical):
                    clean = quant_rows(kind, physical, 2 * n, rng)
                    raw = clean
                    first = -(-logical // block) * block  # first block wholly past k_logical
                    for column in range(first, physical, block):
                        raw = poison_block(kind, raw, slice(None), column, (np.nan, np.inf)[column // block % 2])
                    ids, x = rng.integers(0, 2, rows), rng.uniform(-1, 1, (rows, logical)).astype(np.float32)
                    _, _, decoded, _ = run_indexed("fast", kind, physical, logical, n, 2, clean.tobytes(), ids, x, 1)
                    self.indexed_case(f"indexed-padding:{name}:N{n}:K{logical}/{physical}:R{rows}", kind, physical,
                                      logical, n, 2, raw.tobytes(), ids, x, 1, view=(logical + 1, n, 0), decoded=decoded)

    def test_nonfinite_sets_error_word_indexed(self):
        """Review Focus 3 (indexed): NaN/Inf in x or in a used block scale sets bit 1, oracle and fast."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3832, kind)
            raw, x, ids = quant_rows(kind, 2560, 2 * 33, rng), rng.uniform(-1, 1, (3, 2560)).astype(np.float32), [1, 0, 1]
            for label, value in (("nan", np.nan), ("inf", np.inf)):
                poisoned = x.copy()
                poisoned[1, 100] = value
                bad = poison_block(kind, raw, 33 + 5, 96, value)  # row 5 of expert 1
                for mode in ("oracle", "fast"):
                    with self.subTest(kernel=name, mode=mode, poison=label):
                        for data, xs in ((raw, poisoned), (bad, x)):
                            code, _, _, stderr = run_indexed(mode, kind, 2560, 2560, 33, 2, data.tobytes(), ids, xs, 1)
                            self.assertEqual((code, stderr), (NONFINITE, b"qwen38 Metal: error bits 1\n"))

    def test_rows_bit_independent_indexed(self):
        """Review Focus 4 (indexed): an output's bits depend neither on the number of rows, nor on
        its row, nor on its n_out position, nor on the expert's id, nor on the x divisor."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3833, kind)
            n, k, m, experts = 37, 2560, 19, 4
            raw, x, ids = quant_rows(kind, k, experts * n, rng), rng.uniform(-1, 1, (m, k)).astype(np.float32), rng.integers(0, experts, m)
            with self.subTest(kernel=name):
                full = self.indexed_case(f"indexed-rows:{name}:N{n}:K{k}:R{m}", kind, k, k, n, experts, raw.tobytes(), ids, x, 1)
                fast = lambda data, i, xs, d, nn=n: run_indexed("fast", kind, k, k, nn, experts, data, i, xs, d)[1]
                for rows in ([0], [m - 1], list(range(3, 9)), list(range(m - 1, -1, -1)), [4] * 9):
                    with self.subTest(kernel=name, rows=rows[:4]):
                        np.testing.assert_array_equal(fast(raw.tobytes(), ids[rows], x[rows], 1).view("<u4"), full[rows].view("<u4"))
                shifted = raw.reshape(experts, n, -1)[:, 1:].tobytes()  # weight row j+1 becomes row j
                np.testing.assert_array_equal(fast(shifted, ids, x, 1, n - 1).view("<u4"), full[:, 1:].view("<u4"))
                relabeled = raw.reshape(experts, n, -1)[::-1].tobytes()  # expert e becomes experts-1-e
                np.testing.assert_array_equal(fast(relabeled, experts - 1 - ids, x, 1).view("<u4"), full.view("<u4"))
                gate = fast(raw.tobytes(), ids[:18], x[:6], 3)  # 6 tokens x 3 slots against x repeated per slot
                np.testing.assert_array_equal(gate.view("<u4"), fast(raw.tobytes(), ids[:18], np.repeat(x[:6], 3, 0), 1).view("<u4"))

    def test_strided_offset_views_indexed(self):
        """Review Focus 5 (indexed): pitches above K*4 and n*4 and nonzero offsets change no bit."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3834, kind)
            n, k, rows, experts = 33, 768, 6, 3
            raw, ids = quant_rows(kind, k, experts * n, rng).tobytes(), rng.integers(0, experts, rows)
            x = rng.uniform(-1, 1, (rows // 2, 640)).astype(np.float32)
            with self.subTest(kernel=name):
                plain = self.indexed_case(f"indexed-views:{name}:N{n}:K640/{k}", kind, k, 640, n, experts, raw, ids, x, 2)
                for view in ((640 + 12, n + 7, 37), (640, n, 5 * 640), (640 + 1, n + 1, 1)):
                    with self.subTest(kernel=name, view=view):
                        strided = self.indexed_case(f"indexed-views:{name}:N{n}:K640/{k}:{view}", kind, k, 640, n,
                                                    experts, raw, ids, x, 2, view=view)
                        np.testing.assert_array_equal(strided.view("<u4"), plain.view("<u4"))

    def test_expert_id_bounds(self):
        """Review Focus 5 (indexed): an expert id >= n_experts sets bit 2 and leaves its row
        unwritten (still the harness's NaN), oracle and fast; the other rows are unchanged."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3835, kind)
            n, experts = 33, 3
            raw, x = quant_rows(kind, 768, experts * n, rng).tobytes(), rng.uniform(-1, 1, (6, 640)).astype(np.float32)
            ids, bad = np.array([0, 1, 3, 2, 7, 0xFFFFFFFF]), np.array([False, False, True, False, True, True])
            for mode in ("oracle", "fast"):
                with self.subTest(kernel=name, mode=mode):
                    code, y, _, stderr = run_indexed(mode, kind, 768, 640, n, experts, raw, ids, x, 1)
                    self.assertEqual((code, stderr), (GPU_ERROR, b"qwen38 Metal: error bits 2\n"))
                    _, want, _, _ = run_indexed(mode, kind, 768, 640, n, experts, raw, np.where(bad, 0, ids), x, 1)
                    np.testing.assert_array_equal(y[~bad].view("<u4"), want[~bad].view("<u4"))
                    np.testing.assert_array_equal(y[bad].view("<u4"), np.full((3, n), np.nan, np.float32).view("<u4"))

    INDEXED_SHAPE = {16: (2560, 2560, 10), 10: (768, 640, 1), 12: (2560, 2560, 10), 39: (640, 640, 1)}  # physical, logical, divisor

    def indexed_separating_case(self, kind):
        """A gate (x divisor 10) or down (divisor 1) case of 2 tokens x 10 slots whose fast and
        oracle bits differ (else the selector tests prove nothing)."""
        rng, (physical, logical, divisor) = seeded(3836, kind), self.INDEXED_SHAPE[kind]
        n, experts = 33, 12
        raw = quant_rows(kind, physical, experts * n, rng).tobytes()
        ids = np.concatenate([rng.permutation(experts)[:10] for _ in range(2)])
        x = rng.uniform(-1, 1, (20 // divisor, logical)).astype(np.float32)
        args = (kind, physical, logical, n, experts, raw, ids, x, divisor)
        fast, y_oracle = run_indexed("fast", *args)[1], run_indexed("oracle", *args)[1]
        self.assertFalse(np.array_equal(fast.view("<u4"), y_oracle.view("<u4")), "case does not separate the paths")
        return args, fast, y_oracle

    def test_default_selects_fast_indexed(self):
        """R1.5: q38_packed_encode_indexed runs the indexed fast kernel unless `indexed` is
        selected, also on the real expert slices; a format without one (Q8_0) keeps the oracle."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            args, fast, _ = self.indexed_separating_case(kind)
            for env in ({}, {"Q38_REF_KERNELS": "0"}, {"Q38_REF_KERNELS": ""}, {"Q38_REF_KERNELS": "linear"},
                        {"Q38_REF_KERNELS": "linear,moe"}):
                with self.subTest(kernel=name, env=env):
                    np.testing.assert_array_equal(run_indexed("encode", *args, env=env)[1].view("<u4"), fast.view("<u4"))
            for label, _, physical, logical, data, x, saved in [c for c in CASES if c[1] == kind and c[0].startswith("real:")]:
                with self.subTest(case=label):
                    ids = np.tile(np.arange(len(saved)), len(x))
                    real = (kind, physical, logical, 1, len(saved), data, ids, x, len(saved))
                    np.testing.assert_array_equal(run_indexed("encode", *real)[1].view("<u4"),
                                                  run_indexed("fast", *real)[1].view("<u4"))
        rng = np.random.default_rng(3837)
        q8 = (8, 2560, 2560, 33, 3, packed.packed_fixture(8, 2560, 99, rng), [0, 2, 1, 2], rng.uniform(-1, 1, (4, 2560)).astype(np.float32), 1)
        with self.subTest(kernel="q8_0 (no indexed fast kernel)"):
            self.assertEqual(run_indexed("fast", *q8)[0], UNSUPPORTED)
            np.testing.assert_array_equal(run_indexed("encode", *q8)[1].view("<u4"), run_indexed("oracle", *q8)[1].view("<u4"))

    def test_reference_env_selects_oracle_indexed(self):
        """Q38_REF_KERNELS naming `indexed` (or 1, all, or a typo: ruling R18) restores the
        indexed oracle bit for bit."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            args, _, y_oracle = self.indexed_separating_case(kind)
            for value in ("indexed", "1", "all", "linear,indexed", "indexd"):
                with self.subTest(kernel=name, value=value):
                    np.testing.assert_array_equal(run_indexed("encode", *args, env={"Q38_REF_KERNELS": value})[1].view("<u4"),
                                                  y_oracle.view("<u4"))

    def indexed_real_within_tolerance(self, kind):
        """The real expert rows of one format, each its own expert (n_out 1), every saved and
        seeded x row against every expert as a gate dispatch (x divisor = experts)."""
        real = [case for case in CASES if case[1] == kind and case[0].startswith("real:")]
        self.assertTrue(real, f"no real {NAMES[kind]} slices")
        for label, _, physical, logical, data, x, saved in real:
            experts = len(saved)
            with self.subTest(case=label):
                ids = np.tile(np.arange(experts), len(x))
                y = self.indexed_case(f"indexed-{label}", kind, physical, logical, 1, experts, data, ids, x, experts)
                self.assertEqual(y.shape, (len(x) * experts, 1))

    def test_iq2_xxs_real_indexed_within_tolerance(self):
        """IQ2_XXS slices: ffn_gate/up_exps of layers 0, 23, 47."""
        self.indexed_real_within_tolerance(16)

    def test_q2_k_real_indexed_within_tolerance(self):
        """Q2_K slices: ffn_down_exps (640 of 768) of layers 0, 23, 47."""
        self.indexed_real_within_tolerance(10)

    def test_q4_k_real_indexed_within_tolerance(self):
        """Q4_K slices: ffn_gate/up_exps of the mixed layer 48 (and the MTP head)."""
        self.indexed_real_within_tolerance(12)

    def test_mxfp4_real_indexed_within_tolerance(self):
        """MXFP4 slices: ffn_down_exps (K 640) of the mixed layer 48 (and the MTP head)."""
        self.indexed_real_within_tolerance(39)

    def assert_decode_matches_oracle(self, kind, packed, columns, batch, flush=False):
        """One-hot x through the indexed kernel: y[p, j] = w[j, p]. Fast equals the CPU decoder's
        value (+0 == -0) and the oracle's bits; rows in batches the transport allows. flush: the
        GPU multiply flushes a subnormal operand (oracle and fast alike, even for x = 2^64), so a
        weight below 2^-126 is expected as zero: no dot product can see more of it."""
        x, rows = np.eye(columns, dtype=np.float32), len(packed)
        for start in range(0, rows, batch):
            chunk = packed[start:start + batch]
            with self.subTest(kind=NAMES[kind], rows=start):
                code, fast, decoded, stderr = run_indexed("fast", kind, columns, columns, len(chunk), 1, chunk.tobytes(),
                                                    np.zeros(columns), x, 1)
                self.assertEqual((code, stderr), (0, b""))
                _, y_oracle, _, _ = run_indexed("oracle", kind, columns, columns, len(chunk), 1, chunk.tobytes(),
                                                np.zeros(columns), x, 1)
                expected = np.where(np.abs(decoded) < 2.0**-126, 0, decoded) if flush else decoded
                np.testing.assert_array_equal(fast.T, expected)
                np.testing.assert_array_equal(fast.view("<u4"), y_oracle.view("<u4"))

    def assert_scale_nonfinite(self, kind, block):
        """Every nonfinite fp16 d (and Q2_K dmin) pattern sets bit 1 through the fast kernel."""
        patterns = np.arange(65536, dtype=np.uint16)
        nonfinite = patterns[(patterns & 0x7C00) == 0x7C00]
        for at in SCALES[kind]:
            bad = np.tile(block, (len(nonfinite), 1))
            bad[:, at:at + 2] = nonfinite.view(np.uint8).reshape(-1, 2)
            with self.subTest(kind=NAMES[kind], field=at):
                code, _, _, stderr = run_indexed("fast", kind, 256, 256, len(bad), 1, bad.tobytes(), [0], np.ones((1, 256), np.float32), 1)
                self.assertEqual((code, stderr), (NONFINITE, b"qwen38 Metal: error bits 1\n"))

    def test_iq2_xxs_decode_matches_oracle(self):
        """Every grid index at every (group, slot) position, every 7-bit sign word (8th sign by
        parity) in every slot, every group scale, and every finite fp16 d (cycled over the
        rows): exact, the oracle's bits."""
        finite = np.arange(65536, dtype=np.uint16)
        finite = finite[(finite & 0x7C00) != 0x7C00]
        index = np.arange(65536)  # position (g, s) holds grid index (index + 4g + s) % 256
        start, sign_id, scale = index % 256, index // 256 % 128, (index + index // 256) % 16
        words = (sign_id * (1 | 1 << 7 | 1 << 14 | 1 << 21) | scale << 28).astype("<u4")
        packed = np.zeros((len(index), 66), np.uint8)
        packed[:, :2] = finite[index % len(finite)].view(np.uint8).reshape(-1, 2)
        for g in range(8):
            packed[:, 2 + 8 * g:6 + 8 * g] = (start[:, None] + 4 * g + np.arange(4)) % 256
            packed[:, 6 + 8 * g:10 + 8 * g] = words.view(np.uint8).reshape(-1, 4)
        self.assert_decode_matches_oracle(16, packed, 256, 16384)
        self.assert_scale_nonfinite(16, packed[0])

    def test_q2_k_decode_matches_oracle(self):
        """Every scale/min byte at every scale position, every code at every column, and every
        finite fp16 d and dmin (cycled over the rows): exact, the oracle's bits."""
        finite = np.arange(65536, dtype=np.uint16)
        finite = finite[(finite & 0x7C00) != 0x7C00]
        rows = np.arange(65536)
        packed = np.zeros((len(rows), 84), np.uint8)
        packed[:, :16] = (rows[:, None] + 17 * np.arange(16)) % 256
        packed[:, 16:80] = (rows[:, None] * 29 + 7 * np.arange(64)) % 256
        packed[:, 80:82] = finite[rows % len(finite)].view(np.uint8).reshape(-1, 2)
        packed[:, 82:84] = finite[(rows * 7919) % len(finite)].view(np.uint8).reshape(-1, 2)
        self.assert_decode_matches_oracle(10, packed, 256, 16384)
        self.assert_scale_nonfinite(10, packed[0])

    def test_q2_k_zero_weights_give_zero(self):
        """R1.9c / R42: d = dmin, scale 1, min 3, codes 3 make every weight fl(3d) - 3d = 0, so
        S = 0 and the oracle gives 0. The fast kernel must too: this is why the factored form
        dl*sum(q x) - ml*sum(x) was rejected (its two sums round apart: up to 3.8e-6, (b) fails)."""
        rng = seeded(3841, 10)
        raw = quant_rows(10, 256, 8, rng).reshape(8, 84)
        raw[:, :16], raw[:, 16:80] = 0x31, 0xFF
        raw[:, 82:84] = raw[:, 80:82]
        x = rng.uniform(-1, 1, (4, 256)).astype(np.float32)
        for mode in ("oracle", "fast"):
            with self.subTest(mode=mode):
                code, y, _, stderr = run_indexed(mode, 10, 256, 256, 8, 1, raw.tobytes(), np.zeros(4), x, 1)
                self.assertEqual((code, stderr), (0, b""))
                np.testing.assert_array_equal(y, np.zeros((4, 8), np.float32))

    def test_q4_k_decode_matches_oracle(self):
        """Every 6-bit scale/min byte at every one of the 12 positions (so every scale and min of
        the 8 sub-blocks, low and high bits), every code at every column, and every finite fp16
        d and dmin (cycled over the rows): exact, the oracle's bits."""
        finite = np.arange(65536, dtype=np.uint16)
        finite = finite[(finite & 0x7C00) != 0x7C00]
        rows = np.arange(65536)
        packed = np.zeros((len(rows), 144), np.uint8)
        packed[:, 0:2] = finite[rows % len(finite)].view(np.uint8).reshape(-1, 2)
        packed[:, 2:4] = finite[(rows * 7919) % len(finite)].view(np.uint8).reshape(-1, 2)
        packed[:, 4:16] = (rows[:, None] + 17 * np.arange(12)) % 256
        packed[:, 16:144] = (rows[:, None] * 29 + 7 * np.arange(128)) % 256
        self.assert_decode_matches_oracle(12, packed, 256, 16384)
        self.assert_scale_nonfinite(12, packed[0])

    def test_mxfp4_decode_matches_oracle(self):
        """All 256 E8M0 exponents x 16 E2M1 codes, each code in both nibbles. Every finite value
        (exponent field 126 + m/2 + e - 127 <= 254 for magnitude m; results below 2^-126
        included) decodes exactly to the oracle's bits; the codes that overflow at e = 253
        (m >= 6) and 254 (m >= 4) and every code of the 255 escape set bit 1. The 16 subnormal
        weights (e = 0, m 1..3; e = 1, m 1) reach y as zero in both kernels (see flush)."""
        nibble = np.arange(16, dtype=np.uint8)
        packed = np.zeros((255, 17), np.uint8)
        packed[:, 0], packed[:, 1:] = np.arange(255), nibble | nibble << 4
        overflow = {253: (6, 7), 254: (4, 5, 6, 7)}
        for exponent, big in overflow.items():  # their finite codes only, still in both nibbles
            kept = np.array([c for c in range(16) if c & 7 not in big], np.uint8)
            packed[exponent, 1:] = np.resize(kept | kept << 4, 16)
        self.assert_decode_matches_oracle(39, packed, 32, 16384, flush=True)
        x = np.ones((1, 32), np.float32)
        nonfinite = [(e, c) for e, big in overflow.items() for c in range(16) if c & 7 in big] + [(255, c) for c in range(16)]
        for exponent, code in nonfinite:
            block = np.zeros((1, 17), np.uint8)
            block[0, 0], block[0, 1] = exponent, code
            with self.subTest(exponent=exponent, code=code):
                for mode in ("oracle", "fast"):
                    code_, _, _, stderr = run_indexed(mode, 39, 32, 32, 1, 1, block.tobytes(), [0], x, 1)
                    self.assertEqual((code_, stderr), (NONFINITE, b"qwen38 Metal: error bits 1\n"), mode)

    # ---- R1.5b fused gate+up+SwiGLU (q38_packed_encode_gate_up) ----

    def gate_up_case(self, label, kind, physical, logical, n, experts, gate, up, ids, x, divisor, view=(), decoded=None):
        """Fused kernel on one case: supported, bit-identical to the unfused fast sequence (when
        act is dense), finite, (a) against FP64 and (b) against the oracle sequence."""
        code, fused, dg, du, stderr = run_gate_up("fused", kind, physical, logical, n, experts, gate, up, ids, x, divisor, view=view)
        self.assertNotEqual(code, UNSUPPORTED, f"{label}: {NAMES[kind]} has no fused gate+up")
        self.assertEqual((code, stderr), (0, b""), label)
        dense = view[1:2] in ((), (n,))
        plain = view[:1] + (n,) + view[2:] if view else ()
        if dense:
            code, sequence, _, _, stderr = run_gate_up("sequence", kind, physical, logical, n, experts, gate, up, ids, x, divisor, view=plain)
            self.assertEqual((code, stderr), (0, b""), label)
            np.testing.assert_array_equal(fused.view("<u4"), sequence.view("<u4"), label)
        code, y_oracle, _, _, stderr = run_gate_up("oracle", kind, physical, logical, n, experts, gate, up, ids, x, divisor, view=plain)
        self.assertEqual((code, stderr), (0, b""), label)
        dg, du = (dg, du) if decoded is None else decoded
        ref, magnitude, flush = gate_up_fp64(dg, du, ids, x, divisor, logical, n)
        stats = gate_up_against_fp64(fused, ref, magnitude, flush, logical) | dict(vs_oracle=against_oracle(fused, y_oracle, magnitude))
        RESULTS.setdefault(label, {})["fused"] = stats
        CORPUS.setdefault("moe_gate_up", {})[label] = scaled_difference(fused, y_oracle, magnitude).ravel()
        tol = TOLERANCES["kernels"].get("moe_gate_up", {}).get("tol")
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertIsNotNone(tol, "moe_gate_up: fused kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return fused

    def test_fused_gate_up_bit_identical_to_sequence(self):
        """Fused act has the bits of the unfused fast sequence (IDX gate, IDX up, ACT SILU, MUL):
        the gate layer's K 2560 (n 160: the decoded rows cross stdout) at 1 and 3 tokens x 10 slots
        (x divisor 10), and shapes
        off the tile and chunk (Review Focus 1), every format with a row dot."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng, block = seeded(3840, kind), oracle.FORMATS[kind][0]
            shapes = [(160, 2560, 2560, 10, 1, 10), (160, 2560, 2560, 10, 3, 10), (1, block, block, 1, 2, 3),
                      (33, 768, 600, 2, 3, 3), (3, 2560 if block == 256 else 64, 2560 if block == 256 else 40, 10, 1, 10)]
            for n, physical, logical, slots, tokens, experts in shapes:
                with self.subTest(kernel=name, n=n, physical=physical, logical=logical, slots=slots, tokens=tokens):
                    ids = np.concatenate([rng.permutation(experts)[:slots] for _ in range(tokens)])
                    x = rng.uniform(-2, 2, (tokens, logical)).astype(np.float32)
                    gate, up = (quant_rows(kind, physical, experts * n, rng).tobytes() for _ in range(2))
                    self.gate_up_case(f"gate-up:{name}:N{n}:K{logical}/{physical}:T{tokens}x{slots}", kind, physical,
                                      logical, n, experts, gate, up, ids, x, slots)

    def test_fused_gate_up_real_within_tolerance(self):
        """The real IQ2_XXS and Q4_K gate/up rows (layers 0, 23, 47, 48, MTP): each row an expert's
        gate, the next row its up, every saved and seeded x row against every expert."""
        for kind in (16, 12):
            real = [case for case in CASES if case[1] == kind and case[0].startswith("real:")]
            self.assertTrue(real, f"no real {NAMES[kind]} slices")
            for label, _, physical, logical, data, x, saved in real:
                experts = len(saved)
                up = np.roll(np.frombuffer(data, np.uint8).reshape(experts, -1), 1, axis=0).tobytes()
                with self.subTest(case=label):
                    self.gate_up_case(f"gate-up-{label}", kind, physical, logical, 1, experts, data, up,
                                      np.tile(np.arange(experts), len(x)), x, experts)

    def test_logical_k_padding_gate_up(self):
        """Review Focus 2 (gate+up): poisoned padding past k_logical (the 256-column
        formats: K 600 of 1024, the block 768..1023 wholly padded, a partial chunk at 600) never
        reaches act and x is never read past k_logical (NaN there)."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng, block = seeded(3841, kind), oracle.FORMATS[kind][0]
            physical, logical, n, rows = 768 if block == 32 else 1024, 600, 33, 4
            clean = [quant_rows(kind, physical, 2 * n, rng) for _ in range(2)]
            raw = clean
            for column in range(-(-logical // block) * block, physical, block):
                raw = [poison_block(kind, r, slice(None), column, np.inf) for r in raw]
            ids, x = rng.integers(0, 2, rows), rng.uniform(-1, 1, (rows, logical)).astype(np.float32)
            _, _, dg, du, _ = run_gate_up("fused", kind, physical, logical, n, 2, clean[0].tobytes(), clean[1].tobytes(), ids, x, 1)
            with self.subTest(kernel=name):
                self.gate_up_case(f"gate-up-padding:{name}:N{n}:K{logical}/{physical}", kind, physical, logical, n, 2,
                                  raw[0].tobytes(), raw[1].tobytes(), ids, x, 1, view=(logical + 1, n, 0), decoded=(dg, du))

    def test_nonfinite_sets_error_word_gate_up(self):
        """Review Focus 3 (gate+up): NaN/Inf in x, a nonfinite gate or up block scale, and a finite
        g and u whose product overflows (act = Inf) each set bit 1."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3842, kind)
            gate, up = (quant_rows(kind, 2560, 2 * 33, rng) for _ in range(2))
            x, ids = rng.uniform(-1, 1, (3, 2560)).astype(np.float32), [1, 0, 1]
            cases = [("x nan", gate, up, np.nan), ("x inf", gate, up, np.inf),
                     ("gate scale", poison_block(kind, gate, 33 + 5, 96, np.inf), up, None),
                     ("up scale", gate, poison_block(kind, up, 33 + 5, 96, np.nan), None), ("overflow", gate, up, 1e22)]
            for label, g, u, value in cases:
                xs = x.copy()
                if value is not None:
                    xs[1] = value if label == "overflow" else xs[1]
                    xs[1, 100] = value
                with self.subTest(kernel=name, poison=label):
                    code, act, _, _, stderr = run_gate_up("fused", kind, 2560, 2560, 33, 2, g.tobytes(), u.tobytes(), ids, xs, 1)
                    self.assertEqual((code, stderr), (NONFINITE, b"qwen38 Metal: error bits 1\n"))
                    if label == "overflow":
                        self.assertFalse(np.isfinite(act[1]).all())

    def test_rows_bit_independent_gate_up(self):
        """Review Focus 4 (gate+up): an act element's bits depend neither on the number of tokens or
        slots, nor on its slot, nor on its n_out position, nor on the expert id."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3843, kind)
            n, k, tokens, slots, experts = 37, 2560, 4, 5, 8
            gate, up = (quant_rows(kind, k, experts * n, rng) for _ in range(2))
            x, ids = rng.uniform(-1, 1, (tokens, k)).astype(np.float32), rng.integers(0, experts, tokens * slots)
            fused = lambda g, u, i, xs, d, nn=n: run_gate_up("fused", kind, k, k, nn, experts, g.tobytes(), u.tobytes(), i, xs, d)[1]
            with self.subTest(kernel=name):
                full = self.gate_up_case(f"gate-up-rows:{name}:N{n}:T{tokens}x{slots}", kind, k, k, n, experts,
                                         gate.tobytes(), up.tobytes(), ids, x, slots)
                for t in ([0], [3], [1, 2], [3, 2, 1, 0]):  # token subsets and order
                    rows = (np.array(t)[:, None] * slots + np.arange(slots)).ravel()
                    np.testing.assert_array_equal(fused(gate, up, ids[rows], x[t], slots).view("<u4"), full[rows].view("<u4"))
                np.testing.assert_array_equal(fused(gate, up, ids, np.repeat(x, slots, 0), 1).view("<u4"), full.view("<u4"))
                order = np.arange(slots)[::-1]  # slots reversed within each token
                rows = (np.arange(tokens)[:, None] * slots + order).ravel()
                np.testing.assert_array_equal(fused(gate, up, ids[rows], x, slots).view("<u4"), full[rows].view("<u4"))
                shift = lambda w: w.reshape(experts, n, -1)[:, 1:].copy()  # weight row j+1 becomes row j
                np.testing.assert_array_equal(fused(shift(gate), shift(up), ids, x, slots, n - 1).view("<u4"), full[:, 1:].view("<u4"))
                flip = lambda w: w.reshape(experts, n, -1)[::-1].copy()  # expert e becomes experts-1-e
                np.testing.assert_array_equal(fused(flip(gate), flip(up), experts - 1 - ids, x, slots).view("<u4"), full.view("<u4"))

    def test_strided_offset_views_gate_up(self):
        """Review Focus 5 (gate+up): pitches above K*4 and n*4 and nonzero offsets change no bit."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3844, kind)
            n, k, rows, experts = 33, 768, 6, 3
            gate, up = (quant_rows(kind, k, experts * n, rng).tobytes() for _ in range(2))
            ids, x = rng.integers(0, experts, rows), rng.uniform(-1, 1, (rows // 2, 640)).astype(np.float32)
            with self.subTest(kernel=name):
                plain = self.gate_up_case(f"gate-up-views:{name}", kind, k, 640, n, experts, gate, up, ids, x, 2)
                for view in ((640 + 12, n + 7, 37), (640, n, 5 * 640), (640 + 1, n + 1, 1)):
                    with self.subTest(kernel=name, view=view):
                        strided = self.gate_up_case(f"gate-up-views:{name}:{view}", kind, k, 640, n, experts, gate, up,
                                                    ids, x, 2, view=view)
                        np.testing.assert_array_equal(strided.view("<u4"), plain.view("<u4"))

    def test_expert_id_bounds_gate_up(self):
        """Review Focus 5 (gate+up): an expert id >= n_experts sets bit 2 and leaves its act row
        unwritten (still NaN); the other rows are unchanged."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3845, kind)
            n, experts = 33, 3
            gate, up = (quant_rows(kind, 768, experts * n, rng).tobytes() for _ in range(2))
            x = rng.uniform(-1, 1, (6, 640)).astype(np.float32)
            ids, bad = np.array([0, 1, 3, 2, 7, 0xFFFFFFFF]), np.array([False, False, True, False, True, True])
            with self.subTest(kernel=name):
                code, act, _, _, stderr = run_gate_up("fused", kind, 768, 640, n, experts, gate, up, ids, x, 1)
                self.assertEqual((code, stderr), (GPU_ERROR, b"qwen38 Metal: error bits 2\n"))
                want = run_gate_up("fused", kind, 768, 640, n, experts, gate, up, np.where(bad, 0, ids), x, 1)[1]
                np.testing.assert_array_equal(act[~bad].view("<u4"), want[~bad].view("<u4"))
                np.testing.assert_array_equal(act[bad].view("<u4"), np.full((3, n), np.nan, np.float32).view("<u4"))

    def test_gate_up_selection(self):
        """q38_packed_encode_gate_up answers UNSUPPORTED (the caller runs the unfused sequence) for
        a format without a row dot (Q8_0), gate and up formats that differ, and whenever the
        indexed oracle is selected (indexed, 1, all, a typo: R18); otherwise it fuses."""
        rng = np.random.default_rng(3846)
        x, ids = rng.uniform(-1, 1, (1, 2560)).astype(np.float32), [0, 2]
        rows = {kind: quant_rows(kind, 2560, 3 * 4, rng).tobytes() for kind in (16, 12)}
        args = lambda kinds, g, u: (kinds, 2560, 2560, 4, 3, g, u, ids, x, 2)
        for env in ({}, {"Q38_REF_KERNELS": "0"}, {"Q38_REF_KERNELS": "linear"}, {"Q38_REF_KERNELS": "moe"}):
            with self.subTest(env=env):
                self.assertEqual(run_gate_up("fused", *args(16, rows[16], rows[16]), env=env)[0], 0)
        for value in ("indexed", "1", "all", "linear,indexed", "indexd"):
            with self.subTest(value=value):
                self.assertEqual(run_gate_up("fused", *args(16, rows[16], rows[16]), env={"Q38_REF_KERNELS": value})[0], UNSUPPORTED)
        with self.subTest(case="gate IQ2_XXS, up Q4_K"):
            self.assertEqual(run_gate_up("fused", *args((16, 12), rows[16], rows[12]))[0], UNSUPPORTED)
        q8 = packed.packed_fixture(8, 2560, 12, rng)
        with self.subTest(case="q8_0"):
            self.assertEqual(run_gate_up("fused", *args(8, q8, q8))[0], UNSUPPORTED)

    # ---- R1.5c fused routed down + weighted reduce (q38_packed_encode_down_reduce) ----

    def down_case(self, label, kind, physical, logical, n, experts, data, ids, inputs, view=(), decoded=None):
        """Fused kernel on one case: supported, y AND routed bit-identical to the unfused fast
        sequence (IDX down + q38f_reduce, same views), finite, (a) against FP64 and (b) against
        the oracle sequence (indexed oracle down + q38f_reduce)."""
        run = lambda mode: run_down_reduce(mode, kind, physical, logical, n, experts, data, ids, *inputs, view=view)
        code, y, routed, cpu, stderr = run("fused")
        self.assertNotEqual(code, UNSUPPORTED, f"{label}: {NAMES[kind]} has no fused down+reduce")
        self.assertEqual((code, stderr), (0, b""), label)
        code, y_seq, routed_seq, _, stderr = run("sequence")
        self.assertEqual((code, stderr), (0, b""), label)
        np.testing.assert_array_equal(y.view("<u4"), y_seq.view("<u4"), label)
        np.testing.assert_array_equal(routed.view("<u4"), routed_seq.view("<u4"), label)
        code, y_oracle, routed_oracle, _, stderr = run("oracle")
        self.assertEqual((code, stderr), (0, b""), label)
        refs = down_reduce_fp64(cpu if decoded is None else decoded, ids, *inputs, logical, n)
        slots, tol = inputs[1].shape[1], TOLERANCES["kernels"].get("moe_down_reduce", {}).get("tol")
        for name, out, out_oracle, (ref, magnitude, ulps) in (("routed", routed, routed_oracle, refs[0]), ("y", y, y_oracle, refs[1])):
            stats = down_reduce_against_fp64(out, ref, magnitude, ulps, logical, slots) | dict(
                vs_oracle=against_oracle(out, out_oracle, magnitude))
            RESULTS.setdefault(label, {})[name] = stats
            CORPUS.setdefault("moe_down_reduce", {})[f"{label}:{name}"] = scaled_difference(out, out_oracle, magnitude).ravel()
            self.assertTrue(stats["finite"] and stats["condition_a"], (label, name, stats))
            self.assertIsNotNone(tol, "moe_down_reduce: fused kernel without a measured tolerance")
            self.assertLessEqual(stats["vs_oracle"]["max"], tol, (label, name))
        return y, routed

    def test_fused_down_bit_identical_to_sequence(self):
        """Fused y and routed have the bits of the unfused fast sequence (IDX down, q38f_reduce): the
        down layer's K 640 of 768 (MXFP4: 640; n 160: the decoded rows cross stdout) at 1 and 3
        tokens x 10 slots, and shapes off the tile and chunk (Review Focus 1), every format with a
        row dot."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng, block = seeded(3850, kind), oracle.FORMATS[kind][0]
            down = (640, 640) if block == 32 else (768, 640)
            shapes = [(160, *down, 10, 1, 10), (160, *down, 10, 3, 10), (1, block, block, 1, 2, 3),
                      (33, 768, 600, 2, 3, 3), (3, 2560 if block == 256 else 64, 2560 if block == 256 else 40, 10, 1, 10)]
            for n, physical, logical, slots, tokens, experts in shapes:
                with self.subTest(kernel=name, n=n, physical=physical, logical=logical, slots=slots, tokens=tokens):
                    ids = np.concatenate([rng.permutation(experts)[:slots] for _ in range(tokens)])
                    data = quant_rows(kind, physical, experts * n, rng).tobytes()
                    self.down_case(f"down-reduce:{name}:N{n}:K{logical}/{physical}:T{tokens}x{slots}", kind, physical,
                                   logical, n, experts, data, ids, down_inputs(rng, tokens, slots, logical, n, 2.0))

    def test_fused_down_real_within_tolerance(self):
        """The real Q2_K and MXFP4 down rows (layers 0, 23, 47, 48, MTP): each row an expert (n 1),
        every saved and seeded x row a token whose slots are all those experts."""
        for kind in (10, 39):
            real = [case for case in CASES if case[1] == kind and case[0].startswith("real:")]
            self.assertTrue(real, f"no real {NAMES[kind]} slices")
            rng = seeded(3851, kind)
            for label, _, physical, logical, data, x, saved in real:
                experts = len(saved)
                _, scores, shared, sigmoid = down_inputs(rng, len(x), experts, logical, 1)
                with self.subTest(case=label):
                    self.down_case(f"down-reduce-{label}", kind, physical, logical, 1, experts, data,
                                   np.tile(np.arange(experts), len(x)), (np.repeat(x, experts, 0), scores, shared, sigmoid))

    def test_logical_k_padding_down_reduce(self):
        """Review Focus 2 (down+reduce): poisoned padding past k_logical (K 600 of 768 or 1024, a
        partial chunk at 600, whole padded blocks) never reaches y or routed, and act is never
        read past k_logical (NaN in the gap of a pitch-601 view)."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng, block = seeded(3852, kind), oracle.FORMATS[kind][0]
            physical, logical, n, tokens, slots = 768 if block == 32 else 1024, 600, 33, 2, 3
            clean = quant_rows(kind, physical, 3 * n, rng)
            raw = clean
            for column in range(-(-logical // block) * block, physical, block):
                raw = poison_block(kind, raw, slice(None), column, np.inf)
            ids, inputs = rng.integers(0, 3, tokens * slots), down_inputs(rng, tokens, slots, logical, n)
            decoded = run_down_reduce("fused", kind, physical, logical, n, 3, clean.tobytes(), ids, *inputs)[3]
            with self.subTest(kernel=name):
                self.down_case(f"down-reduce-padding:{name}:N{n}:K{logical}/{physical}", kind, physical, logical, n, 3,
                               raw.tobytes(), ids, inputs, view=(logical + 1, n, 0), decoded=decoded)

    def test_nonfinite_sets_error_word_down_reduce(self):
        """Review Focus 3 (down+reduce): NaN/Inf in act, a nonfinite down block scale, Inf in shared,
        NaN in sigmoid, and finite dots whose y overflows (shared at FLT_MAX) each set bit 1."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3853, kind)
            raw, ids = quant_rows(kind, 2560, 2 * 33, rng), [1, 0, 1, 0]
            act, scores, shared, sigmoid = down_inputs(rng, 2, 2, 2560, 33)
            big = np.full_like(shared, np.finfo(np.float32).max)
            cases = [("act nan", "act", np.nan), ("act inf", "act", np.inf), ("down scale", "raw", None),
                     ("shared inf", "shared", np.inf), ("sigmoid nan", "sigmoid", np.nan), ("overflow", "big", None)]
            for label, target, value in cases:
                a, w, sh, sg = act.copy(), raw, shared, sigmoid.copy()
                if target == "act":
                    a[2, 100] = value
                elif target == "raw":
                    w = poison_block(kind, raw, 33 + 5, 96, np.inf)
                elif target == "shared":
                    sh = shared.copy(); sh[1, 7] = value
                elif target == "sigmoid":
                    sg[1] = value
                else:
                    a, sh, sg = act * 1e33, big, np.ones_like(sigmoid)
                with self.subTest(kernel=name, poison=label):
                    code, y, routed, _, stderr = run_down_reduce("fused", kind, 2560, 2560, 33, 2, w.tobytes(), ids,
                                                                 a, scores, sh, sg)
                    self.assertEqual((code, stderr), (NONFINITE, b"qwen38 Metal: error bits 1\n"))
                    if label == "overflow":
                        self.assertTrue(np.isfinite(routed).all() and not np.isfinite(y).all())

    def test_rows_bit_independent_down_reduce(self):
        """Review Focus 4 (down+reduce): a token's y and routed bits depend neither on the number
        or order of the tokens in the dispatch, nor on n's position, nor on the expert ids."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3854, kind)
            n, k, tokens, slots, experts = 37, 2560, 4, 5, 8
            raw = quant_rows(kind, k, experts * n, rng)
            ids, (act, scores, shared, sigmoid) = rng.integers(0, experts, tokens * slots), down_inputs(rng, tokens, slots, k, n)
            fused = lambda w, i, t, nn=n, cols=slice(None): run_down_reduce(
                "fused", kind, k, k, nn, experts, w.tobytes(), i, act[(np.array(t)[:, None] * slots + np.arange(slots)).ravel()],
                scores[t], shared[t][:, cols], sigmoid[t])[1:3]
            with self.subTest(kernel=name):
                full = self.down_case(f"down-reduce-rows:{name}:N{n}:T{tokens}x{slots}", kind, k, k, n, experts,
                                      raw.tobytes(), ids, (act, scores, shared, sigmoid))
                pick = lambda t: ids[(np.array(t)[:, None] * slots + np.arange(slots)).ravel()]
                for t in ([0], [3], [1, 2], [3, 2, 1, 0]):  # token subsets and order
                    for got, want in zip(fused(raw, pick(t), t), full):
                        np.testing.assert_array_equal(got.view("<u4"), want[t].view("<u4"))
                every = list(range(tokens))
                shift = raw.reshape(experts, n, -1)[:, 1:].copy()  # weight row j+1 becomes row j
                for got, want in zip(fused(shift, ids, every, n - 1, slice(1, None)), full):
                    np.testing.assert_array_equal(got.view("<u4"), want[:, 1:].view("<u4"))
                flip = raw.reshape(experts, n, -1)[::-1].copy()  # expert e becomes experts-1-e
                for got, want in zip(fused(flip, experts - 1 - ids, every), full):
                    np.testing.assert_array_equal(got.view("<u4"), want.view("<u4"))

    def test_strided_offset_views_down_reduce(self):
        """Review Focus 5 (down+reduce): act and y pitches above K*4 and n*4 and nonzero offsets
        change no bit of y or routed (the sequence runs on the same views)."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3855, kind)
            n, k, tokens, slots, experts = 33, 768, 3, 2, 3
            raw = quant_rows(kind, k, experts * n, rng).tobytes()
            ids, inputs = rng.integers(0, experts, tokens * slots), down_inputs(rng, tokens, slots, 640, n)
            with self.subTest(kernel=name):
                plain = self.down_case(f"down-reduce-views:{name}", kind, k, 640, n, experts, raw, ids, inputs)
                for view in ((640 + 12, n + 7, 37), (640, n, 5 * 640), (640 + 1, n + 1, 1)):
                    with self.subTest(kernel=name, view=view):
                        strided = self.down_case(f"down-reduce-views:{name}:{view}", kind, k, 640, n, experts, raw, ids,
                                                 inputs, view=view)
                        for got, want in zip(strided, plain):
                            np.testing.assert_array_equal(got.view("<u4"), want.view("<u4"))

    def test_expert_id_bounds_down_reduce(self):
        """Review Focus 5 (down+reduce): an expert id >= n_experts in any slot of a token sets bit 2
        (that token's y is unspecified); the other tokens' y and routed are unchanged."""
        for name, kind in sorted(INDEXED_KINDS.items()):
            rng = seeded(3856, kind)
            n, experts, slots = 33, 3, 2
            raw = quant_rows(kind, 768, experts * n, rng).tobytes()
            inputs = down_inputs(rng, 4, slots, 640, n)
            ids, bad = np.array([0, 1, 2, 3, 1, 2, 0xFFFFFFFF, 0]), np.array([False, True, False, True])
            with self.subTest(kernel=name):
                code, y, routed, _, stderr = run_down_reduce("fused", kind, 768, 640, n, experts, raw, ids, *inputs)
                self.assertEqual((code, stderr), (GPU_ERROR, b"qwen38 Metal: error bits 2\n"))
                _, y_ok, routed_ok, _, _ = run_down_reduce("fused", kind, 768, 640, n, experts, raw, np.where(ids >= experts, 0, ids),
                                                           *inputs)
                np.testing.assert_array_equal(y[~bad].view("<u4"), y_ok[~bad].view("<u4"))
                np.testing.assert_array_equal(routed[~bad].view("<u4"), routed_ok[~bad].view("<u4"))

    def test_down_reduce_selection(self):
        """q38_packed_encode_down_reduce answers UNSUPPORTED (the caller runs the unfused sequence) for
        a format without a row dot (Q8_0) and whenever the indexed oracle is selected (indexed, 1,
        all, a typo: R18); otherwise (unset, 0, linear, moe: set explicitly) it fuses."""
        rng = np.random.default_rng(3857)
        inputs, ids = down_inputs(rng, 1, 2, 2560, 4), [0, 2]
        q2 = quant_rows(10, 2560, 3 * 4, rng).tobytes()
        for env in ({}, {"Q38_REF_KERNELS": "0"}, {"Q38_REF_KERNELS": "linear"}, {"Q38_REF_KERNELS": "moe"}):
            with self.subTest(env=env):
                self.assertEqual(run_down_reduce("fused", 10, 2560, 2560, 4, 3, q2, ids, *inputs, env=env)[0], 0)
        for value in ("indexed", "1", "all", "linear,indexed", "indexd"):
            with self.subTest(value=value):
                self.assertEqual(run_down_reduce("fused", 10, 2560, 2560, 4, 3, q2, ids, *inputs,
                                                 env={"Q38_REF_KERNELS": value})[0], UNSUPPORTED)
        with self.subTest(case="q8_0"):
            q8 = packed.packed_fixture(8, 2560, 12, rng)
            self.assertEqual(run_down_reduce("fused", 8, 2560, 2560, 4, 3, q8, ids, *inputs)[0], UNSUPPORTED)


    def assert_route_identical(self, label, logits, slots, norm, experts=None):
        """q38_moe_route_fast has the bits of q38f_route in ids, weights, probs and error word."""
        fast, oracle = (run_route(m, logits, slots, norm, experts) for m in ("fast", "oracle"))
        for name, got, want in zip(("error word", "ids", "weights", "probs"), fast, oracle):
            self.assertEqual(np.asarray(got).tobytes(), np.asarray(want).tobytes(), f"{label}: {name}")
        return oracle

    def test_route_fast_bit_identical(self):
        """Bit identity with q38f_route over tokens x experts x slots x norm, row shapes cycling (random,
        wide, ties, equal, +-0, ties across the k-th place, p = 0 ties, ramps, ULP neighbours). The ids also
        follow the (larger p, smaller id) rule computed independently from the returned probabilities."""
        rng = np.random.default_rng(3911)
        for tokens in (1, 3, 32, 128):
            for experts, slots in ((512, 1), (512, 10), (33, 1), (33, 10), (1, 1)):
                for norm in (0, 1):
                    with self.subTest(tokens=tokens, experts=experts, slots=slots, norm=norm):
                        logits = route_logits(rng, tokens, experts, offset=tokens + experts + slots)
                        word, ids, weights, probs = self.assert_route_identical("route", logits, slots, norm)
                        self.assertEqual(word, 0)
                        np.testing.assert_array_equal(ids, route_reference_ids(probs, slots))

    def test_route_fast_every_row_shape(self):
        """Every row shape on its own row (one token: no neighbour hides a wrong tie), at the boundary sizes."""
        rng = np.random.default_rng(3913)
        for experts, slots in ((512, 10), (33, 10), (10, 10), (1024, 10)):
            for offset in range(ROUTE_KINDS):
                with self.subTest(experts=experts, offset=offset):
                    logits = route_logits(rng, 1, experts, offset)
                    word, ids, _, probs = self.assert_route_identical("shape", logits, slots, offset % 2)
                    self.assertEqual(word, 0)
                    np.testing.assert_array_equal(ids, route_reference_ids(probs, slots))

    def test_route_fast_pitch(self):
        """A router row pitch past the expert count (NaN in the gap) is respected, as the oracle does."""
        rng = np.random.default_rng(3917)
        logits = route_logits(rng, 5, 512, 3)
        padded = np.full((5, 512 + ROUTE_PAD), np.nan, np.float32)
        padded[:, :512] = logits
        word, ids, weights, probs = self.assert_route_identical("pitch", padded, 10, 1, 512)
        self.assertEqual(word, 0)
        for a, b in zip(run_route("fast", logits, 10, 1), (word, ids, weights, probs)):
            np.testing.assert_array_equal(a, b)

    def test_route_fast_nonfinite_row(self):
        """A NaN, +Inf or -Inf logit sets bit 1 and gives that row ids 0..k-1, zero weights and untouched
        probs (the oracle's fallback), the finite rows around it unchanged: bit for bit."""
        rng = np.random.default_rng(3919)
        logits = route_logits(rng, 7, 512, 1)
        for row, value in ((1, np.nan), (3, np.inf), (4, -np.inf), (6, np.nan)):
            logits[row, int(rng.integers(512))] = value
        for slots in (1, 10):
            with self.subTest(slots=slots):
                word, ids, weights, probs = self.assert_route_identical("nonfinite", logits, slots, 1)
                self.assertEqual(word, 1)
                for row in (1, 3, 4, 6):
                    np.testing.assert_array_equal(ids[row], np.arange(slots))
                    self.assertEqual(weights[row].tobytes(), bytes(4 * slots))
                    self.assertEqual(probs[row].tobytes(), b"\xa5" * (4 * 512))
                for row in (0, 2, 5):
                    self.assertTrue(np.isfinite(probs[row]).all() and abs(float(probs[row].sum()) - 1) < 1e-4)

    def test_route_fast_capacity_guard(self):
        """Beyond the kernel's 1024 experts, or with k > E, the kernel writes nothing and sets bit 2 (the
        host keeps q38f_route there; this is the kernel's own guard)."""
        rng = np.random.default_rng(3923)
        for experts, slots in ((2048, 10), (4, 6)):
            with self.subTest(experts=experts, slots=slots):
                word, ids, weights, probs = run_route("fast", route_logits(rng, 2, experts), slots, 1)
                self.assertEqual(word, 2)
                for part in (ids, weights, probs):
                    self.assertEqual(part.tobytes(), b"\xa5" * part.nbytes)

    # ---- R1.6a parallel RMSNorm (q38c_rms_fast, family fuse) ----
    def rms_case(self, label, x, gain, z=None, **view):
        """Fast and oracle on one case: clean error word, the expected dispatches, (a) for both, (b) for the
        fast kernel; recorded in the corpus. Returns the fast y."""
        word, dispatches, barriers, y = run_rms("fast", x, gain, z, **view)
        self.assertEqual((word, dispatches, barriers), (0, *RMS_DISPATCHES["fast"]), label)
        word, dispatches, barriers, y_oracle = run_rms("oracle", x, gain, z, **view)
        self.assertEqual((word, dispatches, barriers), (0, *RMS_DISPATCHES["oracle"]), label)
        ref, magnitude = rms_fp64(x, gain, z)
        stats, oracle_stats = (rms_against_fp64(out, ref, magnitude, z is not None) for out in (y, y_oracle))
        stats["vs_oracle"] = against_oracle(y, y_oracle, magnitude)
        RESULTS.setdefault(label, {}).update(fast=stats, oracle=oracle_stats)
        CORPUS.setdefault("rms_fast", {})[label] = scaled_difference(y, y_oracle, magnitude).ravel()
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertTrue(oracle_stats["finite"] and oracle_stats["condition_a"], (label, oracle_stats))
        tol = TOLERANCES["kernels"].get("rms_fast", {}).get("tol")
        self.assertIsNotNone(tol, "rms_fast: fast kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return y

    def test_rms_fast_against_oracle_and_fp64(self):
        """The graph's shapes: HC, final, GDN-block and PLE RMS [t, 4, 2560] with the real [4, 2560]
        hc_attn_norm gain (per stream, forward_gpu.m hc_read), the MTP h-norm [t, 1, 10240] (same gain
        flattened) and e-norm [t, 1, 2560], QSA q/k norms (24 and 2 heads x 256), indexer [t, 4, 128]
        (real q_norm gain), pooled [t, 1, 128], a 160 width, GDN norm gated (48 x 128); 1, 3, 32 tokens."""
        rng = np.random.default_rng(3929)
        hc, index = real_gain("blk.0.hc_attn_norm.weight"), real_gain("blk.3.indexer.q_norm.weight")
        shapes = [(f"hc{t}", t, 4, 2560, dict(gain=hc.reshape(4, 2560))) for t in (1, 3, 32)] + [
            (f"mtp_h{t}", t, 1, 10240, dict(gain=hc)) for t in (1, 3, 32)] + [
            ("mtp_e", 3, 1, 2560, {}), ("q_norm", 3, 24, 256, {}), ("k_norm", 32, 2, 256, {}),
            ("indexer", 3, 4, 128, dict(gain=index)), ("pooled", 3, 1, 128, dict(gain=index)),
            ("w160", 3, 8, 160, {})] + [(f"gdn{t}", t, 48, 128, dict(gated=True)) for t in (1, 3, 32)]
        for name, t, groups, width, extra in shapes:
            with self.subTest(case=name):
                self.rms_case(f"rms:{name}", *rms_inputs(rng, t, groups, width, **extra))

    def test_rms_fast_widths_not_multiple_of_threadgroup(self):
        """Widths around and below the 256-thread group (1, 33, 255, 257, 1000, 4097), 1, 3 and 32 tokens,
        one group with a [width] gain and three groups with a [groups, width] gain."""
        rng = np.random.default_rng(3931)
        for width in (1, 33, 255, 257, 1000, 4097):
            for t in (1, 3, 32):
                for groups, rank in ((1, 1), (3, 2)):
                    with self.subTest(width=width, t=t, groups=groups):
                        self.rms_case(f"rms:w{width}t{t}g{groups}", *rms_inputs(rng, t, groups, width, rank))

    def test_rms_fast_rows_bit_independent(self):
        """Review Focus 4: a row's bits depend neither on the number of tokens in the dispatch nor on the
        token's position (chunk, decode and MTP stay bit-identical among fast paths)."""
        rng = np.random.default_rng(3937)
        for groups, width, rank, gated in ((4, 2560, 2, False), (1, 10240, 1, False), (3, 33, 2, False), (48, 128, 1, True)):
            with self.subTest(groups=groups, width=width):
                x, gain, z = rms_inputs(rng, 32, groups, width, rank, gated)
                y = run_rms("fast", x, gain, z)[3]
                for lo, hi in ((0, 1), (17, 18), (31, 32), (5, 8), (29, 32)):
                    part = run_rms("fast", x[lo:hi], gain, None if z is None else z[lo:hi])
                    self.assertEqual(part[:3], (0, *RMS_DISPATCHES["fast"]))
                    np.testing.assert_array_equal(part[3].view("<u4"), y[lo:hi].view("<u4"), (lo, hi))

    def test_rms_fast_nonfinite_sets_error_word(self):
        """Review Focus 3 with the check folded into the kernel: NaN/Inf in x, gain or z at the first, a
        middle or the last element, and a finite x whose sum of squares overflows, give the error word of
        check + q38c_rms (bit 1); with bit 4 already set, a nonfinite input still adds bit 1 (the check
        always runs) and an overflow adds nothing (q38c_rms exits early)."""
        rng = np.random.default_rng(3941)
        x, gain, z = rms_inputs(rng, 3, 48, 128, 2, True)
        last = (x.shape[0] - 1, x.shape[1] - 1, x.shape[2] - 1)
        for part, where, value in (("x", (0, 0, 0), np.nan), ("x", (1, 20, 77), np.inf), ("x", last, -np.inf),
                                   ("gain", (0, 0), np.nan), ("gain", (47, 127), np.inf), ("gain", (13, 5), -np.inf),
                                   ("z", (0, 0, 0), np.nan), ("z", last, np.inf), ("x", (2, 3, 9), 3e38)):
            for gated in (False, True):
                if part == "z" and not gated:
                    continue
                # GDN_NORM takes a [width] gain: its row 0 (the [groups, width] gain's last index there)
                inputs = dict(x=x.copy(), gain=gain[0].copy() if gated else gain.copy(), z=z.copy() if gated else None)
                inputs[part][where[1:] if part == "gain" and gated else where] = value
                for initial, want in ((0, 1), (4, 4 if np.isfinite(value) else 5)):
                    with self.subTest(part=part, where=where, value=value, gated=gated, initial=initial):
                        words = [run_rms(mode, inputs["x"], inputs["gain"], inputs["z"], initial=initial)[0]
                                 for mode in ("fast", "oracle")]
                        self.assertEqual(words, [want, want])
        with self.subTest(case="clean with bit 4 set"):
            self.assertEqual([run_rms(m, x, gain, initial=4)[0] for m in ("fast", "oracle")], [4, 4])

    def test_rms_fast_strided_offset_views(self):
        """Review Focus 5: a row pitch past the width (NaN in the gap: a read there would set bit 1) and a
        nonzero offset change no bit of y; no float outside the y view is written (the harness checks)."""
        rng = np.random.default_rng(3943)
        for groups, width, rank, gated in ((4, 2560, 2, False), (1, 10240, 1, False), (3, 33, 2, False), (48, 128, 1, True),
                                           (2, 1, 2, False)):
            with self.subTest(groups=groups, width=width):
                x, gain, z = rms_inputs(rng, 3, groups, width, rank, gated)
                y = run_rms("fast", x, gain, z)[3]
                for pitch, offset in ((width + 5, 3), (width + 64, 0), (width, 7)):
                    word, dispatches, barriers, strided = run_rms("fast", x, gain, z, pitch=pitch, offset=offset)
                    self.assertEqual((word, dispatches, barriers), (0, *RMS_DISPATCHES["fast"]))
                    np.testing.assert_array_equal(strided.view("<u4"), y.view("<u4"), (pitch, offset))

    def test_rms_fast_inplace(self):
        """y over x (the admitted RMS alias): the fast kernel reads the whole row before any thread writes,
        so the bits equal the out-of-place run; the oracle's in-place run equals its own out-of-place one."""
        rng = np.random.default_rng(3947)
        for groups, width, rank in ((1, 10240, 1), (3, 257, 2)):
            x, gain, _ = rms_inputs(rng, 3, groups, width, rank)
            for mode in ("fast", "oracle"):
                with self.subTest(mode=mode, width=width):
                    out, same = run_rms(mode, x, gain), run_rms(mode, x, gain, pitch=width + 2, offset=1, inplace=True)
                    self.assertEqual(same[:3], out[:3])
                    np.testing.assert_array_equal(same[3].view("<u4"), out[3].view("<u4"))

    def test_rms_selection(self):
        """q38c_gpu_encode runs q38c_rms_fast (1 dispatch, 1 barrier, the fast bits) unless Q38_REF_KERNELS
        selects fuse at creation (fuse, 1, all, a list with fuse, a typo: R18), then exactly check + barrier
        + q38c_rms + barrier with the oracle's bits; RMS and GDN_NORM alike."""
        rng = np.random.default_rng(3949)
        for gated in (False, True):
            x, gain, z = rms_inputs(rng, 3, 48, 128, 1, gated)
            want = {mode: run_rms(mode, x, gain, z) for mode in ("fast", "oracle")}
            for value in (None, "", "0", "linear", "moe,indexed", "fuse", "1", "all", "linear,fuse", "fuze"):
                mode = "oracle" if value in ("fuse", "1", "all", "linear,fuse", "fuze") else "fast"
                with self.subTest(gated=gated, value=value):
                    got = run_rms("encode", x, gain, z, env={} if value is None else {"Q38_REF_KERNELS": value})
                    self.assertEqual(got[:3], (0, *RMS_DISPATCHES[mode]))
                    np.testing.assert_array_equal(got[3].view("<u4"), want[mode][3].view("<u4"))


    # ---- R2.1 split decode attention (q38d_attend_encode, family attention) ----
    def attend_case(self, label, inputs):
        """Fast and oracle on one case: clean words, 1 dispatch for the oracle and for one split, 2 with the merge;
        both within (a); fast within (b) of the oracle."""
        fast, oracle = (run_attend(mode, *inputs) for mode in ("fast", "oracle"))
        splits = min(64, max(1, -(-len(inputs[4]) // 32)))
        self.assertEqual((fast[:2], oracle[:2]), ((0, 1 if splits == 1 else 2), (0, 1)), label)
        ref, magnitude, factor = attend_fp64(*inputs)
        stats, oracle_stats = (attend_against_fp64(out[2], ref, magnitude, factor) for out in (fast, oracle))
        stats["vs_oracle"] = against_oracle(fast[2], oracle[2], magnitude)
        RESULTS.setdefault(label, {}).update(fast=stats, oracle=oracle_stats)
        CORPUS.setdefault("attend_fast", {})[label] = scaled_difference(fast[2], oracle[2], magnitude).ravel()
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertTrue(oracle_stats["finite"] and oracle_stats["condition_a"], (label, oracle_stats))
        tol = TOLERANCES["kernels"].get("attend_fast", {}).get("tol")
        self.assertIsNotNone(tol, "attend_fast: fast kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return fast

    def test_attend_fast_against_oracle_and_fp64(self):
        """The real dims (24 heads over 2 KV heads, d 256) from 1 key to the 2048-key budget and past the 64-split
        cap (keys per split grow), then groups off the 4x4 simdgroup slices (1, 3, 5, 12, 16 heads per KV
        head), d = 32 (one feature per lane) and 128, and sharp logits (one key dominates)."""
        rng = np.random.default_rng(4099)
        for m in (1, 2, 31, 32, 33, 64, 65, 100, 1000, 2047, 2048, 2049, 4100):
            with self.subTest(m=m):
                self.attend_case(f"attend:{ATTEND_REAL}m{m}", attend_inputs(rng, ATTEND_REAL, max(m, 4 * m // 3), m))
        for dims, m in (((4, 4, 32), 70), ((3, 1, 128), 33), ((10, 2, 64), 200), ((12, 1, 256), 129),
                        ((32, 2, 96), 500), ((16, 16, 256), 1)):
            with self.subTest(dims=dims, m=m):
                self.attend_case(f"attend:{dims}m{m}", attend_inputs(rng, dims, m + 7, m))
        for sharp in (16.0, 64.0):
            with self.subTest(sharp=sharp):
                self.attend_case(f"attend:sharp{sharp}", attend_inputs(rng, ATTEND_REAL, 3000, 2048, sharp))

    def test_attend_fast_bits_depend_on_keys_only(self):
        """Review Focus 4 for attention: the output bits depend on the selected keys only, not on the cache length
        n or where the keys sit in it (the split layout is a function of m)."""
        rng = np.random.default_rng(4111)
        q, gate, k, v, ids = attend_inputs(rng, ATTEND_REAL, 300, 200)
        base = run_attend("fast", q, gate, k[ids], v[ids], np.arange(200, dtype=np.uint32))
        for n, offset in ((400, 0), (4000, 1234)):
            kk, vv = (np.full((n, *k.shape[1:]), np.nan, np.float32) for _ in range(2))
            spots = np.sort(rng.choice(n - offset, 200, replace=False)).astype(np.uint32) + offset
            kk[spots], vv[spots] = k[ids], v[ids]
            got = run_attend("fast", q, gate, kk, vv, spots)
            self.assertEqual(got[:2], base[:2])
            np.testing.assert_array_equal(got[2].view("<u4"), base[2].view("<u4"), (n, offset))

    def test_attend_fast_error_word(self):
        """Review Focus 3: NaN/Inf in q, gate, a selected key or value -> bit 1 (an unselected NaN row is never
        read: clean); an id past the cache, a repeated or decreasing id, or a gate outside [0, 1] -> bit 2;
        each with one split and with the merge, as the oracle."""
        rng = np.random.default_rng(4127)
        for m in (20, 300):
            q, gate, k, v, ids = attend_inputs(rng, ATTEND_REAL, m + 50, m)
            unused = np.setdiff1d(np.arange(m + 50), ids)[0]
            faults = [("q", lambda a: a[0].__setitem__((3, 7), np.nan), 1),
                      ("gate", lambda a: a[1].__setitem__((5, 0), np.nan), 1),
                      ("key", lambda a: a[2].__setitem__((ids[m // 2], 1, 9), np.inf), 1),
                      ("value", lambda a: a[3].__setitem__((ids[-1], 0, 0), np.nan), 1),
                      ("unused", lambda a: a[2].__setitem__((unused, 0, 0), np.nan), 0),
                      ("id range", lambda a: a[4].__setitem__(-1, m + 50), 2),
                      ("id order", lambda a: a[4].__setitem__(m // 2, a[4][m // 2 - 1]), 2),
                      ("gate range", lambda a: a[1].__setitem__((0, 1), 1.5), 2)]
            for name, poke, bit in faults:
                for mode in ("fast", "oracle"):
                    with self.subTest(m=m, fault=name, mode=mode):
                        inputs = [a.copy() for a in (q, gate, k, v, ids)]
                        poke(inputs)
                        self.assertEqual(run_attend(mode, *inputs)[0], bit)

    def test_index_scores_and_selection_bit_identical(self):
        """R2.2: the exact fast scores, rank and compact give the oracle's bytes (scores, IDs, error word): real
        index dims (4 heads x 128) from one block to 32K tokens, ties (equal scores: smaller ID wins), a
        visible tail, other head counts/dims on and off the fast path (h > 8, d % 16 != 0 fall back), and
        NaN/Inf in the query or one block key (bit 1)."""
        rng = np.random.default_rng(4139)
        cases = [((4, 128), visible, 4, 2048) for visible in (4, 7, 2049, 8192, 8195, 32768)]
        cases += [((1, 16), 300, 4, 64), ((8, 128), 1000, 4, 256), ((3, 48), 999, 3, 300), ((9, 32), 400, 4, 64),
                  ((2, 24), 400, 4, 64)]
        for (h, d), visible, ratio, budget in cases:
            n = visible // ratio
            iq, blocks = rng.normal(0, 1, (h, d)).astype(np.float32), rng.normal(0, 1, (n, d)).astype(np.float32)
            variants = [("plain", iq, blocks)]
            if n > 4:
                tied = blocks.copy(); tied[1::3] = tied[0]
                variants += [("ties", iq, tied), ("nan key", iq, np.where(np.arange(n)[:, None] == n // 2, np.nan, blocks)),
                             ("inf query", np.where(np.arange(h)[:, None] == h - 1, np.inf, iq), blocks)]
            for name, q, k in variants:
                with self.subTest(dims=(h, d), visible=visible, case=name):
                    fast, oracle = (run_index(mode, q, k, visible, ratio, budget) for mode in ("fast", "oracle"))
                    self.assertEqual(fast[0], 1 if name in ("nan key", "inf query") else 0)
                    self.assertEqual(fast[0], oracle[0])
                    if not fast[0]:
                        self.assertEqual(fast[1:], oracle[1:])

    # ---- R2.4 fast GDN scan (q38c_gdn_scan_fast, family fuse) ----
    def test_gdn_scan_fast_against_oracle_and_fp64(self):
        """Real GDN dims (hk 16, hv 48, dk = dv 128) and smaller ones (dk 32 / 64, hv/hk 1 and 3): one token
        within (a) for out and state, fast and oracle; 1, 3 and 32 tokens within (b) of the oracle for out
        (S = sum|s' q| of the FP64 last token) and the final state (S = scan_magnitudes' running sum)."""
        rng = np.random.default_rng(4157)
        corpus = CORPUS.setdefault("gdn_scan_fast", {})
        tol = TOLERANCES["kernels"].get("gdn_scan_fast", {}).get("tol")
        self.assertIsNotNone(tol, "gdn_scan_fast: fast kernel without a measured tolerance")
        for dims in (GDN_REAL, (2, 6, 32, 8), (4, 4, 64, 12)):
            for t in (1, 3, 32):
                with self.subTest(dims=dims, t=t):
                    inputs = scan_inputs(rng, t, dims)
                    fast, oracle = (run_scan(mode, *inputs, dims) for mode in ("fast", "oracle"))
                    self.assertEqual((fast[:2], oracle[:2]), ((0, 1), (0, 2)))
                    mag, mag_s = scan_magnitudes(*inputs, dims)
                    label = f"scan:{dims}t{t}"
                    d_out = scaled_difference(fast[2][-1], oracle[2][-1], mag).ravel()
                    d_state = scaled_difference(fast[3], oracle[3], mag_s).ravel()
                    if t == 1:
                        ref, ref_s, _, _, bound, bound_s = scan_fp64(*inputs, dims)
                        for name, got in (("fast", fast), ("oracle", oracle)):
                            stats = dict(out=float((np.abs(got[2][0] - ref) / bound).max()),
                                         state=float((np.abs(got[3] - ref_s) / (bound_s + np.spacing(np.abs(ref_s).astype(np.float32)))).max()))
                            RESULTS.setdefault(label, {})[name] = stats
                            self.assertTrue(stats["out"] <= 1 and stats["state"] <= 1, (label, name, stats))
                    corpus[label] = np.concatenate([d_out, d_state])
                    RESULTS.setdefault(label, {})["vs_oracle"] = dict(out=float(d_out.max()), state=float(d_state.max()))
                    self.assertLessEqual(max(d_out.max(), d_state.max()), tol, label)

    def test_gdn_scan_fast_chunk_equals_steps(self):
        """Review Focus 4 for the scan: 7 tokens in one call give the bits of 7 one-token calls chained
        through the state (chunked prefill vs decode), outputs and final state."""
        rng = np.random.default_rng(4159)
        qkv, g, beta, state = scan_inputs(rng, 7, GDN_REAL)
        whole = run_scan("fast", qkv, g, beta, state, GDN_REAL)
        s, outs = state, []
        for i in range(7):
            word, _, out, s = run_scan("fast", qkv[i:i + 1], g[i:i + 1], beta[i:i + 1], s, GDN_REAL)
            self.assertEqual(word, 0)
            outs.append(out)
        np.testing.assert_array_equal(np.concatenate(outs).view("<u4"), whole[2].view("<u4"))
        np.testing.assert_array_equal(s.view("<u4"), whole[3].view("<u4"))

    def test_gdn_scan_fast_error_word(self):
        """Review Focus 3: NaN/Inf in q, k, v, g, beta or the state set bit 1 and g > 0 or beta outside [0, 1]
        bit 4 (q38c_check's rules; an infinite beta both), the same word as the oracle; with the word preset to
        4 an input fault still adds its bits; an overflow of the update gives bit 1, nothing when preset (the
        op is skipped, as the oracle)."""
        rng = np.random.default_rng(4161)
        dims = (2, 6, 32, 8)
        clean = scan_inputs(rng, 2, dims)
        faults = [(0, (1, 3), np.nan), (0, (0, 2 * 32 + 5), np.inf), (0, (1, 4 * 32 + 9), -np.inf),
                  (1, (0, 4), np.nan), (2, (1, 0), np.inf), (3, (5, 7, 31), np.nan), (1, (1, 2), 0.5),
                  (2, (0, 3), 1.5), (2, (1, 5), -0.25)]
        for which, at, value in faults:
            for mode in ("fast", "oracle"):
                with self.subTest(input=which, at=at, mode=mode):
                    inputs = [a.copy() for a in clean]
                    inputs[which][at] = value
                    want = (1 if not np.isfinite(value) else 0) | (4 if which in (1, 2) and (value > 0 if which == 1
                                                                                            else not 0 <= value <= 1) else 0)
                    self.assertEqual(run_scan(mode, *inputs, dims)[0], want)
                    self.assertEqual(run_scan(mode, *inputs, dims, initial=4)[0], want | 4)
        big = [a.copy() for a in clean]
        big[3][:] = 3e38
        big[0][:, 2 * 32:4 * 32] = 1 / np.sqrt(32)  # aligned k: s.k = 3e38 * sqrt(32) overflows
        for mode in ("fast", "oracle"):
            with self.subTest(overflow=mode):
                self.assertEqual(run_scan(mode, *big, dims)[0], 1)
                self.assertEqual(run_scan(mode, *big, dims, initial=4)[0], 4)

    def test_attend_selector(self):
        """encode follows Q38_REF_KERNELS: unset -> fast (bit-identical to forced fast), attention or 1 -> oracle
        (bit-identical to forced oracle), another family -> fast."""
        rng = np.random.default_rng(4133)
        inputs = attend_inputs(rng, ATTEND_REAL, 500, 300)
        want = {mode: run_attend(mode, *inputs) for mode in ("fast", "oracle")}
        for value, mode in ((None, "fast"), ("attention", "oracle"), ("1", "oracle"), ("linear,moe", "fast")):
            with self.subTest(value=value):
                got = run_attend("encode", *inputs, env={} if value is None else {"Q38_REF_KERNELS": value})
                self.assertEqual(got[:2], want[mode][:2])
                np.testing.assert_array_equal(got[2].view("<u4"), want[mode][2].view("<u4"))

    # ---- R1.8 parallel GDN prepare (q38c_gdn_prepare_fast, family fuse) ----
    def prep_case(self, label, inputs, dims, **view):
        """Fast and oracle on one case: clean word, (1, 1) against (2, 2) dispatches/barriers, the value copy,
        log_g and beta bit-identical (same expressions per element), the norms within (a) and (b)."""
        fast, oracle = (run_prep(mode, inputs, dims, **view) for mode in ("fast", "oracle"))
        self.assertEqual((fast[:3], oracle[:3]), ((0, 1, 1), (0, 2, 2)), label)
        keys = 2 * dims[0] * dims[2]
        for got, want in ((fast[3][:, keys:], oracle[3][:, keys:]), (fast[4], oracle[4]), (fast[5], oracle[5])):
            np.testing.assert_array_equal(got.view("<u4"), want.view("<u4"), label)
        ref, magnitude = prep_fp64(inputs[0], dims)
        stats, oracle_stats = (rms_against_fp64(out[3][:, :keys], ref, magnitude, False) for out in (fast, oracle))
        stats["vs_oracle"] = against_oracle(fast[3][:, :keys], oracle[3][:, :keys], magnitude)
        RESULTS.setdefault(label, {}).update(fast=stats, oracle=oracle_stats)
        CORPUS.setdefault("gdn_prepare_fast", {})[label] = scaled_difference(fast[3][:, :keys], oracle[3][:, :keys],
                                                                              magnitude).ravel()
        self.assertTrue(stats["finite"] and stats["condition_a"], (label, stats))
        self.assertTrue(oracle_stats["finite"] and oracle_stats["condition_a"], (label, oracle_stats))
        tol = TOLERANCES["kernels"].get("gdn_prepare_fast", {}).get("tol")
        self.assertIsNotNone(tol, "gdn_prepare_fast: fast kernel without a measured tolerance")
        self.assertLessEqual(stats["vs_oracle"]["max"], tol, label)
        return fast

    def test_gdn_prepare_fast_against_oracle_and_fp64(self):
        """The real GDN dims (hk 16, hv 48, dk = dv 128: 32 q/k heads, 4 threadgroups per token) at 1, 3 and
        32 tokens, then dims off the 8-head threadgroup and the 32-lane simdgroup: dk 1, 33, 200 (lanes loop),
        2 hk = 2, 10, 48 heads, hv*dv below and above the 256 threads of a token's group."""
        rng = np.random.default_rng(3967)
        for dims, tokens in ((GDN_REAL, (1, 3, 32)), ((1, 3, 33, 5), (1, 3)), ((2, 4, 1, 1), (1, 3)),
                             ((5, 10, 200, 7), (1, 3)), ((24, 48, 32, 64), (1, 3))):
            for t in tokens:
                with self.subTest(dims=dims, t=t):
                    self.prep_case(f"prep:{dims}t{t}", prep_inputs(rng, t, dims), dims)

    def test_gdn_prepare_fast_rows_bit_independent(self):
        """Review Focus 4: a token's bits depend neither on the number of tokens nor on its position."""
        rng = np.random.default_rng(3969)
        inputs = prep_inputs(rng, 32, GDN_REAL)
        whole = run_prep("fast", inputs, GDN_REAL)
        for lo, hi in ((0, 1), (17, 18), (31, 32), (5, 8)):
            part = run_prep("fast", tuple(a[lo:hi] if a.ndim == 2 else a for a in inputs), GDN_REAL)
            self.assertEqual(part[:3], (0, 1, 1))
            for got, want in zip(part[3:], whole[3:]):
                np.testing.assert_array_equal(got.view("<u4"), want[lo:hi].view("<u4"), (lo, hi))

    def test_gdn_prepare_fast_strided_offset_views_and_inplace(self):
        """Review Focus 5: row pitches past the width (NaN in the gaps: a read there sets bit 1) and nonzero
        offsets change no bit and write nothing outside the outputs (the harness checks); in place (the
        admitted aliases), every element is read and written by one thread, so the bits are unchanged too."""
        rng = np.random.default_rng(3971)
        for dims in (GDN_REAL, (5, 10, 200, 7)):
            inputs = prep_inputs(rng, 3, dims)
            for mode in ("fast", "oracle"):
                plain = run_prep(mode, inputs, dims)
                for gap, offset, inplace in ((5, 3, False), (64, 0, False), (0, 7, False), (0, 0, True), (3, 1, True)):
                    with self.subTest(dims=dims, mode=mode, gap=gap, offset=offset, inplace=inplace):
                        got = run_prep(mode, inputs, dims, gap=gap, offset=offset, inplace=inplace)
                        self.assertEqual(got[:3], plain[:3])
                        for a, b in zip(got[3:], plain[3:]):
                            np.testing.assert_array_equal(a.view("<u4"), b.view("<u4"))

    def test_gdn_prepare_fast_nonfinite_sets_error_word(self):
        """Review Focus 3 with the check folded in: NaN/Inf in a q, k or value column, alpha, br, A or dt
        give bit 1 and A > 0 bit 4 as check + q38c_gdn_prepare; a finite overflow of the op (sum of
        squares, alpha + dt) gives bit 1. With bit 4 preset, a nonfinite input still adds bit 1 (the check
        always runs), an overflow adds nothing (the op exits early)."""
        rng = np.random.default_rng(3973)
        dims = (2, 4, 33, 5)
        clean = prep_inputs(rng, 3, dims)
        keys = 2 * dims[0] * dims[2]
        faults = [(0, (0, 0), np.nan), (0, (1, 2 * 33 + 7), np.inf), (0, (2, keys + 19), -np.inf), (1, (0, 3), np.nan),
                  (2, (2, 0), np.inf), (3, (1,), np.nan), (4, (3,), -np.inf), (3, (2,), 0.5), (0, (1, 40), 3e38),
                  (1, (2, 1), 3e38)]
        for part, where, value in faults:
            inputs = [a.copy() for a in clean]
            inputs[part][where] = value
            if part == 1 and value == 3e38:
                inputs[4][1] = 3e38  # alpha + dt overflows
            want = 4 if value == 0.5 else 1
            for initial in (0, 4):
                expected = want | initial if not np.isfinite(value) or value == 0.5 or not initial else initial
                with self.subTest(part=part, where=where, value=value, initial=initial):
                    self.assertEqual([run_prep(m, tuple(inputs), dims, initial=initial)[0] for m in ("fast", "oracle")],
                                     [expected, expected])
        with self.subTest(case="clean with bit 4 set"):
            self.assertEqual([run_prep(m, clean, dims, initial=4)[0] for m in ("fast", "oracle")], [4, 4])

    def test_gdn_prepare_selection(self):
        """q38c_gpu_encode runs q38c_gdn_prepare_fast (1 dispatch, 1 barrier, the fast bits) unless
        Q38_REF_KERNELS selects fuse at creation (fuse, 1, all, a list with fuse, a typo: R18)."""
        inputs = prep_inputs(np.random.default_rng(3977), 3, GDN_REAL)
        want = {mode: run_prep(mode, inputs, GDN_REAL) for mode in ("fast", "oracle")}
        for value in (None, "", "0", "linear", "moe,indexed", "fuse", "1", "all", "linear,fuse", "fuze"):
            mode = "oracle" if value in ("fuse", "1", "all", "linear,fuse", "fuze") else "fast"
            with self.subTest(value=value):
                got = run_prep("encode", inputs, GDN_REAL, env={} if value is None else {"Q38_REF_KERNELS": value})
                self.assertEqual(got[:3], want[mode][:3])
                self.assertEqual(got[:3], (0, 1, 1) if mode == "fast" else (0, 2, 2))
                for a, b in zip(got[3:], want[mode][3:]):
                    np.testing.assert_array_equal(a.view("<u4"), b.view("<u4"))
    def test_hc_tail_fused_bit_identical_to_sequence(self):
        """R2.6: the fused HC tail writes act, gate, mixed and inject byte for byte as the fast sequence
        (HC_LOW, F16 up, HC_READ, HC_INJECT) in one dispatch instead of 7 (5 without inject): the real
        site (K 320, h 4, E 2560) at 1 and 3 tokens, and shapes with no chunk tail (K 256), no head
        (K 8), a partial last chunk (K 100, 300), h 1..4 and K 1024. A NaN in low sets bit 1 in both.
        Under Q38_REF_KERNELS=linear or fuse it answers UNSUPPORTED (8): the caller runs the sequence."""
        shapes = [(1, 320, 4, 2560), (3, 320, 4, 2560), (2, 100, 3, 37), (1, 1024, 4, 64), (1, 256, 4, 50),
                  (2, 8, 1, 9), (1, 300, 2, 33)]
        for shape in shapes:
            for inject in (True, False):
                with self.subTest(shape=shape, inject=inject):
                    got = run_hctail(shape, inject)
                    self.assertEqual(got, dict(status=0, equal=1, words=(0, 0), dispatches=(1, 7 if inject else 5)))
        got = run_hctail((1, 320, 4, 2560), poison=True)
        self.assertEqual((got["status"], got["words"][0] & 1, got["words"][1] & 1), (0, 1, 1))
        for family in ("linear", "fuse", "1"):
            with self.subTest(reference=family):
                self.assertEqual(run_hctail((1, 320, 4, 2560), env={"Q38_REF_KERNELS": family})["status"], 8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--slices", type=Path, action="append", default=[])
    parser.add_argument("--tolerances", type=Path, default=ROOT / "tests/qwen38_kernel_tolerances.json")
    parser.add_argument("--report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if (not os.environ.get("PICOFORGE_GPU_LOCK_FD") or os.environ.get("MTL_DEBUG_LAYER") != "1"
            or not os.environ.get("Q38_PACKED_LIBRARY")):
        parser.error("requires the GPU lock wrapper, MTL_DEBUG_LAYER=1 and Q38_PACKED_LIBRARY")
    NATIVE, SLICES = args.native.resolve(), [p.resolve() for p in args.slices]
    TOLERANCES, CASES = json.loads(args.tolerances.read_text()), list(cases())
    start = time.monotonic()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    report = dict(schema_version=1, success=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=[t.id() for t, _ in result.failures], errors=[t.id() for t, _ in result.errors],
                  seconds=time.monotonic() - start, command=sys.argv, numpy=np.__version__,
                  native_sha256=oracle.digest(NATIVE.read_bytes()),
                  metallib_sha256=oracle.digest(Path(os.environ["Q38_PACKED_LIBRARY"]).read_bytes()),
                  slices={str(p.relative_to(ROOT)): oracle.digest((p / "manifest.json").read_bytes()) for p in SLICES},
                  tolerances=TOLERANCES, cases=RESULTS,
                  # Per kernel over the whole corpus: what tests/qwen38_kernel_tolerances.json records.
                  fast_corpus={name: dict(max=float(np.concatenate(list(parts.values())).max()),
                                          samples=sum(map(len, parts.values())),
                                          p99=float(np.percentile(np.concatenate(list(parts.values())), 99)))
                               for name, parts in CORPUS.items()})
    with args.report.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
