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
