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
