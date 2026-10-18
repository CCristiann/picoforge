"""Fast GPU kernels against the oracle kernel and an FP64 reference (R1.1).

Cases: the real GGUF rows in the --slices folders (several layers/experts per
format, extracted once by qwen38-test-assets: the checkpoint is never opened)
plus seeded synthetic shapes. The CPU decoder's rows give the FP64 product r
and S = sum|w*x| per output. A fast kernel must meet both conditions:
  (a) |y - r| <= gamma(2K+8) * S + 2 ulp32(r)       (holds for any sum order)
  (b) |y_fast - y_oracle| / max(S, 2^-126) <= tol[kernel]
with tol from tests/qwen38_kernel_tolerances.json. Run through the GPU lock.
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
UNSUPPORTED = 8  # Q38_UNSUPPORTED: the format has no fast kernel yet
BANNER = re.compile(rb"(?m)^.*Metal API Validation Enabled\r?\n")
NATIVE, SLICES, TOLERANCES, CASES, ORACLE, RESULTS, CORPUS = None, [], {}, [], {}, {}, {}


def native(mode, kind, physical, logical, data, x, env=None):
    """(status, y [M,N], CPU-decoded rows [N,physical]); status 8 has no data."""
    n = len(data) // (physical // oracle.FORMATS[kind][0] * oracle.FORMATS[kind][1])
    environment = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"} | (env or {})
    lock = os.environ.get("PICOFORGE_GPU_LOCK_FD")
    result = subprocess.run([str(NATIVE), mode, str(kind), str(physical), str(logical), str(n), str(len(x))],
                            input=data + x.astype("<f4").tobytes(), capture_output=True, timeout=60,
                            pass_fds=(int(lock),) if lock else (), env=environment)
    stderr = BANNER.sub(b"", result.stderr)
    if result.returncode == UNSUPPORTED and not result.stdout and not stderr:
        return UNSUPPORTED, None, None
    if result.returncode or stderr:
        raise AssertionError(f"{mode} kind {kind}: exit {result.returncode}, {stderr!r}")
    values = np.frombuffer(result.stdout, dtype="<f4")
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


def scaled_difference(y, y_oracle, magnitude):
    """Condition (b)'s measured side: |y - y_oracle| / max(S, 2^-126)."""
    return np.abs(y.astype(np.float64) - y_oracle) / np.maximum(magnitude, 2.0**-126)


def against_oracle(y, y_oracle, magnitude):
    scaled = scaled_difference(y, y_oracle, magnitude)
    return dict(max=float(scaled.max()), p99=float(np.percentile(scaled, 99)))


class FastKernelTests(unittest.TestCase):
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
                    continue
                stats = against_fp64(fast, ref, magnitude, logical) | dict(vs_oracle=against_oracle(fast, y_oracle, magnitude))
                RESULTS.setdefault(label, {})["fast"] = stats
                CORPUS.setdefault(NAMES[kind], []).append(scaled_difference(fast, y_oracle, magnitude).ravel())
                tol = TOLERANCES["kernels"].get(NAMES[kind], {}).get("tol")
                self.assertTrue(stats["finite"] and stats["condition_a"], stats)
                self.assertIsNotNone(tol, f"{NAMES[kind]}: fast kernel without a measured tolerance")
                self.assertLessEqual(stats["vs_oracle"]["max"], tol)


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
                  fast_corpus={name: dict(max=float(np.concatenate(parts).max()), samples=sum(map(len, parts)),
                                          p99=float(np.percentile(np.concatenate(parts), 99)))
                               for name, parts in CORPUS.items()})
    with args.report.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
