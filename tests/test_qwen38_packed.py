"""Check the external CPU reference against analytical packed-weight fixtures.

No native picoforge decoder exists in this increment. A missing/unverified
reference fails the suite; these tests never silently replace the oracle.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("packed_oracle", ROOT / "tools/oracle/qwen38_packed.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
REFERENCE = None


class PackedReferenceTests(unittest.TestCase):
    def decode(self, data, kind=10, columns=256, rows=1):
        return oracle.decode(REFERENCE, data, kind, columns, rows)

    def assert_bits(self, actual, expected):
        expected = np.asarray(expected, dtype="<f4").reshape(actual.shape)
        np.testing.assert_array_equal(actual.view("<u4"), expected.view("<u4"))

    def test_q2_known_values(self):
        raw = bytes([0x23]) * 16 + bytes([0xe4]) * 64 + struct.pack("<ee", 1, 0.5)
        expected = np.tile(np.repeat([-1, 2, 5, 8], 32), 2)
        self.assert_bits(self.decode(raw), expected)

    def test_q2_all_scale_min_code_combinations(self):
        packed, expected = bytearray(), []
        for scale in range(16):
            for minimum in range(16):
                for code in range(4):
                    packed.extend(bytes([scale | minimum << 4]) * 16)
                    packed.extend(bytes([code * 0x55]) * 64)
                    packed.extend(struct.pack("<ee", 0.125, 0.0625))
                    expected.append([0.125 * scale * code - 0.0625 * minimum] * 256)
        self.assert_bits(self.decode(packed, rows=len(expected)), expected)

    def test_q2_each_column_independently(self):
        packed = bytearray()
        # Encode one impulse per row by bit plane, independently of the scalar
        # decoder's scale-group indexing. Expected rows are the identity matrix.
        for column in range(256):
            half, local = divmod(column, 128)
            plane, lane = divmod(local, 32)
            codes = bytearray(64)
            codes[half * 32 + lane] = 1 << (2 * plane)
            packed.extend(bytes([1]) * 16 + codes + struct.pack("<ee", 1, 0))
        self.assert_bits(self.decode(packed, rows=256), np.eye(256, dtype=np.float32))

    def test_q2_scale_group_boundaries(self):
        raw = bytes(range(16)) + bytes([0x55]) * 64 + struct.pack("<ee", 1, 0)
        self.assert_bits(self.decode(raw), np.repeat(np.arange(16), 16))
        raw = bytes(i << 4 for i in range(16)) + bytes(64) + struct.pack("<ee", 0, 1)
        expected = -np.repeat(np.arange(16), 16).astype(np.float32)
        expected[0:16] = 0  # +0 - +0 is +0 with round-to-nearest.
        self.assert_bits(self.decode(raw), expected)

    def test_q2_multiple_rows_and_physical_padding(self):
        def block(value):
            return bytes([1]) * 16 + bytes([0x55]) * 64 + struct.pack("<ee", value, 0)
        raw = b"".join(block(value) for value in [1, 2, 3, 4, 5, 6])
        physical = self.decode(raw, columns=768, rows=2)
        self.assert_bits(physical, np.repeat(np.arange(1, 7), 256).reshape(2, 768))
        self.assertEqual(physical[0, 639], 3)
        self.assertEqual(physical[0, 640], 3)  # Padded weights need not be zero.
        self.assertEqual(physical[1, 0], 4)   # Next row starts after 252 bytes.

    def test_logical_padding_never_contributes_to_reference_product(self):
        weights = np.ones((2, 768), dtype=np.float32)
        weights[1, :640] = 2
        x = np.ones((3, 640), dtype=np.float32)
        weights[:, 640:] = 65504
        y, magnitude = oracle.matmul_reference(weights, x, 640)
        np.testing.assert_array_equal(y, [[640, 1280]] * 3)
        np.testing.assert_array_equal(magnitude, y)
        weights[:, 640:] = -65504
        changed, _ = oracle.matmul_reference(weights, x, 640)
        np.testing.assert_array_equal(changed, y)
        with self.assertRaises(ValueError):
            oracle.matmul_reference(weights, np.ones((3, 768), dtype=np.float32), 640)

    def test_q2_finite_half_scale_extremes(self):
        for scale in [2**-24, -(2**-24), 2**-14, -2**-14, 65504, -65504]:
            raw = bytes([1]) * 16 + bytes([0x55]) * 64 + struct.pack("<ee", scale, 0)
            self.assert_bits(self.decode(raw), [scale] * 256)

    def test_iq2_known_grid_byte_order_and_sign_parity(self):
        # Grid 1 starts with 43, then seven 8s. Sign index 1 negates positions
        # zero AND seven; local scale zero and d=1 give a factor of 1/8.
        group = bytes([1, 0, 0, 0]) + struct.pack("<I", 1)
        raw = struct.pack("<e", 1) + group * 8
        expected = [-43 / 8, 1, 1, 1, 1, 1, 1, -1] + [1] * 24
        self.assert_bits(self.decode(raw, kind=16), expected * 8)

    def test_iq2_every_group_and_grid_position(self):
        raw, expected = bytearray(), []
        for position in range(32):
            raw.extend(struct.pack("<e", 1))
            row = []
            for group in range(8):
                ids = [int(group * 4 + j == position) for j in range(4)]
                raw.extend(bytes(ids) + struct.pack("<I", group << 28))
                factor = (2 * group + 1) / 8
                for grid_id in ids:
                    row.extend([factor * (43 if grid_id else 8)] + [factor * 8] * 7)
            expected.append(row)
        self.assert_bits(self.decode(raw, kind=16, rows=32), expected)

    def test_iq2_all_grid_sign_scale_combinations(self):
        grid = oracle.iq2_grid(REFERENCE).astype(np.float32)
        for scale in range(16):
            packed, expected = bytearray(), []
            for sign_id in range(128):
                # Derive the eighth sign from even parity, not the upstream table.
                signs = [-1 if sign_id & (1 << j) else 1 for j in range(7)]
                signs.append(-1 if sign_id.bit_count() % 2 else 1)
                expected.append(grid * signs * ((2 * scale + 1) / 8))
                for start in range(0, 256, 32):
                    packed.extend(struct.pack("<e", 1))
                    for first in range(start, start + 32, 4):
                        packed.extend(bytes(range(first, first + 4)))
                        sign_word = sum(sign_id << (7 * j) for j in range(4))
                        packed.extend(struct.pack("<I", sign_word | scale << 28))
            with self.subTest(scale=scale):
                self.assert_bits(self.decode(packed, kind=16, rows=1024), expected)

    def test_mathematical_interpretation_matches_external_reference(self):
        rng = np.random.default_rng(3811)
        grid = oracle.iq2_grid(REFERENCE)
        for kind, columns, blocks in [(10, 768, 3), (16, 2560, 10)]:
            raw = rng.integers(0, 256, (7 * blocks, oracle.FORMATS[kind][1]), dtype=np.uint8)
            # Nontrivial finite half scales; random packed data exercises all planes.
            scales = rng.uniform(-1, 1, (len(raw), 1)).astype("<f2").view(np.uint8)
            if kind == 10:
                raw[:, 80:82] = scales
                raw[:, 82:84] = np.full((len(raw), 1), 0.0625, dtype="<f2").view(np.uint8)
            else:
                raw[:, :2] = scales
            packed = raw.tobytes()
            external = self.decode(packed, kind, columns, 7)
            mathematical = oracle.mathematical_rows(packed, kind, columns, grid=grid)
            self.assert_bits(external, mathematical)

    def test_iq2_half_scale_extremes(self):
        group = bytes(8)
        for scale in [2**-24, -(2**-24), 2**-14, -2**-14, 65504, -65504]:
            raw = struct.pack("<e", scale) + group * 8
            self.assert_bits(self.decode(raw, kind=16), [scale] * 256)

    def test_q8_all_signed_codes_and_half_scale(self):
        codes = np.arange(-128, 128, dtype=np.int16)
        for scale in [0.125, -0.125, 2**-24, 65504]:
            raw = b"".join(struct.pack("<e", scale) + codes[start:start + 32].astype("i1").tobytes()
                           for start in range(0, 256, 32))
            self.assert_bits(self.decode(raw, kind=8), codes.astype(np.float32) * scale)

    def test_all_finite_f16_and_bf16_bit_patterns(self):
        bits = np.arange(65536, dtype="<u2")
        for kind, exponent_mask in [(1, 0x7c00), (30, 0x7f80)]:
            finite = bits[(bits & exponent_mask) != exponent_mask]
            if kind == 1:
                expected = finite.view("<f2").astype("<f4")
            else:
                expected = (finite.astype("<u4") << 16).view("<f4")
            with self.subTest(format=kind):
                self.assert_bits(self.decode(finite.tobytes(), kind, len(finite), 1), expected)

    def test_scalar_nonfinite_classes_are_reference_data(self):
        # The external reference transports IEEE values. Production's future
        # ENONFINITE policy belongs in the native decoder tests, not this adapter.
        for kind in [0, 1, 30]:
            values = np.array([0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan], dtype="<f4")
            if kind == 0:
                raw = values.tobytes()
            elif kind == 1:
                raw = values.astype("<f2").tobytes()
            else:
                raw = (values.view("<u4") >> 16).astype("<u2").tobytes()
            actual = self.decode(raw, kind, 7, 1).ravel()
            self.assert_bits(actual[:6], values[:6])
            self.assertTrue(np.isnan(actual[6]))

    def test_reference_transport_rejects_malformed_requests(self):
        raw = bytes(84)
        cases = [("10", "256", "1", raw[:-1]), ("10", "256", "1", raw + b"x"),
                 ("10", "640", "1", b""), ("16", "0", "1", b""),
                 ("10", "256", "0", b""), ("12", "256", "1", b""),
                 ("10", "18446744073709551615", "2", b""),
                 ("10", "256", "18446744073709551615", b""),
                 ("10", "-256", "1", b""), ("10", "256x", "1", b"")]
        for kind, columns, rows, data in cases:
            with self.subTest(kind=kind, columns=columns, rows=rows, bytes=len(data)):
                r = subprocess.run([str(REFERENCE), kind, columns, rows], input=data,
                                   capture_output=True, timeout=30)
                self.assertEqual(r.returncode, 2, r.stderr.decode())
                self.assertFalse(r.stdout)
                self.assertIn(b"qwen38 reference:", r.stderr)
                self.assertNotIn(b"Sanitizer", r.stderr)

    def test_python_transport_rejects_invalid_shapes(self):
        for kind, cols, rows in [(12, 256, 1), (10, 640, 1), (10, 0, 1),
                                  (10, 256, 0), (10, 2**64, 2), (10, 256.0, 1)]:
            with self.subTest(kind=kind, columns=cols, rows=rows):
                with self.assertRaises(ValueError):
                    self.decode(b"", kind, cols, rows)

    def test_slice_io_retries_interrupts_and_short_reads(self):
        with patch.object(oracle.os, "pread", side_effect=[InterruptedError(), b"ab", b"c"]) as read:
            self.assertEqual(oracle.pread_exact(7, 100, 3), b"abc")
            self.assertEqual([c.args for c in read.call_args_list],
                             [(7, 3, 100), (7, 3, 100), (7, 1, 102)])

    def test_slice_io_rejects_eof_and_errors_without_stale_data(self):
        for failure in [b"", OSError("read failed")]:
            with patch.object(oracle.os, "pread", side_effect=[b"a", failure]):
                with self.assertRaises((EOFError, OSError)):
                    oracle.pread_exact(7, 100, 3)
        for offset, size in [(-1, 1), (0, 0), (0, oracle.BYTE_LIMIT + 1)]:
            with self.assertRaises(ValueError):
                oracle.pread_exact(7, offset, size)

    def test_oracle_refuses_missing_or_changed_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            binary = Path(directory) / "reference"
            binary.write_bytes(Path(REFERENCE).read_bytes())
            with self.assertRaises(FileNotFoundError):
                oracle.decode(binary, bytes(84), 10, 256, 1)
            original = json.loads(Path(REFERENCE).with_name("manifest.json").read_text())
            for field, value in [("upstream_revision", "different"), ("decoder", "picoforge"),
                                 ("binary_sha256", "0" * 64)]:
                manifest = {**original, field: value}
                binary.with_name("manifest.json").write_text(json.dumps(manifest))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    oracle.decode(binary, bytes(84), 10, 256, 1)

    def test_saved_fixture_replay_and_corruption_detection(self):
        raw = (bytes([1]) * 16 + bytes([0x55]) * 64 + struct.pack("<ee", 1, 0)) * 3
        blobs = {"packed": raw, "decoded.f32": np.ones(768, dtype="<f4").tobytes(),
                 "input.f32": np.ones((2, 640), dtype="<f4").tobytes(),
                 "product.f64": np.full(2, 640, dtype="<f8").tobytes(),
                 "sum_abs.f64": np.full(2, 640, dtype="<f8").tobytes()}
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            directory = Path(directory)
            artifacts = {}
            for name, data in blobs.items():
                (directory / name).write_bytes(data)
                artifacts[name] = dict(file=name, bytes=len(data), sha256=oracle.digest(data))
            manifest = dict(schema_version=1, complete=True, kind=10, bytes_read=len(raw),
                            records=[dict(k_physical=768, k_logical=640, input_rows=2,
                                          artifacts=artifacts)])
            (directory / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(oracle.replay_slices(REFERENCE, directory), 1)
            (directory / "packed").write_bytes(bytes(len(raw)))
            with self.assertRaisesRegex(ValueError, "checksum"):
                oracle.replay_slices(REFERENCE, directory)
            manifest["complete"] = False
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                oracle.replay_slices(REFERENCE, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args, remaining = parser.parse_known_args()
    REFERENCE = args.reference.resolve()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    result = program.result
    if args.report:
        report = dict(schema_version=1, success=result.wasSuccessful(), tests_run=result.testsRun,
                      failures=[test.id() for test, _ in result.failures],
                      errors=[test.id() for test, _ in result.errors],
                      skipped=[(test.id(), reason) for test, reason in result.skipped],
                      python=sys.version, numpy=np.__version__, command=sys.argv,
                      reference_manifest=json.loads(REFERENCE.with_name("manifest.json").read_text()),
                      source_sha256={str(p.relative_to(ROOT)): oracle.digest(p.read_bytes()) for p in
                                     [Path(__file__).resolve(), Path(oracle.__file__), oracle.ADAPTER]})
        with args.report.open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
