"""Check the external CPU reference against analytical packed-weight fixtures.

No native picoforge decoder exists in this increment. A missing/unverified
reference fails the suite; these tests never silently replace the oracle.
"""
import argparse
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import unittest

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

    def test_q2_finite_half_scale_extremes(self):
        for scale in [2**-24, -(2**-24), 2**-14, -2**-14, 65504, -65504]:
            raw = bytes([1]) * 16 + bytes([0x55]) * 64 + struct.pack("<ee", scale, 0)
            self.assert_bits(self.decode(raw), [scale] * 256)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    REFERENCE = args.reference.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
