"""Encoder qw4q: compat bytes against DwarfStar goldens, decoder against the oracle."""

import ctypes
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qw4q  # noqa: E402
from golden_inputs import golden_inputs  # noqa: E402
from qwen4_packed import mathematical_rows  # noqa: E402

FIX = ROOT / "tests/fixtures/qw4q"


def oracle_grid() -> np.ndarray:
    text = (ROOT / "src/qwen4_iq2_grid.inc").read_text()
    words = [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]+)ULL", text)]
    return np.array(words, dtype="<u8").view(np.uint8).reshape(256, 8)


class Iq2XxsCompat(unittest.TestCase):
    def setUp(self):
        self.x, self.imp = golden_inputs(1234, 64, 2560)

    def test_matches_dwarfstar_golden(self):
        self.assertEqual(qw4q.quantize(qw4q.IQ2_XXS, self.x, self.imp), (FIX / "iq2_xxs.bin").read_bytes())

    def test_zero_and_tiny_blocks_decode_to_zero(self):
        y = qw4q.dequantize(qw4q.IQ2_XXS, qw4q.quantize(qw4q.IQ2_XXS, self.x, self.imp), 64, 2560)
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue((y[0, :256] == 0).all())
        self.assertTrue((y[1, 256:512] == 0).all())

    def test_dequantize_matches_oracle(self):
        data = (FIX / "iq2_xxs.bin").read_bytes()
        ours = qw4q.dequantize(qw4q.IQ2_XXS, data, 64, 2560)
        self.assertTrue(np.array_equal(ours, mathematical_rows(data, 16, 2560, grid=oracle_grid())))

    def test_bad_width_is_an_error(self):
        x = np.ones(300, "<f4"); imp = np.ones(300, "<f4"); out = np.empty(66, np.uint8)
        f32 = ctypes.POINTER(ctypes.c_float)
        rc = qw4q.lib().qw4q_quantize_iq2_xxs(x.ctypes.data_as(f32), 1, 300, imp.ctypes.data_as(f32), 0,
                                              out.ctypes.data)
        self.assertEqual(rc, -1)


if __name__ == "__main__":
    unittest.main()
