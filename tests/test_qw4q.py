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


class Q2KCompat(unittest.TestCase):
    """The expert down projection: 640 logical inputs padded with zeros to 768."""

    def setUp(self):
        self.x, self.imp = golden_inputs(99, 64, 768)
        self.x[:, 640:] = 0
        self.imp[640:] = 0

    def test_q2_k_matches_dwarfstar_golden(self):
        self.assertEqual(qw4q.quantize(qw4q.Q2_K, self.x, self.imp), (FIX / "q2_k.bin").read_bytes())

    def test_q2_k_dequantize_matches_oracle(self):
        data = (FIX / "q2_k.bin").read_bytes()
        ours = qw4q.dequantize(qw4q.Q2_K, data, 64, 768)
        self.assertTrue(np.array_equal(ours, mathematical_rows(data, 10, 768)))

    def test_q2_k_bad_width_is_an_error(self):
        x = np.ones(640, "<f4"); imp = np.ones(640, "<f4"); out = np.empty(84 * 3, np.uint8)
        f32 = ctypes.POINTER(ctypes.c_float)
        rc = qw4q.lib().qw4q_quantize_q2_k(x.ctypes.data_as(f32), 1, 640, imp.ctypes.data_as(f32), out.ctypes.data)
        self.assertEqual(rc, -1)


def encoder_objective(x: np.ndarray, imp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per block: sum of imp*sqrt(sigma2 + x^2) * (x - y)^2, the compat encoder's own weighting."""
    rows = x.shape[0]
    xb = x.reshape(rows, -1, 256).astype(np.float64)
    yb = y.reshape(rows, -1, 256).astype(np.float64)
    sigma2 = (xb ** 2).mean(axis=2, keepdims=True)
    w = imp.reshape(1, -1, 256) * np.sqrt(sigma2 + xb ** 2)
    return (w * (xb - yb) ** 2).sum(axis=2)


class Iq2XxsRefine(unittest.TestCase):
    """O1: search against the decoder's real levels {1, 3.125, 5.375}, never worse."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.imp = golden_inputs(1234, 64, 2560)
        cls.q0 = qw4q.quantize(qw4q.IQ2_XXS, cls.x, cls.imp, qw4q.COMPAT)
        cls.q1 = qw4q.quantize(qw4q.IQ2_XXS, cls.x, cls.imp, qw4q.REFINE)
        cls.e0 = encoder_objective(cls.x, cls.imp, qw4q.dequantize(qw4q.IQ2_XXS, cls.q0, 64, 2560))
        cls.e1 = encoder_objective(cls.x, cls.imp, qw4q.dequantize(qw4q.IQ2_XXS, cls.q1, 64, 2560))

    def test_never_worse_per_block(self):
        self.assertTrue((self.e1 <= self.e0 * (1 + 1e-6)).all())

    def test_improves_on_average(self):
        self.assertLess(self.e1.sum(), 0.97 * self.e0.sum())

    def test_zero_blocks_untouched(self):
        b0 = np.frombuffer(self.q0, np.uint8).reshape(-1, 66)
        b1 = np.frombuffer(self.q1, np.uint8).reshape(-1, 66)
        self.assertTrue(np.array_equal(b0[[0, 11]], b1[[0, 11]]))

    def test_deterministic(self):
        self.assertEqual(qw4q.quantize(qw4q.IQ2_XXS, self.x, self.imp, qw4q.REFINE), self.q1)

    def test_output_decodes_finite(self):
        self.assertTrue(np.isfinite(qw4q.dequantize(qw4q.IQ2_XXS, self.q1, 64, 2560)).all())


if __name__ == "__main__":
    unittest.main()
