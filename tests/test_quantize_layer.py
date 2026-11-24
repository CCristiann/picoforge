"""quantize_layer.py on a tiny synthetic checkpoint: 2 experts, hidden 512, expert ff 256."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
sys.path.insert(0, str(ROOT / "tests"))
import qw4q  # noqa: E402
import quantize_layer  # noqa: E402
from imatrix import Imatrix  # noqa: E402
from test_quant_io import write_imatrix  # noqa: E402

E, H, FF, LAYER = 2, 512, 256, 3
GATE_UP = f"model.language_model.layers.{LAYER}.mlp.experts.gate_up_proj"
DOWN = f"model.language_model.layers.{LAYER}.mlp.experts.down_proj"


def bf16(x: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(x, "<f4").view("<u4") >> 16).astype("<u2")


def f32(u: np.ndarray) -> np.ndarray:
    return (u.astype("<u4") << 16).view("<f4")


class QuantizeLayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        (d / "src/tensors").mkdir(parents=True)
        (d / "src/config.json").write_text(json.dumps(
            {"text_config": {"num_experts": E, "hidden_size": H, "moe_intermediate_size": FF}}))
        self.gate_up = bf16(rng.standard_t(3, (E, 2 * FF, H)) * 0.02)
        self.down = bf16(rng.standard_t(3, (E, H, FF)) * 0.02)
        for name, arr in ((GATE_UP, self.gate_up), (DOWN, self.down)):
            (d / f"src/tensors/{name}.bin").write_bytes(arr.tobytes())
            (d / f"src/tensors/{name}.json").write_text(json.dumps({"dtype": "BF16", "shape": list(arr.shape)}))
        counts = np.array([3, 0], np.float32)   # expert 1 never routed: weight-energy fallback
        entries = {f"blk.{LAYER}.ffn_{p}_exps.weight": (rng.random((E, w), dtype=np.float32), counts)
                   for p, w in (("gate", H), ("up", H), ("down", FF))}
        self.imx_path = d / "imatrix.gguf"
        self.imx_sha = write_imatrix(self.imx_path, entries)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra) -> int:
        return quantize_layer.main(["--source", str(self.dir / "src"), "--revision", "rev1",
                                    "--imatrix", str(self.imx_path), "--imatrix-sha", self.imx_sha,
                                    "--layer", str(LAYER), "--out", str(self.dir / "out"), "--threads", "2", *extra])

    def expected(self, name: str, rows: np.ndarray, physical: int = 0) -> bytes:
        im, out = Imatrix(self.imx_path, self.imx_sha), b""
        for e in range(E):
            x = f32(rows[e])
            imp = im.expert_importance(name, e, x)
            if physical:
                x = np.pad(x, ((0, 0), (0, physical - x.shape[1])))
                imp = np.pad(imp, (0, physical - imp.shape[0]))
            out += qw4q.quantize(qw4q.Q2_K if physical else qw4q.IQ2_XXS, x, imp)
        return out

    def test_compat_matches_direct_library_call(self):
        self.assertEqual(self.run_tool("--projection", "gate-up", "--mode", "compat"), 0)
        gate = (self.dir / f"out/blk.{LAYER}.ffn_gate_exps.weight.bin").read_bytes()
        up = (self.dir / f"out/blk.{LAYER}.ffn_up_exps.weight.bin").read_bytes()
        self.assertEqual(gate, self.expected(f"blk.{LAYER}.ffn_gate_exps.weight", self.gate_up[:, :FF]))
        self.assertEqual(up, self.expected(f"blk.{LAYER}.ffn_up_exps.weight", self.gate_up[:, FF:]))

    def test_down_is_padded_with_zero_importance(self):
        self.assertEqual(self.run_tool("--projection", "down", "--mode", "compat", "--physical-input", "512"), 0)
        got = (self.dir / f"out/blk.{LAYER}.ffn_down_exps.weight.bin").read_bytes()
        self.assertEqual(got, self.expected(f"blk.{LAYER}.ffn_down_exps.weight", self.down, physical=512))

    def test_resume_skips_verified_and_redoes_corrupted(self):
        self.run_tool("--projection", "gate-up", "--mode", "compat")
        path = self.dir / f"out/blk.{LAYER}.ffn_gate_exps.weight.bin"
        good = path.read_bytes()
        path.write_bytes(b"\0" + good[1:])
        self.assertEqual(self.run_tool("--projection", "gate-up", "--mode", "compat"), 0)
        self.assertEqual(path.read_bytes(), good)

    def test_nan_in_source_fails_before_writing(self):
        bad = self.gate_up.copy(); bad[1, 5, 7] = 0x7FC0
        (self.dir / f"src/tensors/{GATE_UP}.bin").write_bytes(bad.tobytes())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_tool("--projection", "gate-up", "--mode", "compat")
        self.assertFalse((self.dir / f"out/blk.{LAYER}.ffn_gate_exps.weight.bin").exists())

    def test_mode_change_in_same_dir_is_refused(self):
        self.run_tool("--projection", "gate-up", "--mode", "compat")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.run_tool("--projection", "gate-up", "--mode", "o1")


if __name__ == "__main__":
    unittest.main()
