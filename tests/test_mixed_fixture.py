"""The picoforge mixed-experts synthetic fixture and the oracle's reassembly of it."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/synth")); sys.path.insert(0, str(ROOT / "tools/oracle"))
import make_qwen4_graph as synth  # noqa: E402
import qwen4_forward as oracle  # noqa: E402


class MixedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "mixed.gguf"
        synth.write_graph(cls.path, synth.graph_config(), 3811, mixed_hot=1)
        cls.w = oracle.Weights(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_map_is_a_bijection_over_cold_and_hot(self):
        ne = self.w.hf["num_experts"]
        for layer in range(self.w.hf["num_hidden_layers"]):
            v = self.w.vector(f"blk.{layer}.ffn_exps_map.weight").astype(int)
            cold, hot = sorted(x for x in v if x >= 0), sorted(-1 - x for x in v if x < 0)
            self.assertEqual(cold, list(range(ne - 1)))
            self.assertEqual(hot, [0])

    def test_reassembly_takes_each_expert_from_its_tensor(self):
        p = "blk.2."
        full = oracle.experts(self.w, p, "down", 64)
        cold, hot = self.w.matrix(p + "ffn_down_exps.weight", 64), self.w.matrix(p + "ffn_down_exps_hot.weight", 64)
        for e, v in enumerate(self.w.vector(p + "ffn_exps_map.weight").astype(int)):
            np.testing.assert_array_equal(full[e], cold[v] if v >= 0 else hot[-1 - v])
        self.assertEqual(full.shape, (self.w.hf["num_experts"], self.w.hf["hidden_size"], 64))

    def test_mtp_layer_is_not_mixed(self):
        self.assertNotIn(f"blk.{self.w.hf['num_hidden_layers']}.ffn_exps_map.weight", self.w.tensors)


if __name__ == "__main__":
    unittest.main()
