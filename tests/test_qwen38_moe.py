"""Small CPU-only tests. Run with the existing tools/venv Python.

The hash-pinned source cache must be prepared explicitly; tests never fetch.
"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/oracle"))
import qwen38_moe as moe

torch.set_num_threads(1)


class UpstreamOracleTests(unittest.TestCase):
    def test_pinned_source_and_default(self):
        self.assertEqual(moe.upstream_class().__name__, moe.CLASSES[-1])

    def test_corrupt_source_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in moe.HASHES:
                data = (moe.CACHE / name).read_bytes()
                (Path(tmp) / name).write_bytes(data + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                moe.upstream_class(tmp)

    def test_seed_and_zero_input(self):
        x, weights = moe.synthetic()
        x2, weights2 = moe.synthetic()
        np.testing.assert_array_equal(x, x2)
        for name in weights:
            np.testing.assert_array_equal(weights[name], weights2[name])
        result = moe.upstream_forward(moe.FixtureConfig(), x * 0, weights)
        np.testing.assert_array_equal(result["y"], np.zeros_like(x))
        np.testing.assert_allclose(result["weights"], 0.5, atol=0, rtol=0)

    def test_analytical_routed_shared_and_no_residual(self):
        c = moe.FixtureConfig(hidden_size=1, num_experts=2, num_experts_per_tok=1,
                              moe_intermediate_size=1, shared_expert_intermediate_size=1)
        x, weights = moe.synthetic(c, tokens=1)
        x[:] = 1
        weights["gate.weight"][:] = [[2], [1]]
        weights["experts.gate_up_proj"][:] = [[[1], [3]], [[2], [4]]]
        weights["experts.down_proj"][:] = [[[5]], [[6]]]
        weights["shared_expert.gate_proj.weight"][:] = 1
        weights["shared_expert.up_proj.weight"][:] = 2
        weights["shared_expert.down_proj.weight"][:] = 3
        weights["shared_expert_gate.weight"][:] = 0
        result = moe.upstream_forward(c, x, weights)
        sigmoid1 = 1 / (1 + np.exp(-1.0))
        np.testing.assert_array_equal(result["ids"], [[0]])
        np.testing.assert_array_equal(result["weights"], [[1]])
        np.testing.assert_allclose(result["routed"], 15 * sigmoid1, rtol=2e-6)
        np.testing.assert_allclose(result["shared"], 6 * sigmoid1, rtol=2e-6)
        np.testing.assert_allclose(result["y"], 18 * sigmoid1, rtol=2e-6)
        raw = moe.upstream_forward(replace(c, norm_topk_prob=False), x, weights)
        np.testing.assert_allclose(raw["weights"], sigmoid1, rtol=2e-6)
        np.testing.assert_allclose(raw["y"], 15 * sigmoid1**2 + 3 * sigmoid1, rtol=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
