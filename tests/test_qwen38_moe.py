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


class DifferentialTests(unittest.TestCase):
    def compare(self, config, x, weights):
        actual = moe.numpy_forward(config, x, weights)
        expected = moe.upstream_forward(config, x, weights)
        np.testing.assert_array_equal(actual["ids"], expected["ids"])
        for key in expected:
            if key != "ids":
                atol, rtol = (2e-5, 2e-4) if key == "y" else (2e-6, 2e-5)
                np.testing.assert_allclose(actual[key], expected[key], atol=atol, rtol=rtol,
                                           err_msg=key)
        delta = actual["y"].astype(np.float64) - expected["y"]
        rms = np.sqrt(np.mean(expected["y"].astype(np.float64)**2))
        self.assertLessEqual(np.sqrt(np.mean(delta**2)) / max(rms, 1e-6), 1e-4)
        return actual

    def test_random_asymmetric_block(self):
        for seed in range(12):
            for k in (1, 2, 4):
                for norm in (False, True):
                    with self.subTest(seed=seed, k=k, norm=norm):
                        c = replace(moe.FixtureConfig(), num_experts_per_tok=k, norm_topk_prob=norm)
                        x, w = moe.synthetic(c, tokens=7, seed=seed)
                        self.compare(c, x, w)

    def test_512_experts_top10_small_float_weights(self):
        c = replace(moe.FixtureConfig(), num_experts=512, num_experts_per_tok=10,
                    moe_intermediate_size=2)
        x, w = moe.synthetic(c, tokens=3)
        self.compare(c, x, w)

    def test_routed_intermediates_against_torch_float64(self):
        c = moe.FixtureConfig()
        x, w = moe.synthetic(c)
        trace = self.compare(c, x, w)
        # Independent library primitives at higher precision, not another decoder.
        for token, row in enumerate(x):
            for slot, expert in enumerate(trace["ids"][token]):
                xt = torch.tensor(row, dtype=torch.float64)
                gu = torch.tensor(w["experts.gate_up_proj"][expert], dtype=torch.float64)
                gate, up = torch.nn.functional.linear(xt, gu).chunk(2)
                mid = torch.nn.functional.silu(gate) * up
                down = torch.nn.functional.linear(
                    mid, torch.tensor(w["experts.down_proj"][expert], dtype=torch.float64))
                for name, ref in (("gate", gate), ("up", up), ("mid", mid), ("down", down)):
                    np.testing.assert_allclose(trace[name][token, slot], ref.numpy(),
                                               atol=2e-6, rtol=2e-5, err_msg=name)

    def test_shared_gate_zero_and_saturation(self):
        c = moe.FixtureConfig()
        x, w = moe.synthetic(c, tokens=1)
        x[:] = 1
        for logit, factor in ((0, 0.5), (-1000, 0.0), (1000, 1.0)):
            w["shared_expert_gate.weight"][:] = logit / c.hidden_size
            actual = self.compare(c, x, w)
            np.testing.assert_allclose(actual["y"], actual["routed"] + factor*actual["shared"],
                                       atol=2e-6, rtol=2e-5)

    def test_invalid_inputs_and_projection_overflow(self):
        c = moe.FixtureConfig()
        x, w = moe.synthetic(c)
        for bad in (np.nan, np.inf, -np.inf):
            broken = x.copy()
            broken[0, 0] = bad
            for run in (moe.numpy_forward, moe.upstream_forward):
                with self.assertRaises(ValueError):
                    run(c, broken, w)
        for run in (moe.numpy_forward, moe.upstream_forward):
            with self.assertRaises(ValueError):
                run(c, x.astype(np.float64), w)
            with self.assertRaises(ValueError):
                run(c, x[:, :-1], w)
            broken = dict(w, **{"gate.weight": w["gate.weight"].T})
            with self.assertRaises(ValueError):
                run(c, x, broken)
            broken = dict(w, **{"gate.weight": np.full_like(w["gate.weight"], 3e38)})
            with self.assertRaises(FloatingPointError):
                run(c, np.full_like(x, 3), broken)
        for field, value in (("num_experts", 0), ("num_experts_per_tok", 5),
                             ("hidden_size", True), ("hidden_act", "relu"),
                             ("norm_topk_prob", 1)):
            with self.assertRaises(ValueError):
                replace(c, **{field: value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
