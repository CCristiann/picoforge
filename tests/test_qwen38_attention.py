"""Independent CPU oracle tests; no native/GPU parity is claimed here."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_attention as oracle

SOURCE = ROOT / "build/qwen38-attention/modeling_qwen4_exp.py"


def prefix(inputs, length):
    return oracle.Inputs(**{name: value if name.startswith("gamma_") else value[:length]
                            for name, value in inputs.arrays().items()})


class AttentionOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.hf = oracle.load_reference(SOURCE)

    def setUp(self):
        self.cfg = oracle.Config()
        self.inputs = oracle.synthetic(self.cfg, 17)

    def evaluate(self, inputs=None, start=0, cfg=None):
        return oracle.evaluate(self.hf, cfg or self.cfg, inputs or self.inputs, start)

    def test_source_is_pinned_and_tampering_rejected(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), oracle.SOURCE_HASH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                oracle.load_reference(path)
            with self.assertRaises(FileExistsError):
                oracle.fetch_source(path)

    def test_dimensions_dtype_and_nonfinite_are_rejected(self):
        for args in ({"budget": 9}, {"heads": 3}, {"rotary_dim": 3},
                     {"rotary_dim": 10}, {"ratio": 0}, {"eps": float("nan")}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                oracle.Config(**args)
        for mutation in (lambda x: setattr(x, "v", x.v.astype(np.float64)),
                         lambda x: setattr(x, "iq", x.iq[:, :1]),
                         lambda x: x.k.__setitem__((0, 0, 0), np.nan)):
            x = copy.deepcopy(self.inputs)
            mutation(x)
            with self.assertRaises(ValueError):
                self.evaluate(x)
        with self.assertRaises(ValueError):
            self.evaluate(start=17)

    def test_dense_softmax_matches_independent_sdpa(self):
        result = self.evaluate()
        for name in ("attention", "gated"):
            np.testing.assert_allclose(result[f"{name}_eager"], result[f"{name}_fp64"],
                                       atol=3e-5, rtol=2e-4)
        self.assertFalse(np.triu(result["mask"], 1).any())
        self.assertTrue(np.isfinite(result["gated_eager"]).all())

    def test_uniform_scores_are_mean_of_selected_values_and_half_gate(self):
        x = self.inputs
        x.qg.fill(0)
        x.iq.fill(0)
        result = self.evaluate(x)
        for p, mask in enumerate(result["mask"]):
            means = x.v[mask].astype(np.float64).mean(axis=0)
            expected = np.repeat(means, self.cfg.heads // self.cfg.kv_heads, axis=0)
            np.testing.assert_allclose(result["attention_fp64"][p], expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(result["gated_eager"][p], expected * .5, atol=3e-7, rtol=3e-6)

    def test_single_visible_token_gate_is_elementwise(self):
        x = prefix(self.inputs, 1)
        x.qg[0, :, 1] = np.array([-80, 0, 80, -80, 0, 80, 0, 0], dtype=np.float32)
        result = self.evaluate(x)
        repeated = np.repeat(x.v, 2, axis=1)
        factor = 1 / (1 + np.exp(-x.qg[:, :, 1].astype(np.float64)))
        np.testing.assert_allclose(result["gated_eager"], repeated * factor, atol=1e-7, rtol=1e-6)

    def test_pooling_precedes_norm_and_uses_first_position(self):
        result = self.evaluate()
        pooled = self.inputs.ik[:16].astype(np.float64).reshape(4, 4, 8).mean(axis=1)
        np.testing.assert_allclose(result["pooled_last"], pooled, atol=1e-7, rtol=1e-6)
        norm = pooled / np.sqrt(np.mean(pooled ** 2, axis=-1, keepdims=True) + self.cfg.eps)
        norm *= self.inputs.gamma_ik
        expected = norm.copy()
        half = self.cfg.rotary_dim // 2
        for b in range(4):
            for i in range(half):
                angle = b * self.cfg.ratio * self.cfg.theta ** (-2 * i / self.cfg.rotary_dim)
                c, s = np.cos(angle), np.sin(angle)
                expected[b, i] = norm[b, i] * c - norm[b, i + half] * s
                expected[b, i + half] = norm[b, i + half] * c + norm[b, i] * s
        np.testing.assert_allclose(result["block_keys_last"], expected, atol=2e-5, rtol=2e-5)

    def test_nonrotary_features_and_interleaved_q_gate(self):
        result = self.evaluate()
        raw = self.inputs.qg[:, :, 0].astype(np.float64)
        norm = raw / np.sqrt(np.mean(raw ** 2, axis=-1, keepdims=True) + self.cfg.eps)
        norm *= self.inputs.gamma_q
        np.testing.assert_allclose(result["q"][..., 4:], norm[..., 4:], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(result["gate"], self.inputs.qg[:, :, 1])

    def test_block_completion_can_exclude_current_token(self):
        x = prefix(self.inputs, 13)
        x.iq.fill(0)
        x.iq[..., -1] = 1  # No rotated component: an analytical ranking.
        x.ik.fill(0)
        x.ik[:8, -1] = 1
        x.ik[8:, -1] = -1
        x.gamma_iq.fill(1)
        x.gamma_ik.fill(1)
        mask = self.evaluate(x)["mask"]
        np.testing.assert_array_equal(np.flatnonzero(mask[10]), np.arange(11))
        np.testing.assert_array_equal(np.flatnonzero(mask[11]), np.arange(8))
        np.testing.assert_array_equal(np.flatnonzero(mask[12]), np.r_[np.arange(8), 12])

    def test_real_budget_boundary_and_tied_scores(self):
        cfg = oracle.Config(budget=2048)
        x = oracle.synthetic(cfg, 2053)
        x.iq.fill(0)
        result = self.evaluate(x, start=2046, cfg=cfg)
        np.testing.assert_array_equal(result["mask"].sum(-1), [2047, 2048, 2049, 2050, 2051, 2048, 2049])
        for row, p in zip(result["mask"], range(2046, 2053)):
            self.assertFalse(row[p + 1:].any())
            tail = (p + 1) // 4 * 4
            self.assertTrue(row[tail:p + 1].all())
        # torch.topk does not specify which equal-score blocks win. Only
        # valid set/cardinality is asserted here, never native tie parity.

    def test_chunks_append_and_prefix_replay(self):
        full = self.evaluate()
        boundaries = [1, 4, 6, 11, 12, 13, 17]
        start = 0
        for end in boundaries:
            result = self.evaluate(prefix(self.inputs, end), start=start)
            np.testing.assert_array_equal(result["mask"], full["mask"][start:end, :end])
            np.testing.assert_allclose(result["gated_eager"], full["gated_eager"][start:end],
                                       atol=3e-5, rtol=2e-4)
            start = end
        for p in range(17):
            result = self.evaluate(prefix(self.inputs, p + 1), start=p)
            np.testing.assert_array_equal(result["mask"][0], full["mask"][p, :p + 1])

    def test_future_values_do_not_change_past(self):
        full = self.evaluate()
        changed = copy.deepcopy(self.inputs)
        changed.k[6:] = 1000
        changed.v[6:] = -1000
        changed.ik[6:] = 1000
        result = self.evaluate(changed)
        np.testing.assert_array_equal(result["mask"][:6], full["mask"][:6])
        np.testing.assert_array_equal(result["gated_eager"][:6], full["gated_eager"][:6])

    def test_real_head_shapes_and_partial_rotary(self):
        cfg = oracle.Config(width=2560, heads=24, kv_heads=2, head_dim=256,
                            index_heads=4, index_dim=128, rotary_dim=64,
                            budget=2048, theta=10_000_000.0)
        x = oracle.synthetic(cfg, 9)
        result = self.evaluate(x, cfg=cfg)
        self.assertEqual(result["q"].shape, (9, 24, 256))
        self.assertEqual(result["iq"].shape, (9, 4, 128))
        np.testing.assert_allclose(result["attention_eager"], result["attention_fp64"],
                                   atol=3e-5, rtol=2e-4)

    def test_inputs_and_rng_are_unchanged_and_sessions_are_independent(self):
        before = {k: v.copy() for k, v in self.inputs.arrays().items()}
        rng = torch.random.get_rng_state().clone()
        first = self.evaluate()
        self.evaluate(oracle.synthetic(self.cfg, 5, seed=777))
        second = self.evaluate()
        for k in first:
            np.testing.assert_array_equal(first[k], second[k])
        for k, v in self.inputs.arrays().items():
            np.testing.assert_array_equal(v, before[k])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_matrix_budget_and_gamma_roundtrip_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "matrix budget"):
            self.evaluate(oracle.synthetic(self.cfg, 2000))
        self.inputs.gamma_q[0] = np.float32(1e-10)
        with self.assertRaisesRegex(ValueError, "round-trip"):
            self.evaluate()

    def test_export_hashes_and_refuses_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            record = oracle.write_fixture(self.hf, SOURCE, output, "tiny")
            self.assertEqual(json.loads((output / "manifest.json").read_text()), record)
            artifact = output / "fixture.npz"
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), record["artifact_sha256"])
            with np.load(artifact, allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), set(record["arrays"]))
                for key in arrays:
                    value = arrays[key]
                    meta = record["arrays"][key]
                    self.assertEqual(list(value.shape), meta["shape"])
                    self.assertEqual(value.dtype.str, meta["dtype"])
                    self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(), meta["sha256"])
            with self.assertRaises(FileExistsError):
                oracle.write_fixture(self.hf, SOURCE, output, "tiny")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *remaining])
