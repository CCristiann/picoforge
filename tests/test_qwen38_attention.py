"""Independent CPU oracle tests; no native/GPU parity is claimed here."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import struct
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_attention as oracle

SOURCE = ROOT / "build/qwen38-attention/modeling_qwen4_exp.py"
NATIVE = None  # Explicit --native runs the sanitized C harness, never a stub.


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
            for key in ("k", "v"):
                np.testing.assert_array_equal(result[key], full[key][:end])
            np.testing.assert_array_equal(result["block_keys_last"], full["block_keys_last"][:end // 4])
            np.testing.assert_array_equal(result["pooled_last"], full["pooled_last"][:end // 4])
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

    def test_sparse_selection_matches_independent_sort_with_margin(self):
        compared, tied = 0, 0
        for seed in (3811, 77, 125, 902):
            result = self.evaluate(oracle.synthetic(self.cfg, 25, seed))
            for row, scores in enumerate(result["indexer_scores_fp64_unscaled"]):
                visible = (row + 1) // self.cfg.ratio
                block_ids = np.arange(visible)
                order = np.lexsort((block_ids, -scores[:visible]))
                budget = self.cfg.budget // self.cfg.ratio
                if visible > budget:
                    gap = scores[order[budget - 1]] - scores[order[budget]]
                    if gap == 0:
                        # ReLU legitimately produces equal zero scores. Check
                        # the admissible top-k set without guessing HF tie order.
                        block_mask = result["mask"][row, :visible * 4].reshape(visible, 4)
                        np.testing.assert_array_equal(block_mask.any(-1), block_mask.all(-1))
                        chosen = block_mask.all(-1)
                        threshold = scores[order[budget - 1]]
                        self.assertTrue(chosen[scores[:visible] > threshold].all())
                        self.assertFalse(chosen[scores[:visible] < threshold].any())
                        self.assertEqual(int(chosen.sum()), budget)
                        tied += 1
                        continue
                    self.assertGreater(gap, 4e-4)
                    compared += 1
                selected = order[:budget]
                expected = np.zeros(25, dtype=bool)
                for b in selected:
                    expected[b * 4:(b + 1) * 4] = True
                expected[visible * 4:row + 1] = True
                np.testing.assert_array_equal(result["mask"][row], expected)
        self.assertGreater(compared, 10)
        self.assertGreater(tied, 0)

    def test_near_ties_are_reported_without_claiming_exact_rank_parity(self):
        x = prefix(self.inputs, 12)
        x.iq.fill(0)
        x.iq[..., -1] = 1
        x.ik.fill(0)
        x.ik[..., -2] = .8
        delta = np.spacing(np.float32(.6))
        x.ik[:, -1] = np.repeat(np.array([.6 + delta, .6, .6 - delta], dtype=np.float32), 4)
        x.gamma_iq.fill(1)
        x.gamma_ik.fill(1)
        result = self.evaluate(x, start=11)
        scores = result["indexer_scores_fp64_unscaled"][0]
        self.assertLess(float(np.ptp(scores)), 4e-4)
        blocks = result["mask"][0].reshape(3, 4)
        np.testing.assert_array_equal(blocks.any(-1), blocks.all(-1))
        self.assertEqual(int(blocks.all(-1).sum()), 2)
        # A close score is not silently turned into a native/HF parity PASS.

    def test_softmax_normalizes_across_blocks_not_within_each_block(self):
        x = prefix(self.inputs, 8)
        x.qg.fill(0)
        x.qg[:, :, 0, -1] = 1
        x.k.fill(0)
        x.k[:4, :, -1] = 1
        x.k[4:, :, -1] = -1
        x.v[:4] = 0
        x.v[4:] = 1
        result = self.evaluate(x, start=7)
        for h in range(self.cfg.heads):
            logits = result["k"][:, h // 2].astype(np.float64) @ result["q"][0, h].astype(np.float64)
            logits /= np.sqrt(self.cfg.head_dim)
            weights = np.exp(logits - np.max(logits))
            expected = weights[4:].sum() / weights.sum()
            self.assertGreater(abs(expected - .5), .1)
            np.testing.assert_allclose(result["attention_fp64"][0, h], expected, atol=1e-12, rtol=1e-12)

    def test_softmax_large_logits_remains_finite(self):
        x = prefix(self.inputs, 8)
        x.gamma_q.fill(32)
        x.gamma_k.fill(32)
        result = self.evaluate(x)
        self.assertTrue(np.isfinite(result["attention_eager"]).all())
        np.testing.assert_allclose(result["attention_eager"], result["attention_fp64"], atol=3e-5, rtol=2e-4)

    def test_long_prefix_chunk_boundaries_without_large_dense_oracle(self):
        cfg = oracle.Config(budget=2048)
        inputs = oracle.synthetic(cfg, 8193)
        for end in (1024, 1025, 2048, 2053, 8192, 8193):
            chunk = self.evaluate(prefix(inputs, end), start=end - 3, cfg=cfg)
            for p in range(end - 3, end):
                decode = self.evaluate(prefix(inputs, p + 1), start=p, cfg=cfg)
                np.testing.assert_array_equal(decode["mask"][0], chunk["mask"][p - end + 3, :p + 1])
                np.testing.assert_allclose(decode["gated_eager"][0], chunk["gated_eager"][p - end + 3],
                                           atol=3e-5, rtol=2e-4)

    def test_online_tiles_and_splits_match_independent_sdpa(self):
        result = self.evaluate()
        for tile in (1, 3, 8, 32):
            for splits in (1, 2, 7, 17):
                for precision in ("float32", "float64"):
                    with self.subTest(tile=tile, splits=splits, precision=precision):
                        online = oracle.online_attention(self.cfg, result, tile, splits, precision)
                        atol, rtol = (3e-5, 2e-4) if precision == "float32" else (1e-12, 1e-12)
                        np.testing.assert_allclose(online["attention"], result["attention_fp64"],
                                                   atol=atol, rtol=rtol)
                        # First query leaves every later split empty. Its
                        # normalization is exactly one, regardless of tiling.
                        np.testing.assert_array_equal(online["denominator"][0], 1)
                        self.assertTrue(np.isfinite(online["logsumexp"]).all())

    def test_online_normalization_matches_dense_logsumexp(self):
        result = self.evaluate()
        online = oracle.online_attention(self.cfg, result, 3, 7, "float64")
        for row in range(17):
            for head in range(self.cfg.heads):
                scores = (result["k"][:, head // 2].astype(np.float64)
                          @ result["q"][row, head].astype(np.float64)) / np.sqrt(self.cfg.head_dim)
                selected = scores[result["mask"][row]]
                # NumPy's pairwise logaddexp reduction is separate from the
                # online maximum/denominator recurrence under test.
                expected = np.logaddexp.reduce(selected)
                np.testing.assert_allclose(online["logsumexp"][row, head], expected,
                                           atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(online["maximum"][row, head], selected.max(), places=12)

    def test_online_rescales_earlier_tiles_for_large_later_maximum(self):
        x = prefix(self.inputs, 8)
        x.qg.fill(0)
        x.qg[:, :, 0, -1] = 1
        x.gamma_q.fill(32)
        x.gamma_k.fill(32)
        x.k.fill(0)
        x.k[:4, :, -1] = -1
        x.k[4:, :, -1] = 1
        x.v[:4] = -1
        x.v[4:] = 1
        result = self.evaluate(x, start=7)
        for tile, splits in ((1, 1), (3, 2), (5, 7)):
            online = oracle.online_attention(self.cfg, result, tile, splits)
            np.testing.assert_array_equal(online["attention"], 1)
            np.testing.assert_allclose(online["attention"], result["attention_fp64"], atol=1e-12)
            np.testing.assert_array_equal(online["denominator"], 4)

    def test_online_rejects_empty_rows_and_invalid_layouts(self):
        result = self.evaluate()
        for options in ({"tile_size": 0}, {"splits": 18}, {"splits": True},
                        {"precision": "float16"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                oracle.online_attention(self.cfg, result, **options)
        for name, replacement in (("mask", np.zeros_like(result["mask"])),
                                  ("k", result["k"][:, :1]),
                                  ("v", result["v"].astype(np.float64)),
                                  ("q", np.full_like(result["q"], np.nan))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                oracle.online_attention(self.cfg, {**result, name: replacement})

    def test_indexer_relu_precedes_head_sum_without_head_softmax(self):
        cfg = oracle.Config(budget=4)
        x = oracle.synthetic(cfg, 12)
        x.iq.fill(0)
        x.iq[:, 0, -2] = 1
        x.iq[:, 1, -1] = 1
        x.ik.fill(0)
        directions = np.array([[.8, .6], [.96, -.28], [-.6, .8]], dtype=np.float32)
        x.ik[:, -2:] = np.repeat(directions, 4, axis=0)
        x.gamma_iq.fill(1)
        x.gamma_ik.fill(1)
        result = self.evaluate(x, start=11, cfg=cfg)
        # No rotated coordinates: evaluate the two independent dot products
        # analytically, including RMS epsilon and input F32 rounding.
        directions = directions.astype(np.float64)
        key_norm = np.sqrt((directions ** 2).sum(-1) / cfg.index_dim + cfg.eps)
        query_norm = np.sqrt(1 / cfg.index_dim + cfg.eps)
        dots = directions / (key_norm[:, None] * query_norm)
        expected = np.maximum(dots, 0).sum(-1)
        np.testing.assert_allclose(result["indexer_scores_fp64_unscaled"][0], expected,
                                   atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(np.flatnonzero(result["mask"][0]), np.arange(4))
        # Softmax per head would select a DIFFERENT block on this fixture.
        weights = np.exp(dots - dots.max(axis=0))
        self.assertNotEqual(int((weights / weights.sum(axis=0)).sum(-1).argmax()), 0)
        # Opposite heads must not cancel before ReLU. Block 2 is uniquely
        # strongest by absolute projection; sum-then-ReLU gives all zeros.
        x.iq[:, 0, -2] = 0
        x.iq[:, 0, -1] = -1
        x.ik[:, -2:] = np.repeat(np.array([[1, 0], [.8, .6], [0, 1]], np.float32), 4, axis=0)
        result = self.evaluate(x, start=11, cfg=cfg)
        np.testing.assert_array_equal(np.flatnonzero(result["mask"][0]), np.arange(8, 12))
        np.testing.assert_allclose(result["indexer_scores_fp64_unscaled"][0],
                                   np.array([0, .6, 1]) / (1 / cfg.index_dim + cfg.eps),
                                   atol=2e-5, rtol=2e-5)

    def test_ratio_one_single_block_and_oversized_budgets(self):
        for cfg in (oracle.Config(ratio=1, budget=1), oracle.Config(budget=4),
                    oracle.Config(budget=64)):
            x = oracle.synthetic(cfg, 13)
            full = self.evaluate(x, cfg=cfg)
            for p, row in enumerate(full["mask"]):
                complete, tail = divmod(p + 1, cfg.ratio)
                expected_count = min(complete, cfg.budget // cfg.ratio) * cfg.ratio + tail
                self.assertEqual(int(row.sum()), expected_count)
                self.assertFalse(row[p + 1:].any())
                self.assertTrue(row[complete * cfg.ratio:p + 1].all())
                if complete:
                    blocks = row[:complete * cfg.ratio].reshape(complete, cfg.ratio)
                    np.testing.assert_array_equal(blocks.any(-1), blocks.all(-1))
            for start, end in ((0, 3), (3, 4), (4, 9), (9, 13)):
                chunk = self.evaluate(prefix(x, end), start=start, cfg=cfg)
                np.testing.assert_array_equal(chunk["mask"], full["mask"][start:end, :end])
                np.testing.assert_allclose(chunk["gated_eager"], full["gated_eager"][start:end],
                                           atol=3e-5, rtol=2e-4)

    def test_completing_a_block_preserves_past_queries_and_replaces_raw_tail(self):
        before = self.evaluate(prefix(self.inputs, 11), start=8)
        after = self.evaluate(prefix(self.inputs, 12), start=8)
        np.testing.assert_array_equal(before["mask"], after["mask"][:3, :11])
        np.testing.assert_allclose(before["gated_eager"], after["gated_eager"][:3],
                                   atol=3e-5, rtol=2e-4)
        self.assertEqual(len(before["pooled_last"]), 2)
        self.assertEqual(len(after["pooled_last"]), 3)
        np.testing.assert_array_equal(before["block_keys_last"], after["block_keys_last"][:2])
        expected_pool = self.inputs.ik[8:12].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(after["pooled_last"][-1], expected_pool, atol=1e-7, rtol=1e-6)
        # Raw keys across the append boundary form one block. Norming or
        # rotating a tail early and then pooling it would violate this check.
        mutated = copy.deepcopy(prefix(self.inputs, 12))
        mutated.ik[11] += 4
        changed = self.evaluate(mutated, start=8)
        np.testing.assert_array_equal(before["mask"], changed["mask"][:3, :11])
        np.testing.assert_array_equal(before["block_keys_last"], changed["block_keys_last"][:2])
        np.testing.assert_allclose(changed["pooled_last"][-1], expected_pool + 1,
                                   atol=2e-7, rtol=1e-6)


class NativeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if NATIVE is None:
            raise unittest.SkipTest("Pass --native build/qwen38-d/test_attention for C parity")
        torch.set_num_threads(1)
        cls.hf = oracle.load_reference(SOURCE)

    def call(self, mode, dims, *arrays, dtype=np.float32):
        payload = struct.pack("=4I", *dims) + b"".join(x.tobytes() for x in arrays)
        run = subprocess.run([str(NATIVE), mode], input=payload, capture_output=True, check=True)
        self.assertEqual(run.stderr, b"")
        code, = struct.unpack("=i", run.stdout[:4])
        return code, np.frombuffer(run.stdout[4:], dtype=dtype)

    def test_native_rope_matches_official_positions(self):
        cfg = oracle.Config()
        x = oracle.synthetic(cfg, 8193)
        for pos in (0, 1, 3, 4, 1024, 2048, 8192):
            result = oracle.evaluate(self.hf, cfg, prefix(x, pos + 1), pos)
            raw = x.qg[pos, :, 0].astype(np.float64)
            norm = raw / np.sqrt(np.mean(raw ** 2, -1, keepdims=True) + cfg.eps) * x.gamma_q
            code, got = self.call("rope", (cfg.heads, cfg.head_dim, cfg.rotary_dim, pos),
                                  np.array([cfg.theta], np.float32), norm.astype(np.float32))
            self.assertEqual(code, 0)
            np.testing.assert_allclose(got.reshape(cfg.heads, cfg.head_dim), result["q"][0],
                                       atol=2e-5, rtol=2e-5)

    def test_native_scores_and_selection_match_independent_reference(self):
        cfg = oracle.Config()
        result = oracle.evaluate(self.hf, cfg, oracle.synthetic(cfg, 25))
        for pos in range(25):
            n = (pos + 1) // cfg.ratio
            code, scores = self.call("scores", (cfg.index_heads, cfg.index_dim, n, 0),
                                     result["iq"][pos], result["block_keys_last"][:n])
            self.assertEqual(code, 0)
            reference = result["indexer_scores_fp64_unscaled"][pos, :n]
            np.testing.assert_allclose(scores, reference, atol=2e-4, rtol=2e-5)
            code, ids = self.call("select", (n, 2, 0, 0), scores, dtype=np.uint32)
            self.assertEqual(code, 0)
            expected = np.sort(np.lexsort((np.arange(n), -scores))[:2])
            np.testing.assert_array_equal(ids, expected)
            if n > 2:
                ranked = np.sort(reference)[::-1]
                gap = ranked[1] - ranked[2]
                error = np.max(np.abs(scores - reference))
                if gap > 2 * error:
                    mask = result["mask"][pos, :n * 4].reshape(n, 4).all(-1)
                    np.testing.assert_array_equal(ids, np.flatnonzero(mask))

    def test_native_exact_ties_and_nonfinite_scores(self):
        for values in ([0, -0., 0, 0, 0], [3, 3, 1, 4, 4], [1, 2, 3, 4, 5]):
            scores = np.array(values, np.float32)
            for budget in (1, 2, 3, 5, 8):
                code, ids = self.call("select", (5, budget, 0, 0), scores, dtype=np.uint32)
                self.assertEqual(code, 0)
                np.testing.assert_array_equal(ids, np.sort(np.lexsort((np.arange(5), -scores))[:budget]))
        for value in (np.nan, np.inf, -np.inf):
            code, _ = self.call("select", (1, 1, 0, 0), np.array([value], np.float32), dtype=np.uint32)
            self.assertEqual(code, 16)  # Q38_NONFINITE, contract v1.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--native", type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    NATIVE = args.native.resolve() if args.native else None
    unittest.main(argv=[sys.argv[0], *remaining])
