"""Independent oracle with explicit optional native CPU and private Metal gates."""
import argparse
import copy
import hashlib
import json
import re
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
METAL = None
METAL_LIBRARY = ROOT / "build/qwen38-d/attention.metallib"
SLICE_FIXTURE = None
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

    def test_native_real_rotary_at_long_absolute_positions(self):
        cfg, _, _ = oracle.fixture_case("real-shape")
        x = oracle.synthetic(cfg, 1).qg[:, :, 0].copy()
        rope = self.hf.Qwen4ExpTextRotaryEmbedding(cfg.hf())
        for pos in (0, 2051, 8192, 32768):
            with self.subTest(pos=pos):
                cos, sin = rope(torch.zeros(1, 1, 1), torch.tensor([[[pos]]]))
                expected = self.hf.apply_rotary_pos_emb(torch.from_numpy(x)[None], cos=cos, sin=sin,
                                                        unsqueeze_dim=2).numpy()
                code, got = self.call("rope", (cfg.heads, cfg.head_dim, cfg.rotary_dim, pos),
                                      np.array([cfg.theta], np.float32), x)
                self.assertEqual(code, 0)
                np.testing.assert_allclose(got.reshape(expected.shape), expected, atol=2e-5, rtol=2e-5)

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

    def attend(self, cfg, result, row, ids=None):
        if ids is None:
            ids = np.flatnonzero(result["mask"][row]).astype(np.uint32)
        gate = torch.sigmoid(torch.from_numpy(result["gate"][row])).numpy()
        return self.call("attend", (cfg.heads, cfg.kv_heads, cfg.head_dim, len(result["k"])),
                         result["q"][row], gate, result["k"], result["v"],
                         np.array([len(ids)], np.uint32), ids)

    def test_native_tiled_softmax_and_gate_match_sdpa(self):
        for case in ("tiny", "boundary", "real-shape"):
            cfg, tokens, start = oracle.fixture_case(case)
            result = oracle.evaluate(self.hf, cfg, oracle.synthetic(cfg, tokens), start)
            for row in range(tokens - start):
                code, y = self.attend(cfg, result, row)
                self.assertEqual(code, 0)
                np.testing.assert_allclose(y.reshape(cfg.heads, cfg.head_dim), result["gated_fp64"][row],
                                           atol=3e-5, rtol=2e-4)

    def test_native_tiles_rescale_and_skip_poisoned_unselected_keys(self):
        cfg = oracle.Config(budget=64)
        x = oracle.synthetic(cfg, 33)
        x.qg.fill(0); x.qg[:, :, 0, -1] = 1
        x.gamma_q.fill(32); x.gamma_k.fill(32)
        x.k.fill(0); x.k[:16, :, -1] = -1; x.k[16:, :, -1] = 1
        x.v[:16] = -1; x.v[16:] = 1
        result = oracle.evaluate(self.hf, cfg, x)
        code, y = self.attend(cfg, result, 32)
        self.assertEqual(code, 0)
        np.testing.assert_array_equal(y, .5)
        result["k"][1:] = np.nan; result["v"][1:] = np.nan
        code, y = self.attend(cfg, result, 0)
        self.assertEqual(code, 0)
        np.testing.assert_allclose(y.reshape(cfg.heads, cfg.head_dim), result["gated_fp64"][0],
                                   atol=3e-5, rtol=2e-4)
        code, _ = self.attend(cfg, result, 1)
        self.assertEqual(code, 16)

    def test_native_attention_rejects_invalid_selected_ids(self):
        cfg = oracle.Config()
        result = oracle.evaluate(self.hf, cfg, oracle.synthetic(cfg, 3))
        for ids in ([], [1, 1], [2, 1], [3]):
            code, _ = self.attend(cfg, result, 2, np.array(ids, np.uint32))
            self.assertEqual(code, 5 if not ids else 4)  # SHAPE / RANGE

    def append(self, cfg, x, result, ends):
        config = struct.pack("=9I2f", cfg.heads, cfg.kv_heads, cfg.head_dim,
                             cfg.index_heads, 1, cfg.index_dim, cfg.ratio, cfg.budget,
                             cfg.rotary_dim, cfg.eps, cfg.theta)
        header = struct.pack("=2I", len(x.qg), len(ends)) + np.array(ends, np.uint32).tobytes()
        gate = torch.sigmoid(torch.from_numpy(result["gate"])).numpy()
        arrays = [result["q"], gate, result["k"], result["v"], result["iq"], x.ik,
                  result["block_keys_last"]]
        payload = config + header + b"".join(a.tobytes() for a in arrays)
        run = subprocess.run([str(NATIVE), "append"], input=payload, capture_output=True, check=True)
        self.assertEqual(run.stderr, b"")
        code, = struct.unpack("=i", run.stdout[:4])
        return code, np.frombuffer(run.stdout[4:], np.float32).reshape(result["q"].shape)

    def test_native_lifecycle_capacity_epoch_owner_and_busy(self):
        run = subprocess.run([str(NATIVE), "lifecycle"], capture_output=True, check=True)
        self.assertEqual(run.stdout, b"lifecycle OK\n")
        self.assertEqual(run.stderr, b"")

    def test_native_prefill_decode_chunk_append_and_zero_token(self):
        cfg = oracle.Config()
        x = oracle.synthetic(cfg, 33)
        # Unique analytical scores; all selected sets must match HF. This
        # fixture does not turn a near-tie mask disagreement into a PASS.
        x.iq.fill(0); x.iq[..., -1] = 1
        x.ik.fill(0)
        angles = np.linspace(.1, .9, 9, dtype=np.float32).repeat(4)[:33]
        x.ik[:, -1] = angles; x.ik[:, -2] = np.sqrt(1 - angles ** 2)
        x.gamma_iq.fill(1); x.gamma_ik.fill(1)
        result = oracle.evaluate(self.hf, cfg, x)
        scores = result["indexer_scores_fp64_unscaled"][-1]
        self.assertGreater(float(np.diff(scores).min()), 4e-4)
        outputs = []
        for ends in ([33], list(range(1, 34)), [0, 1, 3, 4, 4, 7, 12, 16, 17, 31, 32, 33]):
            code, y = self.append(cfg, x, result, ends)
            self.assertEqual(code, 0)
            np.testing.assert_allclose(y, result["gated_fp64"], atol=3e-5, rtol=2e-4)
            outputs.append(y)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], outputs[2])

    def test_native_numeric_failure_invalidates_append_until_reset(self):
        cfg = oracle.Config()
        x = oracle.synthetic(cfg, 13)
        result = oracle.evaluate(self.hf, cfg, x)
        for key in ("q", "k", "v", "iq", "block_keys_last"):
            changed = copy.deepcopy(result)
            changed[key].flat[-1] = np.nan
            code, _ = self.append(cfg, x, changed, [3, 7, 13])
            self.assertEqual(code, 16, key)
        x.ik[-1, -1] = np.inf
        code, _ = self.append(cfg, x, result, [3, 7, 13])
        self.assertEqual(code, 16)

    def test_native_pool_is_mean_of_raw_keys_before_normalization(self):
        cfg, _, _ = oracle.fixture_case("real-shape")
        x = oracle.synthetic(cfg, 12)
        result = oracle.evaluate(self.hf, cfg, x)
        for block in range(3):
            raw = x.ik[block * 4:(block + 1) * 4]
            code, pooled = self.call("pool", (4, cfg.index_dim, 0, 0), raw)
            self.assertEqual(code, 0)
            np.testing.assert_allclose(pooled, raw.astype(np.float64).mean(0), atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(pooled, result["pooled_last"][block], atol=2e-5, rtol=2e-5)
        code, _ = self.call("pool", (4, 2, 0, 0), np.full((4, 2), np.finfo(np.float32).max, np.float32))
        self.assertEqual(code, 16)

    def test_native_real_index_dimensions_and_selection_margin(self):
        rng = np.random.default_rng(3811)
        iq = rng.uniform(-1, 1, (4, 128)).astype(np.float32)
        blocks = rng.uniform(-1, 1, (513, 128)).astype(np.float32)
        code, scores = self.call("scores", (4, 128, 513, 0), iq, blocks)
        self.assertEqual(code, 0)
        expected = np.maximum(iq.astype(np.float64) @ blocks.astype(np.float64).T, 0).sum(0)
        np.testing.assert_allclose(scores, expected, atol=2e-4, rtol=2e-5)
        for budget in (1, 10, 512):
            code, ids = self.call("select", (513, budget, 0, 0), scores, dtype=np.uint32)
            self.assertEqual(code, 0)
            order = np.lexsort((np.arange(513), -scores))
            np.testing.assert_array_equal(ids, np.sort(order[:budget]))
            ref_order = np.lexsort((np.arange(513), -expected))
            gap = expected[ref_order[budget - 1]] - expected[ref_order[budget]]
            error = np.max(np.abs(scores - expected))
            # Zero-score ties are deliberate contract tie tests, not HF
            # end-to-end parity. Otherwise demand a measured stable margin.
            if gap > 0:
                self.assertGreater(gap, 2 * error)
                np.testing.assert_array_equal(ids, np.sort(ref_order[:budget]))

    def test_native_append_crosses_first_sparse_position(self):
        cfg = oracle.Config(budget=2048)
        x = oracle.synthetic(cfg, 2053)
        x.iq.fill(0); x.iq[..., -1] = 1
        x.ik.fill(0)
        coordinates = np.linspace(.1, .9, 514, dtype=np.float32).repeat(4)[:2053]
        x.ik[:, -1] = coordinates; x.ik[:, -2] = np.sqrt(1 - coordinates ** 2)
        x.gamma_iq.fill(1); x.gamma_ik.fill(1)
        pieces = []
        # Bounded independent oracle; never materialize the full T*T mask.
        for start in range(0, 2053, 128):
            end = min(start + 128, 2053)
            pieces.append(oracle.evaluate(self.hf, cfg, prefix(x, end), start))
        result = dict(pieces[-1])
        for name in ("q", "iq", "gate", "gated_fp64"):
            result[name] = np.concatenate([part[name] for part in pieces])
        self.assertGreater(float(np.diff(result["indexer_scores_fp64_unscaled"][-1]).min()), 4e-4)
        outputs = []
        for ends in ([2053], [0, 1, 1023, 1024, 1025, 2047, 2048, 2051, 2052, 2053]):
            code, y = self.append(cfg, x, result, ends)
            self.assertEqual(code, 0)
            np.testing.assert_allclose(y, result["gated_fp64"], atol=3e-5, rtol=2e-4)
            outputs.append(y)
        np.testing.assert_array_equal(*outputs)


    def public(self, cfg, x, ends):
        return oracle.native_public(NATIVE,cfg,x,ends)

    def test_public_prep_gain_gate_pooling_and_chunk_equivalence(self):
        for case in ("tiny", "real-shape"):
            cfg, n, _ = oracle.fixture_case(case)
            x = oracle.synthetic(cfg, n)
            # Stable, nondegenerate index ranking; other projections/gains
            # and gates remain random and non-unit as in the original fixture.
            x.iq.fill(0); x.iq[..., -1] = 1
            coords = np.linspace(.1,.9,(n+3)//4,dtype=np.float32).repeat(4)[:n]
            x.ik.fill(0); x.ik[:, -1] = coords; x.ik[:, -2] = np.sqrt(1-coords**2)
            ref = oracle.evaluate(self.hf, cfg, x)
            outputs = []
            for ends in ([n], list(range(1,n+1)), [0,1,3,4,4,7,n]):
                code, got = self.public(cfg, x, ends)
                self.assertEqual(code, 0)
                np.testing.assert_allclose(got["out"], ref["gated_fp64"], atol=3e-5, rtol=2e-4)
                np.testing.assert_allclose(got["k"], ref["k"], atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(got["blocks"], ref["block_keys_last"], atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(got["ik"], x.ik)
                hd = cfg.heads * cfg.head_dim
                for p in sorted(set(end-1 for end in ends if end)):
                    np.testing.assert_allclose(got["trace"][p,:hd], ref["q"][p].ravel(), atol=2e-5, rtol=2e-5)
                    gate = torch.sigmoid(torch.from_numpy(ref["gate"][p])).numpy().ravel()
                    np.testing.assert_allclose(got["trace"][p,hd:2*hd], gate, atol=1e-5, rtol=1e-5)
                    np.testing.assert_allclose(got["trace"][p,2*hd:], ref["iq"][p].ravel(), atol=2e-5, rtol=2e-5)
                outputs.append(got["out"])
            np.testing.assert_array_equal(outputs[0],outputs[1])
            np.testing.assert_array_equal(outputs[0],outputs[2])

    def test_public_nonfinite_invalidates_state(self):
        cfg = oracle.Config()
        for name in oracle.synthetic(cfg, 13).arrays():
            x = oracle.synthetic(cfg, 13)
            x.arrays()[name].flat[-1] = np.nan
            code, _ = self.public(cfg, x, [3,7,13])
            self.assertEqual(code,16,name)

    def test_public_ties_preserve_contract_and_record_uncertain_margin(self):
        cfg=oracle.Config()
        records=[]
        for near in (False,True):
            x=oracle.synthetic(cfg,13); x.qg.fill(0); x.iq.fill(0); x.ik.fill(0)
            if near:
                x.iq[..., -1]=1; x.ik[..., -2]=.8
                step=np.spacing(np.float32(.6))
                x.ik[:, -1]=np.repeat(np.array([.6+step,.6,.6-step,.6],np.float32),4)[:13]
            x.gamma_iq.fill(1); x.gamma_ik.fill(1)
            ref=oracle.evaluate(self.hf,cfg,x)
            code,got=self.public(cfg,x,[3,7,11,12,13]); self.assertEqual(code,0)
            order=np.lexsort((np.arange(3),-got["scores"]))
            ids=np.r_[np.concatenate([np.arange(b*4,b*4+4) for b in sorted(order[:2])]),12]
            np.testing.assert_array_equal(got["ids"],ids)
            expected=np.repeat(x.v[ids].astype(np.float64).mean(0),2,axis=0)*.5
            np.testing.assert_allclose(got["out"][-1],expected,atol=3e-5,rtol=2e-4)
            scores64=ref["indexer_scores_fp64_unscaled"][-1]
            rank=np.sort(scores64)[::-1]; gap=float(rank[1]-rank[2])
            error=float(np.max(np.abs(got["scores"]-scores64)))
            records.append(dict(near=near,gap=gap,max_score_error=error,
                scores_fp32=got["scores"].tolist(),scores_fp64=scores64.tolist(),
                selected_native=got["ids"].tolist(),selected_hf=np.flatnonzero(ref["mask"][-1]).tolist(),
                end_to_end_parity_certified=False,
                decision="Exact FP32 contract tested; boundary ties/near-ties require integration review"))
            self.assertLess(gap,4e-4)
        folder=Path(tempfile.mkdtemp(prefix="selection-",dir=ROOT/"build/qwen38-d"))
        (folder/"manifest.json").write_text(json.dumps(records,indent=2)+"\n")

    def test_public_first_sparse_boundary(self):
        cfg=oracle.Config(budget=2048); x=oracle.synthetic(cfg,2053)
        x.iq.fill(0); x.iq[..., -1]=1; x.ik.fill(0)
        coords=np.linspace(.1,.9,514,dtype=np.float32).repeat(4)[:2053]
        x.ik[:, -1]=coords; x.ik[:, -2]=np.sqrt(1-coords**2)
        results=[]
        for ends in ([2053],[0,1,1023,1024,1025,2047,2048,2051,2052,2053]):
            code,got=self.public(cfg,x,ends); self.assertEqual(code,0); results.append(got)
        np.testing.assert_array_equal(results[0]["out"],results[1]["out"])
        ref=oracle.evaluate(self.hf,cfg,x,2046)
        np.testing.assert_allclose(results[0]["out"][2046:],ref["gated_fp64"],atol=3e-5,rtol=2e-4)
        np.testing.assert_array_equal(results[0]["ids"],np.flatnonzero(ref["mask"][-1]))


class MetalAttentionTests(unittest.TestCase):
    """D stages and exact block selection on GPU; layer composition tested in G2."""
    @classmethod
    def setUpClass(cls):
        if METAL is None or NATIVE is None:
            raise unittest.SkipTest("Pass --metal and --native under the shared GPU lock")
        torch.set_num_threads(1)
        cls.hf=oracle.load_reference(SOURCE)
        cls.metrics={"calls":0,"peak_metal_allocated_bytes":0,"max_error_vs_cpu":{},"slice_error":None,"selection":[]}

    @classmethod
    def tearDownClass(cls):
        cls.metrics.update(source_sha256=oracle.SOURCE_HASH,
            metal_sha256=hashlib.sha256(METAL_LIBRARY.read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="Private Metal stages and device selection; no full graph or BF16 loss")
        folder=Path(tempfile.mkdtemp(prefix="metal-",dir=ROOT/"build/qwen38-d"))
        (folder/"manifest.json").write_text(json.dumps(cls.metrics,indent=2)+"\n")

    def call(self,mode,dims,*arrays,dtype=np.float32):
        if mode=="select":
            code,got=self.call("select_tokens",(dims[0],1,dims[1],0),*arrays,dtype=dtype)
            cpu_code,cpu=NativeAttentionTests.call(self,mode,dims,*arrays,dtype=dtype)
            self.assertEqual(code,cpu_code)
            if code==0: np.testing.assert_array_equal(got,cpu)
            return code,got
        payload=struct.pack("=4I",*dims)+b"".join(x.tobytes() for x in arrays)
        run=subprocess.run([str(METAL),str(METAL_LIBRARY),mode],input=payload,capture_output=True,check=True)
        log=re.sub(rb"(?m)^\d{4}-\d\d-\d\d [0-9:.]+ test_attention_gpu\[\d+:\d+\] Metal API Validation Enabled\n",b"",run.stderr)
        sizes=re.findall(rb"(?m)^Q38_D_METAL_BYTES=(\d+)\n",log); self.assertEqual(len(sizes),1)
        self.metrics["peak_metal_allocated_bytes"]=max(self.metrics["peak_metal_allocated_bytes"],int(sizes[0]))
        log=re.sub(rb"(?m)^Q38_D_METAL_BYTES=\d+\n",b"",log)
        self.assertEqual(log,b"")
        code=struct.unpack("=i",run.stdout[:4])[0]; got=np.frombuffer(run.stdout[4:],dtype=dtype)
        self.metrics["calls"]+=1
        if mode not in ("prep","select_tokens","sparse_attend"):
            cpu_code,cpu=NativeAttentionTests.call(self,mode,dims,*arrays,dtype=dtype)
            self.assertEqual(code,cpu_code)
            if code==0 and len(got):
                error=float(np.max(np.abs(got.astype(np.float64)-cpu)))
                previous=self.metrics["max_error_vs_cpu"].get(mode,0.)
                self.metrics["max_error_vs_cpu"][mode]=max(previous,error)
        return code,got

    attend=NativeAttentionTests.attend
    test_rope=NativeAttentionTests.test_native_rope_matches_official_positions
    test_real_rope=NativeAttentionTests.test_native_real_rotary_at_long_absolute_positions
    test_scores_and_device_selection=NativeAttentionTests.test_native_scores_and_selection_match_independent_reference
    test_selection_ties_and_nonfinite=NativeAttentionTests.test_native_exact_ties_and_nonfinite_scores
    test_attention_sdpa=NativeAttentionTests.test_native_tiled_softmax_and_gate_match_sdpa
    test_online_rescale_and_poison=NativeAttentionTests.test_native_tiles_rescale_and_skip_poisoned_unselected_keys
    test_pool=NativeAttentionTests.test_native_pool_is_mean_of_raw_keys_before_normalization
    test_real_index_margin=NativeAttentionTests.test_native_real_index_dimensions_and_selection_margin

    def test_device_selection_tail_sparse_boundary_and_32k(self):
        rng=np.random.default_rng(3813)
        for visible in (0,1,3,4,7,2047,2048,2049,2050,2051,2052,2053,32768):
            n=visible//4
            for pattern in ("random","ties","near"):
                scores=rng.uniform(-2,2,n).astype(np.float32)
                if pattern=="ties": scores.fill(0)
                if pattern=="near":
                    scores=1+np.arange(n,dtype=np.float32)*np.spacing(np.float32(1))
                    rng.shuffle(scores)
                code,ids=self.call("select_tokens",(visible,4,2048,0),scores,dtype=np.uint32)
                self.assertEqual(code,0)
                blocks=np.sort(np.lexsort((np.arange(n),-scores))[:512])
                expected=np.r_[(blocks[:,None]*4+np.arange(4)).ravel(),np.arange(n*4,visible)]
                np.testing.assert_array_equal(ids,expected)
                self.assertTrue(np.all(np.diff(ids.astype(np.int64))>0))
                self.assertTrue(np.all(ids<visible))
                gap=None if n<=512 else float(np.sort(scores)[-512]-np.sort(scores)[-513])
                self.metrics["selection"].append(dict(visible=visible,pattern=pattern,gap=gap,
                    identical_fp32_scores=True,ids_bit_exact=True))
        # Poison at a nontrivial selection boundary must not overrun IDs.
        for value in (np.nan,np.inf,-np.inf):
            scores=np.arange(513,dtype=np.float32); scores[257]=value
            code,_=self.call("select_tokens",(2053,4,2048,0),scores,dtype=np.uint32)
            self.assertEqual(code,16)

    def test_selection_to_attention_one_command_at_32k(self):
        rng=np.random.default_rng(3820)
        for n in (2051,2052,2053,32768):
            h,j,d,r,budget=4,2,8,4,2048
            q=rng.uniform(-1,1,(h,d)).astype(np.float32)
            gate=rng.uniform(0,1,(h,d)).astype(np.float32)
            k=rng.uniform(-1,1,(n,j,d)).astype(np.float32)
            v=rng.uniform(-1,1,(n,j,d)).astype(np.float32)
            scores=rng.uniform(-1,1,n//r).astype(np.float32)
            blocks=np.sort(np.lexsort((np.arange(n//r),-scores))[:budget//r])
            ids=np.r_[(blocks[:,None]*r+np.arange(r)).ravel(),np.arange(n//r*r,n)].astype(np.uint32)
            logits=np.einsum("hd,mhd->hm",q.astype(np.float64),k[ids].astype(np.float64).repeat(h//j,axis=1))/np.sqrt(d)
            probs=np.exp(logits-logits.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
            ref=np.einsum("hm,mhd->hd",probs,v[ids].astype(np.float64).repeat(h//j,axis=1))*gate
            unused=np.ones(n,bool); unused[ids]=False
            k[unused]=np.nan; v[unused]=np.nan
            code,y=self.call("sparse_attend",(h,j,d,n),q,gate,k,v,np.array([r,budget],np.uint32),scores)
            self.assertEqual(code,0)
            np.testing.assert_allclose(y.reshape(h,d),ref,atol=3e-5,rtol=2e-4)
            cpu_code,cpu=NativeAttentionTests.call(self,"attend",(h,j,d,n),q,gate,k,v,
                np.array([len(ids)],np.uint32),ids)
            self.assertEqual(cpu_code,0)
            np.testing.assert_allclose(y,cpu,atol=3e-5,rtol=2e-4)

    def test_c_norm_and_sigmoid_then_d_rope_one_command(self):
        for cfg in (oracle.Config(),oracle.fixture_case("real-shape")[0]):
            x=oracle.synthetic(cfg,9)
            rng=np.random.default_rng(19)
            x.qg[:,:,1]=rng.uniform(-3,3,x.qg[:,:,1].shape)
            x.gamma_q[:]=np.linspace(.5,1.5,cfg.head_dim,dtype=np.float32)
            # HF offset convention must round-trip exactly to the direct gain.
            x.gamma_q[:]=(x.gamma_q-1)+1
            ref=oracle.evaluate(self.hf,cfg,x)
            for pos in (0,3,4,8):
                code,got=self.call("prep",(cfg.heads,cfg.head_dim,cfg.rotary_dim,pos),
                    np.array([cfg.theta,cfg.eps],np.float32),x.qg[pos,:,0].copy(),x.gamma_q,x.qg[pos,:,1].copy())
                self.assertEqual(code,0)
                q,gate=got.reshape(2,cfg.heads,cfg.head_dim)
                np.testing.assert_allclose(q,ref["q"][pos],atol=2e-5,rtol=2e-5)
                expected=torch.sigmoid(torch.from_numpy(x.qg[pos,:,1])).numpy()
                np.testing.assert_allclose(gate,expected,atol=1e-5,rtol=1e-5)

    def test_q2_slice_device_stages(self):
        if SLICE_FIXTURE is None:
            self.skipTest("Pass --slice-fixture generated by the bounded A+B CPU gate")
        manifest=json.loads((SLICE_FIXTURE/"manifest.json").read_text())
        artifact=SLICE_FIXTURE/"fixture.npz"
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(),manifest["fixture_sha256"])
        self.assertEqual(manifest["source_sha256"],oracle.SOURCE_HASH)
        cfg=oracle.Config(**manifest["catalog"]["config"])
        with np.load(artifact,allow_pickle=False) as saved:
            x=oracle.Inputs(**{name:saved["input_"+name].copy() for name in oracle.Inputs.__dataclass_fields__})
        ref=oracle.evaluate(self.hf,cfg,x); n=len(x.qg)
        def prep(raw,gain,pos,gate=None):
            if gate is None: gate=np.zeros_like(raw)
            h,d=raw.shape
            code,got=self.call("prep",(h,d,cfg.rotary_dim,pos),np.array([cfg.theta,cfg.eps],np.float32),
                raw.copy(),gain,gate.copy()); self.assertEqual(code,0)
            return got.reshape(2,h,d)
        q=[]; gates=[]; keys=[]; iq=[]; blocks=[]
        for pos in range(n):
            pair=prep(x.qg[pos,:,0],x.gamma_q,pos,x.qg[pos,:,1]); q.append(pair[0]); gates.append(pair[1])
            keys.append(prep(x.k[pos],x.gamma_k,pos)[0]); iq.append(prep(x.iq[pos],x.gamma_iq,pos)[0])
        for b in range(n//cfg.ratio):
            code,pooled=self.call("pool",(cfg.ratio,cfg.index_dim,0,0),x.ik[b*cfg.ratio:(b+1)*cfg.ratio])
            self.assertEqual(code,0)
            np.testing.assert_allclose(pooled,ref["pooled_last"][b],atol=2e-5,rtol=2e-5)
            blocks.append(prep(pooled[None],x.gamma_ik,b*cfg.ratio)[0][0])
        for values,name in ((q,"q"),(keys,"k"),(iq,"iq"),(blocks,"block_keys_last")):
            np.testing.assert_allclose(values,ref[name],atol=2e-5,rtol=2e-5)
        keys=np.array(keys); blocks=np.array(blocks); output=[]
        for pos in range(n):
            complete=(pos+1)//cfg.ratio
            code,scores=self.call("scores",(cfg.index_heads,cfg.index_dim,complete,0),iq[pos],blocks[:complete])
            self.assertEqual(code,0)
            np.testing.assert_allclose(scores,ref["indexer_scores_fp64_unscaled"][pos,:complete],atol=2e-4,rtol=2e-5)
            code,ids=self.call("select",(complete,cfg.budget//cfg.ratio,0,0),scores,dtype=np.uint32)
            self.assertEqual(code,0)
            tokens=np.array([int(b)*cfg.ratio+i for b in ids for i in range(cfg.ratio)]+
                            list(range(complete*cfg.ratio,pos+1)),np.uint32)
            np.testing.assert_array_equal(tokens,np.flatnonzero(ref["mask"][pos]))
            code,y=self.call("attend",(cfg.heads,cfg.kv_heads,cfg.head_dim,pos+1),q[pos],gates[pos],
                keys[:pos+1],x.v[:pos+1],np.array([len(tokens)],np.uint32),tokens)
            self.assertEqual(code,0); output.append(y.reshape(cfg.heads,cfg.head_dim))
        np.testing.assert_allclose(output,ref["gated_fp64"],atol=3e-5,rtol=2e-4)
        self.metrics["slice_error"]=float(np.max(np.abs(output-ref["gated_fp64"])))
        self.metrics["slice_fixture_sha256"]=manifest["fixture_sha256"]

    def test_kernel_bounds_and_nonfinite_error_bits(self):
        cfg=oracle.Config(); ref=oracle.evaluate(self.hf,cfg,oracle.synthetic(cfg,3))
        for ids in ([1,1],[2,1],[3]):
            code,_=self.attend(cfg,ref,2,np.array(ids,np.uint32)); self.assertEqual(code,4)
        for mode,dims,arrays in (
            ("rope",(1,8,4,1),[np.array([10000],np.float32),np.full(8,np.nan,np.float32)]),
            ("scores",(1,8,1,0),[np.full(8,np.nan,np.float32),np.ones(8,np.float32)])):
            code,_=self.call(mode,dims,*arrays); self.assertEqual(code,16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--metal", type=Path)
    parser.add_argument("--slice-fixture", type=Path)
    parser.add_argument("--metal-library", type=Path, default=METAL_LIBRARY)
    args, remaining = parser.parse_known_args()
    SLICE_FIXTURE = args.slice_fixture
    METAL = args.metal.resolve() if args.metal else None
    METAL_LIBRARY = args.metal_library.resolve()
    SOURCE = args.source
    NATIVE = args.native.resolve() if args.native else None
    unittest.main(argv=[sys.argv[0], *remaining])
