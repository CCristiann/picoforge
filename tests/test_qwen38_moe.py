"""Small CPU-only tests. Run with the existing tools/venv Python.

The hash-pinned source cache must be prepared explicitly; tests never fetch.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import struct
import subprocess
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
            for corrupt in moe.HASHES:
                for name in moe.HASHES:
                    data = (moe.CACHE / name).read_bytes()
                    (Path(tmp) / name).write_bytes(data + (b"\n" if name == corrupt else b""))
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
        xt = torch.tensor(x, dtype=torch.float64)
        linear = lambda name, inp: torch.nn.functional.linear(inp, torch.tensor(w[name], dtype=torch.float64))
        gs = linear("shared_expert.gate_proj.weight", xt)
        us = linear("shared_expert.up_proj.weight", xt)
        ms = torch.nn.functional.silu(gs) * us
        ys = linear("shared_expert.down_proj.weight", ms)
        for name, expected in (("shared_gate", gs), ("shared_up", us),
                               ("shared_mid", ms), ("shared", ys)):
            np.testing.assert_allclose(trace[name], expected.numpy(), atol=2e-6, rtol=2e-5)

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


class EdgeCaseTests(unittest.TestCase):
    def test_analytical_tied_experts_have_distinct_outputs(self):
        c = moe.FixtureConfig(hidden_size=1, moe_intermediate_size=1,
                              shared_expert_intermediate_size=1)
        x, weights = moe.synthetic(c, tokens=1)
        x[:] = 1
        weights["gate.weight"][:] = 0
        weights["experts.gate_up_proj"][:] = 1
        weights["experts.down_proj"][:] = np.arange(1, 5).reshape(4, 1, 1)
        weights["shared_expert.down_proj.weight"][:] = 0
        actual = moe.numpy_forward(c, x, weights)
        np.testing.assert_array_equal(actual["ids"], [[0, 1]])
        np.testing.assert_allclose(actual["y"], 1.5 / (1 + np.exp(-1.0)), atol=2e-6, rtol=2e-5)
        # HF result is correct for ITS selected IDs; identical output isn't promised at ties.
        hf = moe.upstream_forward(c, x, weights)
        expected = (hf["ids"].mean() + 1) / (1 + np.exp(-1.0))
        np.testing.assert_allclose(hf["y"], expected, atol=2e-6, rtol=2e-5)

    def test_probability_ties_choose_lower_id(self):
        cases = (([0, 0, 0, 0], 2, [0, 1]),
                 ([1, 2, 2, 0], 1, [1]),
                 ([0, 1e-9, 0, -1e-9], 2, [0, 1]),
                 ([-1000, -1001, 1000], 2, [2, 0]))
        for values, k, expected in cases:
            with self.subTest(values=values):
                logits = np.array([values], dtype=np.float32)
                p, ids, scores = moe.route_logits(logits, k)
                np.testing.assert_array_equal(ids, [expected])
                c = moe.FixtureConfig(hidden_size=1, num_experts=len(values),
                                      num_experts_per_tok=k, moe_intermediate_size=1,
                                      shared_expert_intermediate_size=1)
                x, w = moe.synthetic(c, tokens=1)
                x[:] = 1
                w["gate.weight"][:] = logits.T
                upstream = moe.upstream_forward(c, x, w)
                # HF may choose different tied IDs; verify its selected probabilities.
                np.testing.assert_allclose(scores, upstream["weights"], atol=2e-6, rtol=2e-5)
                chosen = np.take_along_axis(p, upstream["ids"], -1)
                np.testing.assert_array_equal(chosen, np.sort(p, axis=-1)[:, ::-1][:, :k])

    def test_512_uniform_shift_and_extreme_logits(self):
        logits = np.zeros((2, 512), dtype=np.float32)
        p, ids, weights = moe.route_logits(logits, 10)
        np.testing.assert_array_equal(ids, np.tile(np.arange(10), (2, 1)))
        np.testing.assert_allclose(weights, 0.1, atol=2e-6, rtol=2e-5)
        logits = np.array([[0, 1, 2, 3]], dtype=np.float32)
        for norm in (False, True):
            baseline = moe.route_logits(logits, 3, norm)
            shifted = moe.route_logits(logits + 128, 3, norm)
            for a, b in zip(baseline, shifted):
                np.testing.assert_array_equal(a, b)
        with np.errstate(all="raise"):
            p, ids, weights = moe.route_logits(np.array([[-3e38, 3e38]], dtype=np.float32), 2)
        np.testing.assert_array_equal(p, [[0, 1]])
        np.testing.assert_array_equal(ids, [[1, 0]])
        for k in (0, 3, True):
            with self.assertRaises(ValueError):
                moe.route_logits(p, k)

    def test_chunks_empty_strides_and_input_immutability(self):
        c = moe.FixtureConfig()
        for count in (0, 1, 2, 31, 32, 33, 64, 65, 1024):
            with self.subTest(tokens=count):
                x, weights = moe.synthetic(c, tokens=count)
                storage = np.zeros((count, 2*c.hidden_size), dtype=np.float32)
                storage[:, ::2] = x
                strided = storage[:, ::2]
                before = {name: w.copy() for name, w in weights.items()}
                full = moe.numpy_forward(c, strided, weights)
                hf = moe.upstream_forward(c, strided, weights)
                np.testing.assert_array_equal(full["ids"], hf["ids"])
                np.testing.assert_allclose(full["y"], hf["y"], atol=2e-5, rtol=2e-4)
                if count:
                    parts, offset = [], 0
                    while offset < count:
                        size = (1, 7, 32)[len(parts) % 3]
                        parts.append(moe.numpy_forward(c, x[offset:offset+size], weights))
                        offset += size
                    np.testing.assert_array_equal(np.concatenate([p["ids"] for p in parts]), full["ids"])
                    np.testing.assert_allclose(np.concatenate([p["y"] for p in parts]), full["y"],
                                               atol=2e-5, rtol=2e-4)
                self.assertEqual(full["y"].shape, (count, c.hidden_size))
                np.testing.assert_array_equal(strided, x)
                for name in weights:
                    np.testing.assert_array_equal(weights[name], before[name])

    def test_repeated_tokens_and_independent_shared_contribution(self):
        c = moe.FixtureConfig()
        x, weights = moe.synthetic(c, tokens=1)
        x = np.repeat(x, 65, axis=0)
        first = moe.numpy_forward(c, x, weights)
        np.testing.assert_array_equal(first["ids"], np.repeat(first["ids"][:1], 65, axis=0))
        # No token capacity/drop: every repeated token receives the complete result.
        np.testing.assert_allclose(first["y"], np.repeat(first["y"][:1], 65, axis=0),
                                   atol=2e-5, rtol=2e-4)
        for target in ("experts.down_proj", "shared_expert.down_proj.weight"):
            changed = {name: value.copy() for name, value in weights.items()}
            changed[target][:] = 0
            actual = moe.numpy_forward(c, x, changed)
            hf = moe.upstream_forward(c, x, changed)
            np.testing.assert_allclose(actual["y"], hf["y"], atol=2e-5, rtol=2e-4)
            expected = (first["shared"] * moe.sigmoid(first["shared_logit"])
                        if target.startswith("experts.") else first["routed"])
            np.testing.assert_allclose(actual["y"], expected, atol=2e-5, rtol=2e-4)
        # An unrelated call cannot leave stale route/state in the next result.
        moe.numpy_forward(c, -x, weights)
        np.testing.assert_array_equal(moe.numpy_forward(c, x, weights)["y"], first["y"])


class FixtureExportTests(unittest.TestCase):
    def test_export_replay_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "fixtures"
            report = moe.export_fixtures(folder)
            self.assertTrue(report["complete"])
            self.assertEqual(report["upstream_hashes"], moe.HASHES)
            self.assertEqual(report, json.loads((folder / "manifest.json").read_text()))
            self.assertLess(sum(f["payload_bytes"] for f in report["fixtures"]), 256*1024)
            for fixture in report["fixtures"]:
                path = folder / fixture["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), fixture["sha256"])
                with np.load(path, allow_pickle=False) as arrays:
                    w = {name.removeprefix("weight/"): arrays[name]
                         for name in arrays.files if name.startswith("weight/")}
                    c = moe.FixtureConfig(**fixture["config"])
                    result = moe.upstream_forward(c, arrays["input"], w)
                    for name, array in result.items():
                        np.testing.assert_array_equal(array, arrays["upstream/"+name])
                    for name in arrays.files:
                        self.assertEqual(list(arrays[name].shape), fixture["arrays"][name]["shape"])
            before = (folder / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                moe.export_fixtures(folder)
            self.assertEqual(before, (folder / "manifest.json").read_bytes())

    def test_failed_export_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "failed"
            with self.assertRaises(FileNotFoundError):
                moe.export_fixtures(folder, cache=Path(tmp) / "missing")
            self.assertFalse(json.loads((folder / "manifest.json").read_text())["complete"])


@unittest.skipUnless(os.environ.get("Q38_F_NATIVE"), "set Q38_F_NATIVE to the isolated ASan/UBSan harness")
class NativeRoutingTests(unittest.TestCase):
    def route(self, logits, k, normalize=True):
        t, e = logits.shape
        payload = struct.pack("<4I", t, e, k, int(normalize)) + logits.astype("<f4").tobytes()
        proc = subprocess.run([os.environ["Q38_F_NATIVE"], "--route"], input=payload,
                              capture_output=True, check=True)
        status, = struct.unpack_from("<I", proc.stdout)
        if status:
            self.assertEqual(len(proc.stdout), 4)
            return status, None
        self.assertEqual(len(proc.stdout), 4 + 4*t*(e+2*k))
        probabilities = np.frombuffer(proc.stdout, "<f4", t*e, 4).reshape(t, e)
        ids = np.frombuffer(proc.stdout, "<u4", t*k, 4+4*t*e).reshape(t, k)
        scores = np.frombuffer(proc.stdout, "<f4", t*k, 4+4*t*(e+k)).reshape(t, k)
        return status, (probabilities, ids, scores)

    def test_native_shape_stride_errors(self):
        proc = subprocess.run([os.environ["Q38_F_NATIVE"]], capture_output=True, text=True, check=True)
        self.assertIn("routing:", proc.stdout)

    def test_native_differential_and_ties(self):
        rng = np.random.default_rng(3811)
        for t, e, k in ((0,4,2), (1,1,1), (7,4,1), (7,4,4), (33,512,10),
                         (3,513,11), (2,1025,3), (1024,17,5)):
            logits = rng.normal(0, 3, (t,e)).astype(np.float32)
            for norm in (False, True):
                with self.subTest(t=t, e=e, k=k, norm=norm):
                    status, actual = self.route(logits, k, norm)
                    self.assertEqual(status, 0)
                    expected = moe.route_logits(logits, k, norm)
                    np.testing.assert_array_equal(actual[1], expected[1])
                    for j in (0,2):
                        np.testing.assert_allclose(actual[j], expected[j], atol=2e-6, rtol=2e-5)
                    hp = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                    np.testing.assert_allclose(actual[0], hp, atol=2e-6, rtol=2e-5)
                    if norm:
                        np.testing.assert_allclose(actual[2].sum(-1), 1, atol=2e-6, rtol=0)
        for logits, k in ((np.zeros((3,512), np.float32),10),
                           (np.array([[0,1e-9,0,-1e-9]],np.float32),2),
                           (np.array([[-3e38,3e38,0,-1]],np.float32),4)):
            status, actual = self.route(logits, k)
            self.assertEqual(status,0)
            expected = moe.route_logits(logits, k)
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_allclose(actual[2], expected[2], atol=2e-6, rtol=2e-5)

    def test_native_upstream_routes_from_real_projection_formula(self):
        for seed in range(4):
            c = replace(moe.FixtureConfig(), num_experts=512, num_experts_per_tok=10,
                        moe_intermediate_size=2)
            x, weights = moe.synthetic(c, tokens=7, seed=seed)
            expected = moe.upstream_forward(c, x, weights)
            status, actual = self.route(expected["logits"], 10)
            self.assertEqual(status, 0)
            np.testing.assert_array_equal(actual[1], expected["ids"])
            np.testing.assert_allclose(actual[2], expected["weights"], atol=2e-6, rtol=2e-5)

    def test_native_chunk_invariance_and_nonfinite(self):
        logits = np.random.default_rng(48).normal(size=(65,512)).astype(np.float32)
        _, full = self.route(logits,10)
        parts = [self.route(logits[a:b],10)[1] for a,b in ((0,1),(1,8),(8,40),(40,65))]
        for i in range(3):
            np.testing.assert_array_equal(np.concatenate([p[i] for p in parts]), full[i])
        for bad in (np.nan,np.inf,-np.inf):
            broken=logits.copy(); broken[-1,-1]=bad
            status, _ = self.route(broken,10)
            self.assertNotEqual(status,0)


@unittest.skipUnless(os.environ.get("Q38_F_NATIVE"), "set Q38_F_NATIVE to the isolated ASan/UBSan harness")
class NativeGroupingTests(unittest.TestCase):
    route = NativeRoutingTests.route

    def group(self, ids, experts):
        t, k = ids.shape
        payload = struct.pack("<3I", t, experts, k) + ids.astype("<u4").tobytes()
        proc = subprocess.run([os.environ["Q38_F_NATIVE"], "--group"], input=payload,
                              capture_output=True, check=True)
        status, = struct.unpack_from("<I", proc.stdout)
        if status:
            self.assertEqual(len(proc.stdout), 4)
            return status, None
        self.assertEqual(len(proc.stdout), 4 + 8*(experts+1+2*t*k))
        values = np.frombuffer(proc.stdout, "<u8", offset=4).astype(np.int64)
        return status, (values[:experts+1], values[experts+1:experts+1+t*k],
                        values[experts+1+t*k:])

    def test_grouping_against_independent_stable_sort(self):
        rng = np.random.default_rng(3812)
        for t, e, k in ((0,4,2), (1,1,1), (1,512,10), (7,4,4), (33,512,10),
                         (65,513,11), (1024,17,5)):
            for seed in range(4):
                ids = np.empty((t,k), np.uint32)
                for row in ids:
                    row[:] = rng.choice(e, k, replace=False)
                if seed == 0 and t:
                    ids[:] = ids[0]  # concentrated load; empty expert groups
                with self.subTest(t=t, e=e, k=k, seed=seed):
                    status, (offsets, p2r, r2p) = self.group(ids,e)
                    self.assertEqual(status,0)
                    expected = np.argsort(ids.ravel(), kind="stable")
                    counts = np.bincount(ids.ravel(), minlength=e)
                    np.testing.assert_array_equal(r2p, expected)
                    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(counts)])
                    np.testing.assert_array_equal(r2p[p2r], np.arange(t*k))
                    np.testing.assert_array_equal(p2r[r2p], np.arange(t*k))

    def test_route_group_scatter_preserves_weights_and_pair_identity(self):
        c = replace(moe.FixtureConfig(), num_experts=17, num_experts_per_tok=5)
        for seed in range(4):
            x, weights = moe.synthetic(c,tokens=33,seed=seed)
            expected = moe.numpy_forward(c,x,weights)
            _, (_, ids, scores) = self.route(expected["logits"],5)
            np.testing.assert_array_equal(ids,expected["ids"])
            _, (_, p2r, r2p) = self.group(ids,17)
            # Existing floating oracle produces expert outputs; grouping only
            # reorders and restores these values. No second runtime matmul.
            pairs = expected["down"].reshape(33*5,c.hidden_size)
            grouped = pairs[r2p]
            restored = grouped[p2r].reshape(33,5,c.hidden_size)
            np.testing.assert_array_equal(restored,expected["down"])
            routed = (restored*scores[...,None]).sum(axis=1)
            np.testing.assert_allclose(routed,expected["routed"],atol=2e-5,rtol=2e-4)
            # Splitting token batches may change row IDs but not restored pairs.
            for start, end in ((0,1),(1,8),(8,33)):
                _, (_, sub_p2r, sub_r2p) = self.group(ids[start:end],17)
                block = pairs[start*5:end*5]
                np.testing.assert_array_equal(block[sub_r2p][sub_p2r],block)

    def test_invalid_ids_fail_before_publishing_groups(self):
        for ids in (np.array([[0,1],[2,2]],np.uint32),
                    np.array([[0,1],[2,4]],np.uint32),
                    np.array([[0,1],[2,2**32-1]],np.uint32)):
            status, value = self.group(ids,4)
            self.assertNotEqual(status,0)
            self.assertIsNone(value)


def forward_weights(c, weights):
    gu = weights["experts.gate_up_proj"]
    return [weights["gate.weight"], gu[:,:c.moe_intermediate_size], gu[:,c.moe_intermediate_size:],
            weights["experts.down_proj"], weights["shared_expert.gate_proj.weight"],
            weights["shared_expert.up_proj.weight"], weights["shared_expert.down_proj.weight"],
            weights["shared_expert_gate.weight"]]


def native_forward(c, x, matrices, payloads=None, trace=True, sparse=False):
    physical = matrices[3].shape[-1]
    payload = struct.pack("<9I",c.hidden_size,c.num_experts,c.num_experts_per_tok,
                          c.moe_intermediate_size,c.shared_expert_intermediate_size,physical,
                          int(c.norm_topk_prob),len(x),int(trace)+2*int(sparse))
    for j, matrix in enumerate(matrices):
        experts = matrix.shape[0] if matrix.ndim == 3 else 1
        kind, data = (0,matrix.astype("<f4").tobytes()) if payloads is None else payloads[j]
        payload += struct.pack("<4I",kind,matrix.shape[-1],matrix.shape[-2],experts) + data
    payload += x.astype("<f4").tobytes()
    proc = subprocess.run([os.environ["Q38_F_NATIVE"],"--forward"],input=payload,
                          capture_output=True,check=True,timeout=60)
    if proc.stderr:
        raise AssertionError(proc.stderr.decode())
    blob, offset, records = proc.stdout,0,[]
    while struct.unpack_from("<Q",blob,offset)[0] != 2**64-1:
        stage, expert, token, rows, cols, outputs = struct.unpack_from("<6Q",blob,offset)
        offset += 48
        a = np.frombuffer(blob,"<f4",rows*cols,offset).reshape(rows,cols); offset += rows*cols*4
        b = np.frombuffer(blob,"<f4",rows*outputs,offset).reshape(rows,outputs); offset += rows*outputs*4
        records.append((stage,expert,token,a,b))
    _, status, budget = struct.unpack_from("<3Q",blob,offset); offset += 24
    if status:
        assert offset == len(blob)
        return status,{},records,budget
    t,h,k = len(x),c.hidden_size,c.num_experts_per_tok
    result = {}
    for key, dtype, shape in (("y","<f4",(t,h)),("ids","<u4",(t,k)),("weights","<f4",(t,k))):
        count = np.prod(shape).item()
        result[key] = np.frombuffer(blob,dtype,count,offset).reshape(shape); offset += count*4
    assert offset == len(blob)
    return status,result,records,budget


@unittest.skipUnless(os.environ.get("Q38_F_NATIVE"), "requires F native harness")
class NativeForwardTests(unittest.TestCase):
    def verify(self,c,x,weights,matrices=None,payloads=None):
        matrices = forward_weights(c,weights) if matrices is None else matrices
        status,actual,records,budget = native_forward(c,x,matrices,payloads)
        self.assertEqual(status,0); self.assertLessEqual(budget,32*1024*1024)
        expected = moe.numpy_forward(c,x,weights)
        upstream = moe.upstream_forward(c,x,weights)
        np.testing.assert_array_equal(actual["ids"],expected["ids"])
        for reference in (expected,upstream):
            np.testing.assert_allclose(actual["weights"],reference["weights"],atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(actual["y"],reference["y"],atol=2e-5,rtol=2e-4)
            self.assertLessEqual(moe.error_metrics(actual["y"],reference["y"])["nrmse"],1e-4)
        previous_activation = None
        for stage,expert,token,a,b in records:
            if stage < 8:
                matrix = matrices[stage][expert] if matrices[stage].ndim==3 else matrices[stage]
                matrix = matrix[:,:a.shape[1]].astype(np.float64)
                reference = a.astype(np.float64) @ matrix.T
                magnitude = np.abs(a.astype(np.float64)) @ np.abs(matrix.T)
                nu = (2*a.shape[1]+8)*2**-24
                bound = nu/(1-nu)*magnitude + 2*np.abs(np.spacing(reference.astype(np.float32)))
                self.assertTrue(np.all(np.abs(b-reference)<=bound),f"B bound stage {stage}")
            elif stage in (8,10):
                at = torch.from_numpy(a.copy()).to(torch.float64)
                reference = (torch.nn.functional.silu(at) if stage==8 else torch.sigmoid(at)).numpy()
                np.testing.assert_allclose(b,reference,atol=2e-6,rtol=2e-5)
                if stage==8: previous_activation=b
            elif stage==9:
                np.testing.assert_array_equal(b,previous_activation*a)
            elif stage in (11,12):
                key = "routed" if stage==11 else "y"
                np.testing.assert_allclose(b,expected[key][token:token+len(b)],atol=2e-5,rtol=2e-4)
        return actual,records

    def test_float_composed_stages_chunks_and_shared(self):
        for t in (0,1,2,31,32,33,65,1024):
            for k in (1,2,4):
                c = replace(moe.FixtureConfig(),num_experts_per_tok=k)
                x, weights = moe.synthetic(c,tokens=t,seed=3813+t+k)
                with self.subTest(t=t,k=k): self.verify(c,x,weights)
        c=moe.FixtureConfig(); x,weights=moe.synthetic(c,tokens=65,seed=44)
        full,_ = self.verify(c,x,weights)
        parts = [self.verify(c,x[a:b],weights)[0] for a,b in ((0,1),(1,8),(8,40),(40,65))]
        for key in full: np.testing.assert_array_equal(full[key],np.concatenate([p[key] for p in parts]))
        for name in ("experts.down_proj","shared_expert.down_proj.weight","shared_expert_gate.weight"):
            changed={n:v.copy() for n,v in weights.items()}; changed[name][:]=0
            self.verify(c,x,changed)

    def test_float_nonfinite_and_zero_input(self):
        c=moe.FixtureConfig(); x,weights=moe.synthetic(c,tokens=2,seed=44)
        self.verify(c,x*0,weights)
        for value in (np.nan,np.inf):
            broken=x.copy(); broken[-1,-1]=value
            self.assertNotEqual(native_forward(c,broken,forward_weights(c,weights))[0],0)
        for name in ("gate.weight","shared_expert.down_proj.weight"):
            changed={n:v.copy() for n,v in weights.items()}; changed[name][:]=np.nan
            self.assertNotEqual(native_forward(c,x,forward_weights(c,changed))[0],0)

    def test_packed_composition_and_640_768_padding(self):
        import importlib.util
        root=Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("b_test",root/"tests/test_qwen38_packed.py")
        b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
        reference = Path(os.environ.get("Q38_F_PACKED_REFERENCE",root/"build/qwen38-f/packed-reference/reference"))
        if not reference.exists(): self.skipTest("requires prepared B independent reference")
        c=moe.FixtureConfig(hidden_size=256,num_experts=4,num_experts_per_tok=2,
                            moe_intermediate_size=640,shared_expert_intermediate_size=32)
        rng=np.random.default_rng(3814)
        matrices,payloads=[],[]
        dims=((4,256),(4,640,256),(4,640,256),(4,256,768),
              (32,256),(32,256),(256,32),(1,256))
        for stage,(kind,shape) in enumerate(zip((0,16,16,10,8,8,8,0),dims)):
            raw=b.packed_fixture(kind,shape[-1],int(np.prod(shape[:-1])),rng)
            decoded=b.oracle.decode(reference,raw,kind,shape[-1],int(np.prod(shape[:-1]))).reshape(shape)
            matrices.append(decoded); payloads.append((kind,raw))
        self.assertTrue(np.any(matrices[3][...,640:]!=0))
        weights={"gate.weight":matrices[0],"experts.gate_up_proj":np.concatenate(matrices[1:3],axis=1),
                 "experts.down_proj":matrices[3][...,:640],"shared_expert.gate_proj.weight":matrices[4],
                 "shared_expert.up_proj.weight":matrices[5],"shared_expert.down_proj.weight":matrices[6],
                 "shared_expert_gate.weight":matrices[7]}
        for t in (1,2,33):
            x=(rng.normal(size=(t,256))*0.001).astype(np.float32)
            with self.subTest(t=t): self.verify(c,x,weights,matrices,payloads)
        # Physical trailing coefficients are arbitrary, but inputs remain 640.
        # Replace the whole down tensor by F32 with huge finite trailing poison;
        # runtime B must never read beyond the ACT view's logical columns.
        x=(rng.normal(size=(2,256))*0.001).astype(np.float32)
        poisoned=matrices[3].copy(); poisoned[...,640:]=1e30
        mixed=list(payloads); mixed[3]=(0,poisoned.astype("<f4").tobytes())
        alt=list(matrices); alt[3]=poisoned
        actual,_=self.verify(c,x,weights,matrices,payloads)
        other,_=self.verify(c,x,weights,alt,mixed)
        for key in actual: np.testing.assert_array_equal(actual[key],other[key])

    def test_real_bounded_layers(self):
        from types import SimpleNamespace
        import qwen38_packed as b
        root=Path(__file__).resolve().parents[1]/"build/qwen38-f"
        if not (root/"slice-0/manifest.json").exists(): self.skipTest("prepare bounded real slices explicitly")
        if not os.environ.get("PICOFORGE_GPU_LOCK_FD"): self.skipTest("real slice replay requires common lock")
        reference=root/"packed-reference/reference"
        reports=[]
        for layer in (0,3,47):
            folder=root/f"slice-{layer}"; manifest=json.loads((folder/"manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            def read(record):
                data=(folder/record["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),record["sha256"])
                return data
            h,e,k,i,s,physical,norm=manifest["config"]
            c=moe.FixtureConfig(h,e,k,i,s,bool(norm))
            x=np.frombuffer(read(manifest["input"]),"<f4").reshape(1,h)
            payloads,shapes=[],[]
            for stage,tensor in enumerate(manifest["tensors"]):
                segments=tensor["segments"]
                raw=struct.pack("<I",len(segments))
                for segment in segments:
                    data=read(segment)
                    raw += struct.pack("<2Q",segment["offset_in_tensor"],len(data))+data
                payloads.append((tensor["type"],raw))
                shape=(tensor["experts"],tensor["rows"],tensor["physical"]) if 1<=stage<=3 else (tensor["rows"],tensor["physical"])
                shapes.append(SimpleNamespace(shape=shape,ndim=len(shape)))
            status,actual,records,budget=native_forward(c,x,shapes,payloads,sparse=True)
            self.assertEqual(status,0); self.assertLessEqual(budget,32*1024*1024)
            np.testing.assert_array_equal(actual["ids"],manifest["selected"])
            def decoded(stage,expert=0):
                t=manifest["tensors"][stage]
                segment=next(r for r in t["segments"] if r["expert"]==expert)
                return b.decode(reference,read(segment),t["type"],t["physical"],t["rows"])
            shared={name:decoded(stage) for stage,name in enumerate(("gate.weight",None,None,None,
                "shared_expert.gate_proj.weight","shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight","shared_expert_gate.weight")) if name is not None}
            router=shared["gate.weight"]
            logits=(torch.from_numpy(x.copy()) @ torch.from_numpy(router.copy()).T).numpy()
            probs,ids,scores=moe.route_logits(logits,k,bool(norm))
            np.testing.assert_array_equal(actual["ids"],ids)
            np.testing.assert_allclose(actual["weights"],scores,atol=2e-6,rtol=2e-5)
            native_logits=next(r[4] for r in records if r[0]==0)
            native_probs=moe.route_logits(native_logits,k,bool(norm))[0]
            p_error=float(np.abs(native_probs-probs).max())
            self.assertGreater(manifest["probability_margin_k"],2*p_error)
            sorted_probs=np.sort(probs[0])[::-1]
            internal_margin=float(np.min(sorted_probs[:k]-sorted_probs[1:k+1]))
            self.assertGreater(internal_margin,2*p_error)
            xd=torch.from_numpy(x.copy()).double()
            def linear64(a,matrix): return torch.nn.functional.linear(a,torch.from_numpy(matrix.copy()).double())
            shared64=linear64(torch.nn.functional.silu(linear64(xd,shared["shared_expert.gate_proj.weight"])) *
                              linear64(xd,shared["shared_expert.up_proj.weight"]),shared["shared_expert.down_proj.weight"])
            shared64=(shared64*torch.sigmoid(linear64(xd,shared["shared_expert_gate.weight"]))).numpy()
            routed=np.zeros((1,h),np.float64); max_bound_ratio=0
            for expert in ids[0]:
                gate,up,down=decoded(1,int(expert)),decoded(2,int(expert)),decoded(3,int(expert))
                one=replace(c,num_experts=1,num_experts_per_tok=1)
                single=dict(shared,**{"gate.weight":np.zeros((1,h),np.float32),
                    "experts.gate_up_proj":np.concatenate((gate,up))[None],
                    "experts.down_proj":down[None,:,:i]})
                out=moe.upstream_forward(one,x,single)
                expert64=linear64(torch.nn.functional.silu(linear64(xd,gate))*linear64(xd,up),down[:,:i]).numpy()
                np.testing.assert_allclose(out["routed"],expert64,atol=2e-5,rtol=2e-4)
                slot=int(np.where(ids[0]==expert)[0][0]); routed+=float(scores[0,slot])*expert64
                for stage,ex,token,a,v in records:
                    if not 1<=stage<=3 or ex!=expert: continue
                    matrix=(gate,up,down)[stage-1][:,:a.shape[1]].astype(np.float64)
                    ref=a.astype(np.float64) @ matrix.T
                    mag=np.abs(a.astype(np.float64)) @ np.abs(matrix.T)
                    nu=(2*a.shape[1]+8)*2**-24
                    bound=nu/(1-nu)*mag+2*np.abs(np.spacing(ref.astype(np.float32)))
                    self.assertTrue(np.all(np.abs(v-ref)<=bound))
                    max_bound_ratio=max(max_bound_ratio,float(np.max(np.abs(v-ref)/np.maximum(bound,1e-30))))
            np.testing.assert_allclose(out["shared"]*moe.sigmoid(out["shared_logit"]),shared64,atol=2e-5,rtol=2e-4)
            expected=routed+shared64
            for stage,ex,token,a,v in records:
                if stage in (8,10):
                    at=torch.from_numpy(a.copy()).double()
                    ref=(torch.nn.functional.silu(at) if stage==8 else torch.sigmoid(at)).numpy()
                    np.testing.assert_allclose(v,ref,atol=2e-6,rtol=2e-5)
                if stage in (0,4,5,6,7):
                    mat=decoded(stage).astype(np.float64)
                    ref=a.astype(np.float64) @ mat.T; mag=np.abs(a.astype(np.float64)) @ np.abs(mat.T)
                    nu=(2*a.shape[1]+8)*2**-24
                    self.assertTrue(np.all(np.abs(v-ref)<=nu/(1-nu)*mag+2*np.abs(np.spacing(ref.astype(np.float32)))))
                if stage in (11,12):
                    ref=routed if stage==11 else expected
                    metrics=moe.error_metrics(v,ref)
                    self.assertLessEqual(metrics["nrmse"],2e-4)
                    self.assertLessEqual(metrics["max_abs"],5e-4*max(1,float(np.sqrt(np.mean(ref**2)))))
            reports.append(dict(layer=layer,bytes_read=manifest["bytes_read"],workspace_bytes=budget,
                y=moe.error_metrics(actual["y"],expected),probability_error=p_error,
                probability_margin=manifest["probability_margin_k"],internal_margin=internal_margin,
                max_projection_bound_ratio=max_bound_ratio))
        (root/"slice-results.json").write_text(json.dumps(reports,indent=2)+"\n")
        print("bounded MoE slices:",json.dumps(reports))


if __name__ == "__main__":
    unittest.main(verbosity=2)
