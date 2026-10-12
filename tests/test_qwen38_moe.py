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


if __name__ == "__main__":
    unittest.main(verbosity=2)
