"""Small CPU-only oracle checks; no models, Metal, MPS or network access."""

import argparse
import ctypes as ct
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
import qwen38_recurrent as oracle

SOURCE = ROOT / oracle.UPSTREAM_DEFAULT
FIXTURES = ROOT / "tests/fixtures/qwen38/c"
CPU_LIB = None


class ReferenceTests(unittest.TestCase):
    measurements = []

    @classmethod
    def setUpClass(cls):
        cls.ref = oracle.load_upstream(SOURCE)

    def test_pinned_definitions(self):
        self.assertEqual(set(self.ref.removed_decorators), set(oracle.NODES))
        self.assertTrue(self.ref.removed_decorators["torch_chunk_gated_delta_rule"])
        self.assertTrue(self.ref.removed_decorators["Qwen4ExpTextRMSNormGated"])

    def test_corrupt_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.py"
            path.write_bytes(oracle.source_bytes(SOURCE) + b"\nraise RuntimeError('executed')\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                oracle.load_upstream(path)

    def test_head_order(self):
        self.assertEqual(oracle.head_permutation(2, 6).tolist(), [0, 3, 1, 4, 2, 5])
        self.assertEqual(oracle.head_permutation(3, 3).tolist(), [0, 1, 2])
        for shape in [(0, 3), (2, 5), (2, 0)]:
            with self.assertRaises(ValueError):
                oracle.head_permutation(*shape)

    def test_conv_taps_and_raw_history(self):
        x = torch.tensor([[1.], [2.], [3.], [4.]])
        taps = torch.tensor([[1., 10., 100.]])
        history = torch.zeros(2, 1)
        out, final = oracle.conv_reference(x, taps, history, apply_silu=False)
        torch.testing.assert_close(out[:, 0], torch.tensor([100., 210., 321., 432.]))
        self.assertEqual(final[:, 0].tolist(), [3., 4.])
        self.assertEqual(history[:, 0].tolist(), [0., 0.])
        activated, raw_final = oracle.conv_reference(x, taps, history)
        torch.testing.assert_close(activated, torch.nn.functional.silu(out))
        self.assertTrue(torch.equal(raw_final, final))

    def test_gdn_scalar_analytic(self):
        for case in json.loads((FIXTURES / "analytic.json").read_text())["cases"]:
            with self.subTest(case=case["name"]):
                q, k, v = (torch.tensor(case[name], dtype=torch.float32).reshape(-1, 1, 1) for name in ("q", "k", "v"))
                decay, beta = (torch.tensor(case[name], dtype=torch.float32).reshape(-1, 1) for name in ("log_decay", "beta"))
                state = torch.tensor([[[case["state_initial"]]]], dtype=torch.float32)
                expected = (torch.tensor(case["output"]).reshape(-1, 1, 1),
                            torch.tensor([[[case["state_final"]]]]))
                results = [oracle.gdn_reference(self.ref, q, k, v, decay, beta, state,
                           algorithm=algorithm, normalize_qk=False) for algorithm in ("chunk", "recurrent")]
                results.append(oracle.affine_reference(q, k, v, decay, beta, state, False))
                for result in results:
                    for got, ref in zip(result, expected):
                        torch.testing.assert_close(got.double(), ref.double(), atol=1e-7, rtol=0)
                self.assertEqual(state.item(), case["state_initial"])

    def test_gdn_rectangular_chunk_matches_recurrent(self):
        gen = torch.Generator(device="cpu").manual_seed(3812)
        q, k = (torch.randn(65, 2, 3, generator=gen) for _ in range(2))
        v = torch.randn(65, 6, 5, generator=gen)
        decay = -torch.rand(65, 6, generator=gen)
        beta = torch.rand(65, 6, generator=gen)
        state = torch.randn(6, 5, 3, generator=gen)
        args = (self.ref, q, k, v, decay, beta, state)
        a = oracle.gdn_reference(*args)
        b = oracle.gdn_reference(*args, algorithm="recurrent")
        for got, expected in zip(a, b):
            torch.testing.assert_close(got, expected, rtol=0, atol=5e-5)

    def test_conv_dilation_append_empty_and_kernel_one(self):
        x = torch.arange(1., 14.).reshape(-1, 1)
        for kernel in (1, 2, 4):
            for dilation in (1, 3):
                with self.subTest(kernel=kernel, dilation=dilation):
                    taps = torch.arange(1., kernel + 1).reshape(1, kernel)
                    history = torch.full(((kernel - 1) * dilation, 1), -2.)
                    expected, last = oracle.conv_reference(x, taps, history, dilation)
                    pieces, state, pos = [], history, 0
                    for size in (0, 1, 3, 0, 2, 7):
                        out, state = oracle.conv_reference(x[pos:pos + size], taps, state, dilation)
                        pieces.append(out)
                        pos += size
                    self.assertTrue(torch.equal(torch.cat(pieces), expected))
                    self.assertTrue(torch.equal(state, last))
        taps = torch.tensor([[1., 10., 100., 1000.]])
        y, _ = oracle.conv_reference(x, taps, torch.zeros(9, 1), 3, False)
        self.assertEqual(y[9].item(), 10000 + 700 + 40 + 1)

    def test_hc_zero_gates_and_grouped_norm(self):
        x = torch.tensor([[[1., 2., 3.], [10., -20., 30.]]])
        gain = torch.tensor([[0.5, 1., -2.], [2., -1., 0.5]])
        down, up, inject = torch.zeros(5, 6), torch.zeros(6, 5), torch.zeros(2, 6)
        block = torch.tensor([[3., -2., 1.]])
        out = oracle.hc_reference(self.ref, x, gain, down, up, inject, block)
        norm = x * gain / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        torch.testing.assert_close(out["normed"], norm)
        torch.testing.assert_close(out["mixed"], norm.mean(1) / 2)
        self.assertTrue(torch.equal(out["injection"], torch.ones(1, 2)))
        self.assertTrue(torch.equal(out["residual"], x + block.unsqueeze(1)))
        no_inject = oracle.hc_reference(self.ref, x, gain, down, up)
        self.assertNotIn("injection", no_inject)
        torch.testing.assert_close(no_inject["mixed"], out["mixed"])

    def test_gdn_norm_sigmoid_and_direct_gain(self):
        x = torch.tensor([[[1., -2., 3.], [4., 5., -6.]]])
        z, gain = torch.zeros_like(x), torch.tensor([0.5, -1., 2.])
        got = oracle.gdn_norm_reference(self.ref, x, z, gain)
        expected = x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gain * 0.5
        torch.testing.assert_close(got, expected)

    def test_physical_head_sentinels_and_state_orientation(self):
        # sqrt(Dk)=2 gives exact binary arithmetic: this test isolates layout.
        q = torch.tensor([[[1., 0., 0., 0.], [0., 2., 0., 0.]]])
        k = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        v = torch.arange(1., 19.).reshape(1, 6, 3)
        args = (q, k, v, torch.zeros(1, 6), torch.ones(1, 6), torch.zeros(6, 3, 4))
        expected = v * torch.tensor([1., 2., 1., 2., 1., 2.]).reshape(1, 6, 1) / 2
        for algorithm in ("chunk", "recurrent"):
            out, state = oracle.gdn_reference(self.ref, *args, algorithm=algorithm, normalize_qk=False)
            self.assertTrue(torch.equal(out, expected))
            self.assertTrue(torch.equal(state[::2, :, 0], v[0, ::2]))
            self.assertTrue(torch.equal(state[1::2, :, 1], v[0, 1::2]))
            self.assertEqual(torch.count_nonzero(state[::2, :, 1]).item(), 0)
            self.assertEqual(torch.count_nonzero(state[1::2, :, 0]).item(), 0)
            self.assertEqual(torch.count_nonzero(state[:, :, 2:]).item(), 0)

    def test_gdn_affine_fp64_and_chunk_boundaries(self):
        gen = torch.Generator(device="cpu").manual_seed(3813)
        for tokens in (1, 3, 4, 5, 63, 64, 65, 127, 128, 129):
            with self.subTest(tokens=tokens):
                q, k = (torch.randn(tokens, 2, 3, generator=gen) for _ in range(2))
                v = torch.randn(tokens, 6, 5, generator=gen)
                decay, beta = -torch.rand(tokens, 6, generator=gen), torch.rand(tokens, 6, generator=gen)
                state = torch.randn(6, 5, 3, generator=gen)
                args = (q, k, v, decay, beta, state)
                expected = oracle.affine_reference(*args)
                for algorithm in ("chunk", "recurrent"):
                    result = oracle.gdn_reference(self.ref, *args, algorithm=algorithm)
                    for got, ref in zip(result, expected):
                        torch.testing.assert_close(got.double(), ref, atol=5e-5, rtol=0)

    def test_gdn_append_causality_and_reset(self):
        gen = torch.Generator(device="cpu").manual_seed(3814)
        q, k = (torch.randn(129, 2, 3, generator=gen) for _ in range(2))
        v = torch.randn(129, 6, 5, generator=gen)
        decay, beta = -torch.rand(129, 6, generator=gen), torch.rand(129, 6, generator=gen)
        initial = torch.randn(6, 5, 3, generator=gen)
        full, last = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial)
        state, pos, pieces = initial, 0, []
        for size in (0, 1, 3, 2, 65, 58):
            part = [a[pos:pos + size] for a in (q, k, v, decay, beta)]
            out, state = oracle.gdn_reference(self.ref, *part, state)
            pos += size
            pieces.append(out)
        torch.testing.assert_close(torch.cat(pieces), full, atol=5e-5, rtol=0)
        torch.testing.assert_close(state, last, atol=5e-5, rtol=0)
        changed = v.clone()
        changed[65:] *= -100
        y, _ = oracle.gdn_reference(self.ref, q, k, changed, decay, beta, initial)
        self.assertTrue(torch.equal(y[:65], full[:65]))
        reset = torch.zeros_like(initial)
        y0, s0 = oracle.gdn_reference(self.ref, q, k, v, decay, beta, reset)
        y1, s1 = oracle.gdn_reference(self.ref, q, k, v, decay, beta, torch.zeros_like(initial))
        self.assertTrue(torch.equal(y0, y1) and torch.equal(s0, s1))

    def test_invalid_gdn_gates_rejected(self):
        q = torch.ones(1, 1, 1)
        for decay, beta in [(1., .5), (float("-inf"), .5), (0., -1.), (0., 2.), (0., float("nan"))]:
            with self.assertRaises(ValueError):
                oracle.gdn_reference(self.ref, q, q, q, torch.tensor([[decay]]),
                                      torch.tensor([[beta]]), q)

    def test_gate_extremes_and_l2_epsilon(self):
        a = torch.tensor([-100., -20., -1., 0., 1., 20., 100.]).reshape(-1, 1)
        dt, ssm_a = torch.tensor([.25]), torch.tensor([-.5])
        decay, beta = oracle.gate_reference(a, -a, dt, ssm_a)
        expected = -.5 * np.logaddexp(0., a.double().numpy() + .25)
        np.testing.assert_allclose(decay.numpy(), expected, atol=1e-5, rtol=0)
        np.testing.assert_allclose(beta.numpy(), np.exp(-np.logaddexp(0., a.double().numpy())), atol=1e-7)
        self.assertTrue(torch.isfinite(decay).all() and (decay <= 0).all())
        q = torch.tensor([[0., 0., 0.], [1e-6, -1e-6, 0.], [1., 2., 3.], [100., -100., 100.]])
        expected = q.double() / (q.double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        torch.testing.assert_close(self.ref.l2norm(q).double(), expected, rtol=0, atol=1e-7)
        with self.assertRaises(ValueError):
            oracle.gate_reference(a, a, dt, -ssm_a)

    def test_conv_toeplitz_fp64_and_independent_channels(self):
        gen = torch.Generator(device="cpu").manual_seed(3815)
        for channels in (1, 31, 33):
            for dilation in (1, 3):
                for kernel in (1, 2, 4):
                    halo = (kernel - 1) * dilation
                    for tokens in sorted({0, 1, max(0, halo - 1), halo, halo + 1}):
                        taps = torch.randn(channels, kernel, generator=gen)
                        x = torch.randn(tokens, channels, generator=gen)
                        history = torch.randn(halo, channels, generator=gen)
                        out, final = oracle.conv_reference(x, taps, history, dilation, False)
                        raw = torch.cat((history, x)).double().numpy()
                        matrix = np.zeros((channels, tokens, halo + tokens))
                        for t in range(tokens):
                            matrix[:, t, t + np.arange(kernel) * dilation] = taps.double().numpy()
                        expected = np.einsum("ctj,jc->tc", matrix, raw)
                        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5, rtol=0)
                        expected_history = torch.cat((history, x))[-halo:] if halo else history
                        self.assertTrue(torch.equal(final, expected_history))
                        if halo:
                            self.assertNotEqual(final.data_ptr(), history.data_ptr())

    def test_norm_axes_gains_and_epsilon_fp64(self):
        x = torch.tensor([[[0., 0., 0.], [1e-6, -2e-6, 3e-6]], [[1., -2., 3.], [100., -20., 1.]]])
        for eps in (1e-6, 1e-3):
            for gain in (torch.tensor([.5, -1., 2.]), torch.tensor([[.5, -1., 2.], [2., .25, -2.]])):
                expected = x.double() * gain.double() / (x.double().square().mean(-1, keepdim=True) + eps).sqrt()
                got = oracle.grouped_rms_reference(self.ref, x, gain, eps)
                torch.testing.assert_close(got.double(), expected, rtol=0, atol=1e-5)
            z = torch.tensor([[[-100., 0., 100.], [20., -20., 1.]]]).expand_as(x)
            gain = torch.tensor([.5, -1., 2.])
            expected = (x.double() * gain.double() / (x.double().square().mean(-1, keepdim=True) + eps).sqrt()
                        * z.double().sigmoid())
            got = oracle.gdn_norm_reference(self.ref, x, z, gain, eps)
            torch.testing.assert_close(got.double(), expected, rtol=0, atol=1e-5)

    def test_hc_nonzero_projections_fp64(self):
        gen = torch.Generator(device="cpu").manual_seed(3816)
        x = torch.randn(5, 3, 7, generator=gen)
        gain = torch.tensor([.25, -.5, 1., 2., -1., 0., 1.5]).expand(3, 7)
        down, up = torch.randn(5, 21, generator=gen) / 5, torch.randn(21, 5, generator=gen) / 5
        inject, block = torch.randn(3, 21, generator=gen) / 5, torch.randn(5, 7, generator=gen)
        got = oracle.hc_reference(self.ref, x, gain, down, up, inject, block)
        xx, gg = x.double().numpy(), gain.double().numpy()
        norm = xx * gg / np.sqrt((xx * xx).mean(-1, keepdims=True) + 1e-6)
        low = norm.reshape(5, 21) @ down.double().numpy().T
        sigmoid = lambda t: np.exp(-np.logaddexp(0., -t))
        gate = ((low / 3) * sigmoid(low / 3) @ up.double().numpy().T).reshape(5, 3, 7)
        logits = norm.reshape(5, 21) @ inject.double().numpy().T
        injection = 2 * sigmoid(logits / 3)
        expected = dict(normed=norm, low=low, gate_logits=gate, mixed=(norm * sigmoid(gate)).mean(1),
                        injection_logits=logits, injection=injection,
                        residual=xx + injection[:, :, None] * block.double().numpy()[:, None, :])
        for name, value in expected.items():
            np.testing.assert_allclose(got[name].numpy(), value, atol=2e-5, rtol=0, err_msg=name)

    def test_conv_matches_upstream_prefill_and_decode(self):
        gen = torch.Generator(device="cpu").manual_seed(3817)
        x, taps = torch.randn(11, 3, generator=gen), torch.randn(3, 4, generator=gen)
        expected, last = oracle.conv_reference(x, taps, torch.zeros(3, 3))
        full = self.ref.causal_conv1d_fn(x.T[None], taps, activation="silu")[0].T
        torch.testing.assert_close(full, expected, atol=1e-5, rtol=0)
        # HF caches K samples; our minimal halo contains K-1. Align explicitly.
        state, pieces = torch.zeros(1, 3, 4), []
        for token in x:
            pieces.append(self.ref.causal_conv1d_update(token[None, :, None], state, taps, activation="silu"))
        torch.testing.assert_close(torch.cat(pieces, -1)[0].T, expected, atol=1e-5, rtol=0)
        self.assertTrue(torch.equal(state[0, :, 1:].T, last))

    def test_gdn_single_tokens_noncontiguous_and_no_alias(self):
        gen = torch.Generator(device="cpu").manual_seed(3818)
        q, k = (torch.randn(65, 2, 6, generator=gen)[:, :, ::2] for _ in range(2))
        v = torch.randn(65, 6, 5, generator=gen)
        decay, beta = -torch.rand(65, 6, generator=gen), torch.rand(65, 6, generator=gen)
        initial = torch.randn(6, 5, 3, generator=gen)
        saved = initial.clone()
        full, last = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial)
        for chunk_size in (1, 3, 4, 128):
            out, state = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial, chunk_size=chunk_size)
            torch.testing.assert_close(out, full, atol=5e-5, rtol=0)
            torch.testing.assert_close(state, last, atol=5e-5, rtol=0)
        state, pieces = initial, []
        for i in range(len(q)):
            part = [a[i:i + 1] for a in (q, k, v, decay, beta)]
            out, state = oracle.gdn_reference(self.ref, *part, state, algorithm="recurrent")
            pieces.append(out)
        torch.testing.assert_close(torch.cat(pieces), full, atol=5e-5, rtol=0)
        torch.testing.assert_close(state, last, atol=5e-5, rtol=0)
        self.assertTrue(torch.equal(initial, saved))
        self.assertNotEqual(state.data_ptr(), initial.data_ptr())
        empty, copy = oracle.gdn_reference(self.ref, q[:0], k[:0], v[:0], decay[:0], beta[:0], initial)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(torch.equal(copy, saved))
        copy.zero_()
        self.assertTrue(torch.equal(initial, saved))

    def test_gdn_long_streams(self):
        gen = torch.Generator(device="cpu").manual_seed(3819)
        for tokens in (1023, 1024, 1025, 8192, 16384, 32768):
            q, k = (torch.randn(tokens, 1, 3, generator=gen) for _ in range(2))
            v = torch.randn(tokens, 2, 5, generator=gen)
            # Nearly no forgetting exercises accumulated roundoff in the state.
            decay, beta = -torch.rand(tokens, 2, generator=gen) * 1e-5, torch.rand(tokens, 2, generator=gen)
            initial = torch.randn(2, 5, 3, generator=gen)
            expected = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial, algorithm="recurrent")
            out, last = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial)
            limit = 5e-5 if tokens <= 1025 else 2e-4
            for name, got, ref, axes in (("output", out, expected[0], 1), ("state", last, expected[1], 2)):
                metrics = oracle.error_metrics(got, ref, axes)
                self.measurements.append(dict(tokens=tokens, comparison=name, limit=limit, **metrics))
                self.assertLessEqual(metrics["scaled_inf"], limit, self.measurements[-1])

    def test_scaled_error_rejects_nonfinite_and_keeps_heads_separate(self):
        expected = torch.tensor([[[100000., 0.], [0., 0.]]])
        got = expected.clone()
        got[0, 1, 0] = .1
        metrics = oracle.error_metrics(got, expected)
        self.assertGreater(metrics["scaled_inf"], .09)
        self.assertEqual(metrics["worst_index"], [0, 1, 0])
        got[0, 1, 0] = float("nan")
        with self.assertRaises(ValueError):
            oracle.error_metrics(got, expected)

    def test_fixture_writer_roundtrip_and_no_overwrite(self):
        tensors = {"state": ("state_initial", ("Hv", "Dv", "Dk"), torch.arange(12.).reshape(2, 2, 3)),
                   "output": ("output", ("T", "C"), torch.arange(15.).double().reshape(3, 5)[:, ::2])}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "bundle"
            manifest = oracle.write_bundle(out, tensors, {"test": True})
            raw = (out / "tensors.bin").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["data"]["sha256"])
            for name, item in manifest["tensors"].items():
                begin, size = item["offset_bytes"], item["nbytes"]
                self.assertEqual(begin % 64, 0)
                piece = raw[begin:begin + size]
                self.assertEqual(hashlib.sha256(piece).hexdigest(), item["sha256"])
                actual = np.frombuffer(piece, dtype=item["dtype"]).reshape(item["shape"])
                np.testing.assert_array_equal(actual, tensors[name][2].numpy())
            with self.assertRaises(FileExistsError):
                oracle.write_bundle(out, tensors, {})
            self.assertEqual((out / "tensors.bin").read_bytes(), raw)
            tensors["bad"] = ("output", ("C",), torch.tensor([float("nan")]))
            with self.assertRaises(ValueError):
                oracle.write_bundle(Path(directory) / "invalid", tensors, {})
            self.assertFalse((Path(directory) / "invalid").exists())

    def test_gdn_fixture_is_reproducible_and_keeps_both_states(self):
        recipe = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(recipe["upstream"]["sha256"], oracle.UPSTREAM_SHA256)
        self.assertEqual(recipe["upstream"]["revision"], oracle.UPSTREAM_REV)
        runs = [oracle.gdn_fixture(self.ref, recipe["gdn"], torch.Generator(device="cpu").manual_seed(recipe["seed"]),
                recipe["rms_epsilon"], recipe["tolerances_proposed"]["gdn"]) for _ in range(2)]
        self.assertEqual(runs[0][1], runs[1][1])
        for name, (role, axes, tensor) in runs[0][0].items():
            self.assertTrue(torch.equal(tensor, runs[1][0][name][2]), name)
            self.assertEqual(len(axes), tensor.ndim)
        self.assertEqual(runs[0][0]["gdn.state_initial"][0], "state_initial")
        self.assertEqual(runs[0][0]["gdn.state_chunk"][0], "state_final")
        self.assertEqual(runs[0][0]["gdn.output_affine64"][2].dtype, torch.float64)

    def test_complete_fixture_export_provenance_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            manifest = oracle.export_synthetic(SOURCE, a)
            oracle.export_synthetic(SOURCE, b)
            for filename in ("manifest.json", "tensors.bin"):
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            metadata = manifest["metadata"]
            self.assertEqual(metadata["analytic_sha256"], hashlib.sha256((FIXTURES / "analytic.json").read_bytes()).hexdigest())
            self.assertEqual(metadata["adapter_sha256"], hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest())
            for prefix in ("gdn", "conv_d1", "conv_d3"):
                roles = {t["role"] for name, t in manifest["tensors"].items() if name.startswith(prefix + ".")}
                self.assertTrue({"state_initial", "state_final", "input", "output"} <= roles)
            self.assertIn("norm.output", manifest["tensors"])
            self.assertIn("hc.result_residual", manifest["tensors"])

    def test_conv_gdn_composition_interleaved_sessions_and_reset(self):
        gen = torch.Generator(device="cpu").manual_seed(3821)
        t, hk, hv, dk, dv, kernel = 129, 2, 6, 3, 5, 4
        channels = 2 * hk * dk + hv * dv
        taps = torch.randn(channels, kernel, generator=gen) / 4
        gain = torch.tensor([.25, -.5, 1., 2., -1.])
        sessions = []
        for _ in range(2):
            x = torch.randn(t, channels, generator=gen)
            z = torch.randn(t, hv, dv, generator=gen)
            a, b = (torch.randn(t, hv, generator=gen) for _ in range(2))
            decay, beta = oracle.gate_reference(a, b, torch.full((hv,), .25), torch.full((hv,), -.5))
            state = torch.randn(hv, dv, dk, generator=gen)
            history = torch.randn(kernel - 1, channels, generator=gen)
            sessions.append((x, z, decay, beta, state, history))

        def run(session, start, stop, state, history, algorithm="chunk"):
            x, z, decay, beta, _, _ = session
            qkv, next_history = oracle.conv_reference(x[start:stop], taps, history)
            q, k, v = torch.split(qkv, (hk * dk, hk * dk, hv * dv), dim=-1)
            n = stop - start
            out, next_state = oracle.gdn_reference(self.ref, q.reshape(n, hk, dk), k.reshape(n, hk, dk),
                v.reshape(n, hv, dv), decay[start:stop], beta[start:stop], state, algorithm=algorithm)
            return oracle.gdn_norm_reference(self.ref, out, z[start:stop], gain), next_state, next_history

        expected = [run(s, 0, t, s[4], s[5]) for s in sessions]
        states = [(s[4], s[5]) for s in sessions]
        pieces, pos = [[], []], 0
        for size in (1, 3, 2, 65, 58):
            for i, session in enumerate(sessions):
                out, state, history = run(session, pos, pos + size, *states[i])
                # Output and both persistent states at every irregular boundary.
                prefix = run(session, 0, pos + size, session[4], session[5])
                torch.testing.assert_close(state, prefix[1], atol=5e-5, rtol=0)
                self.assertTrue(torch.equal(history, prefix[2]))
                states[i] = (state, history)
                pieces[i].append(out)
            pos += size
        for i in range(2):
            torch.testing.assert_close(torch.cat(pieces[i]), expected[i][0], atol=5e-5, rtol=0)
        state, history, pieces = sessions[0][4], sessions[0][5], []
        for pos in range(t):
            out, state, history = run(sessions[0], pos, pos + 1, state, history, "recurrent")
            pieces.append(out)
        torch.testing.assert_close(torch.cat(pieces), expected[0][0], atol=5e-5, rtol=0)
        torch.testing.assert_close(state, expected[0][1], atol=5e-5, rtol=0)
        self.assertTrue(torch.equal(history, expected[0][2]))
        # Reset dirty copies, then compare with a completely fresh session.
        state.zero_()
        history.zero_()
        reset = run(sessions[0], 0, t, state, history)
        fresh = run(sessions[0], 0, t, torch.zeros_like(state), torch.zeros_like(history))
        for got, ref in zip(reset, fresh):
            self.assertTrue(torch.equal(got, ref))
        stale_history = run(sessions[0], 0, t, state, expected[0][2])
        self.assertFalse(torch.allclose(stale_history[0][:3], fresh[0][:3]))

    def test_upstream_norms_do_not_claim_fp64_precision(self):
        x, gain = torch.ones(1, 2, 3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "FP32"):
            oracle.grouped_rms_reference(self.ref, x, gain)
        with self.assertRaisesRegex(ValueError, "FP32"):
            oracle.gdn_norm_reference(self.ref, x, x, gain)

    def test_effective_gain_is_never_silently_rounded(self):
        gain = torch.tensor([1e-10, .5, 2.])
        x = torch.ones(1, 1, 3)
        with self.assertRaisesRegex(ValueError, "reconstructed exactly"):
            oracle.grouped_rms_reference(self.ref, x, gain)
        with self.assertRaisesRegex(ValueError, "reconstructed exactly"):
            oracle.hc_reference(self.ref, x, gain[None], torch.zeros(1, 3), torch.zeros(3, 1))
        # Direct GDN gain is unrelated to the zero-centered HF convention.
        result = oracle.gdn_norm_reference(self.ref, x, torch.zeros_like(x), gain)
        self.assertGreater(result[0, 0, 0].item(), 0.)


class CView(ct.Structure):
    _fields_ = [("data", ct.POINTER(ct.c_float)), ("capacity_bytes", ct.c_size_t), ("rank", ct.c_size_t),
                ("dim", ct.c_size_t * 4), ("stride_bytes", ct.c_size_t * 4)]


class CMemory(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("bytes", ct.c_size_t)]


class CKey(ct.Structure):
    _fields_ = [("model", ct.c_void_p), ("layer", ct.c_uint32), ("epoch", ct.c_uint64)]


class CInfo(ct.Structure):
    _fields_ = [("key", CKey), ("next_pos", ct.c_uint64), ("valid", ct.c_bool), ("busy", ct.c_bool)]


class CSize(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("metadata_bytes", "payload_bytes", "alignment")]


class CError(ct.Structure):
    _fields_ = [("code", ct.c_int), ("os_errno", ct.c_int), ("file_offset", ct.c_uint64), ("row", ct.c_uint64),
                ("requested_bytes", ct.c_size_t), ("completed_bytes", ct.c_size_t),
                ("subject", ct.c_char * 128), ("detail", ct.c_char * 256)]


class CConv(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("channels", "kernel", "dilation")] + [("apply_silu", ct.c_bool)]


class CHc(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("streams", "hidden_size", "low_rank")] + [("rms_eps", ct.c_float)]


class CGdn(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("key_heads", "value_heads", "key_dim", "value_dim")] + [
                ("rms_eps", ct.c_float), ("l2_eps", ct.c_float)]


def c_view(array):
    """Test-side ABI mirror of the frozen header; capacity includes row padding."""
    assert array.dtype == np.float32 and 1 <= array.ndim <= 4 and all(s >= 0 for s in array.strides)
    strides = list(array.strides)
    if not array.size:  # NumPy's zero strides for empty allocations are not a valid C descriptor.
        stride = 4
        for axis in range(array.ndim - 1, -1, -1):
            strides[axis] = stride
            stride *= array.shape[axis]
    span = 4 + sum((d - 1) * s for d, s in zip(array.shape, strides)) if array.size else 0
    return CView(array.ctypes.data_as(ct.POINTER(ct.c_float)), span, array.ndim,
                 (ct.c_size_t * 4)(*array.shape), (ct.c_size_t * 4)(*strides))


class NativeTests(unittest.TestCase):
    measurements = []

    @classmethod
    def setUpClass(cls):
        if CPU_LIB is None:
            raise unittest.SkipTest("native comparisons require --cpu-lib; oracle-only run")
        cls.lib = ct.CDLL(str(CPU_LIB.resolve()))
        cls.ref = oracle.load_upstream(SOURCE)
        error = ct.POINTER(CError)
        cls.lib.qwen38_conv_state_bytes.argtypes = [ct.POINTER(CConv), ct.POINTER(CSize), error]
        cls.lib.qwen38_conv_state_init.argtypes = [ct.POINTER(CConv), CKey, CMemory, CMemory, ct.POINTER(ct.c_void_p), error]
        cls.lib.qwen38_conv_state_reset.argtypes = [ct.c_void_p, ct.c_uint64, error]
        cls.lib.qwen38_conv_state_export.argtypes = [ct.c_void_p, CView, error]
        cls.lib.qwen38_conv_state_copy.argtypes = [ct.c_void_p, ct.c_void_p, error]
        cls.lib.qwen38_conv_state_info.argtypes = [ct.c_void_p]
        cls.lib.qwen38_conv_state_info.restype = CInfo
        cls.lib.qwen38_conv_causal.argtypes = [ct.POINTER(CConv), CView, CView, CView, ct.c_void_p, ct.c_uint64, error]
        cls.lib.qwen38_activate.argtypes = [ct.c_int, CView, CView, error]
        cls.lib.qwen38_rmsnorm_groups.argtypes = [CView, CView, CView, ct.c_float, error]
        cls.lib.qwen38_hc_lowrank_activate.argtypes = [ct.POINTER(CHc), CView, CView, error]
        cls.lib.qwen38_hc_inject_weights.argtypes = [ct.POINTER(CHc), CView, CView, error]
        cls.lib.qwen38_hc_read.argtypes = [ct.POINTER(CHc), CView, CView, CView, error]
        cls.lib.qwen38_hc_add.argtypes = [ct.POINTER(CHc), CView, CView, CView, error]
        cls.lib.qwen38_gdn_state_bytes.argtypes = [ct.POINTER(CGdn), ct.POINTER(CSize), error]
        cls.lib.qwen38_gdn_state_init.argtypes = [ct.POINTER(CGdn), CKey, CMemory, CMemory, ct.POINTER(ct.c_void_p), error]
        cls.lib.qwen38_gdn_state_reset.argtypes = [ct.c_void_p, ct.c_uint64, error]
        cls.lib.qwen38_gdn_state_export.argtypes = [ct.c_void_p, CView, error]
        cls.lib.qwen38_gdn_state_info.argtypes = [ct.c_void_p]
        cls.lib.qwen38_gdn_state_info.restype = CInfo
        cls.lib.qwen38_gdn_prepare.argtypes = [ct.POINTER(CGdn), *([CView] * 8), error]
        cls.lib.qwen38_gdn_scan.argtypes = [ct.POINTER(CGdn), *([CView] * 4), ct.c_void_p, ct.c_uint64, error]
        cls.lib.qwen38_gdn_norm_gate.argtypes = [ct.POINTER(CGdn), *([CView] * 4), error]

    def invoke(self, name, *args):
        error = CError()
        status = getattr(self.lib, "qwen38_" + name)(*args, ct.byref(error))
        self.assertEqual(status, 0, (name, status, error.detail.decode()))
        self.assertEqual(error.code, 0)

    def new_state(self, shape, kind="conv"):
        size = CSize()
        self.invoke(kind + "_state_bytes", ct.byref(shape), ct.byref(size))
        buffers, memory = [], []
        for length in (size.metadata_bytes, size.payload_bytes):
            raw = ct.create_string_buffer(length + size.alignment)
            address = (ct.addressof(raw) + size.alignment - 1) // size.alignment * size.alignment
            buffers.append(raw)
            memory.append(CMemory(address, length))
        handle = ct.c_void_p()
        self.invoke(kind + "_state_init", ct.byref(shape), CKey(1, 0, 1), *memory, ct.byref(handle))
        # Retain caller-owned backings for the entire test; no C allocation/free.
        if not hasattr(self, "state_buffers"):
            self.state_buffers = []
        self.state_buffers.extend(buffers)
        return handle

    def test_conv_against_oracle_with_strides_and_append(self):
        gen = torch.Generator(device="cpu").manual_seed(3822)
        for channels in (1, 31, 33):
            for kernel in (1, 2, 4):
                for dilation in (1, 3):
                    for activation in (False, True):
                        shape = CConv(channels, kernel, dilation, activation)
                        halo, tokens = (kernel - 1) * dilation, 13
                        taps = torch.randn(channels, kernel, generator=gen)
                        initial = torch.randn(halo, channels, generator=gen)
                        x = torch.randn(tokens, channels, generator=gen)
                        expected, history = oracle.conv_reference(x, taps, initial, dilation, activation)
                        padded_x = np.full((tokens, channels + 2), np.nan, np.float32)
                        padded_x[:, :channels] = x.numpy()
                        state = self.new_state(shape)
                        # Warm the public state using exactly the raw initial history.
                        warm = np.empty((halo, channels), np.float32)
                        self.invoke("conv_causal", ct.byref(shape), c_view(initial.numpy()), c_view(taps.numpy()),
                                    c_view(warm), state, 0)
                        pieces, position = [], 0
                        for count in (0, 1, 3, 0, 2, 7):
                            output = np.full((count, channels + 2), -123., np.float32)
                            self.invoke("conv_causal", ct.byref(shape), c_view(padded_x[position:position + count, :channels]),
                                        c_view(taps.numpy()), c_view(output[:, :channels]), state, halo + position)
                            self.assertTrue(np.all(output[:, channels:] == -123))
                            pieces.append(output[:, :channels].copy())
                            position += count
                        got = torch.from_numpy(np.concatenate(pieces))
                        self.assertLessEqual(oracle.error_metrics(got, expected)["scaled_inf"], 1e-5)
                        final = np.empty((halo, channels), np.float32)
                        self.invoke("conv_state_export", state, c_view(final))
                        np.testing.assert_array_equal(final, history.numpy())
                        self.assertEqual(self.lib.qwen38_conv_state_info(state).next_pos, halo + tokens)

    def test_activations_against_fp64_and_moe_gate(self):
        x = np.concatenate((np.array([-100., -20., -1., -0., 0., 1., 20., 100.]), np.linspace(-8, 8, 129))).astype(np.float32)
        sigmoid = np.exp(-np.logaddexp(0., -x.astype(np.float64)))
        expected = (sigmoid, x.astype(np.float64) * sigmoid, np.logaddexp(0., x.astype(np.float64)))
        for kind, reference in enumerate(expected):
            output = np.full_like(x, np.nan)
            self.invoke("activate", kind, c_view(x), c_view(output))
            # Also meets the stricter elementwise F gate on moderate inputs.
            np.testing.assert_allclose(output, reference, atol=2e-6, rtol=2e-5)

    def test_native_rms_group_axes_direct_gain_and_padding(self):
        rng = np.random.default_rng(3823)
        for width in (1, 3, 128, 2560):
            for shared in (False, True):
                storage = np.full((3, 4, width + 2), np.nan, np.float32)
                x = storage[:, :, :width]
                x[:] = rng.normal(size=x.shape)
                x[0] *= 1e-6
                gain = rng.normal(size=(width,) if shared else (4, width)).astype(np.float32)
                gain.flat[0] = 1e-10  # Native GGUF gain is direct, never reconstructed as 1+w.
                expected = x.astype(np.float64) * gain / np.sqrt(np.square(x.astype(np.float64)).mean(-1, keepdims=True) + 1e-6)
                output = np.full_like(storage, -123.)
                self.invoke("rmsnorm_groups", c_view(x), c_view(gain), c_view(output[:, :, :width]), 1e-6)
                metrics = oracle.error_metrics(torch.from_numpy(output[:, :, :width]), torch.from_numpy(expected))
                self.assertLessEqual(metrics["scaled_inf"], 1e-5)
                self.assertTrue(np.all(output[:, :, width:] == -123.))
                self.invoke("rmsnorm_groups", c_view(x), c_view(gain), c_view(x), 1e-6)
                np.testing.assert_array_equal(x, output[:, :, :width])
                self.assertTrue(np.isnan(storage[:, :, width:]).all())

    def test_native_hc_projected_stages_against_upstream(self):
        gen = torch.Generator(device="cpu").manual_seed(3824)
        shape = CHc(3, 7, 5, 1e-6)
        residual = torch.randn(5, 3, 7, generator=gen)
        gain = torch.tensor([.25, -.5, 1., 2., -1., 0., 1.5]).expand(3, 7).contiguous()
        down, up = torch.randn(5, 21, generator=gen) / 5, torch.randn(21, 5, generator=gen) / 5
        inject, block = torch.randn(3, 21, generator=gen) / 5, torch.randn(5, 7, generator=gen)
        ref = oracle.hc_reference(self.ref, residual, gain, down, up, inject, block)
        low = ref["low"].numpy().copy()
        self.invoke("hc_lowrank_activate", ct.byref(shape), c_view(low), c_view(low))
        torch.testing.assert_close(torch.from_numpy(low), torch.nn.functional.silu(ref["low"] / 3), atol=2e-5, rtol=0)
        mixed = np.full((5, 7), np.nan, np.float32)
        self.invoke("hc_read", ct.byref(shape), c_view(ref["normed"].numpy()), c_view(ref["gate_logits"].numpy()), c_view(mixed))
        torch.testing.assert_close(torch.from_numpy(mixed), ref["mixed"], atol=2e-5, rtol=0)
        weights = ref["injection_logits"].numpy().copy()
        self.invoke("hc_inject_weights", ct.byref(shape), c_view(weights), c_view(weights))
        torch.testing.assert_close(torch.from_numpy(weights), ref["injection"], atol=2e-5, rtol=0)
        updated = residual.numpy().copy()
        self.invoke("hc_add", ct.byref(shape), c_view(updated), c_view(block.numpy()), c_view(weights))
        torch.testing.assert_close(torch.from_numpy(updated), ref["residual"], atol=2e-5, rtol=0)

    def seed_gdn(self, shape, initial):
        """Set arbitrary S through public rank-one writes, never poke opaque memory."""
        hk, hv, dk, dv = shape.key_heads, shape.value_heads, shape.key_dim, shape.value_dim
        state = self.new_state(shape, "gdn")
        warm = np.zeros((dk, 2 * hk * dk + hv * dv), np.float32)
        for j in range(dk):
            warm[j, hk * dk + np.arange(hk) * dk + j] = 1
            warm[j, 2 * hk * dk:] = initial[:, :, j].reshape(-1)
        decay, beta, output = np.zeros((dk, hv), np.float32), np.ones((dk, hv), np.float32), np.empty((dk, hv, dv), np.float32)
        self.invoke("gdn_scan", ct.byref(shape), c_view(warm), c_view(decay), c_view(beta), c_view(output), state, 0)
        exported = np.empty_like(initial)
        self.invoke("gdn_state_export", state, c_view(exported))
        np.testing.assert_array_equal(exported, initial)
        return state, dk

    def test_native_gdn_prepare_and_norm_gate(self):
        rng = np.random.default_rng(3825)
        for hk, hv, dk, dv in ((2, 6, 3, 5), (2, 2, 4, 7), (16, 48, 128, 128)):
            shape = CGdn(hk, hv, dk, dv, 1e-6, 1e-6)
            t, kw = 3, hk * dk
            qkv = rng.normal(size=(t, 2 * kw + hv * dv)).astype(np.float32)
            qkv[0, :2 * kw] = 0
            qkv[1, :2 * kw] *= 1e-6
            a, b = rng.normal(size=(t, hv)).astype(np.float32), rng.normal(size=(t, hv)).astype(np.float32)
            dt, weight = rng.normal(size=hv).astype(np.float32), -np.abs(rng.normal(size=hv)).astype(np.float32)
            weight[0] = 0
            prepared, log_g, beta = np.full_like(qkv, np.nan), np.full_like(a, np.nan), np.full_like(b, np.nan)
            self.invoke("gdn_prepare", ct.byref(shape), *map(c_view, (qkv, a, b, weight, dt, prepared, log_g, beta)))
            reference = qkv.astype(np.float64)
            for offset, scale in ((0, 1 / np.sqrt(dk)), (kw, 1.)):
                group = reference[:, offset:offset + kw].reshape(t, hk, dk)
                group *= scale / np.sqrt(np.square(group).sum(-1, keepdims=True) + shape.l2_eps)
            metrics = oracle.error_metrics(torch.from_numpy(prepared), torch.from_numpy(reference))
            self.assertLessEqual(metrics["scaled_inf"], 5e-5)
            np.testing.assert_array_equal(prepared[:, 2 * kw:], qkv[:, 2 * kw:])
            gates = oracle.gate_reference(*map(torch.from_numpy, (a, b, dt, weight)))
            for actual, expected in zip((log_g, beta), gates):
                torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=0)
            inplace, ag, bg = qkv.copy(), a.copy(), b.copy()
            self.invoke("gdn_prepare", ct.byref(shape), *map(c_view, (inplace, ag, bg, weight, dt, inplace, ag, bg)))
            for actual, expected in zip((inplace, ag, bg), (prepared, log_g, beta)):
                np.testing.assert_array_equal(actual, expected)
            o, z, gain = rng.normal(size=(t, hv, dv)).astype(np.float32), rng.normal(size=(t, hv, dv)).astype(np.float32), rng.normal(size=dv).astype(np.float32)
            expected = oracle.gdn_norm_reference(self.ref, *map(torch.from_numpy, (o, z, gain)))
            self.invoke("gdn_norm_gate", ct.byref(shape), *map(c_view, (o, z, gain, o)))
            self.assertLessEqual(oracle.error_metrics(torch.from_numpy(o), expected)["scaled_inf"], 1e-5)

    def check_numeric(self, name, got, expected, tolerance, axes=1):
        expected = expected if isinstance(expected, torch.Tensor) else torch.from_numpy(expected)
        metrics = oracle.error_metrics(torch.from_numpy(got), expected, axes)
        record = dict(comparison=name, limit=tolerance, **metrics)
        self.measurements.append(record)
        self.assertLessEqual(metrics["scaled_inf"], tolerance, record)

    def test_native_gdn_core_short_long_and_affine(self):
        gen = torch.Generator(device="cpu").manual_seed(3826)
        shape = CGdn(2, 6, 3, 5, 1e-6, 1e-6)
        for tokens in (0, 1, 3, 4, 5, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 8192, 16384, 32768):
            q, k = (self.ref.l2norm(torch.randn(tokens, 2, 3, generator=gen)) for _ in range(2))
            v = torch.randn(tokens, 6, 5, generator=gen)
            decay, beta = -torch.rand(tokens, 6, generator=gen) * 1e-5, torch.rand(tokens, 6, generator=gen)
            initial = torch.randn(6, 5, 3, generator=gen)
            prepared = np.concatenate(((q / np.sqrt(3)).numpy().reshape(tokens, 6),
                k.numpy().reshape(tokens, 6), v.numpy().reshape(tokens, 30)), axis=-1)
            state, position = self.seed_gdn(shape, initial.numpy())
            guarded = np.full((tokens, 6, 7), -123., np.float32)
            output = guarded[:, :, :5]
            self.invoke("gdn_scan", ct.byref(shape), c_view(prepared), c_view(decay.numpy()), c_view(beta.numpy()), c_view(output), state, position)
            self.assertTrue(np.all(guarded[:, :, 5:] == -123.))
            last = np.empty((6, 5, 3), np.float32)
            self.invoke("gdn_state_export", state, c_view(last))
            limit = 5e-5 if tokens <= 1025 else 2e-4
            for algorithm in ("chunk", "recurrent"):
                expected = oracle.gdn_reference(self.ref, q, k, v, decay, beta, initial, algorithm=algorithm, normalize_qk=False)
                self.check_numeric(f"C/{algorithm}/output/T{tokens}", output, expected[0], limit)
                self.check_numeric(f"C/{algorithm}/state/T{tokens}", last, expected[1], limit, 2)
            if tokens <= 129:
                # Exact prepared inputs: the affine oracle must not scale Q twice.
                pq = torch.from_numpy(prepared[:, :6].reshape(tokens, 2, 3))
                expected = oracle.affine_reference(pq, k, v, decay, beta, initial, False, False)
                self.check_numeric(f"C/affine/output/T{tokens}", output, expected[0], limit)
                self.check_numeric(f"C/affine/state/T{tokens}", last, expected[1], limit, 2)
            self.assertEqual(self.lib.qwen38_gdn_state_info(state).next_pos, position + tokens)

    def test_native_gdn_chunk_bitwise_and_interleaving(self):
        rng = np.random.default_rng(3827)
        shape = CGdn(2, 6, 4, 5, 1e-6, 1e-6)
        streams = []
        for _ in range(2):
            raw = rng.normal(size=(129, 46)).astype(np.float32)
            a, b = rng.normal(size=(129, 6)).astype(np.float32), rng.normal(size=(129, 6)).astype(np.float32)
            weight, dt = np.full(6, -.5, np.float32), np.full(6, .25, np.float32)
            self.invoke("gdn_prepare", ct.byref(shape), *map(c_view, (raw, a, b, weight, dt, raw, a, b)))
            initial = rng.normal(size=(6, 5, 4)).astype(np.float32)
            full_state, start = self.seed_gdn(shape, initial)
            full, last = np.empty((129, 6, 5), np.float32), np.empty_like(initial)
            self.invoke("gdn_scan", ct.byref(shape), *map(c_view, (raw, a, b, full)), full_state, start)
            self.invoke("gdn_state_export", full_state, c_view(last))
            state, _ = self.seed_gdn(shape, initial)
            streams.append((raw, a, b, full, last, state, start))
        pieces, pos = [[], []], 0
        for count in (0, 1, 3, 2, 65, 0, 58):
            for index, (raw, a, b, full, last, state, start) in enumerate(streams):
                part = np.empty((count, 6, 5), np.float32)
                self.invoke("gdn_scan", ct.byref(shape), *map(c_view, (raw[pos:pos+count], a[pos:pos+count], b[pos:pos+count], part)), state, start + pos)
                pieces[index].append(part)
            pos += count
        for index, (raw, a, b, full, last, state, start) in enumerate(streams):
            np.testing.assert_array_equal(np.concatenate(pieces[index]), full)
            actual = np.empty_like(last)
            self.invoke("gdn_state_export", state, c_view(actual))
            np.testing.assert_array_equal(actual, last)
            self.invoke("gdn_state_reset", state, 2)
            self.invoke("gdn_state_export", state, c_view(actual))
            self.assertTrue(np.all(actual == 0))
            self.assertEqual(self.lib.qwen38_gdn_state_info(state).next_pos, 0)

    def test_native_conv_candidate_copy_is_independent(self):
        shape = CConv(2, 4, 3, True)
        source, candidate = self.new_state(shape), self.new_state(shape)
        taps = np.ones((2, 4), np.float32)
        prefix, out = np.arange(22, dtype=np.float32).reshape(11, 2), np.empty((11, 2), np.float32)
        self.invoke("conv_causal", ct.byref(shape), c_view(prefix), c_view(taps), c_view(out), source, 0)
        self.invoke("conv_state_copy", candidate, source)
        initial = np.empty((9, 2), np.float32)
        copied = np.empty_like(initial)
        self.invoke("conv_state_export", source, c_view(initial))
        self.invoke("conv_state_export", candidate, c_view(copied))
        np.testing.assert_array_equal(copied, initial)
        self.assertEqual(self.lib.qwen38_conv_state_info(candidate).next_pos, 11)
        token, result = np.full((1, 2), -7., np.float32), np.empty((1, 2), np.float32)
        self.invoke("conv_causal", ct.byref(shape), c_view(token), c_view(taps), c_view(result), candidate, 11)
        self.invoke("conv_state_export", source, c_view(copied))
        np.testing.assert_array_equal(copied, initial)
        self.assertEqual(self.lib.qwen38_conv_state_info(source).next_pos, 11)
        self.invoke("conv_state_export", candidate, c_view(copied))
        np.testing.assert_array_equal(copied, np.concatenate((initial[1:], token)))


def run_control_tests(args):
    """Small real coefficients before whole HC/GDN projection slices."""
    measurements = []

    class ControlTests(unittest.TestCase):
        invoke = NativeTests.invoke
        new_state = NativeTests.new_state
        setUpClass = classmethod(NativeTests.setUpClass.__func__)

        def test_real_conv_gain_gates_state(self):
            for directory in args.controls:
                weights = oracle.CompositionWeights(args.packed_reference, directory)
                m = weights.metadata
                self.assertEqual(m["scope"], "checkpoint_slice_controls")
                self.assertLessEqual(m["packed_bytes"], 1200000)
                c, t = 2*m["hk"]*m["dk"] + m["hv"]*m["dv"], m["tokens"]
                rng = np.random.default_rng(3870 + m["layer"])
                x = torch.from_numpy((.2*rng.standard_normal((t, c))).astype(np.float32))
                mix = torch.from_numpy((.2*rng.standard_normal((t, m["hidden"]))).astype(np.float32))
                alpha, raw_beta = weights.linear(6, mix), weights.linear(7, mix)
                taps, gain = weights.constant(10), weights.constant(11).flatten()
                out, prepared = np.empty((t,c), np.float32), np.empty((t,c), np.float32)
                g, beta = np.empty((t,m["hv"]),np.float32), np.empty((t,m["hv"]),np.float32)
                cs, gs = CConv(c,m["kernel"],1,True), CGdn(m["hk"],m["hv"],m["dk"],m["dv"],m["eps"],1e-6)
                conv, state = self.new_state(cs), self.new_state(gs,"gdn")
                history = torch.zeros(m["kernel"]-1,c)
                want_conv, want_history = oracle.conv_reference(x,taps,history)
                for pos, n in ((0,1),(1,t-1)):
                    self.invoke("conv_causal",ct.byref(cs),c_view(x[pos:pos+n].numpy()),c_view(taps.numpy()),c_view(out[pos:pos+n]),conv,pos)
                actual_history = np.empty_like(want_history.numpy())
                self.invoke("conv_state_export",conv,c_view(actual_history))
                np.testing.assert_array_equal(actual_history,want_history.numpy())
                self.invoke("gdn_prepare",ct.byref(gs),*map(c_view,(out,alpha.numpy(),raw_beta.numpy(),
                    weights.constant(8).numpy().flatten(),weights.constant(9).numpy().flatten(),prepared,g,beta)))
                want_g, want_beta = oracle.gate_reference(alpha,raw_beta,weights.constant(9).flatten(),weights.constant(8).flatten())
                q,k,v = oracle.composition_qkv(m,torch.from_numpy(out))
                want_prepared = torch.cat(((self.ref.l2norm(q)/np.sqrt(m["dk"])).flatten(1),self.ref.l2norm(k).flatten(1),v.flatten(1)),-1)
                scan=np.empty((t,m["hv"],m["dv"]),np.float32)
                for pos,n in ((0,1),(1,t-1)):
                    self.invoke("gdn_scan",ct.byref(gs),*map(c_view,(prepared[pos:pos+n],g[pos:pos+n],beta[pos:pos+n],scan[pos:pos+n])),state,pos)
                final=np.empty((m["hv"],m["dv"],m["dk"]),np.float32)
                self.invoke("gdn_state_export",state,c_view(final))
                q,k,v=oracle.composition_qkv(m,torch.from_numpy(prepared))
                want_scan,want_state=oracle.gdn_reference(self.ref,q*np.sqrt(m["dk"]),k,v,torch.from_numpy(g),torch.from_numpy(beta),torch.zeros_like(torch.from_numpy(final)),normalize_qk=False)
                z=torch.from_numpy(rng.standard_normal(scan.shape).astype(np.float32)); norm=np.empty_like(scan)
                self.invoke("gdn_norm_gate",ct.byref(gs),*map(c_view,(scan,z.numpy(),gain.numpy(),norm)))
                want_norm=oracle.gdn_norm_reference(self.ref,torch.from_numpy(scan),z,gain,m["eps"])
                r=np.fromfile(Path(directory)/"input.f32","<f4").reshape(t,m["streams"],m["hidden"])
                hn=np.empty_like(r); hg=weights.constant(0).numpy().reshape(r.shape[1:])
                self.invoke("rmsnorm_groups",c_view(r),c_view(hg),c_view(hn),m["eps"])
                want_hn=oracle.grouped_rms_reference(self.ref,torch.from_numpy(r),torch.from_numpy(hg),m["eps"])
                for name,got,want,tol in (("conv",out,want_conv,1e-5),("prepared",prepared,want_prepared,1e-5),
                        ("log_decay",g,want_g,1e-5),("beta",beta,want_beta,1e-5),("scan",scan,want_scan,5e-5),
                        ("state",final,want_state,5e-5),("norm",norm,want_norm,1e-5),("hc_norm",hn,want_hn,1e-5)):
                    metric=oracle.error_metrics(torch.from_numpy(got),want,local_ndim=2 if name=="state" else 1)
                    measurements.append(dict(layer=m["layer"],stage=name,tolerance=tol,**metric))
                    self.assertLessEqual(max(metric["scaled_inf"],metric["scaled_rms"]),tol)
                self.invoke("gdn_state_reset",state,2)
                self.invoke("gdn_state_export",state,c_view(final)); self.assertFalse(np.any(final))
                self.state_buffers=[]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests))
    if args.report:
        args.report.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),
            measurements=measurements,source_sha256=oracle.UPSTREAM_SHA256,
            fixtures=[dict(path=str(p),sha256=hashlib.sha256((p/"manifest.json").read_bytes()).hexdigest()) for p in args.controls]),indent=2)+"\n")
    return not result.wasSuccessful()


def run_composition_tests(args):
    """Explicit C4 suite: no implicit skips in the existing primitive runner."""
    import subprocess
    weights = oracle.CompositionWeights(args.packed_reference, args.composition)
    m = weights.metadata
    directory = Path(args.composition)
    raw = (directory / "input.f32").read_bytes()
    if hashlib.sha256(raw).hexdigest() != m["input_sha256"]:
        raise ValueError("corrupt composition input")
    residual = torch.from_numpy(np.frombuffer(raw, "<f4").copy().reshape(m["tokens"], m["streams"], m["hidden"]))
    ref = oracle.load_upstream(args.source)
    expected = oracle.composition_reference(ref, weights, residual)
    measurements, native_reports = [], []

    def load_output(out):
        return {name: torch.from_numpy(np.fromfile(out / (name + ".f32"), "<f4").reshape(value.shape))
                for name, value in expected.items()}

    def check(name, got, want, tolerance, scope):
        # Local gates are per row/head, never diluted by other heads or streams.
        if name in ("normed", "gate_logits", "residual"):
            got, want = (v.reshape(m["tokens"], m["streams"], m["hidden"]) for v in (got, want))
        if name in ("scan", "gated_norm", "z"):
            got, want = (v.reshape(m["tokens"], m["hv"], m["dv"]) for v in (got, want))
        if name in ("prepared", "convolved", "qkv_raw"):
            for suffix, g, w in zip(("q", "k", "v"), oracle.composition_qkv(m, got), oracle.composition_qkv(m, want)):
                check(name + "/" + suffix, g, w, tolerance, scope)
            return
        metric = oracle.error_metrics(got, want, local_ndim=2 if name == "state" else 1)
        measurements.append(dict(stage=name, scope=scope, tolerance=tolerance, **metric))
        if max(metric["scaled_inf"], metric["scaled_rms"]) > tolerance:
            raise AssertionError(f"{scope}/{name}: {metric}, tolerance={tolerance}")

    class CompositionTests(unittest.TestCase):
        def test_01_cpu_intermediates_and_independent_oracle(self):
            out = directory / "cpu-whole"
            out.mkdir(exist_ok=True)
            result = subprocess.run([str(args.composed_native.resolve()), "--compose", m["checkpoint"],
                str(m["layer"]), str(directory), str(out), str(m["tokens"]), "0"], text=True, capture_output=True, check=True)
            native_reports.append(json.loads(result.stdout))
            actual = load_output(out)
            measurements.extend(dict(scope="projection", **v) for v in oracle.composition_projection_gate(weights, actual))
            for name, want in oracle.composition_local_reference(ref, weights, residual, actual).items():
                check(name, actual[name], want, 5e-5 if name in ("scan", "state") else 2e-5 if name in ("mixed", "residual") else 1e-5, "local")
            for name, want in expected.items():
                check(name, actual[name], want, 5e-4, "composed")
            # Small affine FP64 oracle uses a different mathematical organization.
            if max(m["dk"], m["dv"], m["hv"]) <= 16:
                q, k, v = oracle.composition_qkv(m, expected["convolved"])
                y, state = oracle.affine_reference(q, k, v, expected["log_decay"], expected["beta"],
                    torch.zeros(m["hv"], m["dv"], m["dk"]))
                check("scan", expected["scan"], y.flatten(1), 5e-5, "upstream_affine64")
                check("state", expected["state"], state, 5e-5, "upstream_affine64")

        def test_02_token_chunk_reset_continuation(self):
            full = load_output(directory / "cpu-whole")
            for size, suffix in ((1, "token"), (7, "chunks")):
                out = directory / ("cpu-" + suffix)
                out.mkdir(exist_ok=True)
                result = subprocess.run([str(args.composed_native.resolve()), "--compose", m["checkpoint"],
                    str(m["layer"]), str(directory), str(out), str(m["tokens"]), str(size)], text=True, capture_output=True, check=True)
                native_reports.append(json.loads(result.stdout))
                for name, value in load_output(out).items():
                    self.assertTrue(torch.equal(value, full[name]), f"{suffix}/{name} not bitwise equal")

        def test_03_explicit_fixture_and_provenance(self):
            exported = directory / "oracle-bundle"
            if not exported.exists():
                tensors = dict(residual_input=("input", ("T", "H", "E"), residual),
                    state_initial=("state_initial", ("Hv", "Dv", "Dk"), torch.zeros_like(expected["state"])),
                    history_initial=("state_initial", ("L", "C"), torch.zeros_like(expected["history"])))
                for name, value in expected.items():
                    role = "state_final" if name in ("state", "history") else "output" if name == "residual" else "intermediate"
                    axes = ("Hv", "Dv", "Dk") if name == "state" else ("L", "C") if name == "history" else ("T", "H", "E") if name == "residual" else ("T", "F")
                    tensors[name] = role, axes, value
                oracle.write_bundle(exported, tensors, dict(upstream=oracle.UPSTREAM_REV, source_sha256=oracle.UPSTREAM_SHA256,
                    catalog=m, state_contract="initial zero; history raw oldest-first; S[Hv,Dv,Dk]", projection="external B decode + FP64 products rounded FP32"))
            self.assertEqual(m["reference_sha256"], hashlib.sha256(args.packed_reference.read_bytes()).hexdigest())
            self.assertEqual(len(m["weights"]), 13)
            self.assertLessEqual(m["packed_bytes"], 96 * 1024**2)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompositionTests))
    if args.report:
        args.report.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun, skipped=len(result.skipped),
            scope=m["scope"], tokens=m["tokens"], upstream=oracle.UPSTREAM_REV, upstream_sha256=oracle.UPSTREAM_SHA256,
            native_sha256=hashlib.sha256(args.composed_native.read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest(),
            tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fixture_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            measurements=measurements, native=native_reports), indent=2) + "\n")
    return not result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cpu-lib", type=Path)
    parser.add_argument("--controls", type=Path, nargs="+")
    parser.add_argument("--composition", type=Path)
    parser.add_argument("--composed-native", type=Path)
    parser.add_argument("--packed-reference", type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    CPU_LIB = args.cpu_lib
    torch.set_num_threads(1)
    if args.controls:
        if not args.cpu_lib or not args.packed_reference:
            parser.error("--controls requires --cpu-lib and --packed-reference")
        sys.exit(run_control_tests(args))
    if args.composition:
        if not args.composed_native or not args.packed_reference:
            parser.error("--composition requires --composed-native and --packed-reference")
        sys.exit(run_composition_tests(args))
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(upstream=oracle.UPSTREAM_REV, torch=torch.__version__,
            numpy=np.__version__, tests=result.testsRun, passed=result.wasSuccessful(),
            upstream_sha256=oracle.UPSTREAM_SHA256,
            adapter_sha256=hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest(),
            tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cpu_library_sha256=hashlib.sha256(CPU_LIB.read_bytes()).hexdigest() if CPU_LIB and CPU_LIB.exists() else None,
            measurements=ReferenceTests.measurements + NativeTests.measurements), indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
