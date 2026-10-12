"""Small CPU-only oracle checks; no models, Metal, MPS or network access."""

import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--report", type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    torch.set_num_threads(1)
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(upstream=oracle.UPSTREAM_REV, torch=torch.__version__,
            numpy=np.__version__, tests=result.testsRun, passed=result.wasSuccessful(),
            upstream_sha256=oracle.UPSTREAM_SHA256,
            adapter_sha256=hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest(),
            tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            measurements=ReferenceTests.measurements), indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
