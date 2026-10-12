"""Small CPU-only oracle checks; no models, Metal, MPS or network access."""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_recurrent as oracle

SOURCE = ROOT / oracle.UPSTREAM_DEFAULT


class ReferenceTests(unittest.TestCase):
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
        q, k, v = (torch.tensor([[[a]]]) for a in (0.5, 1., 4.))
        decay = torch.tensor([[0.5]]).log()
        beta, state = torch.tensor([[0.25]]), torch.tensor([[[2.]]])
        for algorithm in ("chunk", "recurrent"):
            out, final = oracle.gdn_reference(self.ref, q, k, v, decay, beta, state,
                                              algorithm=algorithm, normalize_qk=False)
            torch.testing.assert_close(out, torch.tensor([[[0.875]]]), rtol=0, atol=1e-7)
            torch.testing.assert_close(final, torch.tensor([[[1.75]]]), rtol=0, atol=1e-7)
        self.assertEqual(state.item(), 2.)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    torch.set_num_threads(1)
    unittest.main(argv=[sys.argv[0], *remaining])
