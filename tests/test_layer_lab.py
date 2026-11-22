"""layer_lab on a tiny synthetic layer: 3 experts, hidden 512, expert ff 256, top-2."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
import layer_lab  # noqa: E402
import qw4q  # noqa: E402

E, H, FF, K, L = 3, 512, 256, 2, 5


def bf16(x):
    return (np.ascontiguousarray(x, "<f4").view("<u4") >> 16).astype("<u2")


class LayerLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(3)
        (d / "src/tensors").mkdir(parents=True)
        (d / "src/config.json").write_text(json.dumps(
            {"text_config": {"num_experts": E, "hidden_size": H, "moe_intermediate_size": FF}}))
        self.gu = bf16(rng.standard_normal((E, 2 * FF, H)) * 0.03)
        self.dn = bf16(rng.standard_normal((E, H, FF)) * 0.03)
        for name, arr in ((f"model.language_model.layers.{L}.mlp.experts.gate_up_proj", self.gu),
                          (f"model.language_model.layers.{L}.mlp.experts.down_proj", self.dn)):
            (d / f"src/tensors/{name}.bin").write_bytes(arr.tobytes())
        self.x = rng.standard_normal((40, H)).astype("<f4")
        self.ids = np.stack([rng.permutation(2)[:K] for _ in range(40)]).astype("<u4")   # expert 2 never routed
        g = rng.random((40, K)).astype("<f4")
        self.g = (g / g.sum(1, keepdims=True)).astype("<f4")
        run = d / "q"; run.mkdir()
        imp = np.ones(H, "<f4")
        f32 = lambda u: (u.astype("<u4") << 16).view("<f4")
        (run / f"blk.{L}.ffn_gate_exps.weight.bin").write_bytes(b"".join(qw4q.quantize(16, f32(self.gu[e, :FF]), imp) for e in range(E)))
        (run / f"blk.{L}.ffn_up_exps.weight.bin").write_bytes(b"".join(qw4q.quantize(16, f32(self.gu[e, FF:]), imp) for e in range(E)))
        pad = lambda a: np.pad(a, ((0, 0), (0, 256 - FF % 256 if FF % 256 else 0)))
        (run / f"blk.{L}.ffn_down_exps.weight.bin").write_bytes(
            b"".join(qw4q.quantize(10, pad(f32(self.dn[e])), np.ones(pad(f32(self.dn[e])).shape[1], "<f4")) for e in range(E)))

    def tearDown(self):
        self.tmp.cleanup()

    def measure(self, runs, decode=None, x=None, ids=None, g=None):
        return layer_lab.measure(self.dir / "src", L, self.x if x is None else x, self.ids if ids is None else ids,
                                 self.g if g is None else g, runs, decode=decode)

    def test_lossless_weights_give_zero_error(self):
        exact = lambda run, name, e, rows, cols, qtype: layer_lab.bf16_rows(self.dir / "src", L, name, e)
        res = self.measure([self.dir / "q"], decode=exact)[0]
        self.assertEqual(res["routed"], 0.0)
        self.assertEqual((res["gate"], res["up"], res["down"]), (0.0, 0.0, 0.0))

    def test_quantized_weights_give_positive_error(self):
        res = self.measure([self.dir / "q"])[0]
        self.assertGreater(res["routed"], 0)
        self.assertGreater(res["gate"], 0)

    def test_token_order_does_not_matter(self):
        perm = np.random.default_rng(9).permutation(40)
        a = self.measure([self.dir / "q"])[0]
        b = self.measure([self.dir / "q"], x=self.x[perm], ids=self.ids[perm], g=self.g[perm])[0]
        self.assertAlmostEqual(a["routed"] / b["routed"], 1.0, places=5)

    def test_unrouted_expert_contributes_nothing(self):
        broken = lambda run, name, e, rows, cols, qtype: (np.full((rows, cols), np.nan, "<f4") if e == 2
                                                         else layer_lab.decode_expert(run, name, e, rows, cols, qtype))
        res = self.measure([self.dir / "q"], decode=broken)[0]
        self.assertTrue(np.isfinite(res["routed"]))


if __name__ == "__main__":
    unittest.main()
