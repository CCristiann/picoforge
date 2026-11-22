"""QW4_ACT_LOG on the synthetic CLI checkpoint: captured expert inputs, routes and gate weights."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
from act_log import layer_rows, read_act_log  # noqa: E402
from gguf_header import read_header  # noqa: E402

GGUF = ROOT / "build/qwen4-g2/cli.gguf"
PROMPT = [5, 17, 300, 41, 99, 7, 250, 3]


def route_log(path: Path, layers: int, k: int) -> dict:
    """QW4_ROUTE_LOG: per append, position, rows, layers, top_k, then [layer][row][top_k] u32."""
    data, at, per_layer = path.read_bytes(), 0, {}
    while at < len(data):
        pos, rows, n, kk = np.frombuffer(data, "<u4", 4, at); at += 16
        ids = np.frombuffer(data, "<u4", n * rows * kk, at).reshape(n, rows, kk); at += 4 * ids.size
        for layer in range(n):
            per_layer.setdefault(layer, []).append(ids[layer])
    return {layer: np.concatenate(v) for layer, v in per_layer.items()}


class ActLogTests(unittest.TestCase):
    def run_pf(self, directory: Path, layers: str, act_name="act.bin", prompt=PROMPT, chunk=3):
        env = {**os.environ, "QW4_ACT_LOG": str(directory / act_name), "QW4_ACT_LAYERS": layers,
               "QW4_ROUTE_LOG": str(directory / f"route-{act_name}")}
        return subprocess.run([str(ROOT / "pf"), "--qwen4", str(GGUF), "--tokens", ",".join(map(str, prompt)),
                               "--context", "64", "--chunk", str(chunk), "--max-new", "2"],
                              cwd=ROOT, capture_output=True, env=env)

    def test_capture_matches_router_and_route_log(self):
        self.check_capture(PROMPT, 3)

    def test_long_chunk_rows_land_at_their_offset(self):
        """40 rows in one append: more than one 32-token MoE microbatch when the grouped kernels decline."""
        self.check_capture([(7 * i + 3) % 500 for i in range(40)], 40)

    def check_capture(self, prompt, chunk):
        meta, tensors = read_header(GGUF)
        k = int(meta["qwen4exp.expert_used_count"]); e = int(meta["qwen4exp.expert_count"])
        h = int(meta["qwen4exp.embedding_length"])
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g2") as d:
            d = Path(d)
            done = self.run_pf(d, "1,3", prompt=prompt, chunk=chunk)
            self.assertEqual(done.returncode, 0, done.stderr.decode(errors="replace"))
            records = read_act_log(d / "act.bin")
            self.assertEqual(sum(r["rows"] for r in records), len(prompt) + 1)   # prefill + one decode append
            routes = route_log(d / "route-act.bin", 4, k)
            for layer in (1, 3):
                x, ids, g = layer_rows(d / "act.bin", layer)
                np.testing.assert_array_equal(ids, routes[layer])
                t, dims, off = tensors[f"blk.{layer}.ffn_gate_inp.weight"]
                self.assertEqual((t, dims), (0, [h, e]))
                router = np.memmap(GGUF, "<f4", "r", off, (e, h)).astype(np.float64)
                logits = x.astype(np.float64) @ router.T
                p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                top = np.argsort(-p, axis=1, kind="stable")[:, :k]
                np.testing.assert_array_equal(np.sort(top, 1), np.sort(ids, 1))
                want = np.take_along_axis(p, ids.astype(np.int64), 1)
                want /= want.sum(1, keepdims=True)
                self.assertLessEqual(np.abs(want - g).max(), 1e-5)

    def test_layer_out_of_range_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g2") as d:
            done = self.run_pf(Path(d), "1,9")
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("QW4_ACT_LAYERS", done.stderr.decode(errors="replace"))

    def test_existing_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g2") as d:
            (Path(d) / "act.bin").write_bytes(b"keep")
            done = self.run_pf(Path(d), "1")
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual((Path(d) / "act.bin").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
