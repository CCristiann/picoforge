"""tools/quant I/O: imatrix reader, ranged tensor fetch, resumable manifest. Synthetic data only."""

import hashlib
import http.server
import json
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
from fetch import fetch_tensor  # noqa: E402
from imatrix import Imatrix  # noqa: E402
from manifest import Manifest  # noqa: E402


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_imatrix(path: Path, entries: dict) -> str:
    """entries: tensor base name -> (sums [experts, width] f32, counts [experts] f32). Returns sha256."""
    kv = [("general.type", 8, gguf_string("imatrix")),
          ("imatrix.datasets", 9, struct.pack("<IQ", 8, 1) + gguf_string("synthetic.txt")),
          ("imatrix.chunk_count", 4, struct.pack("<I", 3)),
          ("imatrix.chunk_size", 4, struct.pack("<I", 512))]
    tensors, blobs, offset = [], [], 0
    for base, (sums, counts) in entries.items():
        for name, arr, dims in ((base + ".in_sum2", sums, (sums.shape[1], sums.shape[0])),
                                (base + ".counts", counts, (1, counts.shape[0]))):
            data = np.ascontiguousarray(arr, "<f4").tobytes()
            tensors.append((name, dims, offset))
            blobs.append(data + b"\0" * (-len(data) % 32))
            offset += len(blobs[-1])
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, t, value in kv:
        head += gguf_string(key) + struct.pack("<I", t) + value
    for name, dims, off in tensors:
        head += gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", 0, off)
    head += b"\0" * (-len(head) % 32)
    data = head + b"".join(blobs)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class ImatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "imatrix.gguf"
        rng = np.random.default_rng(0)
        self.sums = rng.random((4, 256), dtype=np.float32) * 10
        self.counts = np.array([5, 0, 2, 7], np.float32)
        self.sha = write_imatrix(self.path, {"blk.0.ffn_gate_exps.weight": (self.sums, self.counts)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_imatrix_rejects_wrong_sha(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            Imatrix(self.path, "0" * 64)

    def test_imatrix_missing_tensor_fails(self):
        with self.assertRaisesRegex(KeyError, "blk.0.ffn_up_exps.weight"):
            Imatrix(self.path, self.sha).expert_importance("blk.0.ffn_up_exps.weight", 0, np.ones((2, 256), "<f4"))

    def test_importance_is_sum_over_count(self):
        got = Imatrix(self.path, self.sha).expert_importance("blk.0.ffn_gate_exps.weight", 2, np.ones((2, 256), "<f4"))
        self.assertTrue(np.array_equal(got, (self.sums[2] / np.float32(2)).astype("<f4")))

    def test_zero_count_uses_weight_energy(self):
        source = np.random.default_rng(1).standard_normal((8, 256)).astype("<f4")
        got = Imatrix(self.path, self.sha).expert_importance("blk.0.ffn_gate_exps.weight", 1, source)
        self.assertTrue(np.array_equal(got, np.square(source, dtype=np.float32).sum(axis=0, dtype=np.float32)))

    def test_metadata_is_exposed(self):
        im = Imatrix(self.path, self.sha)
        self.assertEqual((im.datasets, im.chunk_count, im.chunk_size), (["synthetic.txt"], 3, 512))


def safetensors(tensors: dict) -> bytes:
    header, blob = {}, b""
    for name, arr in tensors.items():
        data = arr.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + blob


class RangeServer:
    """Serves files from memory with HTTP Range support; `broken` paths die after half the body."""

    def __init__(self, files: dict, broken: set = frozenset()):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                data = outer.files.get(self.path)
                if data is None:
                    self.send_error(404); return
                start, end = 0, len(data) - 1
                rng = self.headers.get("Range")
                if rng:
                    a, b = rng.removeprefix("bytes=").split("-")
                    start, end = int(a), int(b)
                body = data[start:end + 1]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.path in outer.broken and len(body) > 64:
                    self.wfile.write(body[:len(body) // 2]); self.wfile.flush()
                    self.connection.close(); return
                self.wfile.write(body)

        self.files, self.broken = files, broken
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.a = np.arange(4 * 64, dtype="<u2").reshape(4, 64)
        self.b = (np.arange(3 * 128, dtype="<u2") * 7).reshape(3, 128)
        shard = safetensors({"m.a": self.a, "m.b": self.b})
        (self.dir / "index.json").write_text(json.dumps({"weight_map": {"m.a": "s1.safetensors", "m.b": "s1.safetensors"}}))
        self.server = RangeServer({"/org/model/resolve/rev1/s1.safetensors": shard,
                                   "/org/bad/resolve/rev1/s1.safetensors": shard},
                                  broken={"/org/bad/resolve/rev1/s1.safetensors"})

    def tearDown(self):
        self.server.close(); self.tmp.cleanup()

    def test_fetch_writes_exact_bytes_and_sidecar(self):
        out = fetch_tensor("org/model", "rev1", self.dir / "index.json", "m.b", self.dir / "t", base_url=self.server.url)
        self.assertEqual(out.read_bytes(), self.b.tobytes())
        side = json.loads(out.with_suffix(".json").read_text())
        self.assertEqual((side["dtype"], side["shape"], side["shard"]), ("BF16", [3, 128], "s1.safetensors"))
        self.assertEqual(side["sha256"], hashlib.sha256(self.b.tobytes()).hexdigest())

    def test_fetch_never_leaves_partial_as_final(self):
        with self.assertRaises(Exception):
            fetch_tensor("org/bad", "rev1", self.dir / "index.json", "m.b", self.dir / "t", base_url=self.server.url)
        self.assertFalse((self.dir / "t" / "m.b.bin").exists())


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.identity = {"source": "rev1", "recipe": "compat"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_resume_detects_changed_file(self):
        f = self.dir / "t.bin"; f.write_bytes(b"abc")
        m = Manifest.load_or_create(self.dir / "manifest.json", self.identity)
        m.record("t", f, note="x")
        again = Manifest.load_or_create(self.dir / "manifest.json", self.identity)
        self.assertTrue(again.done("t", f))
        f.write_bytes(b"abd")
        self.assertFalse(again.done("t", f))

    def test_manifest_identity_mismatch_fails(self):
        Manifest.load_or_create(self.dir / "manifest.json", self.identity).record("t", self._file())
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            Manifest.load_or_create(self.dir / "manifest.json", {"source": "rev2", "recipe": "compat"})

    def _file(self) -> Path:
        f = self.dir / "t.bin"; f.write_bytes(b"abc"); return f


if __name__ == "__main__":
    unittest.main()
