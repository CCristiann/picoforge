"""build_mixed.py on synthetic checkpoints: every expert of a mixed file is the exact bytes of the
template (cold) or of the Q4 source (hot), every other tensor is copied, a pool is transparent.

  python tests/test_build_mixed.py                 unit tests
  python tests/test_build_mixed.py --fixture OUT   write the engine fixture (make target)
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quant"))
import build_mixed as bm  # noqa: E402

TEMPLATE, SOURCE = ROOT / "build/qwen4-g1/graph.gguf", ROOT / "build/qwen4-g1/graph-q4.gguf"


def synthetic_ranking(path: Path, seed=5) -> dict:
    tpl = bm.Gguf(TEMPLATE)
    e = tpl.tensors["blk.0.ffn_down_exps.weight"][1][-1]
    rng = np.random.default_rng(seed)
    ranking = {"rows": 0, "experts": e, "sources": {},
               "layers": {str(l): {"order": rng.permutation(e).tolist(), "mass": []} for l in tpl.trunk_layers()}}
    path.write_text(json.dumps(ranking))
    return ranking


def raw(g: bm.Gguf, name: str) -> bytes:
    with open(g.path, "rb") as f:
        f.seek(g.tensors[name][2])
        return f.read(g.nbytes(name))


class BuildMixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g1")
        d = Path(cls.tmp.name)
        cls.ranking = synthetic_ranking(d / "ranking.json")
        cls.layers = bm.Gguf(TEMPLATE).trunk_layers()
        counts = {l: 2 for l in cls.layers}
        cls.direct, cls.pooled, cls.pool = d / "direct.gguf", d / "pooled.gguf", d / "pool.gguf"
        bm.build(TEMPLATE, SOURCE, cls.ranking, counts, cls.direct, {"test": True})
        bm.pool(SOURCE, cls.ranking, 3, cls.pool)
        bm.build(TEMPLATE, cls.pool, cls.ranking, counts, cls.pooled, {"test": True})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pool_is_transparent(self):
        self.assertEqual(self.direct.read_bytes(), self.pooled.read_bytes())

    def test_every_expert_comes_from_template_or_source(self):
        tpl, src, out = bm.Gguf(TEMPLATE), bm.Gguf(SOURCE), bm.Gguf(self.direct)
        self.assertIs(out.meta["picoforge.experts.mixed"], True)
        for key, value in tpl.meta.items():
            self.assertTrue(np.array_equal(out.meta[key], value) if isinstance(value, np.ndarray) else out.meta[key] == value, key)
        with open(TEMPLATE, "rb") as ft, open(SOURCE, "rb") as fs, open(self.direct, "rb") as fo:
            for layer in self.layers:
                m = np.frombuffer(raw(out, f"blk.{layer}.ffn_exps_map.weight"), "<f4")
                self.assertEqual(sorted(np.nonzero(m < 0)[0]), sorted(self.ranking["layers"][str(layer)]["order"][:2]))
                self.assertEqual(sorted(m[m >= 0]), list(range(len(m) - 2)))
                self.assertEqual(sorted(-1 - m[m < 0]), [0, 1])
                for part in bm.PARTS:
                    name = f"blk.{layer}.ffn_{part}_exps.weight"
                    self.assertEqual(out.tensors[name][0], tpl.tensors[name][0])
                    self.assertEqual(out.tensors[name[:-7] + "_hot.weight"][0], src.tensors[name][0])
                    for x, v in enumerate(m):
                        if v >= 0:
                            self.assertEqual(out.expert(fo, name, int(v)), tpl.expert(ft, name, x))
                        else:
                            self.assertEqual(out.expert(fo, name[:-7] + "_hot.weight", int(-1 - v)), src.expert(fs, name, x))
        for name in tpl.tensors:
            if "_exps." not in name or name.startswith(f"blk.{len(self.layers)}."):
                self.assertEqual(raw(out, name), raw(tpl, name), name)

    def test_rank_orders_by_gate_mass_ties_by_id(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g1") as d:
            log = Path(d) / "act.bin"
            ids = np.array([[2, 0], [2, 3], [1, 3]], "<u4"); g = np.array([[0.5, 0.5], [0.25, 0.75], [0.5, 0.5]], "<f4")
            with open(log, "wb") as f:      # two appends, layer 7 only, H = 0 (routes only)
                for rows in (slice(0, 1), slice(1, 3)):
                    n = rows.stop - rows.start
                    f.write(struct.pack("<7I", 0x41345751, 1, rows.start, n, 1, 0, 2) + struct.pack("<I", 7))
                    f.write(ids[rows].tobytes() + g[rows].tobytes())
            res = bm.rank(Path(d) / "r.json", [log], 4)
            self.assertEqual(res["rows"], 3)
            self.assertEqual(res["layers"]["7"]["mass"], [0.5, 0.5, 0.75, 1.25])
            self.assertEqual(res["layers"]["7"]["order"], [3, 2, 0, 1])

    def test_page_aligned_tensors_stay_page_aligned(self):
        """The disk-only n-gram table must start on a page (gguf_plan.c); the real Q2 aligns data
        to 32 bytes only, so the writer keeps every page-aligned template tensor page-aligned."""
        with tempfile.TemporaryDirectory(dir=ROOT / "build/qwen4-g1") as d:
            out = Path(d) / "w.gguf"
            blob = lambda n: (lambda f: f.write(bytes(n)))
            bm._write(out, bm._kv("general.alignment", 32), 1, 32,
                      [("a", 0, [3], blob(12), 12, False), ("b", 0, [4096], blob(16384), 16384, True)])
            g = bm.Gguf(out)
            self.assertEqual(g.tensors["b"][2] % bm.PAGE, 0)
            self.assertEqual(g.tensors["a"][2] % 32, 0)

    def test_mixed_template_is_refused(self):
        with self.assertRaises(AssertionError):
            bm.build(self.direct, SOURCE, self.ranking, {0: 1}, Path(self.tmp.name) / "again.gguf", {})


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fixture"]:
        out = Path(sys.argv[2])
        ranking = synthetic_ranking(out.with_suffix(".ranking.json"))
        bm.build(TEMPLATE, SOURCE, ranking, {l: 1 for l in bm.Gguf(TEMPLATE).trunk_layers()}, out, {"fixture": True})
        # The oracle reads tensors from a synth-style manifest: the template's, with the new table.
        manifest, g = json.loads(TEMPLATE.with_suffix(".json").read_text()), bm.Gguf(out)
        names = {v[0]: k for k, v in bm.FORMATS.items()}
        manifest["metadata"]["picoforge.experts.mixed"] = True
        manifest["tensors"] = [dict(name=n, shape=d, format=names[t], type_id=t, domain="ordinary",
                                    file_offset=o, size_bytes=g.nbytes(n)) for n, (t, d, o) in g.tensors.items()]
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    else:
        unittest.main()
