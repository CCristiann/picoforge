"""build_mixed.py — assemble a picoforge mixed-experts GGUF (notes M4c).

A mixed file is the published Q2 with, in every trunk layer, the most used
experts ("hot", ranked by gate mass on code) taken byte for byte from the
published Q4. Nothing is re-encoded: every expert is the exact bytes of one of
the two DwarfStar files, so the change is only which bits each expert gets.

Layout per mixed layer (read by src/qwen4_weights.c): the profile tensors
blk.L.ffn_{gate,up,down}_exps.weight keep the cold experts (E_cold), the
blk.L.ffn_{gate,up,down}_exps_hot.weight tensors hold the hot ones (Q4_K,
Q4_K, MXFP4 at the logical input) and blk.L.ffn_exps_map.weight (F32 [E],
exact integers) maps expert x to cold slot v >= 0 or hot slot -1-v. Both slot
lists follow increasing expert id. Metadata: picoforge.experts.mixed = true
and picoforge.experts.recipe (JSON). Every other tensor and key is copied.

  rank   --experts E OUT.json ACT.bin...          gate mass per expert, from QW4_ACT_X=0 captures
  pool   Q4.gguf RANKING.json COUNT OUT.gguf      the COUNT best experts per layer, in rank order
  build  TEMPLATE.gguf SOURCE.gguf RANKING.json (--fraction F | --count N) OUT.gguf

SOURCE is the Q4 file or a pool (a pool names its experts in picoforge.pool.blk.L).
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/quant")); sys.path.insert(0, str(ROOT / "tools/oracle"))
from act_log import read_act_log  # noqa: E402
from gguf_header import _Reader  # noqa: E402
from qwen4_config import FORMATS  # noqa: E402

BLOCK = {t: (b, s) for t, b, s in FORMATS.values()}       # ggml type -> (weights per block, bytes)
PARTS = ("gate", "up", "down")
COPY = 1 << 26                                           # 64 MiB per read while copying
PAGE = 16384                                             # Apple Silicon page: the disk-only n-gram table starts on one


class Gguf:
    """Header of a GGUF v3 file, with the raw bytes of its key/value section kept for copying."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(path, "rb") as f:
            r = _Reader(f)
            assert f.read(4) == b"GGUF" and r.unpack("<I") == 3, f"{path}: not GGUF v3"
            n_tensors, self.n_kv = r.unpack("<Q"), r.unpack("<Q")
            kv_start, self.meta = f.tell(), {}
            for _ in range(self.n_kv):
                key = r.string()
                self.meta[key] = r.value(r.unpack("<I"))
            kv_end = f.tell()
            self.tensors = {}                            # name -> (type, dims, absolute offset), file order
            for _ in range(n_tensors):
                name = r.string()
                dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
                self.tensors[name] = (r.unpack("<I"), dims, r.unpack("<Q"))
            self.align = int(self.meta.get("general.alignment", 32))
            base = (f.tell() + self.align - 1) // self.align * self.align
            f.seek(kv_start)
            self.kv_bytes = f.read(kv_end - kv_start)
        self.tensors = {n: (t, d, base + o) for n, (t, d, o) in self.tensors.items()}

    def nbytes(self, name: str) -> int:
        t, dims, _ = self.tensors[name]
        block, size = BLOCK[t]
        return int(np.prod(dims)) // block * size

    def trunk_layers(self) -> range:
        a = self.meta["general.architecture"]
        return range(int(self.meta[f"{a}.block_count"]) - int(self.meta.get(f"{a}.nextn_predict_layers", 0)))

    def expert(self, f, name: str, slot: int) -> bytes:
        per = self.nbytes(name) // self.tensors[name][1][-1]
        f.seek(self.tensors[name][2] + slot * per)
        return f.read(per)


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, bool):
        return _string(key) + struct.pack("<I?", 7, value)
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    if isinstance(value, int):
        return _string(key) + struct.pack("<II", 4, value)
    a = np.asarray(value, "<i4")                         # expert ids
    return _string(key) + struct.pack("<IIQ", 9, 5, a.size) + a.tobytes()


def _write(out: Path, kv: bytes, n_kv: int, align: int, tensors: list) -> None:
    """tensors: (name, type, dims, writer, size, page) with writer(f) writing exactly `size` bytes;
    page = start on a PAGE boundary of the file (the data base is known before any offset, since
    every table entry has a fixed width)."""
    head = 24 + len(kv) + sum(len(_string(n)) + 4 + 8 * len(d) + 12 for n, _, d, _, _, _ in tensors)
    base = (head + align - 1) // align * align
    table, offset, placed = b"", 0, []
    for name, t, dims, writer, size, page in tensors:
        offset = (offset + align - 1) // align * align
        if page:
            offset = (base + offset + PAGE - 1) // PAGE * PAGE - base
        table += _string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        table += struct.pack("<IQ", t, offset)
        placed.append((offset, writer, size))
        offset += size
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), n_kv) + kv + table
    assert len(header) == head
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(header + bytes(base - len(header)))
        for offset, writer, size in placed:
            f.write(bytes(base + offset - f.tell()))
            start = f.tell()
            writer(f)
            assert f.tell() - start == size, "tensor writer wrote the wrong size"
    tmp.rename(out)


def _copier(src: Gguf, name: str):
    def write(f):
        with open(src.path, "rb") as s:
            s.seek(src.tensors[name][2])
            left = src.nbytes(name)
            while left:
                chunk = s.read(min(COPY, left))
                assert chunk, f"{src.path}: truncated in {name}"
                f.write(chunk); left -= len(chunk)
    return write


def _slot(source: Gguf, layer: int, e: int) -> int:
    pool = source.meta.get(f"picoforge.pool.blk.{layer}")
    if pool is None:
        return e
    hit = np.nonzero(np.asarray(pool) == e)[0]
    assert hit.size == 1, f"{source.path}: expert {e} of layer {layer} is not in the pool"
    return int(hit[0])


def _experts(src: Gguf, slots: list, name: str):
    """Writer of the experts in `slots` of tensor `name`, concatenated in that order."""
    def write(f):
        with open(src.path, "rb") as s:
            for slot in slots:
                f.write(src.expert(s, name, slot))
    return write


def rank(out: Path, logs: list, e: int) -> dict:
    """Gate mass per (layer, expert) summed over every captured row; order = decreasing mass,
    ties by expert id."""
    mass, rows = {}, 0
    for path in logs:
        for rec in read_act_log(path):
            rows += rec["rows"]
            for layer, (_, ids, g) in rec["layers"].items():
                np.add.at(mass.setdefault(layer, np.zeros(e)), ids.ravel(), g.ravel().astype(np.float64))
    result = {"rows": rows, "experts": e,
              "sources": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in logs},
              "layers": {str(l): {"order": np.argsort(-m, kind="stable").tolist(), "mass": m.tolist()}
                         for l, m in sorted(mass.items())}}
    out.write_text(json.dumps(result) + "\n")
    return result


def pool(q4: Path, ranking: dict, count: int, out: Path) -> None:
    src = Gguf(q4)
    kv, n_kv, tensors = _kv("general.alignment", src.align), 1, []
    for layer in src.trunk_layers():
        chosen = ranking["layers"][str(layer)]["order"][:count]
        kv += _kv(f"picoforge.pool.blk.{layer}", chosen); n_kv += 1
        for part in PARTS:
            name = f"blk.{layer}.ffn_{part}_exps.weight"
            t, dims, _ = src.tensors[name]
            per = src.nbytes(name) // dims[-1]
            tensors.append((name, t, dims[:-1] + [count], _experts(src, chosen, name), per * count, False))
    _write(out, kv, n_kv, src.align, tensors)


def build(template: Path, source: Path, ranking: dict, counts: dict, out: Path, recipe: dict) -> dict:
    tpl, src = Gguf(template), Gguf(source)
    assert tpl.meta.get("picoforge.experts.mixed") is None, "the template is already mixed"
    kv = tpl.kv_bytes + _kv("picoforge.experts.mixed", True) + _kv("picoforge.experts.recipe", json.dumps(recipe))
    tensors, maps = [], {}
    for name, (t, dims, _) in tpl.tensors.items():
        layer = int(name.split(".")[1]) if name.startswith("blk.") else -1
        part = next((p for p in PARTS if name == f"blk.{layer}.ffn_{p}_exps.weight"), None)
        n = counts.get(layer, 0)
        if part is None or layer not in tpl.trunk_layers() or n == 0:
            tensors.append((name, t, dims, _copier(tpl, name), tpl.nbytes(name), tpl.tensors[name][2] % PAGE == 0))
            continue
        ne = dims[-1]
        hot = sorted(ranking["layers"][str(layer)]["order"][:n])
        cold = [x for x in range(ne) if x not in set(hot)]
        per = tpl.nbytes(name) // ne
        tensors.append((name, t, dims[:-1] + [len(cold)], _experts(tpl, cold, name), per * len(cold), False))
        hname = f"blk.{layer}.ffn_{part}_exps.weight"
        ht, hdims, _ = src.tensors[hname]
        assert hdims[1] == dims[1] and (part == "down" or hdims[0] == dims[0]), f"{hname}: shape mismatch"
        hper = src.nbytes(hname) // hdims[-1]
        slots = [_slot(src, layer, x) for x in hot]
        tensors.append((f"blk.{layer}.ffn_{part}_exps_hot.weight", ht, hdims[:-1] + [n], _experts(src, slots, hname), hper * n, False))
        if part == "down":
            values = np.empty(ne, "<f4")
            values[cold] = np.arange(len(cold)); values[hot] = -1 - np.arange(n)
            maps[layer] = values
            blob = values.tobytes()
            tensors.append((f"blk.{layer}.ffn_exps_map.weight", 0, [ne], lambda f, b=blob: f.write(b), len(blob), False))
    _write(out, kv, tpl.n_kv + 2, tpl.align, tensors)
    return {"hot": {str(l): sorted(np.nonzero(v < 0)[0].tolist()) for l, v in maps.items()}}


def counts_for(ranking: dict, layers, fraction=None, count=None) -> dict:
    e = ranking["experts"]
    n = count if count is not None else int(round(fraction * e))
    return {l: n for l in layers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--experts", type=int, required=True); r.add_argument("out", type=Path); r.add_argument("logs", type=Path, nargs="+")
    p = sub.add_parser("pool"); p.add_argument("q4", type=Path); p.add_argument("ranking", type=Path)
    p.add_argument("count", type=int); p.add_argument("out", type=Path)
    b = sub.add_parser("build"); b.add_argument("template", type=Path); b.add_argument("source", type=Path)
    b.add_argument("ranking", type=Path); b.add_argument("out", type=Path)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--fraction", type=float); g.add_argument("--count", type=int)
    a = ap.parse_args()
    if a.cmd == "rank":
        res = rank(a.out, a.logs, a.experts)
        print(f"{a.out}: {res['rows']} rows, {len(res['layers'])} layers, {res['experts']} experts")
    elif a.cmd == "pool":
        pool(a.q4, json.loads(a.ranking.read_text()), a.count, a.out)
        print(f"{a.out}: {a.out.stat().st_size} bytes")
    else:
        ranking = json.loads(a.ranking.read_text())
        counts = counts_for(ranking, Gguf(a.template).trunk_layers(), a.fraction, a.count)
        recipe = {"fraction": a.fraction, "count": a.count, "ranking_sha256": hashlib.sha256(a.ranking.read_bytes()).hexdigest(),
                  "ranking_rows": ranking["rows"], "template": a.template.name, "source": a.source.name}
        info = build(a.template, a.source, ranking, counts, a.out, recipe)
        a.out.with_suffix(".json").write_text(json.dumps({"recipe": recipe, **info}, indent=1) + "\n")
        print(f"{a.out}: {a.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
