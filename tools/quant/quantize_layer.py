"""quantize_layer.py — quantize the routed experts of one layer from the BF16 checkpoint.

gate-up: the checkpoint's fused gate_up_proj [experts, 2*ff, hidden] becomes
blk.L.ffn_gate_exps / ffn_up_exps in IQ2_XXS. down: down_proj [experts, hidden,
ff] becomes blk.L.ffn_down_exps in Q2_K, its ff inputs padded with zeros (and
zero importance) to the physical width the engine expects. Importance per
expert comes from the imatrix. Modes: compat (byte-identical to DwarfStar,
hence to the published Q2) and o1 (IQ2_XXS refinement; Q2_K stays compat).

  quantize_layer.py --source DIR --revision SHA --imatrix PATH --layer L \
      --projection {gate-up,down} --mode {compat,o1} --out DIR \
      [--experts N] [--threads 12] [--check-template Q2.gguf] [--physical-input N]

Missing BF16 tensors are fetched into DIR/tensors (needs DIR/model.safetensors.index.json).
"""

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qw4q  # noqa: E402
from fetch import fetch_tensor  # noqa: E402
from gguf_header import read_header  # noqa: E402
from imatrix import Imatrix, file_sha256  # noqa: E402
from manifest import Manifest  # noqa: E402

UNSLOTH_IMATRIX_SHA = "a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154"
MODES = {"compat": qw4q.COMPAT, "o1": qw4q.REFINE}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--revision", required=True)
    p.add_argument("--imatrix", type=Path, required=True)
    p.add_argument("--imatrix-sha", default=UNSLOTH_IMATRIX_SHA)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--projection", choices=("gate-up", "down"), required=True)
    p.add_argument("--mode", choices=tuple(MODES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--experts", type=int, default=0, help="first N experts only (0 = all)")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--check-template", type=Path)
    p.add_argument("--physical-input", type=int, default=0)
    return p.parse_args(argv)


def text_config(source: Path) -> dict:
    cfg = json.loads((source / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    return {"experts": cfg["num_experts"], "hidden": cfg["hidden_size"], "ff": cfg["moe_intermediate_size"]}


def load_bf16(args, name: str, shape: tuple) -> np.ndarray:
    path = args.source / "tensors" / f"{name}.bin"
    if not path.exists():
        fetch_tensor(args.repo, args.revision, args.source / "model.safetensors.index.json", name, path.parent)
    side = json.loads(path.with_suffix(".json").read_text())
    if side["dtype"] != "BF16" or tuple(side["shape"]) != shape:
        raise ValueError(f"{name}: expected BF16 {shape}, found {side['dtype']} {side['shape']}")
    values = np.memmap(path, "<u2", "r", shape=shape)
    if ((values & 0x7F80) == 0x7F80).any():   # exponent all ones: inf or NaN
        raise ValueError(f"{name}: non-finite values in the BF16 source")
    return values


def to_f32(u16: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((u16.astype("<u4") << 16).view("<f4"))


def outputs(args, cfg) -> list:
    """(gguf tensor, BF16 source name, row slice, ggml type)."""
    L, ff = args.layer, cfg["ff"]
    base = f"model.language_model.layers.{L}.mlp.experts."
    if args.projection == "gate-up":
        return [(f"blk.{L}.ffn_gate_exps.weight", base + "gate_up_proj", slice(0, ff), qw4q.IQ2_XXS),
                (f"blk.{L}.ffn_up_exps.weight", base + "gate_up_proj", slice(ff, 2 * ff), qw4q.IQ2_XXS)]
    return [(f"blk.{L}.ffn_down_exps.weight", base + "down_proj", slice(None), qw4q.Q2_K)]


def physical_width(args, cfg, template_meta) -> int:
    if args.projection == "gate-up":
        return cfg["hidden"]
    width = args.physical_input or int(template_meta.get("ds4.qwen4.down.physical_input", 0))
    if width < cfg["ff"] or width % 256:
        raise ValueError(f"down projection needs a physical input width >= {cfg['ff']} and a multiple of 256")
    return width


def encode_expert(values, rows, e, name, qtype, width, mode, imatrix) -> bytes:
    x = to_f32(values[e, rows, :])
    importance = imatrix.expert_importance(name, e, x)
    if width > x.shape[1]:   # zero padding carries zero importance
        x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
        importance = np.pad(importance, (0, width - importance.shape[0]))
    return qw4q.quantize(qtype, x, importance, mode if qtype == qw4q.IQ2_XXS else qw4q.COMPAT)


def check(template: Path, tensors: dict, name: str, data: bytes, n: int) -> int:
    t, dims, offset = tensors[name]
    per = len(data) // n
    same = 0
    with open(template, "rb") as f:
        for e in range(n):
            f.seek(offset + e * per)
            same += f.read(per) == data[e * per:(e + 1) * per]
    print(f"{name}: {same}/{n} experts byte-identical to {template.name}", flush=True)
    return same


def main(argv) -> int:
    args = parse(argv)
    cfg = text_config(args.source)
    n = args.experts or cfg["experts"]
    meta, tensors = read_header(args.check_template) if args.check_template else ({}, {})
    width = physical_width(args, cfg, meta)
    imatrix = Imatrix(args.imatrix, args.imatrix_sha)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load_or_create(args.out / "manifest.json", {
        "repo": args.repo, "revision": args.revision, "imatrix_sha256": args.imatrix_sha,
        "mode": args.mode, "library_sha256": file_sha256(qw4q.LIB_PATH)})
    shape = (cfg["experts"], 2 * cfg["ff"], cfg["hidden"]) if args.projection == "gate-up" \
        else (cfg["experts"], cfg["hidden"], cfg["ff"])
    failed = False
    for name, source, rows, qtype in outputs(args, cfg):
        path = args.out / f"{name}.bin"
        entry = manifest.data["files"].get(name, {})
        if manifest.done(name, path) and entry.get("experts") == n:
            print(f"{name}: verified, skipped", flush=True)
            data = path.read_bytes()
        else:
            values = load_bf16(args, source, shape)
            with cf.ThreadPoolExecutor(args.threads) as pool:
                parts = list(pool.map(lambda e: encode_expert(values, rows, e, name, qtype, width,
                                                              MODES[args.mode], imatrix), range(n)))
            data = b"".join(parts)
            part = path.with_suffix(".bin.part")
            part.write_bytes(data)
            part.replace(path)
            manifest.record(name, path, experts=n, layer=args.layer, projection=args.projection,
                            ggml_type=qtype, physical_input=width)
            print(f"{name}: {n} experts, {len(data)} bytes", flush=True)
        if args.check_template:
            failed |= check(args.check_template, tensors, name, data, n) != n and args.mode == "compat"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
