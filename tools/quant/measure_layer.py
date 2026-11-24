"""measure_layer.py — quantization error of quantized expert tensors against the BF16 source.

For each expert e of a tensor, with x the BF16 rows, q the decoded rows and
imp the imatrix importance per column:
  imatrix-weighted  sum imp * (x - q)^2       ~ expected output error, uncorrelated inputs
  encoder-weighted  sum imp * sqrt(sigma2 + x^2) * (x - q)^2   what the compat encoder minimizes
  unweighted        sum (x - q)^2
Prints each run's totals relative to the first (the baseline).

  measure_layer.py --source DIR --imatrix PATH --layer L --runs DIR_A DIR_B ... [--experts N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qw4q  # noqa: E402
from imatrix import Imatrix  # noqa: E402
from quantize_layer import UNSLOTH_IMATRIX_SHA, text_config, to_f32  # noqa: E402


def errors(x: np.ndarray, q: np.ndarray, imp: np.ndarray) -> np.ndarray:
    x64, d2 = x.astype(np.float64), (x.astype(np.float64) - q) ** 2
    sigma2 = (x64.reshape(x.shape[0], -1, 256) ** 2).mean(axis=2).repeat(256, axis=1)
    enc = imp * np.sqrt(sigma2 + x64 ** 2)
    return np.array([(imp * d2).sum(), (enc * d2).sum(), d2.sum()])


def main(argv) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--imatrix", type=Path, required=True)
    p.add_argument("--imatrix-sha", default=UNSLOTH_IMATRIX_SHA)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--experts", type=int, default=0)
    a = p.parse_args(argv)
    cfg = text_config(a.source)
    n = a.experts or cfg["experts"]
    im = Imatrix(a.imatrix, a.imatrix_sha)
    src = a.source / "tensors" / f"model.language_model.layers.{a.layer}.mlp.experts.gate_up_proj.bin"
    values = np.memmap(src, "<u2", "r", shape=(cfg["experts"], 2 * cfg["ff"], cfg["hidden"]))
    per = cfg["ff"] * cfg["hidden"] // 256 * 66
    totals = np.zeros((len(a.runs), 3))
    ratios = [[] for _ in a.runs]
    for part, proj in enumerate(("gate", "up")):
        name = f"blk.{a.layer}.ffn_{proj}_exps.weight"
        blobs = [np.memmap(r / f"{name}.bin", np.uint8, "r") for r in a.runs]
        for e in range(n):
            x = to_f32(values[e, part * cfg["ff"]:(part + 1) * cfg["ff"], :])
            imp = im.expert_importance(name, e, x).astype(np.float64)
            errs = [errors(x, qw4q.dequantize(qw4q.IQ2_XXS, bytes(b[e * per:(e + 1) * per]), cfg["ff"], cfg["hidden"]), imp)
                    for b in blobs]
            for i, err in enumerate(errs):
                totals[i] += err
                ratios[i].append(err[0] / errs[0][0])
    print(f"layer {a.layer}, {n} experts x gate+up, error relative to {a.runs[0].name} (lower is better)")
    print(f"{'run':28s} {'imatrix-weighted':>17s} {'encoder-weighted':>17s} {'unweighted':>11s} {'median/expert':>14s}")
    for i, r in enumerate(a.runs):
        rel = totals[i] / totals[0] - 1
        print(f"{r.name:28s} {rel[0]:+16.2%} {rel[1]:+16.2%} {rel[2]:+10.2%} {np.median(ratios[i]) - 1:+13.2%}")


if __name__ == "__main__":
    main(sys.argv[1:])
