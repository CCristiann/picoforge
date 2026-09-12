"""plot_quant.py — Phase 3's three trade-offs, drawn from the committed CSVs.

    ./tools/venv/bin/python tools/plot_quant.py

  left    quality against size: dNLL (with its standard error) vs bits per
          weight, one point per format from bench/quant_quality_local.csv
  middle  what a format costs the matmul: time relative to bf16 across batch
          size M, from bench/matmul_quant_m5pro.csv
  right   whole forward passes: tokens/s for prefill, decode and 32-token
          verify, from bench/e2e_quant_m5pro.csv

Writes bench/quant_tradeoffs.png.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"


def rows(name):
    return list(csv.DictReader(open(BENCH / name)))


def main() -> None:
    fig, (ax_q, ax_k, ax_e) = plt.subplots(1, 3, figsize=(17, 5.2))

    for i, r in enumerate(sorted(rows("quant_quality_local.csv"), key=lambda r: float(r["dnll"]))):
        bpw, d, se = float(r["bpw"]), float(r["dnll"]), float(r["dnll_se"])
        color = "tab:blue" if r["format"].startswith("q8") else "tab:red"
        ax_q.errorbar(bpw, d, yerr=se, fmt="o", color=color, capsize=3)
        # Staggered labels: several formats share a bpw and nearly a dNLL.
        ax_q.annotate(r["format"], (bpw, d), textcoords="offset points",
                      xytext=(6, -10 + 9 * (i % 3)), fontsize=7)
    ax_q.axhline(0, color="gray", lw=0.8)
    ax_q.set_yscale("symlog", linthresh=0.01)
    ax_q.set_xlabel("bits per weight (projections + embedding, scales included)")
    ax_q.set_ylabel("dNLL vs fp32, nats/token (symlog)")
    ax_q.set_title("Quality: local corpus, 38 400 scored tokens")

    by_kernel = defaultdict(dict)
    for r in rows("matmul_quant_m5pro.csv"):
        by_kernel[r["kernel"]][int(r["M"])] = float(r["median_s"])
    base = by_kernel.pop("bf16")
    for kernel, series in sorted(by_kernel.items()):
        ms = sorted(series)
        ax_k.plot(ms, [series[m] / base[m] for m in ms], marker="o", label=kernel,
                  ls="-" if "row" in kernel else "--")
    ax_k.axhline(1, color="gray", lw=0.8)
    ax_k.set_xscale("log", base=2)
    ax_k.set_xlabel("batch M (rows of activations), N=3072, K=1024")
    ax_k.set_ylabel("matmul time / bf16 TensorOps time")
    ax_k.set_title("Kernel cost: one op call per row vs per block")
    ax_k.legend(fontsize=8)

    e2e = rows("e2e_quant_m5pro.csv")
    models = list(dict.fromkeys(r["model"] for r in e2e))
    regimes = ["prefill", "decode", "verify32"]
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [next(float(r["tok_s"]) for r in e2e if r["model"] == model and r["regime"] == g)
                for g in regimes]
        xs = [j + i * width for j in range(len(regimes))]
        bars = ax_e.bar(xs, vals, width, label=Path(model).name)
        for b, v in zip(bars, vals):
            ax_e.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax_e.set_xticks([j + 0.4 - width / 2 for j in range(len(regimes))])
    ax_e.set_xticklabels(["prefill 512", "decode @512", "verify 32 @512"])
    ax_e.set_yscale("log")
    ax_e.set_ylabel("tokens / s (GPU time, median of 25)")
    ax_e.set_title("End to end on the M5 Pro")
    ax_e.legend(fontsize=8)

    fig.tight_layout()
    out = BENCH / "quant_tradeoffs.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
