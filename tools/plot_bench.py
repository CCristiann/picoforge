"""plot_bench.py — draw the three sweeps from a benchmark CSV.

Committed alongside the raw numbers so the plots can be redrawn, and the
claims rechecked, by someone who was not in the room.

    ./tools/venv/bin/python tools/plot_bench.py bench/matmul_m5pro.csv
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PEAK_GBPS = 307.0          # M5 Pro memory bandwidth, the ceiling for decode
STYLE = {"naive": ("tab:red", "o"), "simd": ("tab:orange", "s"),
         "tensor": ("tab:blue", "^")}
LABEL = {"naive": "1. naive", "simd": "2. simdgroup", "tensor": "3. TensorOps"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("M", "N", "K"):
            r[k] = int(r[k])
        for k in ("median_s", "p10_s", "p90_s", "gflops", "gbps"):
            r[k] = float(r[k])
    return rows


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/matmul_m5pro.csv")
    rows = load(path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. batch sweep, GFLOP/s. The headline "one kernel, three implementations".
    ax = axes[0]
    for k in STYLE:
        d = sorted((r for r in rows if r["sweep"] == "batch-sweep" and r["kernel"] == k),
                   key=lambda r: r["M"])
        if not d:
            continue
        c, m = STYLE[k]
        ax.plot([r["M"] for r in d], [r["gflops"] / 1000 for r in d],
                color=c, marker=m, label=LABEL[k])
    ax.set_xscale("log", base=2)
    ax.set(xlabel="batch size M (tokens)", ylabel="TFLOP/s",
           title="Throughput vs batch\nN=3072, K=1024 (Qwen3-0.6B MLP)")
    ax.grid(alpha=.3); ax.legend()

    # 2. the same sweep in GB/s against the machine's ceiling. Decode lives at
    #    the far left of this plot, and the flat region is the finding.
    ax = axes[1]
    for k in STYLE:
        d = sorted((r for r in rows if r["sweep"] == "batch-sweep" and r["kernel"] == k),
                   key=lambda r: r["M"])
        if not d:
            continue
        c, m = STYLE[k]
        ax.plot([r["M"] for r in d], [r["gbps"] for r in d], color=c, marker=m,
                label=LABEL[k])
    ax.axhline(PEAK_GBPS, ls="--", color="k", lw=1)
    ax.text(1.2, PEAK_GBPS * 0.94, f"{PEAK_GBPS:.0f} GB/s theoretical", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="batch size M (tokens)", ylabel="effective GB/s",
           title="Bandwidth vs batch\nflat to M=32: the weights dominate, not the work")
    ax.grid(alpha=.3); ax.legend()

    # 3. width at M=1: the decode shape. Shows that the limit there is
    #    parallelism, not bandwidth, until N is large enough.
    ax = axes[2]
    for k in STYLE:
        d = sorted((r for r in rows if r["sweep"] == "width-at-M1" and r["kernel"] == k),
                   key=lambda r: r["N"])
        if not d:
            continue
        c, m = STYLE[k]
        ax.plot([r["N"] for r in d], [r["gbps"] for r in d], color=c, marker=m,
                label=LABEL[k])
    ax.axhline(PEAK_GBPS, ls="--", color="k", lw=1)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="output width N", ylabel="effective GB/s",
           title="Decode (M=1) vs output width\nshort of parallelism, not of bandwidth")
    ax.grid(alpha=.3); ax.legend()

    fig.suptitle("picoforge matmul — one kernel, three implementations — "
                 "Apple M5 Pro, median of 25 runs", fontsize=12)
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
