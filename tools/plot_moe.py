"""plot_moe.py — what one MoE block costs to verify, from the committed CSV.

    ./tools/venv/bin/python tools/plot_moe.py

GPU time of one Qwen3-30B-A3B-shaped MoE block (bench/moe_verify_cost_m5pro.csv)
against the distinct experts the verified tokens touch, one line per number of
tokens. The fitted line and its marginal GB/s are computed here from the same
rows, so the annotation cannot drift from the data.

Colour is an ordinal blue ramp (tokens verified is an ordered quantity), five
steps validated for monotone lightness, visible step gaps and contrast against
the surface. n = 1 is a single point (8 experts) and is left to the CSV.

Writes bench/moe_verify_cost.png.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
RAMP = {2: "#86b6ef", 4: "#3987e5", 8: "#256abf", 16: "#184f95", 32: "#0d366b"}


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "bench/moe_verify_cost_m5pro.csv")))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for n, color in RAMP.items():
        pts = sorted((int(r["distinct_experts"]), float(r["median_s"]) * 1e3)
                     for r in rows if int(r["n"]) == n)
        d, t = zip(*pts)
        ax.plot(d, t, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", ms=8, mec=SURFACE, mew=2, label=f"{n} tokens", zorder=3)

    eight = [r for r in rows if int(r["n"]) == 8]
    gb = np.array([float(r["expert_gb_read"]) for r in eight])
    d = np.array([int(r["distinct_experts"]) for r in eight])
    t = np.array([float(r["median_s"]) for r in eight])
    slope_d, icpt = np.polyfit(d, t, 1)
    slope_gb = np.polyfit(gb, t, 1)[0]
    ax.text(0.02, 0.97,
            f"8 tokens: {icpt * 1e3:.2f} ms + {slope_d * 1e6:.0f} us per distinct expert\n"
            f"each extra expert costs its bytes at {1 / slope_gb:.0f} GB/s "
            f"({100 / slope_gb / 307:.0f}% of 307)",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)

    ax.set_xlabel("distinct experts touched by the verified tokens", color=INK2)
    ax.set_ylabel("GPU time per MoE layer (ms, median of 25)", color=INK2)
    ax.set_title("Verifying a Qwen3-30B-A3B MoE layer on the M5 Pro: priced by experts, not tokens",
                 color=INK, fontsize=11, loc="left")
    ax.set_xticks([8, 16, 32, 48, 64, 96, 128])
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(title="verified at once", loc="lower right", frameon=False, fontsize=9)
    plt.setp(leg.get_texts(), color=INK2)
    leg.get_title().set_color(INK2)
    fig.tight_layout()
    out = ROOT / "bench/moe_verify_cost.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
