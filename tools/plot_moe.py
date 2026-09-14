"""plot_moe.py — what one MoE block costs to verify, from the committed CSV.

    ./tools/venv/bin/python tools/plot_moe.py

GPU time of one Qwen3-30B-A3B-shaped MoE block (bench/moe_verify_cost_m5pro.csv)
against the distinct experts the verified tokens touch, one line per number of
tokens, one panel per expert format on a shared time axis. The fitted line and its marginal GB/s are computed here from the same
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
    formats = [f for f in ("bf16", "q8_row") if any(r["format"] == f for r in rows)]
    fig, axes = plt.subplots(1, len(formats), figsize=(11, 4.6), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    for ax, fmt in zip(axes[0], formats):
        mine = [r for r in rows if r["format"] == fmt]
        ax.set_facecolor(SURFACE)
        for n, color in RAMP.items():
            pts = sorted((int(r["distinct_experts"]), float(r["median_s"]) * 1e3)
                         for r in mine if int(r["n"]) == n)
            d, t = zip(*pts)
            ax.plot(d, t, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
                    marker="o", ms=8, mec=SURFACE, mew=2, label=f"{n} tokens", zorder=3)

        lines = []
        for n in (8, 32):
            sel = [r for r in mine if int(r["n"]) == n]
            d = np.array([int(r["distinct_experts"]) for r in sel])
            gb = np.array([float(r["expert_gb_read"]) for r in sel])
            t = np.array([float(r["median_s"]) for r in sel])
            slope_d, icpt = np.polyfit(d, t, 1)
            slope_gb = np.polyfit(gb, t, 1)[0]
            lines.append(f"{n:2d} tokens: {icpt * 1e3:.2f} ms + {slope_d * 1e6:.0f} us/expert"
                         f"  ({1 / slope_gb:.0f} GB/s marginal)")
        mb = float(mine[0]["expert_gb_read"]) / int(mine[0]["distinct_experts"]) * 1e3
        ax.set_title(f"{fmt} experts ({mb:.2f} MB each)", color=INK, fontsize=10, loc="left")
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK2, family="monospace")
        ax.set_xlabel("distinct experts touched by the verified tokens", color=INK2)
        ax.set_xticks([8, 16, 32, 48, 64, 96, 128])
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2)
    axes[0][0].set_ylabel("GPU time per MoE layer (ms, median of 25)", color=INK2)
    leg = axes[0][-1].legend(title="verified at once", loc="lower right", frameon=False, fontsize=9)
    plt.setp(leg.get_texts(), color=INK2)
    leg.get_title().set_color(INK2)
    fig.suptitle("Verifying one Qwen3-30B-A3B MoE layer on the M5 Pro: priced by distinct experts",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "bench/moe_verify_cost.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
