"""Plot an expert-scheduler experiment against the unchanged adaptive baseline.

Usage: tools/venv/bin/python tools/plot_experts.py bench/EXPERIMENT.csv
Writes a PNG beside the CSV. Ratios use per-prompt medians, then a geometric
mean across prompts. No confidence interval is inferred from a pilot.
"""

import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    path = Path(sys.argv[1])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    table = {(r["prompt"], r["config"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, metric, title in zip(axes, ("median_tok_s", "median_process_s"),
                                 ("Decode throughput", "Complete request")):
        ax.set_facecolor("#fcfcfb")
        for config, label, color, offset in (("model-2", "fixed k=2", "#8b98ab", -.14),
                                            ("model-auto-experts", "auto + experts", "#256abf", .14)):
            ratios = []
            for prompt in prompts:
                current = float(table[prompt, config][metric])
                baseline = float(table[prompt, "model-auto"][metric])
                ratios.append(current / baseline if metric == "median_tok_s" else baseline / current)
            ratios.append(math.exp(statistics.mean(map(math.log, ratios))))
            y = [i + offset for i in range(len(ratios))]
            ax.scatter(ratios, y, color=color, s=50, label=label, zorder=3)
            for value, pos in zip(ratios, y):
                ax.annotate(f"{value:.2f}x", (value, pos), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8, color=color)
        ax.axvline(1, color="#52514e", lw=1, linestyle="--", label="auto baseline")
        ax.grid(axis="x", color="#e8e7e3")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("speedup vs auto (higher is better)", fontsize=9)
        ax.margins(x=.25)
        for spine in ax.spines.values():
            spine.set_color("#e8e7e3")
    axes[0].set_yticks(range(len(prompts) + 1), [p.replace("-", " ") for p in prompts] + ["geometric mean"])
    axes[0].invert_yaxis()
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    reps = sorted({r["reps"] for r in rows})
    fig.suptitle(f"Expert-cost scheduler pilot on Qwen3-30B-A3B q8 — {len(prompts)} prompts, "
                 f"{'/'.join(reps)} measured requests per configuration", fontsize=11)
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
