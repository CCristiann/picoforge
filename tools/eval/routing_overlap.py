"""routing_overlap.py — how many experts do w consecutive tokens touch?

    ./tools/venv/bin/python tools/eval/routing_overlap.py TRACE.bin OUT.csv [FORMAT]

The number a speculative verify step is priced by (bench/moe_verify_cost_m5pro.csv:
per MoE layer, a + b * distinct experts, almost independent of tokens). Reads a
trace written by `picoforge MODEL --routing-trace`, and for windows of w
consecutive tokens (w = 1..32, non-overlapping) reports, averaged over windows
and MoE layers:

  distinct     experts the window touches, measured
  uniform      what w independent tokens would touch under uniform routing,
               E * (1 - (1 - k/E)^w) -- the pessimist's baseline
  est. ms      MoE time of one verify step over all MoE layers, from the cost
               model fitted to the committed CSV for FORMAT at the nearest
               measured token count. ARITHMETIC, not a measurement: it
               assumes every layer costs what the synthetic layer did.
  x decode     that estimate over the w = 1 estimate: what a verify of w
               tokens costs in decode steps. Speculation pays only if a pass
               accepts more tokens than this.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_trace(path: Path):
    raw = path.read_bytes()
    t, layers, k, experts = (int(x) for x in np.frombuffer(raw[:16], dtype=np.uint32))
    return np.frombuffer(raw[16:], dtype=np.uint16).reshape(t, layers, k), k, experts


def cost_model(fmt: str):
    rows = [r for r in csv.DictReader(open(ROOT / "bench/moe_verify_cost_m5pro.csv"))
            if r["format"] == fmt]
    fits = {}
    for n in sorted({int(r["n"]) for r in rows}):
        sel = [r for r in rows if int(r["n"]) == n]
        if len(sel) < 2:
            continue
        d = np.array([int(r["distinct_experts"]) for r in sel], dtype=float)
        t = np.array([float(r["median_s"]) for r in sel])
        fits[n] = np.polyfit(d, t, 1)                        # (b, a)
    return fits


def main() -> None:
    trace, k, E = read_trace(Path(sys.argv[1]))
    out, fmt = Path(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "q8_row")
    moe = [l for l in range(trace.shape[1]) if (trace[:, l, :] != 0xFFFF).any()]
    fits = cost_model(fmt)
    print(f"{trace.shape[0]} tokens, {len(moe)} MoE layers, top-{k} of {E}; cost model: {fmt}")
    print("   w   distinct  uniform  ratio   est. ms   x decode")

    results, base_ms = [], None
    for w in (1, 2, 4, 8, 16, 32):
        windows = trace.shape[0] // w
        if windows == 0:
            break
        per = [len(np.unique(trace[i * w:(i + 1) * w, l, :])) for i in range(windows) for l in moe]
        distinct = float(np.mean(per))
        uniform = E * (1 - (1 - k / E) ** w)
        n_fit = min(fits, key=lambda n: abs(np.log2(n) - np.log2(w)))
        b, a = fits[n_fit]
        est = len(moe) * (a + b * distinct) * 1e3
        base_ms = base_ms or est
        results.append((w, windows, distinct, uniform, est, est / base_ms))
        print(f"  {w:2d}   {distinct:7.1f}  {uniform:7.1f}  {distinct / uniform:5.2f}  "
              f"{est:8.1f}   {est / base_ms:6.2f}")

    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window", "windows", "mean_distinct_experts", "uniform_expectation",
                     "est_moe_verify_ms", "est_x_decode", "cost_format"])
        for w, n, d, u, ms, x in results:
            wr.writerow([w, n, f"{d:.3f}", f"{u:.3f}", f"{ms:.3f}", f"{x:.3f}", fmt])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
