# Quant M3 — output-fitted gains and usage-aware mixed precision (2026-11-22)

All numbers: routed output error of a layer (y = sum_k g_k E_k(x), `tools/quant/layer_lab.py`
logic) on captured inputs of the R7 code windows (`bench/quant/m2_20261122`), relative to the
compat file (the published Q2). Anything fitted or ranked uses windows 0-1; errors are measured
on windows 2-3 (4096 tokens per layer) unless noted. Spike scripts (private): `notes/quant/m3_*.py`.
Raw outputs: `m3-gain.txt`, `m3-gain-all.txt`, `m3-mixed-split.txt`.

## 1. Gain corrections (format unchanged)

Gate and up quantized, down exact (`m3-gain.txt`):

| Layer | compat + row-LS | compat + expert-out | o1 | o1 + expert-out |
|---:|---:|---:|---:|---:|
| 4 | +19.3% | −0.4% | +22.2% | −0.4% |
| 28 | −4.5% | −6.2% | −5.2% | −5.4% |
| 44 | −2.5% | −6.4% | −5.8% | −6.2% |

row-LS (per output row, least squares on the projection) shrinks further and hurts; expert-out
(two gains per expert chosen on a grid for the expert's output error) helps the deep layers.
All three projections quantized, with a third, least-squares down gain (`m3-gain-all.txt`):
−0.1% (layer 4), −6.3% (28), −7.0% (44); median gains gate 0.98-1.00, up 0.94, down 0.92.
Compat relative error |dy|^2/|y|^2 with all projections quantized: 0.105, 0.199, 0.219.

## 2. Usage-aware mixed precision (needs a picoforge-only layout)

The hottest experts take the weights of the published Q4 file (Q4_K gate/up, MXFP4 down), the
rest stay compat. Ranking either by the Unsloth imatrix counts (general text) or by the gate mass
on our calibration code windows 0-1 (`m3-mixed-split.txt`):

| Layer | ranking | top 5% (26 experts) | top 10% (51) | top 20% (102) |
|---:|---|---:|---:|---:|
| 4 | Unsloth | −17.4% (mass 12.9%) | −18.7% (15.5%) | −29.0% (30.6%) |
| 4 | code | **−50.8%** (44.7%) | **−68.2%** (66.5%) | −76.1% (80.2%) |
| 28 | Unsloth | −3.0% (3.0%) | −7.2% (7.1%) | −12.1% (13.9%) |
| 28 | code | **−42.6%** (41.4%) | **−60.1%** (61.4%) | −75.2% (79.9%) |
| 44 | Unsloth | −5.9% (4.9%) | −8.2% (9.5%) | −12.4% (17.2%) |
| 44 | code | **−48.1%** (44.9%) | **−59.3%** (58.3%) | −67.0% (70.6%) |

"mass": share of the evaluation tokens' gate weight routed to the Q4 experts. The error falls
roughly in proportion to that mass, and code concentrates on few experts that general-text
statistics do not pick.

Memory: one expert is 1.49 MB in Q2 (IQ2_XXS gate/up + Q2_K down) and 2.71 MB in Q4, so top 10%
in every one of the 48 layers adds about 3.0 GB to the 41.7 GiB resident Q2 weights (5%: 1.5 GB).

## Limits

- The four windows are all picoforge code (C, ObjC, Metal, Python): held-out windows, but one
  repository and style. Generality to other code needs a broader corpus; prose will get less
  benefit (or none) from a code ranking — not measured yet.
- Three layers of 48; per-layer error, not end-to-end logits. The end-to-end check is R7 KL /
  top-1 / greedy agreement on a full build.
- Mixed formats inside one layer's experts are not supported by the picoforge loader or by
  DwarfStar today: an engine change (with the oracle chain) is needed before a file can run.
- Literature check pending before any novelty claim (frequency-aware bit allocation for MoE
  experts has prior art; the domain-specific ranking on a 64 GB Mac is what we would test).
