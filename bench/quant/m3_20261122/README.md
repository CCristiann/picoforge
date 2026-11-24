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

## 3. Does the code ranking generalize? (2026-11-23)

New captures, same layers, Q4 teacher-forced, one guarded run per window: the 4 R7 prose
windows and 12 windows of `build/quant/code-corpus.json` (`tools/quant/code_corpus.py`: six
permissive repositories at recorded commits, one language each; calibration llama.cpp C++,
flask Python, react JS; evaluation tokio Rust, cobra Go, guava Java). The code ranking is the
gate mass on the calibration repositories plus picoforge code windows 0-1; evaluation sets
never feed a ranking. Raw output: `m3-general.txt`.

| Layer | eval set | compat rel.err | Unsloth 5% | Unsloth 10% | code 5% | code 10% |
|---:|---|---:|---:|---:|---:|---:|
| 4 | other languages (12288 tok) | 0.089 | −1.7% | −3.9% | −36.1% | **−54.0%** |
| 4 | picoforge 2-3 (4096) | 0.105 | −17.4% | −18.7% | −54.5% | **−63.7%** |
| 4 | prose (8192) | 0.114 | −30.5% | −34.6% | −30.6% | **−37.9%** |
| 28 | other languages | 0.189 | −4.1% | −7.0% | −35.7% | **−52.3%** |
| 28 | picoforge 2-3 | 0.199 | −3.0% | −7.2% | −35.9% | **−56.7%** |
| 28 | prose | 0.206 | −7.0% | −13.8% | −6.0% | **−18.1%** |
| 44 | other languages | 0.219 | −7.9% | −15.0% | −25.7% | **−45.5%** |
| 44 | picoforge 2-3 | 0.219 | −5.9% | −8.2% | −46.5% | **−65.4%** |
| 44 | prose | 0.211 | −2.8% | −3.9% | −8.0% | **−16.5%** |

A ranking learned on C++/Python/JS code transfers to Rust/Go/Java, and it does not cost prose:
there it also beats the general-text ranking (the R7 prose is technical Markdown).

Literature (2026-11-23, abstracts only): frequency- or importance-aware bit allocation across
MoE experts is established (MC-MoE arXiv 2410.06270, MoPEQ 2509.02512, GEMQ, Colla-Q 2609.18131,
AlphaQ 2606.04980), and calibration-domain effects on code are reported (EAC-MoE 2508.01625,
AlphaQ). So the idea is not new; what we add is the measured effect on Qwen3.8-Flash-Next at
IQ2/Q4 in a GGUF engine that has to fit 64 GB.

## Limits

- Section 3 answers the single-repository concern for routing-level error; the corpus repos are
  public and probably in training data (fine for comparing quantizations).
- Three layers of 48; per-layer error, not end-to-end logits. The end-to-end check is R7 KL /
  top-1 / greedy agreement on a full build.
- Mixed formats inside one layer's experts are not supported by the picoforge loader or by
  DwarfStar today: an engine change (with the oracle chain) is needed before a file can run.
- Literature check pending before any novelty claim (frequency-aware bit allocation for MoE
  experts has prior art; the domain-specific ranking on a 64 GB Mac is what we would test).
