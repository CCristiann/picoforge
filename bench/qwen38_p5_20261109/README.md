# P5.1 — split attention on simdgroup matrices (2026-11-09)

Real Qwen3.8-Flash-Next Q2, the 8K prompt of P4 (`build/p4/tok8k.txt`, 8063 IDs +
16 teacher-forced), `--context 8192 --chunk 1024`, AC power, every run through the
guarded launcher under the GPU lock (`scripts/`). Base: main 78ea2a8.

## Where the 8K prefill went (profile, GPU seconds)

| run | total | qsa.core |
|---|---|---|
| before, half (`profile/before-profile8k-half`) | 28.31 | 8.654 |
| before, FP32 (`profile/before-profile8k-fp32`) | 48.78 | 8.654 |
| ablation: one key per split (`profile/abl-split`) | 22.53 | 2.978 |
| ablation: rank replaced by "first m blocks" (`profile/abl-rank`) | 28.17 | 8.378 |
| MMA v1, 29 KB threadgroup memory (`profile/v1-29kb-half`) | 30.98 | 10.987 |
| MMA final, 13 KB + shared id scan (`profile/after-half`) | 25.84 | 5.458 |

The ablation metallibs (`scripts/ablation_*.metal.txt`) produce wrong outputs: timing only.

## Logits (`logits/compare.jsonl`, G3-style report, gate 1e-3)

FP32, MMA kernel vs the R2.1 split kernel (`Q38_QSA_SPLIT=1`), first N IDs + 16 forced:
scaled 7.8e-7 (N 16) .. 2.1e-5 (N 5000), argmax 17/17; N 6000 0.067, N 7000 0.026,
8K 0.038, argmax 17/17 throughout. The step between 5000 and 6000 is the router near-tie
regime of R3.1: the GEMM reorder alone (`Q38_REF_KERNELS=gemm`, R2.1 split in both runs)
gives 0.069 at N 5000 and 0.077 at N 6000.

G3 (`g3.log`): pass_fp32, scaled 2.17e-7, argmax 12/12.

## 8K prefill wall (one run each, `logits/8k-*/runs.jsonl`, prefill wait_s)

| | MMA | R2.1 split |
|---|---|---|
| FP32 | 47.07 s | 50.66 s |
| half | 26.61 s | 29.76 s |

Single runs, not the 3+20 protocol: the closure bench of P5 will pair them.
