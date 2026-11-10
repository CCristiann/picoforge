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

# P5.2 — four splits per MMA threadgroup, 128 tokens per dispatch

Ablations on the P5.1 binary (`p52/profile/abl-*`, 8K half, qsa.core 5.46 s): indexer
scores ~0.63 s, merge ~0.84 s, keys past the first per split ~2.23 s, the rest ~1.8 s.

| run | total | qsa.core |
|---|---|---|
| P5.1 (`profile/after-half`) | 25.84 | 5.458 |
| 4 splits per threadgroup, online softmax (`p52/profile/grouped`) | 24.46 | 4.826 |
| + 128 tokens per split/merge dispatch (`p52/profile/grouped-128tok`) | 24.24 | 4.595 |

Kernel against the oracle (`scripts/msweep.py`): every m in 900..1100 plus a sweep to 2051,
232 cases, worst 2.3e-6, no error bit. Logits FP32 vs the R2.1 kernel (`p52/compare.jsonl`):
N 16..950 within 3.1e-6; N 1000 0.074 and 4000 0.043, argmax 17/17. The route logs at N 1000
(`p52/route/*.u32`, `Q38_ROUTE_LOG`: [pos, rows, layers, top_k] then [layer][row][top_k])
agree on every layer of tokens 0..219; the first different expert set is token 220, layer 19
(231 vs 366), the near-tie R3.1 found there (relative gap 4e-8); everything after follows.
G3 (`p52/g3.log`): pass_fp32, 2.17e-7, 12/12.

## P5.2 group size (consistent rebuilds; profile 8K half, prefill qsa.core / decode qsa.core)

The first G=1/G=2 attempt was invalid: the Makefile did not rebuild host objects on a
header change, so the host kept G=4 against a G=2/G=1 metallib (G=2 stopped on error bit 1).
Fixed in 163887f; `p52/group-*` are clean rebuilds.

| splits per threadgroup | prefill qsa.core | decode qsa.core (8K context) |
|---|---|---|
| 1 (`p52/group-1`) | 5.90 s | 2.18 ms/token |
| 2 (`p52/group-2`, chosen) | 5.01 s | 2.51 ms/token |
| 4 (`p52/group-4`) | 4.59 s | 3.52 ms/token |

G=2: -0.9 s per 8K prefill for +0.33 ms per token at 8K (break-even after ~2700 generated
tokens); G=4 breaks even after ~1000. Gates at G=2: attend_fast (max 2.23e-4), g2-parts, g2,
g2-state, MTP, speculative, d-gpu green; G3 2.17e-7, 12/12.

# P5.3 — batch indexer scores in tiles (bit-identical)

Tiles of 32 blocks x 8 tokens, q and k staged 32 features at a time; every dot keeps its
increasing-feature order and the head order of the clamped sum. Real Q2 FP32 logits at N 1000
and 8K equal byte for byte to the P5.2 binary (`p53/bits/logits.sha256`); g2-parts green.
Profile 8K half (`p53/profile-half`): qsa.core 5.014 -> 4.877 s (total 24.63 -> 24.64, noise).
8K FP32 prefill wall 46.01 / 46.12 s (one run each): no measurable end-to-end change.
