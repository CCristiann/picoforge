# R7 — quality of Q2 against Q4, 2026-11-10

Same model (Qwen3.8-Flash-Next), two quantisations of the routed experts: Q2 (the resident file)
and Q4 (Q4_K gate/up + MXFP4 down, streamed from SSD). Every other tensor is identical in the two
files. Q4 is the **reference**: it is a proxy, because the BF16 original does not fit in 64 GiB,
so every number below measures how far Q2 is from Q4, not from the original.

## Method

- **Corpus** (`tools/eval/qwen38_r7_corpus.py 33f6d86`): picoforge's own files added after
  2026-10-12, after Qwen3.8-Flash-Next shipped, so the model cannot have seen them. Two streams:
  prose (13 `.md` files, Italian and English, 35 707 tokens) and code (65 C/ObjC/Metal/Python
  files, 213 681 tokens). 4 windows of 2048 tokens per stream, evenly spaced. The window IDs have
  sha256 `c2c47468…` (`docs/corpus.lock`).
- **Runs** (`run.sh`): one fresh process per window and model, `--tokens W --teacher-forced 2047
  --context 2048 --logits`, which gives 2048 FP32 logits rows, of which 2047 have a next token.
  Each process ran through the guarded launcher under the GPU lock, on AC power. Binary
  `d6f3f5f2…` at 33f6d86, default FP32 kernels.
- **Metrics** (`tools/eval/qwen38_r7_quality.py`, FP64): per position KL(P_Q4 ‖ P_Q2), the top-1
  agreement and the NLL of the true next token for each model. The Q4 top-1/top-2 log-prob
  margin is kept to check flips against near-ties. 95% CIs come from a bootstrap over
  128-position blocks, because neighbouring positions are correlated. Per-position arrays are in
  `windows/*.npz`, the aggregate in `report.json`, and the sha256 of every raw logits file
  (deleted afterwards, about 2 GB each) is in `logits-sha256.txt`.
- **Tooling check**: on synthetic logits the KL equals a hand computation to 1e-12, identical
  files give KL 0 and top-1 1.0, and the row alignment holds on real data. Q4's argmax hits the
  true next token at 49.5% on prose-0, 2.1% if the rows were shifted by one.

## Results (16 376 scored positions)

| stream | KL mean [95% CI] | KL p50 / p90 / p99 / max | top-1 agreement [95% CI] | PPL Q4 → Q2 | ΔNLL nats [95% CI] |
|---|---|---|---|---|---|
| prose | 0.513 [0.462, 0.561] | 0.278 / 1.27 / 3.74 / 12.9 | 69.9% [68.4, 71.9] | 12.63 → 13.89 | 0.095 [0.067, 0.125] |
| code | 0.288 [0.248, 0.334] | 0.034 / 0.85 / 2.95 / 12.3 | 84.1% [82.0, 85.9] | 3.35 → 3.57 | 0.063 [0.045, 0.082] |
| all | 0.401 [0.367, 0.441] | 0.128 / 1.09 / 3.37 / 12.9 | 77.0% [75.1, 78.7] | 6.50 → 7.04 | 0.079 [0.063, 0.096] |

Per window: KL 0.36–0.59 and top-1 66–73% on prose, KL 0.19–0.39 and top-1 78–89% on code.
Q2's perplexity is higher in every one of the 8 windows (+5% to +15%).

Reading:
- **Q2 is measurably worse than Q4, not only different.** The ΔNLL CI excludes 0 in both
  streams: PPL +8% overall, +10% on prose, +7% on code.
- **The damage concentrates on uncertain positions.** On code the median KL is 0.03: most
  positions agree closely, and the mean comes from a tail (p90 0.85). Prose has a higher
  entropy, here PPL 12.6, so the tail is wider.
- **Flips are often near-ties.** For 46% of the positions where the top-1 changes, Q4's own top-2
  margin is below 0.5 nats (median 0.56). Those flips cost little NLL, which is why a 23%
  disagreement rate turns into only +0.08 nats.
- Not comparable with published KL tables: the corpus is our own, the reference is Q4 rather than
  BF16, and the windows are 2048 tokens.

## Quality / speed / memory

| | Q2 resident | Q4 streamed |
|---|---|---|
| expert weights / file | 41.7 GiB resident | 62.1 GiB experts in a 165 GiB file, pool ≤ 36 GiB |
| quality vs Q4 (this run) | KL 0.40, top-1 77%, PPL +8% | reference |
| decode, chat (PR #14, warm, with MTP) | ~34 tok/s | — |
| decode, R5.3b (128-token runs, 36 GiB pool) | — | ~18–19 tok/s warm |
| teacher-forced decode here (logits readback on) | 24 tok/s (2047 tok / ~85 s) | 15–18 tok/s at 14421 slots; 3.6–5.5 tok/s at 10000 slots |

The speed rows are not protocol benches: the Q4 protocol bench is the next roadmap item.

## Run notes

- Two Q4 starts (prose-3, code-1) were stopped by the memory guard: system pressure level 2 for
  2 samples while the 14421-slot pool (37.3 GiB) was being wired, with the desktop open (Chrome,
  Spotify, T3). Nothing was retried with a looser guard. The script now waits 15 s between
  processes, and the last three code windows used a 10000-slot pool (25.9 GiB,
  `PICOFORGE_STREAM_EXPERTS=10000`).
- **The pool size does not change the numbers.** code-0 re-run at 10000 slots gives Q4 logits
  byte-identical to the 14421-slot run (sha256 `c4243bfe…`, as in R5.3a).
- **Open anomaly:** at 10000 slots Q4 decoded 4.6x slower (550 s against 121 s for a window).
  Misses rose from about 2.4% to 8.1% (202 GiB read), but only 36 s were spent waiting in
  acquire. One visible difference: at 10000 slots the residency set is attached (33.7 GiB),
  while at 14421 it is skipped. To investigate before the Q4 protocol bench.
- Not done: the frozen task suite (`tools/eval/qwen38_cases.json`) graded on Q2 and Q4 greedy,
  and the comparisons between models, which wait for R6 (QN4).
