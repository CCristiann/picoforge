# P4 — the GDN scan: addresses out of the token loop (2026-11-08)

Branch `perf/qwen38-gdn-chunk`, kernel commit fe3c1c1. Machine: M5 Pro, AC (battery
79% → 75% "charging"), awake, one `caffeinate -i -s` over all runs; every model run
through `tools/qwen38_gpu_lock.py` + `build/r5-checks/launch-any.py` (release wait,
memory guard). No other model process (`pgrep -fl 'ds4|picoforge'`).

## Method

"Before" and "after" are the **same `picoforge` binary** with two metallibs
(`--metallib`): the one built at 90a3393 and the one at fe3c1c1. Only
`src/qwen38_recurrent.metal` changed, so the pair isolates the kernel. Real Q2
checkpoint, the R3 8K prompt (8063 IDs + 16 teacher-forced, context 8192, chunk 1024)
and its first 1000 IDs + 16 for the profile. `scripts/p4run.sh` runs everything,
`scripts/summ.py` writes `summary.txt`. Logits (`--logits`) were written outside the
repository and compared by `summ.py`; they are not committed.

## Isolated kernel

`build/qwen38-r1/test_fast_kernels scanbench fast 1024 16 48 128 128 1` (3 warmup + 20
measured command buffers, median):

| Variant | ms / layer |
|---|---|
| 90a3393 | 24.7 |
| ablation A: addresses hoisted, P = dk/32 at run time | 9.9 |
| ablation B: + P = 4 at compile time (`scripts/ablation_constP.metal.txt`) | 4.9 |
| ablation C: B without the two `simd_sum` (wrong result, floor only; `scripts/ablation_no_simd_sum.metal.txt`) | 4.4 |
| fe3c1c1 (template P, every address hoisted) | 3.68 |
| fe3c1c1 with 32 / 64 / 256 threads per group (kept 128) | 3.61 / 3.62 / 3.69 |

Ablation C says the cross-lane reduction latency is ~10% of the kernel: the rest is
per-element work. A chunked (WY/UT) scan in FP32 does 1.5-2x the arithmetic on the same
ALUs, so its ceiling here is ~1-2 ms per layer per 1024 tokens.

Bit identity old vs new kernel: `scripts/scan_bits.py`, 20 seeded cases (dk 32..256,
P 1..8, 1..333 tokens): error word, outputs and state byte-equal.

## Model (Q2, real)

| Run | before | after |
|---|---|---|
| profile N 1000 FP32: gdn.scan | 0.89 s | 0.13 s |
| profile N 1000 FP32: sum of groups | 6.71 s | 5.92 s |
| profile N 1000 half: gdn.scan | 0.89 s | 0.13 s |
| profile N 1000 half: sum of groups | 3.73 s | 2.98 s |
| 8K prefill FP32, pair 1 / pair 2 | 69.09 / 60.55 s | 55.10 / 54.44 s |
| 8K prefill half, pair 1 / pair 2 | 36.67 / 36.14 s | 31.59 / 30.30 s |

Logits before vs after: **bit-identical** in all 6 pairs (4,221,440 floats each, FP32
and half). The pair-1 FP32 "before" (69.1 s) is the session's first 8K run and an
outlier against R3's 61.0 s and pair 2. Pair 2: −6.1 s FP32 (148 t/s) and −5.8 s half
(266 t/s), consistent with 36 GDN layers × 8 chunks × ~21 ms saved per 1024-token chunk.

Against ds4's 365.9 t/s of the R3 session (not re-paired today): 0.40x FP32, 0.73x half
(R3: 0.36x / 0.59x). Protocol deviation: 2 alternating pairs per mode instead of ≥ 20
repetitions (an 8K prefill takes a minute); powermetrics needs sudo, `pmset` instead.
