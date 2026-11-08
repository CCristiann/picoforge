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

# P4.2-P4.3 — grouped MoE in 32-row tiles (same day, later)

Commits db2b0b3 (FP32) and 99a6187 (half). A 1024-token chunk routes ~20 rows to each of
the 512 experts; the 64-row tiles multiplied ~70% padding rows.

## Isolated kernel

`scripts/gb.sh LIB gbench grouped 16 640 2560 2560 512 10240 10 1` (IQ2_XXS gate+up) and
`ibench grouped 10 2560 768 640 512 10240 1 1` (Q2_K down), 3 + 20, median ms:

| Variant | gate+up | down |
|---|---|---|
| FP32 64x32 (main) | 39.5 | 25.7 |
| ablation A: x row addresses once per tile (`scripts/moe-ablations/A_xrow.metal.txt`) | 38.6 | 26.7 |
| ablation B: no weight decode (wrong, cost only) | 37.8 | 26.2 |
| ablation C: no MMA (wrong, cost only) | 24.2 | 27.4 |
| FP32 32x64 (db2b0b3) | 22.9 | 16.0 |
| half 64x64 (main) | 16.8 | 11.1 |
| half 32x64 (99a6187) | 11.5 | 7.1 |

Old vs new outputs byte-equal in FP32 and half: `scripts/grouped_bits.py` (4 routed
formats x 3 shapes x gate+up/down, partial and multiple tiles).

## Model (Q2, real)

Before = binary + metallib built at main 3039ce4, after = 99a6187 (`scripts/p5run.sh`,
`moe-tiles/summary.txt`). Battery 75% -> 73% "charging", awake.

| Run | before | after |
|---|---|---|
| profile N 1000 FP32: moe.gate_up / moe.down | 1.47 / 1.02 s | 1.07 / 0.74 s |
| profile N 1000 FP32: sum of groups | 5.94 s | 5.27 s |
| profile N 1000 half: moe.gate_up / moe.down | 0.65 / 0.44 s | 0.51 / 0.33 s |
| profile N 1000 half: sum of groups | 3.00 s | 2.74 s |
| 8K prefill FP32, pair 1 / 2 | 63.67 / 54.51 s | 50.24 / 49.36 s |
| 8K prefill half, pair 1 / 2 | 30.76 / 30.33 s | 28.64 / 28.86 s |

Logits before vs after bit-identical in all 6 pairs. The real gain (gate+up -27%) is
below the isolated one (-42%): the real routing is skewed, so fewer tiles were mostly
padding. Pair 1 "before" FP32 is again the session's first 8K run (slow). Against ds4's
365.9 t/s (R3 session): 8K FP32 163 t/s = 0.45x, half 279 t/s = 0.76x.

FP32 ceiling note: gdn.proj runs at ~3.4 TFLOPS useful, the 32x64 MoE near 4 TFLOPS
executed, against the 4.23 TFLOPS simdgroup peak of Phase 2: in FP32 the GEMMs are
close to this machine's limit; the large remaining factor is half/TensorOps.
