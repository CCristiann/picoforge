# Prefill-gap branch — paired short decode against DwarfStar (2026-11-15)

Real Qwen3.8-Flash-Next Q2, M5 Pro 64 GB, AC power, swap delta 0. Harness `tools/eval/qwen38_compare.py run`
(suite `../r0_20261017/suite`, short frontier, 128 forced appends per process, 3 warm-ups + 20 measured per
engine, alternating). DwarfStar `fc80bd6`. Code: branch `perf/qwen38-prefill-gap` at 1f8e407 (Decode VII +
the prefill work; decode kernels as Decode VII).

| engine | median tok/s | p10 | p90 |
|---|---:|---:|---:|
| picoforge | 32.15 | 31.84 | 32.23 |
| DwarfStar | 33.19 | 32.90 | 33.27 |

**0.969x** (Decode IV: 0.919x). The first pairing of Decode VI + VII. During the run the binary was relinked
once with the `--power` option (default 100, no change on this path).

## 8K frontier (`paired-8k`)

Same harness and protocol, 8K frontier (8063-token prompt, then 128 forced appends), 3 warm-ups + 20 measured per
engine, alternating, AC, swap delta 0. Code: 1a4081a (materialized prefill attention, default chunk 1024).

| engine | decode median tok/s | p10 | p90 |
|---|---:|---:|---:|
| picoforge | 29.04 | 28.87 | 31.27 |
| DwarfStar | 31.06 | 30.47 | 32.14 |

**0.935x** (Decode IV: 0.921x). picoforge's prefill of the 8K prompt in the same 20 samples (FP32, chunk 1024):
median **26.86 s**, p10 26.06, p90 27.79 (session start: 46.2 s single run). ds4-bench reports no prefill time here;
DwarfStar's ~14 s is the R3 reference.
