# Decode III — paired short decode against DwarfStar (2026-11-11)

Real Qwen3.8-Flash-Next Q2, M5 Pro 64 GB, AC power, no swap growth. Harness
`tools/eval/qwen38_compare.py run` (suite `../r0_20261017/suite`, short frontier, 128 forced
one-token appends per process, 3 warm-ups + 20 measured per engine, alternating, block 1).
DwarfStar `fc80bd6`. Code: branch `perf/qwen38-decode3` at 417e558.

| engine | median tok/s | p10 | p90 |
|---|---:|---:|---:|
| picoforge | 28.33 | 28.14 | 28.41 |
| DwarfStar | 32.90 | 32.30 | 33.10 |

**0.861x** (Decode II closure: 0.782x, 25.97 tok/s). Raw samples in `paired-short/`.

8K frontier (8063 IDs prefilled, then 128 forced appends), same protocol, run on main b45693d
(the merged branch), `paired-8k/`:

| engine | median tok/s | p10 | p90 |
|---|---:|---:|---:|
| picoforge | 27.64 | 27.31 | 27.71 |
| DwarfStar | 31.97 | 31.45 | 32.06 |

**0.865x** (R2: 0.721x; 8K was not re-measured in Decode II).

What changed since Decode II (A/B on the same machine, 41-ID prompt + 128 forced, ms/token):

| commit | change | bits | decode |
|---|---|---|---|
| 6cfe0a7 | router in one simdgroup, tree exp sum | 1.1e-6 scaled | moe.route 1295 -> 900 us |
| 24615d2 | routed down, slots in parallel | identical | moe.down 3166 -> 2927 us |
| 48e27ef | shared expert beside router and gate+up (2 barriers fewer per layer) | identical | 37.83 -> 37.24 |
| f7aace4 | HC tail, 8 lanes per stream | 1.3e-6 scaled | 37.77 -> 36.39 |
| f1f5f86 | IQ2_XXS gate+up, table decode, coalesced x | 7.9e-7 scaled | 36.42 -> 35.74 |
| a2e81d6 | wider decoder loads, factored Q2_K down dot | 8.7e-7 scaled | 35.71 -> 35.52 |

Gates: fast kernels 96/96, g2-parts, g2, g2-state, cli, session, mtp, speculative, repl, server,
compare green; G3 pass_fp32 scaled 2.53e-7, argmax 12/12. `Q38_REF_KERNELS=exact` restores the
pre-Decode-III kernels.
