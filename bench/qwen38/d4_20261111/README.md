# Decode IV — paired short decode against DwarfStar (2026-11-11)

Real Qwen3.8-Flash-Next Q2, M5 Pro 64 GB, AC power. Harness `tools/eval/qwen38_compare.py run`
(suite `../r0_20261017/suite`, short frontier, 128 forced appends per process, 3 warm-ups + 20
measured per engine, alternating). DwarfStar `fc80bd6`. Code: branch `perf/qwen38-decode4` at fe78e0f.

| engine | median tok/s | p10 | p90 |
|---|---:|---:|---:|
| picoforge | 30.03 | 29.67 | 30.19 |
| DwarfStar | 32.67 | 31.97 | 32.91 |

**0.919x** (Decode III: 0.861x). A first attempt stopped at sample 10 on the memory guard (system
pressure 2 with ~6 GB already compressed by other applications, not the engine); the rerun is clean.

Changes since Decode III (A/B on the same machine, 41-ID prompt + 128 forced, ms/token):

| commit | change | bits | effect |
|---|---|---|---|
| 8c66dc3 | Q8_0 matvec: 4 lanes per block, 2-byte code loads, d * sum(code x) | 8.4e-7 scaled | 35.32 -> 34.20 (metallib A/B) |
| 5ebb653 | GDN conv: check + conv + history in one dispatch | identical | gdn.conv 735 -> 501 us |
| 814abe5 | HC read RMSNorm folded into the F16 down | 9.2e-7 scaled | GPU groups 34.54 -> 33.26 |
| 5c7cfca | HC_ADD in the GDN/QSA output projections | identical | 48 dispatches fewer |
| 52cf29f | HC_ADD in the MoE parallel-slot down | identical | 48 dispatches fewer |
| b1272b7, fe78e0f | router folded into gate+up: **negative** (33.53 -> 35.56), reverted | identical | — |

Whole branch against main b45693d: 35.44 -> 33.20 ms/token (5 alternating pairs).
Gates at fe78e0f: fast kernels 97/97, c-gpu, g2-parts, g2, g2-state, cli, session, mtp,
speculative, repl, server, compare green; G3 pass_fp32 scaled 2.62e-7, argmax 12/12.
