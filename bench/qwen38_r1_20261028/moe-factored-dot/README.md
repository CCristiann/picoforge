# R1.9c factored block dot + fp16 scale conversion — same-session A/B (2026-10-28)

**Shipped: only the fp16 scale conversion (5fbbdc7).** The factored Q2_K chunk of the
"after" tree (d62f163) was reverted by ruling R42 (exact cancellation: zero weights give
a nonzero output); its rows below are evidence of a design **not shipped (R42)**.

Before = 1f995d6 (the R1.9b tree), after = d62f163 (5fbbdc7 fp16 scales via the GPU
half->float conversion, bit-identical; d62f163 factored Q2_K chunk). Harness + metallib
built from each tree (sha256 in manifest.txt; the harness binary is the same, only the
metallib differs). `run.sh` runs both under one GPU lock, 3 rounds, order alternating per
round (B,A / A,B / B,A); each line is the harness's 3 warm-up + 20 measured command buffers
(20 encodes each), median and p10/p90 in runs.jsonl. Synthetic 512-expert tensors (never
the checkpoint), 10 slots per token; m = rows (tokens x 10). GB/s = bytes / median (the
same experts in every encode: SLC-warm, an upper bound).
Conditions (before.txt / after.txt): **battery power** (ruling R38; `pmset -g batt` 51%
discharging), `pmset -g therm` no warning, `caffeinate -i`.

Median of the 3 round medians. Spread in runs.jsonl: every p10 is within 0.4 us of its
median; p90 has outliers in some rounds, up to +53 us (IQ2_XXS before, round 3: median
65.4, p90 118.6; also Q2_K before round 2 p90 88.2, Q4_K after round 1 p90 88.9, Q2_K
after p90 up to +1.3 us). The medians are stable across rounds (within 0.2 us).

| kernel | m | before us (GB/s) | after us (GB/s) | change |
|---|---:|---:|---:|---:|
| down+reduce Q2_K 640/768 -> 2560 | 10 | 75.4 (72.7) | 60.1 (91.2) | -20.3% (factored: not shipped, R42) |
| down+reduce Q2_K | 40 | 280.3 (76.3) | 218.3 (98.0) | -22.1% (factored: not shipped, R42) |
| down+reduce MXFP4 640 -> 2560 | 10 | 94.3 (92.9) | 94.0 (93.2) | -0.3% (no fp16 scale) |
| gate+up IQ2_XXS 2560 -> 640 | 10 | 65.3 (129.8) | 61.6 (137.7) | -5.7% (fp16 only) |
| gate+up IQ2_XXS | 40 | 251.2 (131.7) | 237.4 (139.4) | -5.5% |
| gate+up Q4_K 2560 -> 640 | 10 | 84.2 (219.3) | 77.1 (239.5) | -8.4% (fp16 only) |

Split (pilot, same session, not in runs.jsonl): Q2_K down 75.4 -> 64.8 us from the fp16
conversion (the shipped state), 64.8 -> 60.1 us from the factoring (reverted, R42).
IQ2_XXS factored was measured (61.6 us, no gain) and not shipped. The IQ2_XXS and Q4_K rows
are the fp16 conversion alone (their chunks were never factored), so they are the shipped
numbers. Expected decode effect of the shipped state (synthetic): ~10.6 us x 47 Q2_K down
layers + ~3.7 us x 47 IQ2_XXS gate/up layers ~ 0.7 ms/token. Measured (controller,
profile + g3): pending.
