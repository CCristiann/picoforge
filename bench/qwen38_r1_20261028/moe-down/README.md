# R1.9b fused down+reduce — protocol bench (2026-10-28)

Command: `make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261028/moe-down` at 08d8c62
(manifest.txt), i.e. with the tail-chunk pooling of 382c17e in the down kernel.
Conditions (before.txt / after.txt): **battery power** (ruling R38, user-authorized;
`pmset -g batt` 88-89% discharging), `pmset -g therm` no warning, thermalState nominal.
Synthetic 512-expert tensors (never the checkpoint), 10 slots per token, 3 warm-ups + 20
measured command buffers, 20 encodes per buffer up to 4 tokens. GB/s = bytes of the fused
pass / median (the bench reuses the same experts in all encodes: SLC-warm, an upper bound).
type 10 Q2_K, 39 MXFP4, 16 IQ2_XXS, 12 Q4_K; m = rows (tokens x 10).

| type | mode | m | µs median (p10/p90) | GB/s |
|---|---|---|---|---|
| 10 | fused | 10 | 75.8 (75.7/76.1) | 72.4 |
| 10 | sequence | 10 | 82.0 (82.0/82.0) | 66.9 |
| 10 | fused | 40 | 277.9 (277.8/278.1) | 77.0 |
| 10 | sequence | 40 | 312.4 (312.3/312.5) | 68.5 |
| 10 | fused | 320 | 2201.4 (2198.6/2203.2) | 59.5 |
| 10 | sequence | 320 | 2478.4 (2477.8/2478.9) | 52.8 |
| 10 | fused | 1280 | 8980.0 (8961.7/8999.5) | 29.1 |
| 10 | sequence | 1280 | 9931.9 (9930.9/9933.2) | 26.3 |
| 39 | fused | 10 | 94.8 (94.6/95.0) | 92.4 |
| 39 | sequence | 10 | 99.5 (99.5/99.6) | 88.0 |
| 39 | fused | 40 | 355.6 (355.4/355.8) | 98.5 |
| 39 | sequence | 40 | 383.2 (383.1/383.3) | 91.5 |
| 39 | fused | 320 | 2802.2 (2799.7/2806.0) | 75.5 |
| 39 | sequence | 320 | 3026.4 (3025.5/3027.8) | 69.9 |
| 39 | fused | 1280 | 11224.8 (11212.4/11242.0) | 36.4 |
| 39 | sequence | 1280 | 12031.3 (12030.8/12032.6) | 34.0 |

Result: no gain. Fused Q2_K at 1 token 75.8 µs vs 75.3 at a5bf5f2 (2026-10-21, AC) and 75.4
for the restored original kernel measured right after on the same battery session; MXFP4
94.8 vs 94.4. The pooling was therefore reverted (the shipped down kernel is 461d528's).
The two rbench lines (q38f_route alone) are part of the target, unchanged.
