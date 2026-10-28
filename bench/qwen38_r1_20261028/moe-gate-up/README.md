# R1.9b fused gate+up+SwiGLU — protocol bench (2026-10-28)

Command: `make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261028/moe-gate-up` at
08d8c62 (manifest.txt): gate and up tail chunks pooled over one round of 32 lanes (8a4a1f2).
Conditions (before.txt / after.txt): **battery power** (ruling R38, user-authorized;
`pmset -g batt` 88-89% discharging), `pmset -g therm` no warning, thermalState nominal.
Synthetic 512-expert tensors (never the checkpoint), 10 slots per token, 3 warm-ups + 20
measured command buffers, 20 encodes per buffer up to 4 tokens. GB/s = bytes of the fused
pass / median (the bench reuses the same experts in all encodes: SLC-warm, an upper bound).
type 10 Q2_K, 39 MXFP4, 16 IQ2_XXS, 12 Q4_K; m = rows (tokens x 10).

| type | mode | m | µs median (p10/p90) | GB/s |
|---|---|---|---|---|
| 16 | fused | 10 | 65.3 (65.2/65.9) | 129.9 |
| 16 | sequence | 10 | 81.5 (81.5/81.6) | 104.1 |
| 16 | fused | 40 | 251.2 (251.2/251.3) | 131.7 |
| 16 | sequence | 40 | 294.9 (294.9/295.0) | 112.2 |
| 16 | fused | 320 | 2020.0 (2019.5/2021.4) | 102.6 |
| 16 | sequence | 320 | 2360.3 (2359.7/2361.5) | 87.8 |
| 16 | fused | 1280 | 8076.0 (8075.2/8077.5) | 49.4 |
| 16 | sequence | 1280 | 9417.5 (9414.2/9420.9) | 42.4 |
| 12 | fused | 10 | 84.2 (84.2/84.3) | 219.3 |
| 12 | sequence | 10 | 101.5 (101.4/101.6) | 182.0 |
| 12 | fused | 40 | 330.1 (330.0/330.3) | 212.6 |
| 12 | sequence | 40 | 373.6 (373.3/373.8) | 187.9 |
| 12 | fused | 320 | 2641.8 (2638.2/2647.2) | 168.6 |
| 12 | sequence | 320 | 2966.0 (2965.4/2967.5) | 150.2 |
| 12 | fused | 1280 | 10445.7 (10444.2/10448.2) | 81.4 |
| 12 | sequence | 1280 | 11715.4 (11711.0/11717.0) | 72.6 |

Reference (committed, AC power): bench/qwen38_r1_20261021/moe-gate-up, fused at 1 token
IQ2_XXS 67.7 µs (125 GB/s), Q4_K 87.2 µs. After (this folder, battery): 65.3 µs (130 GB/s)
and 84.2 µs (219 GB/s), about -3.5%, bits unchanged. Before = AC, after = battery, and no
same-session A/B was run (ruling R41): the gain is small enough to sit within condition
noise. The scratch runs at 461d528 (67.6 / 87.1 µs, AC, 2026-10-27) and the ablation
(36.9 µs with the IQ2_XXS decode replaced by a constant) were NOT recorded: no raw file.
The 200 GB/s target for IQ2_XXS is not reached.
