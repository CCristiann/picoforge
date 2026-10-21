# R1.5c fused down + weighted reduce — protocol bench (2026-10-21)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261021/moe-down`
at a5bf5f2 (manifest.txt: commit, harness and metallib SHA256; the only untracked path was
this folder). Synthetic 512-expert down tensors (never the checkpoint), 2560 output rows
per expert, rows of act = tokens x 10 slots with distinct experts per token; 3 warm-ups +
20 measured command buffers per spec, GPU time per encode; 20 encodes per buffer up to 4
tokens (as in the decode graph), 1 above. Fused = `q38_moe_down_reduce_*` (one dispatch +
barrier); sequence = the unfused fast tail it replaces in `q38f_routed_encode`: indexed
fast down into the [pairs, 2560] scratch, barrier, `q38f_reduce` (identity map), barrier.

Conditions (before.txt / after.txt): AC power, `pmset -g therm` no thermal or
performance warning, NSProcessInfo.thermalState nominal, swap 0 before and after.
GB/s = the fused pass's bytes (down rows of each distinct expert once, as a row reads
them: 212 of 252 B for a 640-of-768 Q2_K row; act; shared; y and routed) / median,
against the 307 GB/s ceiling; the same byte count for both modes.

| format | tokens | fused µs median (p10/p90) | sequence µs median (p10/p90) | fused / sequence | fused GB/s |
|---|---|---|---|---|---|
| Q2_K 640/768 → 2560 (trunk) | 1 | 75.3 (75.1/75.6) | 81.7 (81.7/81.8) | 0.92 | 72.8 |
| | 4 | 282.9 (280.4/290.1) | 311.3 (311.3/322.4) | 0.91 | 75.6 |
| | 32 | 2204.6 (2203.5/2205.6) | 2471.3 (2469.5/2731.1) | 0.89 | 59.4 |
| | 128 | 8876.5 (8874.9/8877.8) | 9995.3 (9899.0/10153.6) | 0.89 | 29.4 |
| MXFP4 640 → 2560 (layer 48) | 1 | 94.4 (94.0/100.8) | 101.4 (99.4/126.5) | 0.93 | 92.8 |
| | 4 | 354.5 (354.2/355.5) | 383.3 (383.2/383.5) | 0.92 | 98.9 |
| | 32 | 2786.2 (2782.9/2846.5) | 3027.2 (3025.6/3208.1) | 0.92 | 75.9 |
| | 128 | 11120.1 (11118.1/11425.8) | 12124.3 (12033.3/12356.7) | 0.92 | 36.7 |

The fused kernel is faster at every token count, prefill sizes included (32, 128): it is
the default with no threshold. At 1 token it saves 6.4 µs per Q2_K MoE layer, about
0.3 ms per decoded token over 48 layers (estimate, not measured end to end). GB/s at 32
and 128 tokens is a lower bound (each distinct expert counted once, read once per slot).

## Route timing (input of the Task 13b decision)

`q38f_route` alone, 512 experts, top 10, normalized, as the graph dispatches it (one
thread per token, 64-thread groups, a barrier), 20 per buffer:

| tokens | µs median (p10/p90) |
|---|---|
| 1 | 1233.7 (1229.9/1236.6) |
| 32 | 1297.0 (1294.3/1299.1) |

48 x 1233.7 µs = 59.2 ms per decoded token: 37% of the post-Task-12 token (160.6 ms,
profile-t12), far above the 2% threshold (the post-Task-13 profile is the controller's).
One thread walks 512 probabilities for each of the 10 slots and rechecks the chosen IDs
in device memory: the time is almost independent of the token count.
