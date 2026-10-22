# R1.5d parallel router top-k: exploration bench (2026-10-22, on battery)

STATUS: EXPLORATION. The Mac was on battery (`pmset -g batt`: Battery Power, 84%,
discharging; before.txt/after.txt), not on AC as the protocol requires, so these numbers are
not the protocol evidence. Rerun on AC into a new folder (this one is never overwritten) before
quoting them. No thermal or performance warning, NSProcessInfo.thermalState nominal, swap 0
before and after.

Command: `make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261022/moe-route` (manifest.txt:
commit, harness and metallib SHA256). Synthetic logits (the rbench generator), 512 experts,
top 10, normalized, never the checkpoint; 3 warm-ups + 20 measured command buffers, GPU time
per dispatch, 20 dispatches (each followed by a barrier) per buffer as in the decode graph.
oracle = `q38f_route` (one thread per token, 64-thread groups); fast = `q38_moe_route_fast`
(one 256-thread threadgroup per token). The order alternates between token counts.

| tokens | q38f_route µs median (p10/p90) | q38_moe_route_fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 1232.4 (1228.4/1236.1) | 35.8 (24.3/50.8) | 34x |
| 4 | 1239.6 (1237.1/1242.6) | 24.0 (23.9/48.5) | 52x |
| 32 | 1294.9 (1293.4/1296.9) | 24.6 (24.6/49.7) | 53x |
| 128 | 1311.0 (1310.6/1312.0) | 38.7 (38.7/123.6) | 34x |

Per decoded token (48 MoE layers, t = 1): 48 x 1232.4 = 59.2 ms before; 48 x 35.8 = 1.72 ms
by the median (48 x 24.3 = 1.17 ms by the p10) after. The fast kernel is faster at every token
count, prefill sizes included, so it is the default with no threshold. The spread of the fast
kernel (p90 about twice the median) is the short-dispatch GPU clock ramp, not a data effect;
the 1-token median sits between the idle-clock p90 and the p10.
