# R1.5b fused gate+up+SwiGLU — protocol bench (2026-10-21)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261021/moe-gate-up`
at d3af03b (manifest.txt: commit, harness and metallib SHA256). Synthetic 512-expert gate
and up tensors (never the checkpoint), 640 x 2560 per expert, rows = tokens x 10 slots
with distinct experts per token, x divisor 10; 3 warm-ups + 20 measured command buffers
per spec, GPU time per encode; 20 encodes per buffer up to 4 tokens (as in the decode
graph), 1 above. Fused = `q38_moe_gate_up_*` (one dispatch + barrier); sequence = the
unfused fast path it replaces in `q38f_routed_encode`: indexed fast gate, indexed fast
up, q38c ACT SILU (check + activate), q38f_product, a barrier after each.

Conditions (before.txt / after.txt): AC power, `pmset -g therm` no thermal or
performance warning, NSProcessInfo.thermalState nominal, swap 0 before and after.
GB/s = bytes read (gate and up of each distinct expert once, x, act) / median, against
the 307 GB/s ceiling.

| format | tokens | fused µs median (p10/p90) | sequence µs median (p10/p90) | fused / sequence | fused GB/s |
|---|---|---|---|---|---|
| IQ2_XXS (trunk layers) | 1 | 67.7 (67.6/67.8) | 81.7 (81.6/98.5) | 0.83 | 125.4 |
| | 4 | 261.1 (260.9/263.7) | 294.1 (294.1/306.4) | 0.89 | 126.7 |
| | 32 | 2088.1 (2084.6/2092.6) | 2360.6 (2359.7/2361.5) | 0.88 | 99.3 |
| | 128 | 8278.8 (8272.3/8283.9) | 9462.7 (9415.6/9766.6) | 0.87 | 48.2 |
| Q4_K (layer 48) | 1 | 87.2 (87.2/87.6) | 101.5 (101.4/102.3) | 0.86 | 211.7 |
| | 4 | 340.8 (340.3/341.2) | 372.0 (371.9/372.5) | 0.92 | 205.9 |
| | 32 | 2704.0 (2698.0/2926.4) | 2981.1 (2973.6/2987.4) | 0.91 | 164.7 |
| | 128 | 10656.4 (10642.0/10689.7) | 11713.0 (11709.1/11718.3) | 0.91 | 79.8 |

The fused kernel is faster at every token count, prefill sizes included (32, 128), so it
is the default with no threshold. At 1 token the saving is 14 µs per MoE layer
(IQ2_XXS), about 0.7 ms per decoded token over 48 trunk layers (estimate, not measured
end to end). As in matvec-indexed, GB/s at 32 and 128 tokens is a lower bound: each
distinct expert is counted once, but the kernel reads it once per slot.
