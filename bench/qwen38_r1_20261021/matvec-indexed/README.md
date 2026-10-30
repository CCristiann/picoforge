# R1.5a indexed expert matvecs — protocol bench (2026-10-21)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261021/matvec-indexed`
at ca519f2 (manifest.txt: commit, harness and metallib SHA256). Synthetic 512-expert
tensors (never the checkpoint), rows = tokens x 10 slots with distinct experts per token;
3 warm-ups + 20 measured command buffers per spec, GPU time per encode; 20 encodes per
buffer up to 4 tokens (as in the decode graph), 1 above. Fast = `q38_matvec_indexed_*`,
oracle = `q38_packed_linear_indexed`.

Conditions (before.txt / after.txt): AC power, `pmset -g therm` no thermal or
performance warning, NSProcessInfo.thermalState nominal, swap 0 before and after.
GB/s = bytes actually read (each distinct expert once, x, y) / median; the Q2_K down
counts 212 of the 252 bytes of a 640-of-768 row (ca519f2), against the 307 GB/s ceiling.

| shape | tokens | fast µs median (p10/p90) | oracle µs median | fast GB/s |
|---|---|---|---|---|
| gate IQ2_XXS 2560→640 | 1 | 37.2 (37.2/56.7) | 704.9 | 114.4 |
| | 4 | 141.4 (141.4/141.5) | 1799.3 | 117.5 |
| | 32 | 1162.4 (1161.7/1347.7) | 13055.1 | 91.5 |
| | 128 | 4642.5 (4638.9/4644.5) | 51442.8 | 43.6 |
| down Q2_K 640/768→2560 | 1 | 78.1 (78.1/78.2) | 543.6 | 71.1 |
| | 4 | 307.3 (307.3/307.4) | 1916.5 | 70.5 |
| | 32 | 2464.6 (2464.1/2473.5) | 14618.9 | 54.1 |
| | 128 | 9885.3 (9883.1/10220.4) | 58415.1 | 27.4 |
| gate Q4_K (layer 48) | 1 | 46.9 (46.8/47.0) | 1069.6 | 197.5 |
| | 32 | 1462.7 (1461.3/1647.6) | 17798.9 | 144.4 |
| down MXFP4 (layer 48) | 1 | 95.4 (95.4/95.4) | 372.5 | 92.6 |
| | 32 | 3008.2 (3007.7/3177.8) | 10545.0 | 71.1 |

The fast kernel is faster than the oracle at every measured row count (3.5–23x), so the
default (fast at any row count, `encode_indexed` in src/qwen38_packed_gpu.m) needs no
threshold. Earlier exploration on battery (task 11) had Q4_K gate 86.6 µs at 1 token;
the protocol run on AC gives 46.9. GB/s falls with tokens because the count takes each
distinct expert once while the kernel reads an expert once per slot that selects it (128
tokens = 1280 slots over at most 512 experts): there it is a lower bound.
