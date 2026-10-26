# R1.6a parallel RMSNorm: exploration on battery, not protocol evidence (2026-10-25)

**Exploration on battery, not protocol evidence.** The Mac was on battery power for the
whole run (`pmset -g batt`: 'Battery Power', 75%, discharging, in before.txt and after.txt),
so the GPU clock is not the protocol's. The fast medians at 1 and 4 tokens have wide p90s
(54 and 69 µs against medians of 28 and 29 µs) and the fast kernel is faster at 32 tokens
than at 1. These numbers only show the order of magnitude. The protocol run must be
repeated on AC into a new folder (`.../rms`); do not cite these figures as measured results.

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261025/rms-battery`
at 0b571df (manifest.txt: commit, `git status --porcelain`, harness and metallib SHA256; the
only entry is this untracked folder). Synthetic x and gain (fixed seed), never the checkpoint.
One spec times one `q38c_gpu_encode` of `Q38C_RMS` at the HC width (groups 1, width 10240, as
hc_attn.rms, hc_ffn.rms and final.rms): fast = `q38c_rms_fast` + one barrier (input check folded
in); oracle = `q38c_check` + barrier + `q38c_rms` + barrier. 3 warm-ups + 20 measured command
buffers per spec, 20 encodes per buffer as in the decode graph, GPU time per encode
(`GPUEndTime - GPUStartTime` / 20). Spec order alternates between token counts. The error word
is checked to be 0 after every measured buffer.

Conditions: 17:02:44Z to 17:02:48Z, battery (see above), `pmset -g therm` no thermal,
performance or CPU power warning, NSProcessInfo.thermalState nominal, swap 0 before and after.
`powermetrics` needs sudo and was not used.

| tokens | oracle µs median (p10/p90) | fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 1151.4 (1141.5/1168.5) | 28.2 (20.7/54.4) | 40.8x |
| 4 | 1134.0 (1131.3/1137.2) | 29.0 (28.7/69.4) | 39.1x |
| 32 | 1288.1 (1288.0/1288.2) | 16.7 (15.1/70.4) | 77.1x |
| 128 | 1610.8 (1609.8/1615.3) | 42.3 (42.2/59.2) | 38.1x |

Per decoded token estimate (61 HC norms: 48 hc_ffn + 12 hc_attn + 1 final, t = 1, from the
per-encode time, not an end-to-end measurement; battery): 61 x 1151.4 = 70.2 ms with the oracle,
61 x 28.2 = 1.72 ms with the fast kernel. The t12 profile on AC measured 290 µs per oracle HC
norm in the graph (17.7 ms/token), so this battery oracle time is about 4x the graph's: one
more reason to rerun on AC before using any of these numbers.

## runs.jsonl

One JSON line per spec, in execution order. `mode`: `oracle` or `fast`. `n` and `k`: the width
(10240). `m`: rows = tokens x groups (groups 1). `type`: always 0 (unused here). `reps`: encodes
per command buffer (20). `bytes`: nominal traffic 4 x (2 x rows x width + width) (x read, y
written, gain once), not a measured byte count. `median_us`, `p10_us`, `p90_us`: per-encode GPU
time over the 20 measured buffers. `gbps`: `bytes` / median.

## Reproduce (protocol, on AC)

`caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/rms PYTHON=<venv>/bin/python`
(a new folder every time; an existing one fails). `make -n` with the same arguments lists the 8
`rmsbench` specs. The numerical checks of the fast kernel are in `make test-qwen38-fast-kernels`.

Superseded: wrong shape [t,1,10240]; see bench/qwen38_r1_20261026/rms-battery2/.
