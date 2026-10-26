# R1.6a parallel RMSNorm at the graph's shapes: exploration on battery, not protocol evidence (2026-10-26)

**Exploration on battery, not protocol evidence.** The Mac was on battery power for the
whole run (`pmset -g batt`: 'Battery Power', 90%, discharging, in before.txt and after.txt).
The protocol run must be repeated on AC into a new folder (`.../rms`). Until then, cite these
figures only as an exploration. This folder replaces `bench/qwen38_r1_20261025/rms-battery`,
which timed the wrong shape ([t, 1, 10240], the MTP h-norm).

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261026/rms-battery2`
at f5e4d7f. manifest.txt records the commit, `git status --porcelain` and the harness and
metallib SHA256; the only status entry is this untracked folder. The inputs are synthetic x,
gain and z from a fixed seed; the checkpoint is never read.

What each spec times: one `q38c_gpu_encode`.
- fast = `q38c_rms_fast` + one barrier; the input check is folded into the kernel.
- oracle = `q38c_check` + barrier + `q38c_rms` + barrier.

Protocol: 3 warm-ups + 20 measured command buffers per spec, with 20 encodes per buffer as in
the decode graph. The time reported is GPU time per encode (`GPUEndTime - GPUStartTime` / 20).
Spec order alternates. The error word is checked to be 0 after every measured buffer.

Conditions: 00:20:49Z to 00:20:52Z, on battery (see above). `pmset -g therm` shows no thermal,
performance or CPU power warning. NSProcessInfo.thermalState is nominal. Swap is 0 before and
after. `powermetrics` needs sudo and was not used.

## HC shape [t, 4, 2560], [4, 2560] gain

This shape covers hc_attn, hc_ffn, final, the GDN-block HC norm and the 3 PLE norms.

| tokens | oracle µs median (p10/p90) | fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 288.1 (286.5/289.8) | 25.9 (25.8/26.0) | 11.1x |
| 4 | 306.4 (306.0/310.9) | 26.2 (26.2/26.3) | 11.7x |
| 32 | 377.6 (377.3/379.1) | 21.1 (12.8/54.3) | 17.9x |
| 128 | 649.0 (648.2/653.0) | 39.7 (39.6/80.0) | 16.3x |

The oracle at 1 token (288.1 µs) matches the 290 µs per hc_*.rms call that the t12 profile
measured in the graph on AC. Each oracle thread does a serial pass over 2560 floats.

## Small widths at 1 token (the fast path takes them too)

| shape | calls per token | oracle µs median (p10/p90) | fast µs median (p10/p90) |
|---|---|---|---|
| GDN norm [1, 48, 128] gated | 36 | 61.6 (61.5/62.7) | 20.5 (20.0/21.5) |
| QSA q [1, 24, 256] | 12 | 34.3 (34.0/34.4) | 16.4 (15.9/16.7) |
| QSA k [1, 2, 256] | 12 | 32.8 (31.6/55.5) | 3.4 (3.3/14.9) |
| indexer [1, 4, 128] | 12 | 18.0 (17.9/37.6) | 13.0 (12.6/13.8) |
| pooled [1, 1, 128] | 3 (12 every 4th token) | 21.1 (18.8/38.2) | 13.0 (12.7/13.5) |

No small width is slower on the fast path.

## Per decoded token (estimate on battery)

This is an estimate from the per-encode times at t = 1, not an end-to-end measurement.

Call counts are taken from src/qwen38_forward_gpu.m (48 trunk layers: 36 GDN and 12 QSA, PLE
at layer 1):
- [1, 4, 2560]: 61 calls in the profile's hc_*.rms groups (48 hc_ffn + 12 hc_attn + 1 final).
- [1, 4, 2560]: 39 more calls inside the gdn and ple groups (36 GDN-block HC norms in
  q38c_block_encode, 3 PLE norms). That makes 100 in all.

| | oracle | fast |
|---|---|---|
| 61 x [1, 4, 2560] | 61 x 288.1 = 17.57 ms | 61 x 25.9 = 1.58 ms |
| 100 x [1, 4, 2560] | 28.81 ms | 2.59 ms |
| small widths (counts above) | 3.30 ms | 1.17 ms |
| all RMS / GDN norms (100 HC + small) | 32.11 ms | 3.76 ms |

## runs.jsonl

There is one JSON line per spec, in execution order. The fields are:
- `mode`: `oracle` or `fast`.
- `n` and `k`: the width.
- `m`: rows, which is tokens x groups.
- `specs`: op (0 RMS, 1 gated GDN norm), tokens, groups, width and gain rank.
- `type`: always 0 (unused).
- `reps`: encodes per command buffer (20).
- `bytes`: nominal traffic, 4 x (2 x rows x width + gain + z), not a measured count.
- `median_us`, `p10_us`, `p90_us`: per-encode GPU time over the 20 measured buffers.
- `gbps`: `bytes` / median.

## Reproduce (protocol, on AC)

`caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/rms PYTHON=<venv>/bin/python`

Use a new folder every time; an existing one makes the target fail. `make -n` with the same
arguments lists the 18 `rmsbench` specs. The numerical checks are in
`make test-qwen38-fast-kernels`.
