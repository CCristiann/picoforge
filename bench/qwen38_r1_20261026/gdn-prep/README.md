# R1.8 parallel GDN prepare: protocol bench (2026-10-26)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261026/gdn-prep`
at 570ecdd (manifest.txt: commit, `git status --porcelain`, harness and metallib SHA256; the
untracked entries are this folder and the controller's `profile-t16-0`). Synthetic qkv,
alpha, br, A (made nonpositive) and dt from a fixed seed (the `prepbench` generator), never
the checkpoint. What each spec times: one `q38c_gpu_encode` of `Q38C_GDN_PREP` at the real GDN
dims (hk 16, hv 48, dk = dv 128, qkv width 10240). oracle = `q38c_check` + barrier +
`q38c_gdn_prepare` + barrier, one thread per token; fast = `q38c_gdn_prepare_fast` + one
barrier, 4 threadgroups of 256 threads per token (one simdgroup per q/k head), input check
folded in. 3 warm-ups + 20 measured command buffers per spec, 20 encodes per buffer as in the
decode graph, GPU time per encode (`GPUEndTime - GPUStartTime` / 20). Spec order alternates
(oracle, fast / fast, oracle / ...). The error word is checked to be 0 after every measured
buffer.

Conditions (before.txt / after.txt, 20:47:54Z to 20:47:57Z): AC power (`pmset -g batt`:
'AC Power', 62%, charging; checked again after the run), `pmset -g therm` no thermal,
performance or CPU power warning, NSProcessInfo.thermalState nominal, swap 0 before and
after. `powermetrics` needs sudo and was not used.

## Real dims [t, 10240] (hk 16, hv 48, dk = dv 128)

| tokens | q38c_gdn_prepare µs median (p10/p90) | q38c_gdn_prepare_fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 666.8 (659.6/674.8) | 19.4 (19.0/19.6) | 34.4x |
| 4 | 656.7 (656.1/662.7) | 19.2 (5.3/19.9) | 34.2x |
| 32 | 837.2 (824.6/843.9) | 6.9 (6.9/7.1) | 121x |
| 128 | 1094.6 (1088.6/1105.7) | 24.1 (23.9/28.6) | 45.4x |

The fast kernel is faster at every measured spec, so it is the default with no threshold; the
oracle stays selectable with `Q38_REF_KERNELS=fuse` (or `1`). The fast times at 1 and 4 tokens
are above the 32-token time and the 4-token p10 is 5.3 µs: a short dispatch chain runs at a
lower GPU clock some of the time (the same pattern as the RMS bench,
`bench/qwen38_r1_20261026/rms`); the medians are the honest number.

Per decoded token (t = 1; an estimate from the per-dispatch median, not an end-to-end
measurement): one GDN prepare per GDN layer, 36 per token. 36 x 666.8 = 24.0 ms with the
oracle (the 16-0 profile's gdn.prep: 23.23 ms/token), 36 x 19.4 = 0.70 ms with the fast
kernel (by p10/p90: 23.7/24.3 and 0.68/0.71 ms): about 23.3 ms saved by this estimate.

## runs.jsonl

One JSON line per spec, in execution order. `mode`: `oracle` or `fast`. `n` and `k`: the qkv
width (10240). `m`: tokens. `type`: always 0 (not used by this mode). `reps`: encodes per
command buffer (20). `bytes`: the harness's nominal traffic, 4 x (4 x tokens x hv + 2 x hv)
for the gates plus 4 x tokens x 2 x 10240 for qkv in and prepared out (83072 at t = 1); not a
measured byte count. `median_us`, `p10_us`, `p90_us`: per-encode GPU time over the 20 measured
buffers. `gbps`: `bytes` / median; the 20 encodes of a buffer re-read the same few hundred KB,
which stay in the cache, so values above the 307 GB/s DRAM ceiling (t = 32, 128) are not a
bandwidth result. `specs`: the shape (tokens, hk, hv, dk, dv).

## Reproduce

On AC power, with the Python venv of the main checkout:
`caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/gdn-prep PYTHON=<venv>/bin/python`
(a new folder every time; an existing one fails). `make -n` with the same arguments lists the
8 `prepbench` specs. The tolerance and bit-identity are checked by
`make test-qwen38-fast-kernels`, not by this bench.
