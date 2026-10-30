# R1.6a parallel RMSNorm: protocol bench (2026-10-26)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261026/rms`
at f614264 (manifest.txt: commit, `git status --porcelain`, harness and metallib SHA256; the
only entry is this untracked folder). Synthetic x, gain and z from a fixed seed (the
`rmsbench` generator), never the checkpoint. What each spec times: one `q38c_gpu_encode`.
oracle = `q38c_check` + barrier + `q38c_rms` (or the gated `q38c_gdn_norm`) + barrier, one
thread per row; fast = `q38c_rms_fast` + one barrier, one 256-thread threadgroup per row,
input check folded in. 3 warm-ups + 20 measured command buffers per spec, 20 encodes per
buffer as in the decode graph, GPU time per encode (`GPUEndTime - GPUStartTime` / 20). Spec
order alternates (oracle, fast / fast, oracle / ...). The error word is checked to be 0 after
every measured buffer.

Conditions (before.txt / after.txt, 00:37:59Z to 00:38:02Z): AC power (`pmset -g batt`:
'AC Power', 80%, not charging; checked again after the run), `pmset -g therm` no thermal,
performance or CPU power warning, NSProcessInfo.thermalState nominal, swap 0 before and
after. `powermetrics` needs sudo and was not used.

## HC shape [t, 4, 2560], [4, 2560] gain

Covers hc_attn, hc_ffn, final, the GDN-block HC norm and the PLE norms.

| tokens | q38c_rms µs median (p10/p90) | q38c_rms_fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 286.6 (286.5/287.3) | 25.9 (25.8/26.1) | 11.1x |
| 4 | 305.7 (305.6/306.4) | 27.0 (26.5/27.4) | 11.3x |
| 32 | 378.3 (377.3/378.9) | 12.8 (12.8/12.9) | 29.6x |
| 128 | 652.3 (651.0/653.5) | 39.7 (39.7/73.8) | 16.4x |

## Small widths at t = 1 (rank-1 gain)

| norm (groups x width) | oracle µs median (p10/p90) | fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| GDN gated [48, 128] | 61.5 (61.5/61.9) | 4.5 (4.4/4.5) | 13.7x |
| attention q [24, 256] | 34.3 (34.0/72.8) | 15.7 (15.4/16.6) | 2.2x |
| attention k [2, 256] | 32.3 (32.1/33.5) | 13.3 (12.9/14.3) | 2.4x |
| indexer q [4, 128] | 18.1 (17.9/85.3) | 13.9 (12.0/14.3) | 1.3x |
| pooled key [1, 128] | 22.0 (18.4/39.4) | 13.5 (12.6/14.2) | 1.6x |

The fast kernel is faster at every measured spec, so it is the default with no threshold; the
oracle stays selectable with `Q38_REF_KERNELS=fuse` (or `1`). The wide p90 of some oracle rows
(72.8, 85.3 µs) is one slow buffer out of 20, not a trend.

Per decoded token (t = 1; an estimate from the per-dispatch median, not an end-to-end
measurement):
- HC-shaped [1, 4, 2560] norms: 100 per token = 61 in the profile's rms groups (hc_ffn.rms 48
  + hc_attn.rms 12 + final.rms 1, `profile-t13/summary/summary.md`; 48 layers = 36 GDN + 12
  attention) + 36 GDN-block HC norms (`OP(RMS,w->residual,K(HN),...)` in
  `qwen38_recurrent_gpu.m`, one per GDN layer, inside the profile's `gdn` group) + 3 PLE norms
  (three `OP(RMS,...)` in `qwen38_ple_gpu.m`, one PLE call per token, inside the `ple`
  group; the count of 3 comes from the fix-round report and the three calls in the source,
  their shape was not re-derived here). 100 x 286.6 = 28.66 ms with the oracle, 100 x 25.9 =
  2.59 ms with the fast kernel (by p10/p90: 28.65/28.73 and 2.58/2.61 ms). Of the 100, the 61
  in the profile's rms groups: 17.48 -> 1.58 ms.
- Small-width norms: 36 GDN gated (one per GDN layer), 12 q, 12 k, 12 indexer (one each per
  attention layer, `qwen38_attention_gpu.m`), 3 pooled (the report's count; it depends on
  the position, one per completed indexer block): 3.30 -> 0.72 ms.
- All: 31.96 -> 3.31 ms per token, about 28.7 ms saved by this estimate.

## runs.jsonl

One JSON line per spec, in execution order. `mode`: `oracle` or `fast`. `n` and `k`: the
norm width. `m`: rows = tokens x groups. `type`: always 0 (not used by this mode). `reps`:
encodes per command buffer (20). `bytes`: the harness's nominal traffic,
4 x (gain elements + rows x width for the gated op's z) + 4 x rows x 2 x width for x and y
(122880 at the HC t = 1 spec); not a measured byte count. `median_us`, `p10_us`, `p90_us`:
per-encode GPU time over the 20 measured buffers. `gbps`: `bytes` / median; at these sizes
the kernel is not bandwidth-bound except at t = 128, so it is not a bandwidth result.
`specs`: the shape (`op` 0 = RMS, 1 = gated GDN norm; tokens, groups, width, gain rank).

## Reproduce

On AC power, with the Python venv of the main checkout:
`caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/rms PYTHON=<venv>/bin/python`
(a new folder every time; an existing one fails). `make -n` with the same arguments lists the
18 `rmsbench` specs. The tolerance and bit-identity are checked by
`make test-qwen38-fast-kernels`, not by this bench. The battery exploration of the same specs
is `bench/qwen38_r1_20261026/rms-battery2` (exploration only; superseded by this folder).
