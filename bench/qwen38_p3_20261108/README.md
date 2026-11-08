# Perf III, 2026-11-08 — short chat prompts on the real Q2

All runs: real Qwen3.8-Flash-Next Q2, resident, AC, guarded launcher
(`build/r5-checks/launch-any.py`) under the GPU lock. Scripts in `scripts/`
(`prof.py` sums the prefill GPU time per op from `--profile`; the logits files
it compared were deleted after the comparison, 33 MB).

- `gemm-threshold/`: `bench` 1+5, prompt of 60 IDs. PICOFORGE_GEMM_MIN_ROWS 8/16/32
  give the same 1.282 s: 60 rows are already above the row threshold.
- `profile-short60/`: per-op profile of that prefill (1294 ms GPU): grouped MoE
  gate_up+down 809 ms at 47-64 GB/s, HC down / hc_pre 190 ms at 3-5 GB/s.
- `shape/`: grouped MoE off and GEMM tile thresholds 80/160 on the 60-ID prompt.
- `cross/`: 417/828/2586-ID prompts, grouped on/off; 32 vs 0 tiles at 417 and 60.

| prompt IDs | before | after (32 tiles, grouped from 2560 pairs) |
|---:|---:|---:|
| 60 | 1294 ms | 671 ms |
| 417 | 3601 ms | 3563 ms |

Logits moved by <= 1.5e-4 at 60 IDs (FP32 reorder GEMM vs matvec); at 417 IDs
the matvec/GEMM switch on HC flips router near-ties (0.66), as in R3.1.
Chat check by the user (REPL, thinking, sampled, MTP, ds4's system message):
prefill ~50 t/s, generation ~30 t/s; three lines of reasoning, then the answer.

## First-turn cost and the load-time warm-up

- `overhead/mtp/`: 66-ID prompt, `--mtp --repeat 4 --warmup 1`: first prefill 1.155 s,
  then 0.775-0.779 s (85 t/s); the MTP prompt pass is 0.02 s, decode 34 t/s warm.
- `warmup/`: `--repeat 3`, no warm-up run. Without the touch: load 0.20 s, prefills
  2.213 / 0.945 / 0.784 s. With `q38g_touch` (1 byte per MiB of every weight view at
  load, as ds4): load 4.91 s, prefills 0.896 / 0.786 / 0.788 s (first turn 74 t/s).
