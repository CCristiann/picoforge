# D1 — one decode token in several command buffers

`PICOFORGE_CMD_SPLIT=K` (default 4, 1 = the old single command buffer) commits the token's
command buffer after K-1 layer groups, so the GPU starts on the first layers while the CPU
still encodes the rest. Only decode-sized appends (<= 4 rows, full trunk, no profile/trace)
are split. The numbers do not change: logits and ids are bit-identical for K = 1, 3, 4, 8
(`tests/test_qwen38_cli.py::test_command_split_is_bit_identical`, plus the CLI, REPL, session
and speculative suites on the synthetic graph).

Real model, short frontier (137 ids: prompt + 128 forced), teacher-forced decode, 1 warm-up +
3 measured per process, AC, battery 90%, guarded launcher, one process at a time. Raw lines:
`s*/runs.jsonl`; ms per token = decode_s / 128.

| Run | K | ms/token (3 measured) | median | wall - GPU |
|---|---:|---|---:|---:|
| s1 | 1 | 41.00, 41.00, 41.06 | 41.00 | 2.98 |
| s2 | 4 | 38.92, 38.88, 38.85 | 38.88 | 0.87 |
| s3 | 4 | 38.86, 38.88, 39.06 | 38.88 | 0.88 |
| s4 | 1 | 44.61, 44.60, 43.13 | 44.60 | 3.00 (GPU time itself rose: drift) |
| s5 | 8 | 40.45, 40.04, 41.31 | 40.45 | 0.65 |
| s6 | 4 | 39.02, 38.97, 39.05 | 39.02 | 0.90 |
| s7 | 8 | 39.10, 39.76, 39.03 | 39.10 | 0.64 |
| s8 | 4 | 40.30, 38.97, 39.01 | 39.01 | 0.91 |

K = 4 removes ~2.1 ms/token (-5%, 24.4 -> 25.7 tok/s) by cutting the idle GPU time before the
first dispatch from ~3.0 to ~0.9 ms; the GPU time of the token (38.0 ms) is unchanged. K = 8
leaves less idle time (0.65 ms) but did not run faster overall in two samples; 4 is kept.
The s4 K=1 run is slower in GPU time too (41.2 vs 38.0 ms): a thermal/clock drift, so the
comparison rests on the adjacent pairs s1/s2 and s3/s6.
