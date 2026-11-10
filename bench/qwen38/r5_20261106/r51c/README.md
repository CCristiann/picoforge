# R5.1c — streamed experts on the real Q2, 2026-11-06

Branch `perf/qwen38-r5-streaming` at fa7d287. One chat prompt (21 tokens), 16 greedy tokens,
context 2048, chunk 128, `Q38_REF_KERNELS=gemm` in every run. Each run went through the guarded
launcher (release wait, memory guard, caffeinate) under the GPU lock, on AC, 90% battery, no
thermal warning. Script: `run.sh`.

| run | pool slots | share of 24576 experts | misses | evictions | peak wired GiB | swap |
|---|---|---|---|---|---|---|
| res | — (resident) | — | — | — | 45.94 | 0 |
| s2458 | 2458 (3.4 GiB) | 10% | 6301 / 17280 (36.5%) | 3843 | 15.29 | 0 |
| s12288 | 12288 (17.1 GiB) | 50% | 5248 / 17280 (30.4%) | 0 | 28.84 | 0 |
| s24576 | 24576 (34.1 GiB) | 100% | 5248 / 17280 (30.4%) | 0 | 46.05 | 0 |

Gate: logits (sha256 in `*/logits.sha256`, all four equal) and generated ids bit-identical to
resident at every size; swap 0 and memory pressure level 1 throughout (`memory.tsv`).

The misses at 50% and 100% are the cold first reads (no eviction): the prefill of 21 tokens
touches most experts. Timings in `runs.jsonl` are single cold runs with the page cache warmed by
the previous run; they are NOT measurements (protocol not followed). Speed comes in R5.2.
