# Measurement files

See [the protocol](../docs/BENCHMARKS.md) for timing boundaries and commands.
The reference machine is an M5 Pro with 64 GB memory. These are local results;
no external runtime or independent machine has been measured with this protocol.

## October 11, 2026

- `spec_qwen3_30b_20261011_pilot.csv`: ten prompts, five measured requests per
  configuration, up to 128 tokens. `.runs.jsonl` includes 150 measurements and
  30 warmups; `.meta.json` records the exact runner and inputs. Adaptive decode
  reaches 1.427x versus no draft, fixed `k=2` 1.412x. Full-process ratios are
  1.131x and 1.099x. See the protocol for definitions and memory-accounting limits.
- `routing_overlap_qwen3_30b_20261011.csv`: corrected 4,096-token trace, q8_row,
  48 MoE layers. The `est_*` columns apply the existing synthetic cost surface;
  they are estimates, not measured complete verification latency.
- `profile_qwen3_30b_20261011.csv`: per-operation GPU timings for the 30B.
  Profiling splits the command buffer into encoders and changes its cost.
  Use `TOTAL, uncut` for the baseline; use the other rows to locate bottlenecks.
- `corrected-routing-20261011.meta.json`: checkpoint, corpus, source and output
  hashes, plus commands for both measurements above. The corpus is pinned to
  commit `8697213`, not the current working tree.

## Historical results

`routing_overlap_qwen3_30b_corpus.csv` is **invalid beyond the old 2,048-position
attention bound**. Do not use it to support routing-locality claims. It is
retained to make the correction auditable.

`spec_qwen3_30b_m5pro.csv` is the original five-prompt, three-repetition pilot.
It contains medians but no raw per-request timings or percentiles. It should
not be presented as if it used the current request protocol.

Other historical kernel and quality files retain the conditions documented
in the devlog and benchmark protocol. In particular, 0.6B perplexity results
cannot establish quality for the 30B checkpoint.
