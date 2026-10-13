# Benchmark protocol

Kernel timings and complete generation requests use different protocols.
Keep their timing boundaries separate. Historical files are retained; corrections
and newer runs are identified in `bench/README.md`.

## Native Qwen3.8 FP32

The new resident CLI benchmark is independent of the earlier DwarfStar and
Qwen3 measurements. See [protocol and current status](specs/qwen38/native-next.md)
and [retained evidence](../bench/qwen38_native_20261013/README.md).
`tools/eval/qwen38_native_bench.py` records model load, request TTFT, prefill,
decode `(generated-1)/seconds`, Metal GPU timestamps, bounded n-gram preparation
and logical reads, footprint, system pressure and swap deltas. One model load
per mode, explicit warmups followed by reset/replay repetitions; no cold-cache
claim. Raw warmups, IDs, commands, revision/diff, hashes and memory samples remain
in exclusive output directories. Three repetitions are labelled a pilot.

The original scalar and cooperative QSA paths remain selectable for paired
comparisons. Kernel speedup and complete-engine speedup are reported separately.
The negative DwarfStar stock Metal/FP16 numerical gate remains a FAIL; it is
not used as the FP32 reference and its tolerance has not changed.

## Generation requests

`tools/eval/bench_speculate.py` compares no draft, fixed `k=2`, and adaptive
model drafting by default. It uses ten local prompts in `prompts.jsonl`, the
same Qwen3 single-turn template with thinking disabled, and greedy decoding.
This suite covers prose, code, Italian, repetition and a domain switch. It is
a pilot suite, not a standard benchmark or an independent evaluation set.

```sh
tools/venv/bin/python tools/eval/bench_speculate.py \
  models/Qwen3-30B-A3B-q8_row models/Qwen3-0.6B bench/my-run.csv 25
```

- One discarded warmup per prompt and configuration; 25 measured requests by
  default. Use fewer only for a labelled pilot. Order is shuffled within each
  repetition with a recorded seed. Prompts stay in file order.
- Every request starts a fresh process. Model hashing and a separate ordinary
  greedy reference run happen before measurement for each prompt. These warm
  the file cache; this is not a cold-start storage benchmark.
- `plain` uses the speculative loop with zero drafts. Its output, and every
  other output, must equal the independent `--gpu-greedy` loop token for token.
  A mismatch aborts the run and retains the raw evidence.
- `prefill_s` and `draft_prefill_s` are wall times around the two prompt passes.
  `decode_s` starts after prefill and ends when generation stops. Decode
  throughput is `(generated - 1) / decode_s`: the first token comes from prefill.
- `generation_s` is the sum of those three timers. `process_s` is wall time
  around process launch and completion, including model setup and teardown.
  Neither describes a resident serving process. Time to first streamed token
  is not measured by this runner.
- Process and internal timers use `CLOCK_MONOTONIC`. The runner saves the raw
  record, then aborts on negative/nonfinite times or phases that exceed their
  enclosing timer by more than 100 ms. Earlier archived runners used Python's
  `perf_counter` for process time; a clock disagreement discovered in the
  expert pilot prompted this correction.
- `peak_rss_bytes` is macOS `ru_maxrss`. It can omit GPU-resident mapped weights
  and driver allocations. The October pilot reported only 38–45 MB despite a
  31.2 GB target checkpoint: this metric does **not** measure the engine's full
  unified-memory requirement. Checkpoint bytes are recorded separately in metadata.
- The CSV reports medians, p10/p90 throughput, p90 process time, acceptance,
  tokens per pass and parity. Percentiles select the nearest sorted sample;
  they are not confidence intervals. Speedups divide per-prompt median
  throughputs; the aggregate is their geometric mean. Report the worst prompt too.
- `.runs.jsonl` retains all warmup and measured runs, IDs and draft-length
  histograms. `.meta.json` records source, binary, kernel, checkpoint and prompt
  hashes, the runner source, settings, OS/hardware and thermal warnings. `complete: false` means
  the run did not finish. Existing output files are refused.

Keep the machine plugged in and idle during measurement. Do not run tests or
another GPU workload alongside it. OS thermal warnings are recorded but are
not a clock or power trace. No independent machine has reproduced these results.

The historical `spec_qwen3_30b_m5pro.csv` used five prompts, three repetitions,
fixed configuration order and no retained individual timings. Its adaptive
1.44x result is a pilot and does not follow the new protocol.

### October 11 pilot

`bench/spec_qwen3_30b_20261011_pilot.*`: ten prompts, five measured repetitions
per configuration, one warmup, up to 128 output tokens, context 1,024. All
150 measured and 30 warmup outputs matched the separate ordinary greedy runs.
The run took 30.7 minutes. No OS thermal/performance warning was recorded.

| Configuration | Decode speedup | Generation speedup | Process speedup | Worst decode speedup |
|---|---:|---:|---:|---:|
| No draft | 1.000x | 1.000x | 1.000x | 1.000x |
| Model, k=2 | 1.412x | 1.102x | 1.099x | 1.117x |
| Model, adaptive | 1.427x | 1.136x | 1.131x | 1.073x |

Each aggregate is the geometric mean of the ten per-prompt ratios. Generation
and process ratios divide the baseline's median time by the configuration's
median time. Decode ratios divide median throughput by baseline throughput.

Adaptive drafting is 1.09% ahead of fixed `k=2` on aggregate decode throughput,
and wins on five of ten prompts. Five repetitions do not establish that this
small difference is stable.

Most of the gap between decode and full-request speedups falls inside the
**first target forward pass**, recorded as prefill. For `explain`, the no-draft
median prefill was 6.065 s, generation 9.288 s and process 9.458 s. The prefill
timer includes first-use and memory-residency costs; it is not a measurement
of prompt computation in a resident engine. Separately time first and warmed
forwards before deciding how to reduce this cost.

The pilot metadata archives the exact executed runner. Additional tokenizer
input hashes were captured during the run; the current runner collects them
automatically. These metadata changes do not change the timing loop.

### Expert-cost pilot

The separate six-prompt experiment compares `plain`, `model-2`, `model-auto`
and opt-in `model-auto-experts`, with five measured requests each and up to
96 tokens. Decode speedups over no draft are 1.277x, 1.294x and 1.307x; process
speedups are 1.015x, 1.041x and 1.029x. The expert predictor does not meet its
adoption threshold. See the [full report](phase4-expert-experiment.md) for
per-prompt regressions, CPU overhead, forecast errors and exact artifacts.

The raw file retains an interrupted schedule and a whole prompt repeated
after two internal/process clock disagreements. Metadata specifies which
records enter the final 120-measurement summary, and archives both runners.
This is a labelled pilot with an audit trail, not evidence of stable gains.

## The machine

MacBook Pro 14", Apple **M5 Pro**: 18-core CPU (6 performance + 12 efficiency),
20-core GPU with one Neural Accelerator per core, 64 GB unified memory,
**307 GB/s** memory bandwidth. Historical runs used macOS 26.5.1 and Metal
toolchain 32023.883. The October 11 rerun uses macOS 27.0.1; current generation
metadata records the environment for each run.

That bandwidth figure is the ceiling every decode measurement is reported
against. It is the theoretical peak, not an achievable one; treat 80% of it
as excellent and anything above it as a bug in the measurement.

## GPU kernel and forward-pass procedure

- **0.5 s of warm-up per cell, discarded.** The first dispatch of a pipeline
  pays for shader caches, page faults on freshly allocated buffers, and a GPU
  that may still be at idle clocks. Warm-up is by **time, not count**: until
  Phase 3 it was 5 runs, and on a 0.1 ms kernel that is half a millisecond of
  work. The first quantised sweep measured M=1 at 154 us for every kernel and
  watched cells get faster in the order they were measured, straight across
  kernels doing different work -- the GPU leaving a low-power state. Warmed by
  time, the same cells read 33 us. (The Phase 2 CSV escaped this only because
  its sweep began with the slow naive kernel, which woke the GPU first.)
- **25 measured runs.** One run is an anecdote. A laptop is a noisy
  instrument: other processes, thermal state and the scheduler all leak into
  a single sample.
- **Median, with p10 and p90**, never the mean. A mean lets one descheduled
  run dominate. The spread between p10 and p90 says whether to trust the
  median at all; a wide spread is a result to report, not noise to hide.
- **Buffers allocated and uploaded once**, outside the timed region.
  Otherwise the measurement is of the driver.
- **GPU timestamps, not wall clock.** Timing comes from the command buffer's
  `GPUStartTime` / `GPUEndTime`. Wall time around a submit measures queueing,
  driver work and scheduling as much as the kernel.
- **The dispatch floor is measured and printed** alongside every run: an
  empty kernel, best of 45, currently **6.0 us**. Any measurement within a
  small multiple of it is reporting the command queue, not the work.
- **Raw CSV committed** under `bench/`, with the plotting script, so the
  figures can be redrawn and the claims rechecked by someone who was not here.

### Why this is not ceremony

The same kernel and the same shape, measured two ways:

| | naive, M=1, N=1024, K=1024 |
|---|---|
| single shot, fresh buffers (`--metal-check`) | 8.0 GB/s |
| protocol above (`--bench`) | 36.2 GB/s |

A factor of four and a half, entirely from cold buffers and no warmup. The
first number is not a pessimistic version of the second; it is a measurement
of something else.

## Phase 3: quantisation

Three measurements, each answering a different question, each with its CSV.

**Kernel cost** -- `bench/matmul_quant_m5pro.csv`. The batch sweep above at
gate_proj's shape (N=3072, K=1024), with the bf16 TensorOps kernel as the
baseline in the same file, for Q8/Q4 per-row and Q8_G32, Q4_G64, Q4_G32
blocked kernels. Bytes counted: activations (fp32, or bf16 for Q4), codes,
scales, output.

**End to end** -- `bench/e2e_quant_m5pro.csv`. Whole forward passes on the
GPU, timed by the command buffer: prefill of 512 tokens from position 0,
decode of 1 token at position 512, and a 32-token verify at position 512
(the step speculative decoding would pay). The same depth for decode and
verify, because a decode at position 10 reads almost no cache and flatters
every format equally. Decode GB/s counts every weight once (the embedding in
full, as the LM head) plus the K/V cache up to the position.

**Quality** -- `bench/quant_quality_local.csv` (transformers, fake-quantised,
exact fp32 arithmetic) and `bench/quant_quality_engine.csv` (the engine's own
tokenizer and GPU forward, bf16 activations for Q4 included). Protocol shared
by both so the columns line up:

- Corpus pinned by SHA256 in `docs/corpus.lock`. Windows of 1024 tokens,
  non-overlapping, the second half of each scored (llama.cpp's convention:
  every scored token has at least 512 tokens of context).
- Everything is **paired** against the unquantised model on the same windows:
  dNLL, KL(fp32 || quantised), top-1 agreement. KL and log-softmax in fp64.
- dNLL's standard error is taken across **windows**, not tokens. Neighbouring
  tokens are correlated, and a per-token SE claims certainty the data lacks.

## Phase 4: cost per dispatch

`bench/dispatch_m5pro.csv`. The same protocol, one cell per row, but each
timed command buffer holds `count` copies of one matmul, so the GPU time
divided by `count` is what one more dispatch costs inside a buffer that is
already paid for. Wall time (encode + commit + wait) is recorded beside it.
Sweeps: `count` 1..1024 at three shapes, tile fan-out, work per tile, rows per
dispatch, and tile shape. Every tile-shape variant is checked against the
32x32 TensorOps kernel before it is timed, with C poisoned with NaN first so
a kernel that writes nothing cannot pass; the relative difference is in the
last column.

## Phase 4: MoE verify cost

`bench/moe_verify_cost_m5pro.csv`. One MoE block at Qwen3-30B-A3B's shapes
(synthetic weights: the time depends on shapes and routing, not on what the
weights say), the router matmul skipped and routing forced so that n tokens
touch exactly D distinct experts. The harness reads back how many expert
groups the GPU formed and aborts unless it equals D. Same warm-up, repetitions
and percentiles as above. The x48 column is arithmetic, labelled as such:
real layers route differently from one another and attention is not included.

## Reproducing

```
make
./picoforge models/Qwen3-0.6B --bench bench/matmul_m5pro.csv
./tools/venv/bin/python tools/plot_bench.py bench/matmul_m5pro.csv
```

`--metal-check` runs the correctness comparison against the CPU oracle instead;
its timings are single-shot and are printed for orientation only.

Phase 3:

```
make quant-models
./picoforge models/Qwen3-0.6B --bench-quant bench/matmul_quant_m5pro.csv
for m in Qwen3-0.6B Qwen3-0.6B-q8_row-embed Qwen3-0.6B-q4_g32 Qwen3-0.6B-q4_g32-embed; do
  ./picoforge models/$m --bench-e2e bench/e2e_quant_m5pro.csv; done
tools/venv/bin/python tools/eval/build_local_corpus.py 8697213
tools/venv/bin/python tools/quant/sweep.py build/corpus_local.txt bench/quant_quality_local.csv
./picoforge models/Qwen3-0.6B-q4_g32 --ppl build/corpus_local.txt models/Qwen3-0.6B bench/quant_quality_engine.csv
tools/venv/bin/python tools/plot_quant.py
```

Phase 4:

```
./picoforge models/Qwen3-0.6B --bench-dispatch bench/dispatch_m5pro.csv
./picoforge models/Qwen3-0.6B --profile bench/profile_m5pro.csv
make build/synth-30b-layer
./picoforge build/synth-30b-layer --bench-moe bench/moe_verify_cost_m5pro.csv
./tools/venv/bin/python tools/plot_moe.py
```

Thermal state was read with `pmset -g therm` before and after the Phase 3
and Phase 4 runs: no thermal or performance warning recorded. (`powermetrics` needs root
and was not run: the benchmarks do not require elevated privileges.)

## What is measured

`bench_matmul` sweeps three axes, all on `C[M,N] = A[M,K] * B[N,K]^T` with A
in fp32, B in bf16 and C in fp32 — the shape the engine actually issues, not
a rounded-off version of it.

- **batch-sweep**: M from 1 to 512 at N=3072, K=1024, Qwen3-0.6B's MLP width.
  This is the axis separating decode (M=1) from prefill.
- **width-at-M1**: N from 256 to 16384 at M=1. Isolates whether decode is
  limited by bandwidth or by having too little work to occupy the GPU.
- **square**: 128 to 2048 cubed. Comparable with everyone else's numbers, and
  issued by nothing in this engine.

Reported per row: median/p10/p90 seconds, GFLOP/s at `2*M*N*K`, and effective
GB/s counting the minimum traffic that must cross the memory system
(`M*K*4 + N*K*2 + M*N*4`). Real traffic can exceed that when a kernel
re-reads, which is precisely what the tiled versions exist to avoid.
