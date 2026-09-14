# Benchmark protocol

Every performance number in this repository was produced by the procedure
below, on the machine below, and the raw CSV is committed next to the plot.
A number without those three things is not a result.

## The machine

MacBook Pro 14", Apple **M5 Pro**: 18-core CPU (6 performance + 12 efficiency),
20-core GPU with one Neural Accelerator per core, 64 GB unified memory,
**307 GB/s** memory bandwidth. macOS 26.5.1, Metal toolchain 32023.883.

That bandwidth figure is the ceiling every decode measurement is reported
against. It is the theoretical peak, not an achievable one; treat 80% of it
as excellent and anything above it as a bug in the measurement.

## Procedure

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
and was not run; that is the human's call, not the benchmark's.)

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
