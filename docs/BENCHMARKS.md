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

- **5 warmup runs, discarded.** The first dispatch of a pipeline pays for
  shader caches, page faults on freshly allocated buffers, and a GPU that may
  still be at idle clocks.
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

## Reproducing

```
make
./picoforge models/Qwen3-0.6B --bench bench/matmul_m5pro.csv
./tools/venv/bin/python tools/plot_bench.py bench/matmul_m5pro.csv
```

`--metal-check` runs the correctness comparison against the CPU oracle instead;
its timings are single-shot and are printed for orientation only.

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
