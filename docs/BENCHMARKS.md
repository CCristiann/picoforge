# Benchmark protocol

Every number in this repository is measured on one machine. This file defines
how. Kernel timings and complete generation runs answer different questions,
so their timing boundaries are kept separate. Where the files live is
described in [`bench/README.md`](../bench/README.md). The protocols of the
removed Qwen3 engine are in [`history.md`](history.md).

## The machine

MacBook Pro 14", Apple **M5 Pro**: 18-core CPU, 20-core GPU with one Neural
Accelerator per core, 64 GB unified memory, **307 GB/s** memory bandwidth.
Metal reports a recommended working set of ~51.8 GiB and a single-buffer cap
of ~38.9 GiB. Each run records its OS, toolchain and power state.

Decode is memory-bound, so it is reported against the 307 GB/s ceiling. That
figure is the theoretical peak, not an achievable one: treat 80% of it as
excellent, and anything above it as a bug in the measurement. Prefill is
compute-bound and is reported in tokens/s and effective TFLOPS.

## Rules common to every measurement

- **Warm-up first, discarded.** The first dispatch of a pipeline pays for
  shader caches, page faults on fresh buffers and a GPU at idle clocks.
- **At least 20 measured repetitions**, reduced to the **median with p10 and
  p90**, never the mean. A mean lets one descheduled run dominate; the p10–p90
  spread says whether to trust the median. A baseline too slow for 20
  repetitions is measured with fewer, and the run is labelled a pilot.
- **Raw data committed** under `bench/`, with the script that reduces or plots
  it, so a reader can recheck every claim.
- **One GPU workload at a time**, on AC power, with the thermal state recorded
  (`pmset -g therm` before and after; `powermetrics` needs root and is optional).
- **Memory is not RSS.** `ru_maxrss` misses GPU-mapped weights. Runs record the
  physical footprint, Metal allocations, system memory pressure and swap growth.

## GPU kernels: the microbenchmark

`tools/microbench` measures the machine, not a model. Its CSVs are in
`bench/microbench/`.

- **0.5 s of warm-up per cell, by time, not count.** Five warm-up runs of a
  0.1 ms kernel are half a millisecond of work. A sweep warmed by count once
  measured M=1 at 154 µs for every kernel, with cells getting faster in the
  order they were measured: the GPU was leaving a low-power state. Warmed by
  time, the same cells read 33 µs.
- **25 measured runs**, median with p10/p90.
- **Buffers allocated and uploaded once**, outside the timed region.
  Otherwise the measurement is of the driver.
- **GPU timestamps, not wall clock.** `GPUStartTime` / `GPUEndTime` of the
  command buffer. Wall time around a submit also measures queueing and
  scheduling.
- **The dispatch floor is printed** with every run (an empty kernel, best of
  45, ~6 µs). Anything within a small multiple of it measures the command
  queue, not the work.

The same kernel and shape, measured two ways, shows why this matters:

| naive, M=1, N=1024, K=1024 | |
|---|---|
| single shot, fresh buffers | 8.0 GB/s |
| protocol above | 36.2 GB/s |

The first number is not a pessimistic version of the second: it measures
something else.

**What the sweeps cover.** All on `C[M,N] = A[M,K] · B[N,K]ᵀ`, A in FP32, B in
BF16 (or quantised), C in FP32:

- *batch-sweep*: M from 1 to 512 at N=3072, K=1024. This axis separates
  decode (M=1) from prefill.
- *width-at-M1*: N from 256 to 16384 at M=1, to tell bandwidth limits from
  too little work to fill the GPU.
- *square*: 128³ to 2048³, comparable with published numbers.
- *quant*: the same batch sweep for Q8/Q4 per-row and blocked kernels, with
  the BF16 TensorOps kernel as the baseline in the same file.
- *dispatch*: each command buffer holds `count` copies of one matmul, so GPU
  time / `count` is the cost of one more dispatch. Every tile-shape variant
  is checked against the reference first, with C poisoned with NaN so that a
  kernel that writes nothing cannot pass.

Each row reports median/p10/p90 seconds, GFLOP/s at `2·M·N·K` and effective
GB/s over the minimum traffic (`M·K·4 + N·K·2 + M·N·4`).

```sh
make microbench
./build/microbench --check                                   # kernels against a CPU reference
./build/microbench --matmul   bench/microbench/matmul_m5pro.csv
./build/microbench --quant    bench/microbench/matmul_quant_m5pro.csv
./build/microbench --dispatch bench/microbench/dispatch_m5pro.csv
./tools/venv/bin/python tools/plot_bench.py bench/microbench/matmul_m5pro.csv
```

## Qwen3.8 generation

**Settings.** The same GGUF file for every engine, 8,192 tokens of context,
1,024-token prefill chunks, greedy decoding, FP32 activations unless a
reduced-precision mode is named explicitly.

**Two frontiers.** A *short* prompt, and a *long* one of 8,063 prompt tokens.
Both are followed by 128 teacher-forced decode tokens, as `ds4-bench` does.
Forcing the IDs removes any dependence on which tokens each engine would
choose, so tok/s compares the same work.

**Against DwarfStar.** `tools/eval/qwen38_compare.py` prepares token IDs that
both tokenizers agree on, then alternates the engines (`fc80bd6` for
DwarfStar), one process per sample: ≥ 3 warm-ups, then 20 measured samples.
Decode throughput is `(generated − 1) / decode_s`, because the first token
comes from prefill. The ratio is median against median.

**`picoforge bench`.** Runs 3 warm-ups and 20 repetitions of an ordinary run
inside one process and prints prefill tok/s, decode tok/s and TTFT as median
and p10/p90. `--bench-csv FILE` keeps the raw rows.

**Profiles.** `--profile FILE.csv` splits the command buffer into one encoder per
operation. That changes the cost, so per-operation rows locate the bottleneck,
and the unsplit total remains the baseline.
`tools/eval/qwen38_profile_summary.py` reduces a profile.

**Quality.** Teacher-forced FP32 logits of two files on a frozen corpus, reduced
in FP64 by `tools/eval/qwen38_r7_quality.py`: KL, top-1 agreement and ΔNLL with
confidence intervals. The corpus window IDs are pinned in
[`corpus.lock`](corpus.lock).

**Safety.** Real runs go through `tools/qwen38_gpu_lock.py`: one process at a
time, waiting for the previous process's wired pages to be released. A
memory-pressure guard stops a child that pushes the system into swap. Mapping
the file again within ~1.5 s of an exit froze this Mac twice.

**Speculative paths** (MTP) must emit exactly the tokens of plain greedy
decoding, warm-ups included. A mismatch aborts the run and keeps the evidence.
