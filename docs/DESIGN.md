# Design

picoforge is a single-model LLM inference engine for Apple Silicon, written in
C11 and Metal. It is deliberately narrow, one model family at a time with no ML
framework inside the engine, and small enough to be read end to end. Its
target is **Qwen3.8-Flash-Next** (a 48-layer mixture-of-experts model with a
Gated DeltaNet / gated-attention layout and multi-token prediction), run from
its quantised GGUF file.

The stage plan, with what is done and what is next, is in
[`specs/qwen38/roadmap.md`](specs/qwen38/roadmap.md). An earlier Qwen3
dense/MoE engine (Qwen3-0.6B and Qwen3-30B-A3B) was removed from the tree and
lives at the git tag `archive/pre-qwen38-focus`.

## Principles

Code comments refer to these by number.

1. **Correctness before speed.** A faster path with unexplained drift in
   attention, the K/V cache or the logits is not accepted. Every numerical
   component is validated against a reference before it is optimised.
2. **Chain of oracles.** An independent FP32 reference (`tools/oracle/`) is
   the reference for the C CPU path; the CPU path is the reference for the
   Metal kernels. A divergence is a bug until it is explained and documented.
   Fast kernels change the order of floating-point sums, so each gets a
   tolerance fixed from the measured error distribution.
3. **Small and readable.** Plain C11, Metal Shading Language for kernels,
   Objective-C only where Metal requires it. Few files, short functions,
   comments that explain why. No external dependencies in the engine; Python
   (NumPy, PyTorch, transformers) is used only under `tools/` and `tests/`.
4. **Every number is measured, never guessed.** Benchmarks follow
   [`BENCHMARKS.md`](BENCHMARKS.md): warm-up by time, 25 repetitions, median
   with p10/p90, thermal state recorded, raw CSV and plotting script committed
   under `bench/`. Test tolerances are calibrated on measured values.
5. **Read the model's metadata, never hardcode.** Every model dimension is
   parsed from the GGUF metadata at load time and printed as an architecture
   summary. A kernel specialised for today's shapes takes them as parameters
   and asserts them at load, so a sibling model with the same architecture
   needs a new configuration, not a rewrite.

## Fresh APIs

The M5 Neural Accelerators are reached through Metal 4 tensor operations
(`MetalPerformancePrimitives`, TensorOps), which require an M5-family GPU and
macOS 26.2 or later. The API is young and its headers have been wrong in
detail; every assumption about it is compiled into a probe
(`tools/probe/`) and checked on the device before code depends on it. The
findings are recorded in the kernel comments and in [`devlog.md`](devlog.md).

## Reference machine

All published numbers come from one machine: MacBook Pro 14", Apple M5 Pro,
18-core CPU, 20-core GPU with a Neural Accelerator per core, 64 GB unified
memory, 307 GB/s memory bandwidth. Decode throughput is reported against that
bandwidth ceiling. Metal caps a single buffer at 41.75 GB on this machine,
which is why checkpoints larger than that run quantised.

## Repository layout

| path | contents |
|---|---|
| `src/` | engine C sources, Objective-C Metal glue, `.metal` kernels (`qwen38_*`), GGUF reader |
| `tools/oracle/` | independent reference implementations of the Qwen3.8 operators |
| `tools/synth/` | synthetic GGUF checkpoints for tests |
| `tools/eval/` | benchmark runners, grading and reports |
| `tools/probe/` | standalone Metal 4 API probes |
| `tools/microbench/` | the model-agnostic matmul and dispatch-cost benchmark |
| `tools/ui/qwen38/` | a local web chat in front of the engine (secondary to the terminal) |
| `tests/` | correctness harnesses (`make test-all`, `make test-qwen38`) |
| `bench/` | raw benchmark CSVs, run evidence and plots |
| `docs/` | this file, the benchmark protocol, the development log, specs |

## Conventions

- Small commits, imperative subject lines, one logical change each.
- Model weights are never committed; their SHA256 sums are recorded in
  [`weights.lock`](weights.lock).
- A negative result is committed with its measurement before the code is
  removed, so the comparison stays reproducible from history.
