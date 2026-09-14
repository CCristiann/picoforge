# Design

picoforge is a single-model LLM inference engine for Apple Silicon, written in
C11 and Metal. It is deliberately narrow — one model family at a time, no ML
framework inside the engine — and small enough to be read end to end. Its
reference target is Qwen3: Qwen3-0.6B for the dense engine, Qwen3-30B-A3B for
mixture-of-experts.

## Principles

Code comments refer to these by number.

1. **Correctness before speed.** A faster path with unexplained drift in
   attention, the K/V cache or the logits is not accepted. Every numerical
   component is validated against a reference before it is optimised.
2. **Chain of oracles.** HuggingFace transformers is the reference for the
   NumPy oracle (`tools/oracle/`); the oracle is the reference for the C CPU
   path; the CPU path is the reference for the Metal kernels. A divergence is
   a bug until it is explained and documented.
3. **Small and readable.** Plain C11, Metal Shading Language for kernels,
   Objective-C only where Metal requires it. Few files, short functions,
   comments that explain why. No external dependencies in the engine; Python
   (NumPy, PyTorch, transformers) is used only under `tools/` and `tests/`.
4. **Every number is measured, never guessed.** Benchmarks follow
   [`BENCHMARKS.md`](BENCHMARKS.md): warm-up by time, 25 repetitions, median
   with p10/p90, thermal state recorded, raw CSV and plotting script committed
   under `bench/`. Test tolerances are calibrated on measured values.
5. **Read `config.json`, never hardcode.** Every model dimension is parsed at
   load time and printed as an architecture summary, byte-identical to the
   oracle's.

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
| `src/` | engine C sources, Objective-C Metal glue, `kernels.metal` |
| `tools/oracle/` | NumPy reference implementation and its verification against transformers |
| `tools/quant/` | quantisers (q8_row, q4_g32, ...) |
| `tools/synth/` | synthetic checkpoints at real shapes, sharding |
| `tools/eval/` | perplexity corpus, routing-overlap analysis |
| `tools/probe/` | standalone Metal 4 API probes |
| `tools/ui/` | a local web UI in front of the engine binary |
| `tests/` | correctness harnesses (`make test-all`) |
| `bench/` | raw benchmark CSVs and plots |
| `docs/` | this file, the benchmark protocol, the development log |

## Roadmap

| milestone | scope | status |
|---|---|---|
| 0 | NumPy oracle, logits matched against transformers | done |
| 1 | C engine on CPU: safetensors, byte-level BPE, forward pass, K/V cache, sampling, chat template | done |
| 2 | Metal: naive, simdgroup and TensorOps matmul kernels; full forward pass on the GPU | done |
| 3 | Quantisation: q8 and q4 block formats, perplexity evaluation | done |
| 4 | Qwen3-30B-A3B: MoE routing on the GPU, expert-cost-aware speculative decoding | in progress — see [`phase4-plan.md`](phase4-plan.md) |

## Conventions

- Small commits, imperative subject lines, one logical change each.
- Model weights are never committed; their SHA256 sums are recorded in
  [`weights.lock`](weights.lock), and derived corpora in `corpus.lock`.
- A negative result is committed with its measurement before the code is
  removed, so the comparison stays reproducible from history.
