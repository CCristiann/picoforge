# Design

picoforge is a single-model LLM inference engine for Apple Silicon, written in
C11 and Metal. It is deliberately narrow, one model family at a time with no ML
framework inside the engine, and small enough to be read end to end. Its
target is **Qwen3.8-Flash-Next** (a 48-layer mixture-of-experts model with a
Gated DeltaNet / gated-attention layout and multi-token prediction), run from
its quantised GGUF file.

The stages, with what is done and what is next, are listed below. An earlier
Qwen3 dense/MoE engine (Qwen3-0.6B and Qwen3-30B-A3B) was removed from the
tree and lives at the git tag `archive/pre-qwen38-focus`; its story is in
[`history.md`](history.md).

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

## Stages

Each stage has a measurable exit and closes with its evidence under
`bench/qwen38/` and an entry in [`devlog.md`](devlog.md). The speed reference
is DwarfStar `fc80bd6` on the same file and machine.

| Stage | Content | State |
|---|---|---|
| R0 Measure | Capability probe, per-op GPU profile, bytes and dispatches per token, paired harness with DwarfStar | Done |
| R1 Decode | Cooperative quantised matvecs, fused MoE, leaner graph | Done: 0.62x short, 0.37x at 8K |
| R2 Terminal | `./picoforge` as a daily multi-turn chat with KV reuse; decode closer to DwarfStar | Done: 0.75x short, 0.72x at 8K |
| R3 Prefill | GEMM on simdgroup matrices and TensorOps, grouped experts, batched QSA | Done: 8K prompt 215 s -> 61.0 s (FP32), 37.6 s (half) |
| R4 MTP | Adaptive, cost-aware draft depth; multi-row verify | Done: 1.09-1.65x over plain decode, exact greedy |
| Decode II | Token split over 4 command buffers, fused shared-expert activation | Done: 0.78x (target 0.95x not met) |
| Decode III | Reordered-sum kernels under measured tolerances (router, HC tail, IQ2 gate+up, Q2_K down), shared expert overlapped | Done: 0.86x short, 0.86x at 8K |
| Decode IV | Q8_0 matvec lane map, RMS and HC_ADD folded into neighbouring kernels, one-dispatch GDN conv | Done: 0.92x short, 0.92x at 8K |
| Decode V | GDN prepare folded into the scan, router top-k with per-lane cache | In progress: 33.09 -> 32.68 ms/token (A/B) |
| R5 SSD streaming | Expert pool with parallel reads and event gates; the Q4 file on 64 GiB | Done: Q4 at ~18-19 tok/s with a warm pool |
| P3-P5 Long prefill | Chat-path overheads, GDN scan addressing, MoE tiles, split attention | Done: 8K prompt 46.2 s (FP32), 24.9 s (half) |
| R7 Quality | Q2 against Q4 on a frozen corpus | Done: KL 0.40, top-1 77%, PPL +8% |
| R6 Next generation | Support for the next Qwen release (a 27B dense and a Flash-Next sibling) | Waiting for the release |
| R8 Closure | Final protocol benchmarks, the per-technique ladder, writeup | Next |

**Not goals:** an API server or coding agent, vision, batched sessions, other
GPU backends. **Failures even if fast:** output outside the correctness gate,
unexplained drift, kernels copied from DwarfStar (it is the speed reference;
an adapted idea is cited with its pinned commit), numbers without the
protocol.

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
| `bench/` | measured data by stage (`qwen38/`), microbenchmark CSVs, the Qwen3 archive |
| `docs/` | this file, the benchmark protocol, the development log, the project history, weight and corpus locks |

## Conventions

- Small commits, imperative subject lines, one logical change each.
- Model weights are never committed; their SHA256 sums are recorded in
  [`weights.lock`](weights.lock).
- Specs, plans and handoff notes live in the gitignored `notes/` folder and are
  not published; the public record is the devlog, this file and `bench/`.
- A negative result is committed with its measurement before the code is
  removed, so the comparison stays reproducible from history.
