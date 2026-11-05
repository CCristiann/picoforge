# picoforge

A small, readable inference engine for **Qwen3.8-Flash-Next** on Apple Silicon,
written in C11 and Metal. It runs the full 48-layer graph on the GPU from the
quantised GGUF file, with no ML framework in the engine; Python appears only in
`tools/` and `tests/` for reference implementations and checks.

The goal is to understand where inference time and memory go, so correctness
comes first and every number is measured. All results come from one machine
(MacBook Pro, M5 Pro, 64 GiB) and make no general claim of a lead over other
runtimes. The design follows the philosophy of
[DwarfStar](https://github.com/antirez/ds4): one model at a time, self-contained,
small enough to read end to end.

## Status

| Area | State |
|---|---|
| Decode on the GPU | 0.78x of DwarfStar `fc80bd6` on the same file and machine, 26.0 tok/s at the short frontier |
| Prefill, 8K prompt | 61.0 s in FP32, 37.6 s with the opt-in `--prefill-precision half` |
| MTP (multi-token prediction) | Exact greedy verification; 1.09-1.65x over plain decode |
| Terminal chat | Multi-turn with KV reuse, `/read`, `/ctx`, `/think`, Ctrl+C |
| Next | Explain the routed-MoE kernels with GPU counters to close the rest of the decode gap, then SSD expert streaming for the Q4K file |

Support for the next Qwen generation (its 27B dense model and a Flash-Next
sibling) is planned for after its release. Nothing for it exists yet.

## Requirements

An M5-family Mac, macOS 26.2 or later (the Neural Accelerators are reached through
Metal 4 TensorOps), Xcode with the Metal 4 toolchain, and Python 3.12+ for the
tooling. The Q2 checkpoint is 137 GiB and is never committed:
[`docs/qwen38-q2-baseline.md`](docs/qwen38-q2-baseline.md) records its provenance,
and `tools/eval/download_qwen38_q2.py` fetches and verifies it against
[`docs/weights.lock`](docs/weights.lock). The weights use the Qwen Community License.

## Build and run

```sh
make
export PICOFORGE_MODEL=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf

./picoforge                            # interactive chat: /help /read FILE /ctx N /think /quit
./picoforge -p "What is 2 + 2?" -n 32  # one answer, then exit
./picoforge -p "..." --mtp             # with multi-token prediction
./picoforge bench                      # 3 warm-up + 20 runs: median and p10/p90 tok/s
```

Context is 8K by default, decoding is greedy and thinking is off unless `--thinking`
or `/think` is used. `--mtp` drafts up to three tokens per cycle, chosen by a cost
model measured on this machine (`--mtp-draft 0..3` fixes the depth); the output is
always identical to plain greedy decoding. The long form
`./picoforge --qwen38 FILE.gguf [flags]` keeps every measurement flag.

**Run one process at a time.** About 45 GiB of wired pages are released for roughly
1.5 s after exit, and mapping the file again in that window froze this Mac twice.
Scripted real runs go through `tools/qwen38_gpu_lock.py`.

## How correctness is checked

Kernels are judged by a chain of oracles: an independent FP32 reference implementation,
then the C CPU path, then the Metal kernels. A faster kernel changes the order of the
floating-point sums, so bit-identity is not expected; instead each kernel gets a
tolerance fixed from the measured error distribution. The end-to-end gate does not
move: logits within 1e-3 of the FP32 reference and identical greedy output on a
frozen suite ([G2 spec](docs/specs/qwen38/G2.md)). Speculative paths must emit
exactly the tokens plain greedy decoding emits.

```sh
make test-all          # offline checks: no checkpoint, no long GPU workload
make test-qwen38       # the full Qwen3.8 gate (needs the reference assets, see tools/qwen38.mk)
make microbench && ./build/microbench --check   # the matmul kernels against a CPU reference
```

## Measurements

Benchmarks follow [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): warm-up, at least 20
repetitions, median with p10/p90, alternating order against DwarfStar, raw CSVs
committed. They are small pilots, not general claims.

| Stage | Result | Evidence |
|---|---|---|
| R1 decode | 0.62x DwarfStar short prompt, 0.37x at 8K | [R1](bench/qwen38_r1_20261028/README.md) |
| R2 decode | 24.15 vs 32.33 tok/s short (0.75x), 21.73 vs 30.14 at 8K (0.72x) | [R2](bench/qwen38_r2_20261030/README.md) |
| Decode II | 25.97 vs 33.22 tok/s short (0.78x; R2 was 0.75x); decode token in 4 command buffers, fused shared-expert activation | [D1](bench/qwen38_d1_20261105/README.md) |
| R3 prefill | 8K prompt 215 s to 61.0 s (FP32) / 37.6 s (half) | [R3](bench/qwen38_r3_20261102/README.md) |
| R4 MTP | 33.2 / 25.6 / 27.0 / 41.0 tok/s on code / chat / story / counting | [R4](bench/qwen38_r4_20261103/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Engine C sources and Metal kernels (`qwen38_*`), GGUF reader |
| `tools/` | Python oracles, fixtures, benchmark runners, probes, the matmul microbenchmark |
| `tests/` | Correctness harnesses and fixtures |
| `bench/` | Measured data, run evidence and plots |
| `docs/` | [Design](docs/DESIGN.md), [benchmark protocol](docs/BENCHMARKS.md), [roadmap](docs/specs/qwen38/roadmap.md), [devlog](docs/devlog.md) |

An earlier Qwen3 dense/MoE engine and its tools are no longer in the tree; they remain
in git history at the tag `archive/pre-qwen38-focus`.

## License

MIT, see [LICENSE](LICENSE).
