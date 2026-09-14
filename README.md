# picoforge

A single-model LLM inference engine for Apple Silicon, in C11 and Metal.

picoforge runs Qwen3 end to end — tokenizer, forward pass, K/V cache, sampling,
quantised weights, mixture-of-experts routing and speculative decoding — with no
ML framework inside the engine. The matrix multiplications run on the M5's GPU
Neural Accelerators through Metal 4 tensor operations. Every component is
verified against a reference implementation, and every performance number is
measured on one reference machine by a written protocol.

## Highlights

Measured on an Apple M5 Pro (64 GB, 307 GB/s). Raw data in [`bench/`](bench/),
protocol in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

- **Qwen3-30B-A3B on one laptop GPU:** q8_row (31.2 GB), 21.0 tok/s decode at
  cache depth 512, verified against the CPU path, which is verified against
  the NumPy oracle and transformers on the real weights.
- **Adaptive, lossless speculative decoding for MoE:** Qwen3-0.6B drafts for the
  30B; the draft length is chosen before every pass from measured acceptance
  and measured drafting and verification cost. 1.44x geometric-mean decode
  speedup over five prompts (1.11x-1.67x), output identical to greedy in every
  run, and better than any fixed draft length.
- **Qwen3-0.6B on the GPU:** 4490 tok/s prefill (512 tokens), 75.9 tok/s decode
  at cache depth 512, bf16.
- **Lossless speculative decoding:** greedy verification with a prompt-lookup
  or draft-model drafter; output identical to plain greedy, token for token,
  across the test suite. Up to 1.64x decode on text that repeats its context.
- **Mixture of experts on the GPU:** Qwen3-MoE routing, grouping and expert
  products run inside one command buffer — nine dispatches per layer regardless
  of the routing. bf16 and q8_row experts.
- **A measured verify-cost model for MoE on this machine:** one Qwen3-30B-A3B
  layer costs ~0.35 ms + 22 µs per distinct q8 expert touched, nearly
  independent of how many tokens share them
  ([plot](bench/moe_verify_cost.png)); and the trained router's locality keeps
  that union small: eight consecutive tokens of real text touch 39% of the
  experts uniform routing would.
- **Quantisation:** q8_row (no measurable perplexity cost) and q4_g32 formats,
  consumed natively by the matrix units.

## Status

| milestone | scope | status |
|---|---|---|
| 0 | NumPy reference implementation matched against transformers | done |
| 1 | C engine on CPU: safetensors, byte-level BPE, forward pass, K/V cache, sampling | done |
| 2 | Metal: naive, simdgroup and TensorOps kernels; full forward pass on GPU | done |
| 3 | Quantisation: q8 / q4 formats, perplexity evaluation | done |
| 4 | Qwen3-30B-A3B: GPU MoE, expert-cost-aware speculative decoding | in progress |

Details and the reasoning behind each result are in
[`docs/devlog.md`](docs/devlog.md); the Phase 4 plan and related work are in
[`docs/phase4-plan.md`](docs/phase4-plan.md).

## Requirements

- macOS 26.2 or later with Xcode and the Metal 4 toolchain.
- An M5-family Mac for the TensorOps kernels (the default GPU path).
- Python 3 for the reference implementation, tests and tools (not needed by
  the engine itself).

## Quick start

```sh
make                                   # engine + Metal library

python3 -m venv tools/venv
tools/venv/bin/pip install -r tools/requirements.txt
tools/venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

./picoforge models/Qwen3-0.6B --gpu-chat "Why is decoding limited by memory bandwidth?"
```

Other entry points:

```sh
./picoforge models/Qwen3-0.6B --spec-greedy "def fibonacci(n):" 128 4 out.txt   # speculative decoding
make quant-models                       # q8_row and q4_g32 checkpoints
make ui                                 # local web UI on http://127.0.0.1:8000
make test-all                           # every correctness harness
```

## How correctness is established

A chain of references, each link tested:

1. `tools/oracle/` — a NumPy implementation, matched against HuggingFace
   transformers (fp32, eager attention).
2. The C CPU path, matched against the NumPy oracle on logits, and against
   transformers on generated tokens.
3. The Metal kernels and GPU forward pass, matched against the CPU path.

Tolerances are calibrated on measured values, and verifiers are themselves
tested by injecting deliberate bugs. See [`docs/DESIGN.md`](docs/DESIGN.md).

## Repository layout

```
src/        engine (C11), Metal glue (Objective-C), kernels.metal
tools/      reference implementation, quantisers, synthetic checkpoints, UI
tests/      correctness harnesses
bench/      benchmark CSVs and plots
docs/       design, benchmark protocol, development log
```

## Acknowledgements

The Qwen3 models are released by the Qwen team under Apache-2.0. The engine's
narrow, single-model design is inspired by antirez's DwarfStar. Related work
on Apple Silicon and MoE speculative decoding is cited in
[`docs/phase4-plan.md`](docs/phase4-plan.md).

## License

MIT — see [`LICENSE`](LICENSE).
