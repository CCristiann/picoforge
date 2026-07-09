# picoforge — an educational single-model LLM inference engine for Apple Silicon

## What this project is

A self-contained, readable inference engine written in C + Metal, built by a
computer engineering / AI undergraduate to learn how LLM inference works at the
lowest level — in public. Inspired by the philosophy of antirez's DwarfStar (ds4):
intentionally narrow (one model at a time), self-contained (no ML frameworks in
the engine), small enough that one person can hold the whole codebase in their
head, and ruled by "correctness before speed".

- Target model for Phases 0–3: **Qwen/Qwen3-0.6B** (Apache-2.0).
- Stretch target for Phase 4 (the thesis): a MoE of the "64 GB class",
  e.g. **Qwen3-30B-A3B**.
- The long-term public artifact: benchmarks and writeups measured on the exact
  machine below, including the "one kernel, three implementations" experiment
  on the M5 Neural Accelerators.

## Hardware & environment (the only machine that matters)

- MacBook Pro 14" — Apple **M5 Pro**: 18-core CPU (6 super + 12 performance),
  **20-core GPU with one Neural Accelerator per core**, 64 GB unified memory,
  **307 GB/s** memory bandwidth. All performance reasoning starts from that
  bandwidth number: generation is memory-bandwidth-bound, prefill is
  compute-bound.
- macOS 26.x + Xcode with the Metal 4 SDK. The Metal tensor APIs (TensorOps)
  used by the Neural Accelerators are gated: they require an M5-family chip
  AND a recent macOS (≥ 26.2). Always verify availability with a runtime
  probe — never assume.
- macOS caps GPU-wired memory at ~75% of RAM by default; it can be raised with
  `sudo sysctl iogpu.wired_limit_mb=...`. Never suggest changing system
  settings silently — explain what it does and let the human decide and run it.

## Non-negotiable engineering principles

1. **Correctness before speed.** A faster path with unexplained drift in
   attention, KV cache, or logits is never acceptable. Every numerical
   component is validated against a reference before any optimization.
2. **Chain of oracles.** Python/transformers is the oracle for the C CPU path;
   the C CPU path is the oracle for the Metal kernels. Divergence is a bug
   until explained and documented.
3. **Small and readable.** Plain C11 (no C++), Metal Shading Language for
   kernels, Objective-C only where Metal requires it. Few files, short
   functions, generous comments explaining the *why*. No external ML
   dependencies in the engine itself. Python (numpy / torch / transformers)
   is allowed ONLY under `tools/` for verification, conversion, and analysis.
4. **Every number is measured, never guessed.** Benchmark protocol: ≥ 3 warmup
   runs, ≥ 20 measured repetitions, report median and p10/p90, note thermal
   conditions (`powermetrics`), sweep sizes, report effective GFLOPS and GB/s
   against the 307 GB/s theoretical ceiling, and commit raw CSVs under
   `bench/` together with the plotting script.
5. **Read config.json, don't hardcode.** All model dimensions (layers, heads,
   head_dim, vocab size, rope_theta, norm eps, tied embeddings, ...) are
   parsed from the model's config at load time and printed as an architecture
   summary at startup.

## How to work with me (the human) — MOST IMPORTANT SECTION

I am here to LEARN, not to have code generated at me. Treat every session as
pair programming where I am the junior dev who must end up understanding every
line:

- Before writing any component: explain the concept in 5–15 lines (what it is,
  why it exists, the math if relevant), THEN propose an implementation plan,
  THEN code — in that order.
- Work in small increments: one component per step. After each step, give me a
  short verification task that I run myself (a command + the expected output).
- Never introduce more than ~150 lines of new code in a single step without
  asking me first.
- Regularly ask me 1–2 check questions ("why do we scale by 1/sqrt(head_dim)?",
  "what breaks if the KV cache and the position index disagree?"). If my
  answer is wrong, correct me before moving on.
- When a bug is instructive, show me the failing test and the cause, and let me
  attempt the fix first. Only fix it yourself if I ask or if it's boilerplate.
- If I ever tell you "just write the whole thing", remind me of this section
  and split the work anyway.

## Fresh-API caution (Metal 4 / TensorOps)

The GPU Neural Accelerators and the Metal 4 tensor APIs are only a few months
old. Your training data may be stale or wrong about exact API names,
signatures, and availability. Before writing any TensorOps code: consult
current Apple documentation (developer.apple.com — Metal 4, Metal Performance
Primitives, MTLTensor, tensor operations in MSL) and prefer compiling a
minimal standalone probe over assuming. If web search is available, use it to
confirm API details; if not, say explicitly what you are unsure about.

## Repo conventions

- Layout:
  - `src/` — engine C sources and `.metal` kernels
  - `tools/` — Python oracle, converters, bench analysis (with its own venv)
  - `tests/` — correctness harnesses and tolerance definitions
  - `bench/` — raw CSV results + plots
  - `docs/` — `devlog.md`, per-phase writeups, `BENCHMARKS.md` protocol
- Git: small commits, imperative messages, one logical change each. Model
  weights are never committed (gitignored); record their SHA256 in
  `docs/weights.lock` instead.
- A phase is "done" only when: all its tests are green AND `docs/phaseN.md`
  exists, drafted by me (the human) and reviewed by you — not the other way
  around.

## Phase map (work strictly one phase at a time; never start N+1 early)

- **Phase 0 — The Python oracle.** Full forward pass of Qwen3-0.6B in pure
  NumPy, logits matched against HuggingFace transformers.
- **Phase 1 — The C engine on CPU.** Weights loader (safetensors first),
  byte-level BPE tokenizer, full forward pass, KV cache, sampling, chat
  template. Token-level parity with the oracle.
- **Phase 2 — Metal.** Matmul kernel in three versions: naive →
  simdgroup-based → Metal 4 TensorOps on the Neural Accelerators. Benchmark
  harness per the protocol above; then move the full forward pass to GPU.
  Public deliverable: the "one kernel, three implementations" writeup with
  measured prefill numbers on this machine.
- **Phase 3 — Quantization.** Q8 then Q4 block formats (GGUF-style),
  perplexity evaluation to quantify degradation, speed/memory/quality
  trade-off tables.
- **Phase 4 — The thesis.** The 64 GB-class MoE (Qwen3-30B-A3B): expert
  routing, grouped expert GEMM in the prefill, expert caching strategy,
  published end-to-end benchmarks on this exact machine.
