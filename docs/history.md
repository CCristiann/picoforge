# History: from Qwen3-0.6B to Qwen3.8-Flash-Next

This file condenses the project's earlier reports into one place: the Qwen3
engine (Phases 0–4), the October 2026 model choice, and the first Qwen3.8
steps that came before the native engine. The day-by-day record is in
[`devlog.md`](devlog.md). The Qwen3 engine and its tools were removed from the
tree on 2026-11-05. They remain at the git tag `archive/pre-qwen38-focus`, and
their measurements are in [`bench/archive-qwen3/`](../bench/archive-qwen3/).

## 1. The Qwen3 engine (July – October 2026)

| Phase | What it established |
|---|---|
| 0 | A NumPy forward pass of Qwen3-0.6B, logits matched against transformers |
| 1 | The C CPU engine: safetensors loader, byte-level BPE, KV cache, sampling, chat template; token-level parity with the oracle |
| 2 | Metal: matmul three ways (naive, simdgroup, Metal 4 TensorOps), then the whole forward pass on the GPU |
| 3 | Q8/Q4 per-row and block formats, built around the matrix units; perplexity, KL and top-1 measured against the unquantised model |
| 4 | Lossless speculative decoding, a MoE block (oracle, then CPU, then GPU), and Qwen3-30B-A3B on the GPU |

### Phase 4: speculative decoding for a MoE

**The hypothesis (2026-09-14).** In a MoE, verifying k draft tokens reads the
*union* of their experts. A literature search found work holding half the
idea each: BaseRT (Qwen3-30B-A3B on an M5 Pro, with speculative decoding left
as future work), EcoSpec (draft selection that reuses experts, on H200s) and
EVICT (draft trees cut by profiled verify cost, on A100s). The plan was
expert-cost-aware drafting on unified memory, with quantised experts and
Qwen3-0.6B as the drafter.

**What verification costs here.** Two models were possible: bytes (every extra
expert must be read) or calls (~33 µs per TensorOps dispatch). Measured on a
synthetic 30B-shaped layer, it is **bytes**: 0.35 ms + 43 µs per distinct
expert, for any number of tokens up to 32. 43 µs is one expert's 9.4 MB at
221 GB/s. A grouped kernel runs every expert in one dispatch, so call count
stops mattering.

**Adaptive draft length, measured (2026-10-11).** Qwen3-30B-A3B q8_row as
target, Qwen3-0.6B as drafter. Ten local prompts, five measured requests each,
up to 128 tokens. All 180 outputs matched ordinary greedy decoding.

| Configuration | Decode | Generation | Process | Worst prompt (decode) |
|---|---:|---:|---:|---:|
| No draft | 1.000x | 1.000x | 1.000x | 1.000x |
| Model, k=2 | 1.412x | 1.102x | 1.099x | 1.117x |
| Model, adaptive | 1.427x | 1.136x | 1.131x | 1.073x |

Adaptive drafting was only 1.09% ahead of a fixed k=2, and five repetitions
cannot show that this difference is stable. Most of the gap between decode
and process speedups fell inside the first target forward pass, which included
first-use and residency costs.

**The expert-cost predictor: a negative result.** An opt-in `auto-experts`
mode forecast each verification's expert union from the last 32 accepted
routes. It fit verify time as an intercept plus a cost per distinct expert,
learning from every pass, rejected drafts included. On six separate prompts
(five requests each, 120 measurements, all matching greedy) it reached
**1.009x decode and 0.988x process speed** against `auto`. The adoption
threshold was a stable 10%. Its verify-time error was 2.459 ms against the
plain timing estimate's 1.909 ms. The default stayed `auto`. The run was
interrupted once and one prompt was repeated after a disagreement between
the engine's and the runner's clocks. The runner then moved to the engine's
`CLOCK_MONOTONIC`, and the metadata records exactly which records were kept.

**Correctness fixes found by the 2026-10-11 review.**
- Attention kept a fixed 2,048-float score array, and 4,096-position routing
  traces wrote past it. The kernel now sizes it from the active context and
  checks GPU capacity, with tests at 2,048, 2,049 and 4,096 positions.
- The corrected routing trace touches 29.8 distinct experts per layer across
  eight consecutive tokens, against 51.6 under uniform routing. The earlier
  locality estimate was withdrawn
  (`routing_overlap_qwen3_30b_corpus.csv` is invalid beyond 2,048 positions).
- `ru_maxrss` reported 38–45 MB for a 31.2 GB checkpoint. Process RSS does not
  measure unified-memory use, and later protocols record footprint, Metal
  allocations, pressure and swap instead.

### The Qwen3-era measurement protocols

The matmul and dispatch protocol carried over unchanged; it is in
[`BENCHMARKS.md`](BENCHMARKS.md). The other Qwen3 files followed these rules:

- **Generation requests** (`spec_*`): a fresh process per request, one
  discarded warm-up, shuffled order with a recorded seed. Every output was
  checked against an independent greedy run. Decode throughput is
  `(generated − 1) / decode_s`. Speedups are geometric means of per-prompt
  median ratios, and the worst prompt is reported too.
- **End to end** (`e2e_*`): whole GPU forward passes timed by the command
  buffer. Prefill was 512 tokens, decode one token at position 512 and verify
  32 tokens at position 512, so that decode reads a realistic cache.
- **Quality** (`quant_quality_*`): a corpus pinned by SHA-256, 1,024-token
  windows with the second half scored. Each format is compared pairwise with
  the unquantised model (dNLL, KL, top-1), in FP64, with standard errors taken
  across windows rather than tokens.
- **MoE verify cost** (`moe_verify_cost_*`): one synthetic layer with routing
  forced so that n tokens touch exactly D experts. The harness reads back the
  number of groups the GPU formed and aborts if it differs from D.

## 2. Choosing a larger model (2026-10-11)

The goal changed: run a substantially larger model on this 64 GiB Mac. That
lifted the earlier ban on SSD offload. Checkpoint sizes considered at the time
(file size, not runtime memory):

| Candidate | Total / active | Artifact | Assessment |
|---|---|---|---|
| **Qwen3.8-Flash-Next** | 125B / 6B, plus 51B n-gram and 4B MTP | DwarfStar Q2: 41.73 GiB resident + 95.37 GiB disk-only n-gram tables | Chosen. The main weights fit; n-gram rows are read selectively from the SSD |
| Qwen3.5-122B-A10B | 122B / 10B | UD-IQ3_S, 43.36 GiB | Fully resident alternative |
| Qwen3-Coder-Next | 80B / 3B | Q4_K_M, 45.20 GiB | Coding-oriented comparison; milder compression |
| DeepSeek V4 Flash | 284B / 13B | Q2, ~81 GiB | Needs expert streaming |
| GLM-5.3-Flash | 320B / 18B | Q2, ~90 GiB | Larger still |

A separate look at the 80–100B class recommended Qwen3-Coder-Next and
Qwen3-Next-80B-A3B-Thinking at 4 bits as more conservative alternatives. The
decision stayed with Qwen3.8-Flash-Next Q2, with the caveat that parameter
count and vendor scores say nothing about a Q2 file's quality until it is
measured locally.

The probe at the time read: 64 GiB physical, `recommendedMaxWorkingSetSize`
51.84 GiB, `maxBufferLength` 38.88 GiB, 32 KiB of threadgroup memory. The
working set and the buffer cap are separate limits: several buffers get past
the per-buffer cap, but they add no RAM.

## 3. First Qwen3.8 steps (2026-10-11 – 10-12)

**DwarfStar baseline.** Before writing any Qwen3.8 code, the pinned Q2 file
(`d600fe1`, 147,207,127,040 bytes, SHA-256 in [`weights.lock`](weights.lock))
was run in DwarfStar `fc80bd6` on this Mac. Settings: 8K context, 1,024-token
prefill chunks, greedy, no MTP. Five frozen local tasks ran with one warm-up
and three measured requests each. Decode ran at 30.0–32.3 tok/s, and the
stock `ds4-bench` control gave 600.7 prefill tok/s at 8,063 tokens and 30.9
decode tok/s. Peak Metal allocation was 46.5 GiB, with no sustained swap.
Every prompt ID matched the original Hugging Face tokenizer and template. Four
of five answers passed their executable tests, and two of five also followed
the requested output format. Evidence:
[`bench/qwen38/q2_20261011_phase0/`](../bench/qwen38/q2_20261011_phase0/).

**Native GGUF storage.** `src/gguf_reader.c` and `src/gguf.c` parse GGUF v3
with bounded reads and overflow, alignment, overlap and duplicate checks, and
inventory the eight formats present (F32, F16, BF16, Q8_0, IQ2_XXS, Q2_K, Q4_K,
MXFP4). `src/gguf_plan.c` splits the resident tensors into page-aligned
no-copy `MTLBuffer` views at tensor boundaries and leaves out the 95.37 GiB
n-gram table. On the real file, all 1,256 descriptors agree with an
independent Python inventory, and 1,255 resident tensors fit in two views
(44.8 GB). 3,765 GPU byte probes match `pread`. Evidence:
[`bench/qwen38/q2_20261012_storage/`](../bench/qwen38/q2_20261012_storage/).

From there the native engine was built component by component against
independent oracles: configuration and tokenizer, the compressed formats,
the recurrent operators, QSA attention, n-gram lookup, MoE, then the CPU graph
(G1) and the GPU graph with ordinary generation (G2/G3). The R0–R7 stages that
followed are in the devlog and in [`bench/qwen38/`](../bench/qwen38/).
