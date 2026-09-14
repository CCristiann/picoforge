# Phase 4 plan — speculative decoding for a MoE, priced by this machine

Written 2026-09-14. A plan, not a result: every claim below marked
*expected* is arithmetic waiting for a measurement.

## Where the gap is

A literature search on 2026-09-14 found three pieces of work that each hold
half of the idea, and nothing holding all of it:

- **BaseRT** ([arXiv 2607.19438](https://arxiv.org/abs/2607.19438)) already
  runs Qwen3-30B-A3B on an M5 Pro through TensorOps (prefill 3834 tok/s at
  2048, decode 105 tok/s). Its own future-work section names speculative
  decoding -- "a batched verification step whose matmuls are compute-bound" --
  and leaves it undone.
- **EcoSpec** ([arXiv 2607.12696](https://arxiv.org/abs/2607.12696)) observes
  that in a MoE, verifying k draft tokens reads the *union* of their experts
  ("expert scattering"), and picks drafts that reuse experts. Evaluated on
  8xH200 only, with a separately trained expert predictor, no quantised experts.
- **EVICT** ([arXiv 2605.00342](https://arxiv.org/abs/2605.00342)) truncates
  draft trees by profiled verify cost, on A100s.

The picoforge contribution: **expert-cost-aware speculative decoding on unified
memory and the M5 Neural Accelerators, with quantised experts, drafted by
Qwen3-0.6B** -- the model this engine already runs, which shares the 30B's
tokenizer. Novel as far as that search went; re-search before claiming it.

## Why this machine changes the cost model

Dense 0.6B, measured in Phase 2: verifying 32 tokens costs about as much as
decoding one, because the kernel reads the same weights either way. A MoE
breaks that. Qwen3-30B-A3B (expected config -- to be read from config.json,
never hardcoded): 48 layers, 128 experts, 8 per token, expert width 768,
hidden 2048. If routing were uniform, 32 tokens would touch
128 * (1 - (120/128)^32) = ~112 experts per layer instead of 8.

Two things could be what that costs, and they predict different schedulers:

1. **Bytes.** 104 extra experts x 3 matrices x 2048 x 768 at ~4.5 bits = ~280 MB
   per layer, ~13 GB per verify step. On H200 this is EcoSpec's whole story.
2. **Calls.** Phase 3 measured ~33 us per TensorOps matmul regardless of size.
   112 experts x 3 x 48 layers = 16,128 calls = ~0.5 s -- if that floor also
   holds inside one command buffer, which has never been measured.

Step 4.1 settles which one is real, before anything is built on top of it.

**Measured (step 4.7, synthetic layer):** bytes. One MoE layer costs
0.35 ms + 43 us per distinct expert whatever the number of tokens up to 32,
and 43 us is an expert's 9.4 MB at 221 GB/s. The "calls" arithmetic above
assumed one dispatch per expert; the grouped kernel runs them all in one.

## Steps

Each one is small, verified against an oracle, and committed on its own.

| step | what | needs the 30B? |
|---|---|---|
| 4.1 | Dispatch cost inside one command buffer: time vs number of dispatches, big and tiny shapes -- **done** | no |
| 4.2 | Speculative decoding on the dense 0.6B, lossless: greedy output must equal plain greedy, token for token -- **done** | no |
| 4.3 | MoE block in the NumPy oracle, verified against transformers on a tiny random Qwen3-MoE -- **done** | no |
| 4.4 | MoE on the C CPU path, parity with the oracle on the tiny model -- **done** | no |
| 4.5 | Qwen3-30B-A3B: download (61 GB), quantise, layer-by-layer parity -- **done** | yes |
| 4.6 | MoE on the GPU: routed expert matmuls grouped per expert on TensorOps -- **done**, bf16 and q8_row experts, verified on the real 30B | no |
| 4.7 | The verify cost surface on this machine: time(k tokens, distinct experts) -- **done**, synthetic layer and real routing overlap | partly |
| 4.8 | Expert-cost-aware draft selection, training-free, 0.6B drafting; speedup vs plain speculation vs none -- **cost-aware draft length done** (1.44x geomean); expert-level prediction next | yes |

Correctness rule for 4.2 and 4.8: speculative decoding with greedy
acceptance must not change a single output token. Any difference is a bug.
