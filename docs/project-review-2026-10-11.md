# Project review — 2026-10-11

Reviewed at `a842853`; updated with the correctness and measurement work that
followed. This document separates implemented work from proposed experiments.

**Subsequent direction change:** the user clarified that the goal is to run a
substantially larger model on the 64 GiB M5 Pro. The
[large-model research](large-model-research-2026-10-11.md) supersedes this
review's work order and restriction against SSD offload for future work.
The correctness findings and benchmark limitations below still apply.

## Direction

Keep picoforge small: Qwen3, C11, Metal, one tested machine, readable code and
reproducible measurements. This is a useful systems project. Commercial demand
and a performance advantage over established runtimes have not been demonstrated.

A first release should let another developer build the engine, run a supported
checkpoint, reproduce a result and understand its limits without a hosted service.

## What the review found

- The attention kernel had a 2,048-float score array. Routing measurements used
  4,096 positions and could write past it. The new kernel sizes that array for
  the active context and checks GPU capacity. Boundary tests and the routing
  trace have been rerun. Earlier routing estimates are superseded.
- The scheduler measures acceptance, drafting time and verification time. It
  does not predict which experts the next draft will use. That remains an experiment.
- The old speculative runner kept medians from three repetitions over five
  prompts. Adaptive drafting reached 1.4413x versus no draft; fixed `k=2`
  reached 1.4091x. The adaptive margin was only 2.30%.
- The new runner records raw runs, warmups, randomised order, model and source
  hashes, process memory, and separate prefill/decode/process times. Its ten
  prompts are a local pilot suite, not a standard benchmark or held-out test set.
- The new pilot completed with exact greedy parity. Adaptive decode is only
  1.09% ahead of fixed `k=2`; see [the results](BENCHMARKS.md#october-11-pilot).
  First-use costs inside the target prefill now need their own measurement.
  Process RSS is recorded but does not capture the full GPU memory footprint.
- Q4 quality measurements cover the 0.6B and a local corpus. They do not establish
  30B quality. The GPU MoE path still supports bf16 and q8_row experts only.

## Related work

Primary sources checked on 2026-10-11. External speeds have not been reproduced here.

| Project | What to learn from it |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | A required performance baseline; usable CLI and server interfaces |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon inference, prompt caches and draft-model decoding |
| [BaseRT](https://github.com/basecompute/baseRT) | Separate GEMV decode from TensorOps prefill/verification; the engine is a proprietary binary, unlike its open CLI and bindings |
| [DwarfStar](https://github.com/antirez/ds4) | Keep the supported model set narrow and the whole path understandable |
| [Uzu](https://trymirai.com/blog/speculative-decoding-in-uzu) | Adaptive tree speculation and the relationship between draft quality and verify cost |
| [DFlash-MLX](https://github.com/bstnxbt/dflash-mlx) | Parallel drafting and kernels built for verification |
| [EcoSpec](https://arxiv.org/abs/2607.12696), [EVICT](https://arxiv.org/abs/2605.00342) | Expert cost already informs speculative decoding; that idea alone is not a new contribution |

The [BaseRT paper](https://arxiv.org/html/2607.19438v1) reports about 105 tok/s
for Qwen3-30B-A3B at four bits on its M5 Pro. picoforge's historical 21 tok/s
uses q8 and a different protocol. Dividing those figures would not be a fair
comparison. A local comparison must match prompts, output length, cache state
and precision as closely as possible, then document the remaining differences.

[Ollama](https://docs.ollama.com/api/openai-compatibility) and
[LM Studio](https://lmstudio.ai/docs/developer) already cover local chat and
application APIs. Adding an API can make picoforge easier to use; it would not
by itself establish a reason to choose it.

## Work order

1. **Correctness:** safe attention capacity, boundary checks, corrected routing.
2. **Measurement:** raw request timings, ordinary greedy parity, fixed `k=2`
   and current adaptive baselines. Compare an external runtime separately.
3. **Latency:** separate the first target forward from warmed prompt processing,
   then optimise one measured bottleneck with a numerical check and an
   end-to-end comparison. For decode, the 30B profile points first to expert
   projections, then attention projections and attention.
4. **Expert prediction:** test whether past accepted routes predict future
   verification cost. Include rejected drafts and prediction overhead. Compare
   against `k=0`, `k=2` and the existing scheduler on separate evaluation prompts.
5. **Release:** document supported checkpoints and limits, provide a small demo,
   and have another person reproduce the build and one result.

A proposed acceptance threshold for the predictor is a stable 10% improvement
over the current scheduler, with identical greedy output and per-prompt
regressions reported. This is a decision rule, not a promised result. If it
fails, keep the simpler scheduler and publish the negative experiment.

Avoid expanding the first release into many model families, multimodality,
SSD offload or another GPU backend. A resident process and streaming API can
follow if developers need to embed the engine in an application.

## Publication

Publish a short README, exact commands, raw results and a release note that
states what changed. Do not claim novelty from a literature search or compare
unmatched quantisation formats. A second machine and independent reproduction
remain useful next checks; neither has been completed in this review.
