# Evaluating the 80–100B class on 64 GiB Apple Silicon

Research date: 2026-10-11. Workload: coding and general reasoning.
Subsequent user decision: proceed with **Qwen3.8-Flash-Next Q2**; this document
remains an alternative comparison, not the active implementation target.
This is a follow-up to [the larger-model research](large-model-research-2026-10-11.md),
evaluating a proposed size constraint. No model weights were downloaded,
no model inference was benchmarked, and no new runtime path was implemented.

## Recommendation within this range

Start the selection with **Qwen3-Coder-Next, 80B total / 3B active**, at
4-bit weight precision. Compare **Qwen3-Next-80B-A3B-Thinking** on the same
coding and general-reasoning tasks before choosing the default checkpoint.
Coder-Next is the initial engineering target for the coding workload, not
a demonstrated winner across both workloads.

The two models share the `qwen3_next` architecture family, so the main C/Metal
graph investment can serve both. Their configurations and chat templates
must remain model-specific: for example, Coder-Next uses `rope_theta=5000000`,
while Next-Thinking uses `10000000`. Coder-Next supports non-thinking mode
only; Next-Thinking supports thinking mode only. This is a difference in
training and generation behavior, not a toggle that turns one into the other.
Non-thinking does not mean incapable of reasoning.
[Coder model card][coder], [Thinking model card][thinking],
[Coder configuration][coder-config], [Thinking configuration][thinking-config].

This is a more conservative memory/quantization trade-off than the earlier
Qwen3.8 Q2 proposal. It does **not** establish superior capability. Nor does
an 80B parameter count establish contemporary near-frontier quality: these
models must earn their place against a recent smaller model at higher
precision on the user's actual tasks.

An important scope distinction: 80B BF16 weights do not fit in 64 GiB, but
existing 4-bit releases already fit with established runtimes. A successful
custom engine should demonstrate improved latency, memory use or usable
context at equivalent task quality. Merely loading the checkpoint is not
evidence that custom kernels were necessary to make it fit.

## Verified checkpoint sizes

The following values are sums of actual artifact bytes from Hugging Face's
model metadata, including every shard of a selected checkpoint. They are
**file sizes, not measured runtime peaks**. All values use GiB = 2^30 bytes.

| Model | Format | Checkpoint GiB | Initial assessment |
|---|---|---:|---|
| Qwen3-Coder-Next | MLX affine 4-bit, group 64 | 41.76 | Useful baseline with more workspace headroom |
| Qwen3-Coder-Next | Unsloth Q4_K_S | 42.41 | Second 4-bit layout to consider |
| Qwen3-Coder-Next | Unsloth Q4_K_M | 45.20 | Quality comparator with tighter runtime headroom |
| Qwen3-Coder-Next | Unsloth Q5_K_S / Q5_K_M | 51.27 / 52.94 | Too tight for the proposed resident application budget |
| Qwen3-Next-80B-A3B-Thinking | MLX affine 4-bit, group 64 | 41.76 | General-reasoning comparison |
| Qwen3-Next-80B-A3B-Thinking | Unsloth Q4_K_S / Q4_K_M | 42.38 / 45.18 | Alternative layouts for quality/memory comparison |
| Qwen3-Next-80B-A3B-Thinking | MLX 5-bit | 51.03 | Leaves insufficient headroom within our target budget |

Sources: [Coder MLX][coder-mlx], [Coder GGUF][coder-gguf],
[Thinking MLX 4-bit][thinking-mlx], [Thinking MLX 5-bit][thinking-mlx5],
[Thinking GGUF][thinking-gguf]. The MLX configurations retain the MoE router
and shared-expert gate at 8 bits. Their affine packing is not identical to
GGUF Q4_K or picoforge's existing symmetric Q4 layout. The smaller file is
not automatically the higher-quality or faster format; compare them.

For intuition, idealized 4-bit payloads alone are 37.25 GiB for 80 billion
parameters and 46.57 GiB for 100 billion. Scales, offsets, higher-precision
tensors, padding and runtime allocations come on top. Thus 100B at 4 bits
is considerably tighter than 80B at 4 bits on this machine.

## Context and working-set budget

The prior local Metal probe observed 64.00 GiB physical memory, a recommended
Metal working set of **51.84 GiB**, and a **38.88 GiB single-buffer limit**.
Target 48–50 GiB peak application footprint, one session, and initially 8K
total context. Validate 16K and 32K after correctness and allocation checks.

Both Qwen candidates have 48 layers: 36 Gated DeltaNet layers and 12 gated
attention layers. For the ordinary attention layers, a straightforward
FP16/BF16 K/V cache has this payload, calculated from the model configuration:

```text
KV bytes = 12 layers × 2 (K and V) × tokens × 2 KV heads × 256 × 2 bytes

8,192 tokens  -> 0.1875 GiB
16,384 tokens -> 0.3750 GiB
32,768 tokens -> 0.7500 GiB
65,536 tokens -> 1.5000 GiB
```

The 36 recurrent layers' matrix state at FP32 is approximately 72 MiB per
session: `36 × 32 value heads × 128 × 128 × 4 bytes`. These are analytical
payload estimates, not the full cache or runtime footprint. Convolution
history, padding, activation buffers, prefill workspaces, allocation peaks
and any speculative checkpoints must also be budgeted. Chunked prefill is
essential to bounding temporary memory. The advertised 256K model context
is not a promise that the local engine can serve it within this budget.

For a 41.76 GiB checkpoint, weights + 32K KV + FP32 recurrent matrix state
amount to approximately 42.58 GiB before those remaining costs. For Q4_K_M,
the same subtotal is about 46.02 GiB. Measure the remaining allocations
before claiming either configuration fits comfortably. All experts must
remain resident in this proposal; 3B active parameters do not mean 3B worth
of weight storage.

## Other candidates investigated

| Candidate | Evidence | Why it is not the first target |
|---|---|---|
| openPangu-2.0-Flash, 92B / 6B | Recent hybrid thinking/non-thinking model; published base-only Q3_K_M 40.73 GiB, Q4_K_M 51.48 GiB | License section 3.1 explicitly excludes use within the EU; additionally needs MLA, DSA/SWA, mHC and MoME kernels |
| Solar-Open-100B, actually 102.6B / 12B | GGUF Q3_K_M 45.94 GiB, Q4_K_M 57.98 GiB; official Nota INT4 safetensors total 51.93 GiB | Around the class boundary; inspected 4-bit artifacts leave little or no runtime headroom; would need a different compression/graph path |
| Ring-flash-2.0, marketed 100B / 6.1B | Reasoning-oriented model, MIT; inspected i1-Q3_K_M 45.94 GiB | Worth a later matched-quality comparison, but adds the `bailing_moe` architecture and needs more aggressive compression than the Qwen baseline |
| GLM-4.5-Air-REAP-82B-A12B | Expert-pruned derivative of 106B GLM-4.5-Air; GGUF Q3_K_M 39.93 GiB, Q4_K_S 49.03 GiB, Q4_K_M 52.68 GiB | Useful compression experiment; pruning and quantization losses must be assessed separately, with a different graph and 12B active parameters |
| LLaDA2.0-flash, 100B non-embedding / about 6B active | Diffusion language model with published weights | Different generation algorithm; a separate research direction rather than the shortest path to a useful autoregressive C/Metal engine |

Primary sources and artifact producers: [Pangu architecture][pangu],
[Pangu license, section 3.1][pangu-license], [Pangu conversion][pangu-gguf],
[Solar model][solar], [Solar GGUF][solar-gguf], [Nota conversion][solar-nota],
[Ring model][ring], [Ring conversion][ring-gguf],
[GLM REAP model][reap-model], [GLM REAP conversion][reap-gguf],
[LLaDA model][llada]. This is a screened shortlist, not a claim that every
open checkpoint in the range has been evaluated.

Pangu's `base` files omit the optional three MTP layers, not its instruction
post-training. Do not compare that file size with an artifact including MTP
without accounting for the difference. Its converter documents Apple Silicon
usage with an increased wired-memory limit; that is not proof of fit under
our unchanged local budget. No system limits were modified.

The [REAP paper][reap-paper] adds a relevant compression option to the earlier
survey: rank experts using router weights and output activation norms, then
prune low-contributing experts using domain-representative calibration.
This can be combined with quantization, but retained active expert count
does not shrink in proportion to stored parameters. Consequently, reduced
weight storage does not imply a proportional decode speedup. Treat claims
of near-lossless compression as workload-specific evidence requiring our
own held-out coding and reasoning evaluation.

## What changes for picoforge and Metal

1. Replace the single MTLBuffer assumption with checked buffer/offset views;
   every shortlisted 4-bit Qwen checkpoint exceeds this Mac's single-buffer
   limit. Avoid making another complete copy of the weights.
2. Implement the hybrid `qwen3_next` graph: Gated DeltaNet recurrence,
   convolution and normalization, gated attention, and routed/shared experts.
   The existing `qwen3_moe` 30B path is not compatible merely because both use
   MoE layers.
3. Choose and decode one explicit quantization layout correctly, including
   per-group scale/offset and mixed tensor precision. Use fused quantized
   GEMV for single-token decode and grouped expert matrix multiplication
   with Metal TensorOps/MPP for prefill where profiling justifies it.
   Native INT4 support does not make every 4-bit file a drop-in operand.
4. Bound scratch space through chunked prefill, tiled attention and reusable
   allocations; keep recurrent state at the validated reference precision
   initially. Avoid expanding all expert weights to FP16 in unified memory.
5. Reuse the earlier DwarfStar lessons on no-copy weight views, fused kernels
   and explicit memory budgeting. SSD expert streaming is not needed for
   this initial 80B configuration. MTP/speculation comes after graph parity
   and a measured baseline, with checkpoint support verified separately.

## Selection and success criteria

First benchmark the published Qwen 4-bit artifacts in their supported
reference runtimes; MLX-LM and llama.cpp are comparison tools, not proposed
dependencies of the C inference engine. Compare one loaded model at a time.

Evaluate code changes with executable tests, multi-file debugging, tool-call
validity, and held-out general reasoning in Italian and English. Measure
quality at a specified generation budget, and separately under equal
wall-clock limits: thinking models can spend many more tokens on an answer.
Record time to first token, prefill and decode separately, total time to a
correct solution, peak physical/Metal memory and swap growth at 8K/16K/32K.

Separate three questions: whether our kernels reproduce the same quantized
checkpoint; how much compression changes task performance; and which model
solves the user's tasks best. A proposed engineering target is a 4-bit 80B
with all weights resident, no sustained swap growth, and a demonstrated
advantage over the matched reference runtime in at least one useful metric
without an unacceptable quality regression. No speed target is reported as
achieved or forecast from another Mac's results.

## Artifact revisions inspected

| Repository | Revision |
|---|---|
| Qwen/Qwen3-Coder-Next | `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| Qwen/Qwen3-Next-80B-A3B-Thinking | `e502dd4100cc68c0de57643fd4317ec93a128670` |
| mlx-community/Qwen3-Coder-Next-4bit | `7b9321eabb85ce79625cac3f61ea691e4ea984b5` |
| mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit | `9a2b46347bb170cb2924092175fa21554fe585a9` |
| mlx-community/Qwen3-Next-80B-A3B-Thinking-5bit | `541db33a0eda3cf8010e60098cdc49d95c80103d` |
| unsloth/Qwen3-Coder-Next-GGUF | `ce09c67b53bc8739eef83fe67b2f5d293c270632` |
| unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF | `6e6e89e27c5a838dba47ba7b14501183b3d76e10` |
| openpangu/openPangu-2.0-Flash | `5b78fc70a06dd27c5c4c5e918c46c49531d357b4` |
| mrexodia/openPangu-2.0-Flash-GGUF | `3eaa1c8f1367f178556800275bf1c4e026b48080` |
| mradermacher/Solar-Open-100B-GGUF | `e72e988e2a4e736a4f015261084d03e6315f0dcd` |
| nota-ai/Solar-Open-100B-NotaMoEQuant-Int4 | `54028f45cb27f2bcd4eccc4d4e4dea8a14ecd6a7` |
| mradermacher/Ring-flash-2.0-i1-GGUF | `01b5fbefeee1f263ff2a3a5ac39047efeb5f3858` |
| bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF | `d4ba84e54f918101e1b024a9632f319a2e907445` |

[coder]: https://huggingface.co/Qwen/Qwen3-Coder-Next
[thinking]: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking
[coder-config]: https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb/config.json
[thinking-config]: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking/blob/e502dd4100cc68c0de57643fd4317ec93a128670/config.json
[coder-mlx]: https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit
[coder-gguf]: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
[thinking-mlx]: https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit
[thinking-mlx5]: https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-5bit
[thinking-gguf]: https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF
[pangu]: https://huggingface.co/openpangu/openPangu-2.0-Flash/blob/main/README_EN.md
[pangu-license]: https://huggingface.co/openpangu/openPangu-2.0-Flash/blob/5b78fc70a06dd27c5c4c5e918c46c49531d357b4/LICENSE
[pangu-gguf]: https://huggingface.co/mrexodia/openPangu-2.0-Flash-GGUF
[solar]: https://huggingface.co/upstage/Solar-Open-100B
[solar-gguf]: https://huggingface.co/mradermacher/Solar-Open-100B-GGUF
[solar-nota]: https://huggingface.co/nota-ai/Solar-Open-100B-NotaMoEQuant-Int4
[ring]: https://huggingface.co/inclusionAI/Ring-flash-2.0
[ring-gguf]: https://huggingface.co/mradermacher/Ring-flash-2.0-i1-GGUF
[reap-model]: https://huggingface.co/cerebras/GLM-4.5-Air-REAP-82B-A12B
[reap-gguf]: https://huggingface.co/bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF
[reap-paper]: https://arxiv.org/html/2510.13999v1
[llada]: https://huggingface.co/inclusionAI/LLaDA2.0-flash
