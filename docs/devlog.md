# picoforge devlog

Two lines per session: what was done, what comes next. Newest entry first.

## 2026-09-14 — Phase 3 closed by decision; Phase 4 aimed at a gap

`make test-all` re-run: green. Repository published to
github.com/CCristiann/picoforge (main + phase3). **Decision of the human:**
Phase 3 closes without `docs/phase3.md`, against CLAUDE.md's done-criterion,
and work continues building first and studying after. Phases 0-2 have no
writeup either. phase3 fast-forwarded into main.

The human asked for something that has not been built before. A literature
search found the pieces apart -- BaseRT runs Qwen3-30B-A3B on an M5 Pro but
leaves speculative decoding as future work; EcoSpec and EVICT make speculation
expert-cost-aware, but on H200/A100 servers -- and nothing putting them
together on unified memory. Plan and sources in `docs/phase4-plan.md`.

### Step 4.1 — what a matmul costs inside a command buffer

`--bench-dispatch`, raw CSV in `bench/dispatch_m5pro.csv`. One matmul
repeated 1..1024 times in ONE buffer, then three sweeps to find the axis.

- **The floor holds inside a buffer**, so Phase 3's hypothesis was right in
  number and wrong in cause. TensorOps at 1x3072x1024 is 32.0 us per dispatch
  whether alone or 1 of 1024. But it is not a toll: a 1x32x32 matmul costs
  1.4 us per dispatch.
- **Bytes do not set it either.** An expert projection (1x768x2048, half the
  bytes) costs the same 31 us. What does: work per 32x32 tile grows linearly
  with K (1.4 -> 105 us from K=32 to 8192), and tiles run in parallel for free
  up to 16 of them (27.1 -> 27.5 us), then stop being free (128 tiles: 86 us).
- **Rows are free.** 1 to 32 tokens routed through one expert projection cost
  the same 31 us. The op computes the whole 32-row tile whether one row
  exists or thirty-two.
- **So the tile was the waste.** The same op with an 8x32 tile, output
  bit-identical (max relative difference 0): 19.6 us at decode (1.64x),
  13.8 us for 8 rows (2.2x), 31.8 vs 32.1 us at 32 rows. The header forbids a
  1-row tile with SIMD groups ("M must be a multiple of 8 or 16"); the
  single-thread scope allows it and is 3.5-160x slower.

For the plan: on this machine a MoE verify step is priced neither by bytes
(EcoSpec's H200 model) nor by calls, but by tiles and by how many of them
fit in parallel. That changes what "an expensive draft" means.
*(Superseded by step 4.7: true of one small matmul per dispatch, false of a
grouped MoE matmul at the 30B's shapes, which is bandwidth-bound.)*

One unexplained failure, logged and not chased: a single-thread-scope 1x1024
tile over 8x768 left rows of C unwritten; the check caught it before timing.

In the forward pass (`bench/e2e_tile_m5pro.csv`, both tiles in one run):

                      32x32 tile      8x32 for M <= 8
  prefill 512          4489 tok/s      4490
  decode @512          69.7 tok/s      75.9   (+8.9%)
  verify32 @512        1574 tok/s      1572

Tried at M <= 32 first: decode the same, verify32 2% slower -- four times the
tiles to fan out -- so the threshold is 8, the range the kernel sweep showed
winning. All parity tests green. Half of the 2.4 ms the kernel sweep promised
arrived; the rest of decode's 13.2 ms is still unaccounted for.

### Step 4.1c — where a decode step's 13 ms actually go

`--profile`, raw CSV in `bench/profile_m5pro.csv`: GPU timestamps at the
boundaries of one encoder per op group. Cutting the pass into ~300 encoders
costs 1.2 ms, and both totals are recorded.

                       decode@16  decode@512  verify8@512  verify32@512
  attention              0.41       4.62        4.71         11.01
  mlp proj (84 matmuls)  3.80       3.80        3.59          4.03
  attn proj (112)        3.14       3.14        2.93          3.32
  lm head                1.40       1.40        1.43          1.59
  everything else        0.56       0.56        0.59          0.73
  total, uncut (ms)      8.94      13.18       12.90         20.33

- **Attention is the decode bottleneck at depth**: 35% at 512 and growing
  linearly with the cache (0.41 -> 4.62 ms). At verify32 it is 54% of the pass.
- **Verifying 8 tokens costs less than decoding 1** (12.90 vs 13.18 ms). Any
  accepted draft token is profit for speculation up to 8.
- **Projections cost 28-45 us per matmul in situ**, not the 19.6 us of the
  isolated sweep. Interleaving pipelines is not free; not yet explained.
- **The LM head is the only bandwidth-bound op**: 311 MB in 1.4 ms, 222 GB/s,
  72% of the machine.

The instrument was wrong first: MTLDevice sampleTimestamps' CPU side is
already in nanoseconds, and read as mach ticks (x 125/3) the groups summed to
39x the pass. The profile now dies if the groups do not sum to 80-102% of the
command buffer's own GPU time.

### Step 4.2 — speculative decoding, lossless, on the dense model

`src/speculate.c`, `make test-speculate`. Greedy verification of up to k
drafted tokens per GPU pass; the drafter is prompt lookup (the most recent
earlier occurrence of the last 3/2/1 tokens, and what followed it), which
costs nothing and stands in for the 0.6B drafter of step 4.8. K/V rollback is
free: the cache is indexed by position, and rejected rows are overwritten.

Correctness: 5 prompts x draft lengths {2, 4, 8, 16}, 128 tokens each, every
sequence identical to plain greedy through generate() -- a separate code path,
so the reference does not share the code it judges.

Speed, one run each (orientation, not the protocol), draft 4 vs draft 0:

  continuing "The capital of France is"   1.69 tokens/pass   101.8 -> 167.4 tok/s
  rewriting two functions with hints       1.61               98.0 -> 152.9
  repeating a list                         1.66               98.3 -> 158.8
  "def fibonacci(n):"                      1.43              102.2 -> 141.9
  a poem                                   1.00              100.4 -> 100.8

Where lookup finds nothing, nothing is lost up to draft 8; at 16 the poem
pays 4.5% for 384 rejected drafts, because a 17-row verify leaves the 8-row
tile and grows attention. Long drafts need a drafter that is right, which is
what a model is for.

### Step 4.3 — the MoE block in the oracle

`tools/oracle/make_tiny_moe.py` builds a 3-layer random Qwen3-MoE (8 experts,
top-2, one dense layer, untied head, bf16, the 0.6B's tokenizer);
`make test-oracle-moe` verifies the NumPy oracle against transformers on it.
All three prompt lengths PASS: relative error 1.9e-6, KL 1.2e-11, argmax 40/40.

Three things read before written, each of which would have been a silent bug:
- transformers 5 fuses experts in memory (gate_up_proj [E, 2I, H]) but
  save_pretrained writes one tensor per expert -- the same names as the real
  checkpoint's index (read from the Hub, weights not downloaded).
- transformers 5 writes a different config dialect: num_local_experts, dtype,
  rope_parameters. The real 30B config (saved by 4.51) says num_experts,
  torch_dtype, rope_theta; the tiny config is rewritten in those words and
  checked to read back identically.
- The real 30B has tie_word_embeddings=false. The oracle used the embedding
  as the head unconditionally; it now asks the config.

A verifier that passes first time is itself suspect, so three deliberate bugs
were run through it: router weights not renormalised, head tied, the dense
layer routed. All three caught. Qwen3-0.6B still verifies.

### Step 4.4 — the MoE block in the C engine, on the CPU

`make test-forward-moe`. The config reads qwen3_moe (every MoE key required
once the model type says so), weights bind the router and every expert with
its shape checked, the LM head follows tie_word_embeddings, and the forward
pass runs the oracle's moe_block structurally: softmax over all experts, top-k
by selection with ties to the lower index, renormalised, experts summed into a
scratch before the residual add.

Against the NumPy oracle on the tiny model, batch and incremental: relative
error 1.2e-6, KL 4.3e-12, argmax 15/15. The architecture summary is
byte-identical to the oracle's for both the dense and the MoE model.

The C engine was then run on three wrong configs over the same weights --
router weights not renormalised, head tied, top-1 instead of top-2 -- and
compared with the oracle on the right one: all three caught. Dense, quantised
and generation parity unchanged. The GPU refuses MoE and untied heads loudly.

### Step 4.6 (early) — the MoE layer on the GPU, routing included

Routing is data, and reading it back per layer would cost 48 round trips per
token on the 30B, so it never leaves the GPU. One MoE layer is nine dispatches
whatever the tokens pick: router matmul, `moe_route` (softmax and top-k, the
CPU's arithmetic in the CPU's order), `moe_group` (a counting sort of the
token-expert pairs by expert, one GPU thread), `moe_gather`, grouped gate and
up, swiglu, grouped down, `moe_scatter`. A grouped matmul is one dispatch for
every expert at once -- grid row y is an expert group, column x a 32-wide
tile -- so by step 4.1's measurements it should be priced by distinct experts.

A group holds at most one tile of rows, so a MoE pass takes <= 32 tokens and
gpu_forward splits longer inputs into cached chunks. Bf16 experts only for now;
quantised experts are refused, not misread. The LM head is untied on the GPU.

Against the oracle on the tiny MoE, 1/5/9/40 tokens, GPU batch and incremental:
worst relative error 1.1e-6, KL 4.0e-12 -- the 40-token prompt goes through a
32-token chunk, the 32-row tile, and an 8-token tail. A mutated moe_route (no
renormalisation) fails all four GPU cases. Dense and quantised parity and
test-speculate unchanged.

### Step 4.7 (synthetic) — what verifying a 30B MoE layer costs here

`tools/synth/make_moe_layer.py` writes one decoder layer at Qwen3-30B-A3B's
published shapes (128 experts of 2048x768, real attention), random bf16
weights, vocabulary cut to 1024: 1.25 GB instead of 61. `--bench-moe` times the
GPU MoE block with routing forced -- n tokens touching exactly D distinct
experts -- and dies unless the GPU formed exactly D expert groups. Raw CSV
`bench/moe_verify_cost_m5pro.csv`, plot `bench/moe_verify_cost.png`.

  per layer, 8 tokens:   t = 0.35 ms + 43 us x D     (6 points, max residual 18 us)
            32 tokens:   t = 0.48 ms + 45 us x D     (8 points, max residual 57 us)
  8 experts, 1 -> 32 tokens:   0.65 -> 0.80 ms

- **Priced by experts, almost not by tokens.** Thirty-two tokens through the
  same 8 experts cost 23% more than one token.
- **Each extra expert costs its bytes.** 43 us for 9.4 MB is 221 GB/s, 72% of
  the machine: at the 30B's shapes the grouped kernel runs every expert's tiles
  in parallel and waits on memory. Step 4.1's "tiles, not bytes" described one
  small matmul per dispatch on the 0.6B; it does not carry over. EcoSpec's
  byte-cost model holds on this machine too -- now with its constants measured.

What that says, as arithmetic to be checked against the real model: at bf16,
decode is 48 x 0.65 = 31 ms of MoE per token. A verify of 8 tokens whose
routing were uniform would touch ~51 experts per layer, ~2.5 ms, 3.8x a decode
step, so speculation would need 3.8 accepted tokens per pass to break even.
Every expert two drafts share saves 43 us x 48 = 2 ms. How much real
consecutive tokens share is a property of the trained router, and the one
number here that cannot be synthesised: it needs the 30B.

### Step 4.6b — q8_row experts on the GPU, and a limit that sets the format

A probe of the device before anything else: `maxBufferLength` is **41.75 GB**
and the recommended working set 55.7 GB. The engine binds the checkpoint as
ONE no-copy buffer, and the 30B in bf16 is 61 GB -- it cannot run on this GPU
unquantised, whatever else is done. q8_row (free in quality in Phase 3) puts it
near 31 GB, inside one buffer. So the GPU gained grouped q8_row expert kernels:
the op on the int8 codes, then the per-row scales applied inside the tile.

`make test-forward-moe-quant`: the quantised tiny MoE (48 expert projections
as int8, router and untied head kept bf16) against the oracle on dequantised
weights, strict budgets, GPU batch and incremental: worst relative 1.7e-6,
KL 9e-12. A kernel with the scales removed fails all six GPU cases.

The verify-cost surface, both formats measured in one session
(`bench/moe_verify_cost_m5pro.csv` now carries a format column):

                 <= 8 tokens (8-row tile)          32 tokens (32-row tile)
  bf16           0.36 ms + 44 us/expert  216 GB/s   0.49 ms + 47 us/expert  202 GB/s
  q8_row         0.35 ms + 22 us/expert  218 GB/s   0.57 ms + 38 us/expert  125 GB/s

- **Half the bytes, half the marginal expert** -- 44 -> 22 us at <= 8 tokens,
  at the same ~217 GB/s. The byte model holds across formats.
- **The 32-row tile threw that away.** Past 8 tokens the q8 kernel paid nearly
  bf16's price per expert, because the 32-row op computes rows that are not
  there.

So the grouped kernels now tile rows in eights for groups of any size: grid
row y is group y / r, row tile y % r, with r = ceil(n / 8) passed in the
otherwise unused M. Parity unchanged (bf16 and q8 tiny MoE). Same session,
before -> after, ms per layer:

  q8_row, 16 tokens, 64 experts     2.97 -> 1.78
  q8_row, 32 tokens, 64 experts     3.00 -> 1.87
  q8_row, 32 tokens, 128 experts    5.38 -> 3.34
  bf16,   32 tokens, 64 experts     3.43 -> 3.45   (bandwidth-bound either way)

A q8 expert now costs 21-23 us at every verify size up to 32 tokens. The one
cost model, for the scheduler to use: **per MoE layer, ~0.35-0.6 ms + 22 us
per distinct q8_row expert** (44 us bf16), almost independent of how many
tokens share them.

### Step 4.8a — a draft model in the speculative loop

`--spec-model DRAFT_DIR`: a second GPU model over the same vocabulary proposes
k tokens of its own greedy continuation. It keeps its own K/V cache, which
rolls back as freely as the target's: after a verify accepts i drafts, the
drafter's valid prefix is cut to the accepted text, and what it has not seen
goes in as one catch-up pass next step.

`make test-speculate` now holds 40 runs to plain greedy, all identical: the
five prompts under prompt lookup, under Qwen3-0.6B-q4_g32 drafting for the
bf16 0.6B (drafts 3 and 8), and under the 0.6B drafting for the tiny random MoE
-- a pair that never agrees, so every pass rolls back both caches and the
target is a MoE split across chunks.

Two numbers for the plan, one run each (orientation):
- q4 drafting for bf16 is accepted 55-100% of the time, 2.7-8.5 tokens per pass.
- **A draft token costs the 0.6B about 10 ms** (378 drafts in 3.87 s). Against
  the 30B's estimated decode that is the drafter/target cost ratio every
  speculative speedup is divided by; the 0.6B's attention (35% of its decode at
  depth 512) is where that ratio can still move.

### Step 4.8b — barrier-free attention: a hypothesis the silicon rejected

The attention kernel coordinates 128 threads per (head, token) through sixteen
threadgroup barriers, and at depth 512 each thread owns four positions. The
hypothesis: the GPU spends its time synchronising, so three dispatches where
every thread owns its work (scores per position, softmax per head, one value
sum per output dimension) should be faster. It was built, verified against the
oracle on every model (82 checks green), and measured.

`bench/attention_ab_m5pro.csv`, both kernels interleaved rep by rep in one
process -- the first attempt at a baseline read decode 30% slower than the
morning's while another session loaded the machine (load average 6.2), and
alternating is what makes a comparison survive that:

                   attention, barriers   attention, split    whole pass
  decode@16          0.42 ms               0.54 ms           9.42 -> 9.53
  decode@512         4.69                  6.08             13.70 -> 15.08
  verify8@512        4.77                  7.43             13.37 -> 16.13
  verify32@512      11.18                 12.89             21.04 -> 22.82

Slower everywhere. The likely reason, not yet measured: the value sum strides
one float per cache row, 128 x 16 threads of it, so the barrier-free version
buys independence with memory traffic -- and memory, not synchronisation, is
what this machine runs short of first. The code is committed switched off so
the table can be reproduced, and removed in the next commit.

**Next:** the 30B checkpoint (a download that needs the human's yes): sharded
safetensors, parity, and the routing-overlap measurement that turns this cost
model into a draft scheduler.

## 2026-09-12 (later that night) — Phase 3: quantisation, designed around the matrix units

Q8 and Q4 run end to end on CPU and GPU, every link of the chain of oracles
green (`make test-all`). Formats were chosen by measurement: quality first
(fake-quantised sweep on an uncontaminated local corpus), kernel cost second,
whole passes third.

                  weights  ppl vs fp32  KL     prefill 512  decode@512  verify32@512
  bf16            1192 MB     --         --    3692 tok/s   69.7        1575
  q8_row+embed     597 MB   -0.24%     0.003   3556         73.5        1705
  q4_g32           559 MB  +10.4%      0.165   2069         74.6        1055
  q4_g32+embed     335 MB  +12.1%      0.184   2108         74.3        1007

What the silicon said that the header did not: the TensorOps destination is
overwritten, not accumulated (C = A*B, not A*B + C); int4 is packed low nibble
first, two's complement, behind a `device uchar *`; there is no float x int4,
so Q4 activations go in as bf16; a fixed-width slice is `slice<G, ...>`, not
the header example's `static_slice`.

Four findings worth keeping:
- The scale is ours to place, and where it goes sets the op-call count. Per
  row is one call and costs nothing over bf16 at any M; Q8 per row is also
  free in quality, so Q8 ships that way. Per-row Q4 is +70% perplexity; Q4
  needs blocks of 32, which cost 2.3x in prefill and verify.
- Decode at this size is not bandwidth-bound: 3.5x fewer bytes buys 7%. A
  lone matmul costs ~33 us whether it moves 6 MB or 1.5 MB, and a token issues
  197 matmuls. IF that floor also holds inside one command buffer -- measured
  only one call per buffer so far -- it is ~6.5 ms of decode's 13.4 ms. A
  hypothesis worth a profile, not a result.
- Q4's bf16 narrowing is a step function that turns fp32 noise into whole bf16
  steps, so the GPU diverges from any other implementation by KL ~5e-4 at 9
  tokens -- shown by the CPU diverging from ITSELF that much under a 1e-6
  input perturbation. In perplexity it costs 0.00004 nats, 1% of an SE.
- Two instruments were wrong before any kernel was: a clip counter that
  counted the block extreme as clipped, and a relative error that reported
  cancellation in sums of 1024 signed terms as kernel error. And warm-up by
  count left the GPU asleep for M=1; it is by time now.

The engine reproduces transformers' fp32 perplexity on the corpus to four
decimals (8.5258), which validates tokenizer, cache and GPU pass at 1024 tokens.

**Next:** `docs/phase3.md`, drafted by the human, then Phase 3 is done.
Published numbers still need a comparable corpus (WikiText-2, 733 KB test
split) and a larger post-cutoff one; both are downloads awaiting a yes. Phase 4
needs Qwen3-30B-A3B: 16 bf16 shards, 61.07 GB, awaiting a yes as well.

## 2026-09-12 (night) — the forward pass runs on the GPU

End to end on Metal, twelve parity combinations green (3 prompt lengths x
{CPU, GPU} x {batch, incremental}). Weights are bound with no copy over the
existing mmap, so the GPU reads the same physical pages the CPU does. A whole
forward pass is one command buffer: ~420 dispatches that would otherwise cost
2.5 ms of command-buffer latency per token.

              CPU scalar     GPU
  prefill      4.9 tok/s   223.6 tok/s   45x
  decode       3.7 tok/s    92.8 tok/s   25x, 110 GB/s of 307 (36%)

Two findings. The GPU is MORE accurate than the CPU -- 7.2e-6 relative error
against the oracle versus 1.1e-5 -- because tree reductions accumulate error
as log(n) where a sequential loop accumulates it as n, and NumPy's BLAS sums
in blocks too. And the microbenchmark could not pick the kernel: it found
naive beating TensorOps at M=1 with N=8192, but the model's matmuls are
1024-3072 wide and there TensorOps wins end to end, 92.8 vs 50.0 tok/s.

**Next:** decode sits at 36% of the machine. The gap is worth profiling before
optimising -- attention over the cache and the 151936-wide LM head are the
suspects, not the projections. The batching result from the kernel sweep still
stands unexploited: 32 tokens for the price of one is speculative decoding.

## 2026-09-12 (evening) — Phase 2: one kernel, three implementations

All three matmul kernels written and all three bit-exact against the CPU
oracle. Peak on a 2048 cube: naive 0.72, simdgroup 4.23, TensorOps 7.78
TFLOP/s -- 10.8x end to end. Raw CSV and plot in bench/, protocol in
docs/BENCHMARKS.md.

TensorOps cost four probes. The one that mattered: element types must not be
const, or every is_same<T, float> in the dispatch chain fails and the terminal
static_assert reports "Unsupported type" while naming the DESTINATION type --
the error blames the wrong parameter. Also found: tensor_inline (not the
default tensor_handle) wraps a device pointer; float x bfloat -> float is
supported; and the same chain accepts bfloat x int4b_format and bfloat x
int8_t with fp32 accumulation, which is Phase 3 arriving early and should
shape the quantisation format before one is picked.

Three measured findings:
- Batching is free to M=32. 33 us whether M is 1 or 32, because the kernel is
  reading the same 6 MB of weights either way. 188 -> 6017 GFLOP/s at no cost.
- At M=1 with N=8192 the NAIVE kernel beats TensorOps, 276 vs 213 GB/s, which
  is 90% of the machine's 307. Bandwidth-bound work does not want matrix units.
- Decode is short of parallelism, not bandwidth: throughput climbs with N.

The protocol earns its keep: the same kernel and shape reads 8.0 GB/s
single-shot and 36.2 GB/s warmed with reused buffers.

**Next:** move the forward pass itself onto the GPU, keeping the CPU path as
the oracle. The batching result says the decode loop should be built so a
batch dimension can exist later -- speculative decoding gets 32 tokens for
the price of one.

## 2026-09-12 (later still) — Phase 1 complete: the C engine generates

Steps 1.6-1.9. Byte-level BPE tokenizer (30/30 adversarial prompts identical to
HuggingFace, all round-tripping), full NFC normalisation (fuzzed against Python's
unicodedata on 4017 strings), K/V cache, sampling with temperature/top-k/top-p,
and Qwen3's chat template. Token-level parity reached: three prompts generate
the identical 16-token greedy continuation as transformers, code and multi-byte
characters included.

Three bugs worth remembering, all found by counting rather than by reading:
96 merge rules silently dropped because merges.txt uses '#' for both comments
and data; added tokens matched only at pre-token boundaries, by which point the
regex had already eaten " <|"; and NFC missing entirely, which would have made
every accented prompt copied out of Finder diverge.

Measured baseline, scalar and single-threaded: prefill 4.9 tok/s, decode
3.7 tok/s. Decode moves 4.4 GB/s of the machine's 307 -- 1.4%. That number is
Phase 2's entire subject. Theoretical decode ceiling is 307 / 1.19 GB = 258
tok/s, so there is 70x on the table before physics intervenes.

**Next:** Phase 2 -- Metal. Re-run tools/probe/metal4_probe first (the tensor
API tightened since macOS 26.2: K must be a multiple of 32 in matmul2d_descriptor
or it silently truncates). Then one matmul kernel in three versions: naive,
simdgroup, TensorOps on the Neural Accelerators.

## 2026-09-12 (later) — Phase 1 reaches logit parity

The C engine now runs Qwen3-0.6B end to end. Steps 1.1-1.5: dependency-free
config scanner; mmapped safetensors reader (bf16 stays bf16 in the mapping and
is widened inside the loops — a shift, so free, and half the bytes moved);
matmul / rmsnorm / softmax / rope / silu, each judged by the oracle's own
functions rather than a reimplementation; then the full forward pass.

Measured: argmax agreement with the oracle on every position of three prompts,
worst relative logit error 1.1e-5, worst KL 1.8e-9. That is the same order as
NumPy-vs-transformers (8.4e-6), so the chain of oracles is not degrading.
"The capital of France is" -> " Paris" (65.7%). Prefill is 3.6 tok/s, scalar,
single-threaded — the honest baseline Phase 2 gets measured against.

Two bugs found in our own instruments, both worth remembering: a KL accumulated
in fp32 read negative (Gibbs forbids it) and needed fp64; and the top-5 printer
was softmaxing the logits in place, then re-running the whole forward to
recover them — two seconds of work saved by a 600 KB copy.

**Next:** step 1.6, the byte-level BPE tokenizer, so prompts stop being
hand-typed token ids. Then KV cache (1.7), sampling and chat template (1.8).

## 2026-09-12 — Phase 0 closed: the oracle is verified

Full forward pass committed, then `tools/oracle/verify.py` against transformers
(forced to fp32 + eager attention, so the reference is not the less precise of
the two). Three prompt lengths — 1 / 5 / 40 tokens — all PASS: argmax agreement
40/40, relative logit error 8.4e-6, KL 2.9e-10 worst case. One instrument bug
found on the way: KL accumulated in fp32 read *negative*, which Gibbs' inequality
forbids; fp64 moved the reading by four orders of magnitude. Tolerance budgets
are now calibrated on measured values (1e-4, 1e-8), not guessed. Ecosystem check
after the two-month gap: Metal 4 tensor APIs tightened since macOS 26.2 — K must
be a multiple of 32 in `matmul2d_descriptor` (it silently truncates otherwise),
at least one of M/N a multiple of 16, and no mixing bfloat with half. Landmines
for Phase 2; re-run `tools/probe/metal4_probe` before starting it.

**Next:** Phase 1 — the C engine. Scaffolding and config parsing first (read
config.json, never hardcode), then the safetensors loader, the byte-level BPE
tokenizer, the forward pass, KV cache and sampling.

## 2026-07-09 — Environment setup complete

Toolchain verified (Xcode + Metal Toolchain 17F109, macOS 26.5.1, Apple clang 17).
Metal 4 probe compiled and run on the M5 Pro: Apple10 + Metal4 families supported,
MTLTensor creation works — Phase 2 is viable on this machine (probe:
`tools/probe/metal4_probe.m`). Working set reported: 55.7 GB. Python venv under
`tools/venv` (numpy 2.5.1, torch 2.13.0, transformers 5.13.0), versions frozen in
`tools/requirements.txt`. Qwen3-0.6B downloaded to `models/` (gitignored), SHA256s
in `docs/weights.lock` (verify: `shasum -a 256 -c` from the model dir).

**Next:** Phase 0 — parse config.json, print architecture summary (mind the
explicit `head_dim` gotcha), then the NumPy forward pass component by component.
