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

**Next:** step 4.1, the measurement the whole plan rests on: does a matmul
still cost ~33 us when it is one of hundreds in a single command buffer?

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
