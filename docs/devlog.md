# picoforge devlog

Two lines per session: what was done, what comes next. Newest entry first.

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
