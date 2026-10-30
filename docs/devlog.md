# picoforge devlog

Two lines per session: what was done, what comes next. Newest entry first.

## R2 — close the gap to DwarfStar, short prompt and 8K (from 2026-10-28)

Target (ruling R44): >= 0.80x DwarfStar on the short prompt and >= 0.60x at
8K (R1 ended at 0.62x and 0.37x). Same numerics rules as R1; one closure at
the end. Notes below are newest first.

### R2.9 — check folded into HC_ADD and the activations

HC_ADD (96 per token), ACT (MoE shared SiLU and sigmoid, QSA gate) and
the unfused HC_LOW/HC_INJECT ran q38c_check, then the op: two dispatches
and two barriers each. `q38c_hc_add_fast` and `q38c_activate_fast` test
each thread's own inputs (together every element the check reads) and
then evaluate the op's expression, so the outputs keep their bits
(test_hc_add_fast_bit_identical_to_oracle,
test_activation_fast_bit_identical_to_oracle). The difference is on
faults only: the sequence stops the whole op, the folded kernel may still
write other elements (the session discards the command; the C contract
test now checks "nothing written" with the oracle forced). On a
concurrent command the fault goes to word 1 and the op continues, as
R1.6b's overlapped check. Measured (`profile-folded`, battery): GPU
groups 40.16 -> 38.63 ms/token, dispatches 1560 -> 1356, hc_add+gdn.add
0.95 -> 0.57 ms; real short decode unchanged within battery noise
(24.00 tok/s, p10 22.62: `speed-folded`).

### R2.8 — QSA small stages: RoPE per element, scores per head

At the short prompt qsa.core cost 185 us per layer for a few keys. A
20-encode microbench found two latency chains: RoPE ran one thread per
head (24 threads, each 256 precise cos/sin in series: 33 us per call at
any context, three calls per token) and the R2.2 fast scores one thread
per block (4 head dots interleaved over 128 columns: 31 us even at 3
blocks). RoPE now runs one thread per element with the oracle's
expression (7 us; test_rope_fast_bit_identical_to_oracle), the scores 8
lanes per block, lane h running head h's dot in increasing d, the first
lane adding the clamped dots in head order (10-14 us short, 62 -> 15-21 us
at 8K; the index scores test unchanged). Graph (`profile-qsa-small`,
battery): qsa.core 2.33 -> 0.95 ms/token; real short decode 24.04 tok/s
(`speed-qsa-small`).

### R2.7 — routed down: four rows per simdgroup

The Q2_K down (K 640 = 40 chunks of 16) gave each output row one
simdgroup: 32 chunks in the first lane round, 8 in the second, so a
quarter of the lanes worked half the time. A simdgroup now owns 4 rows
and their tail chunks share rounds (`q38_quant_rows`, 5 rounds instead of
8), each lane keeping its partial sums in q38_quant_row's order:
bit-identical (the fused-down tests unchanged). Isolated, 1 token x 10
slots: 65.4 -> 60.1 us; 2 rows 61.5, 8 rows 72.3 (fewer simdgroups in
flight). A threadgroup copy of the IQ2_XXS grid for the gate/up was tried
and dropped (61.6 -> 63.2 us): the kernels are ALU-bound, ~7 operations
per weight, not limited by the constant cache.

### R2.6 — HC read tail in one dispatch

After down, every HC site ran HC_LOW, the F16 up (320 -> 10240), HC_READ
and HC_INJECT: 7 dispatches (each C op is check + op). One kernel now does
all four (idea: ds4 hc_gate_mix): every threadgroup recomputes
act = silu(low/4) in threadgroup memory, a simdgroup takes the 4 up rows of
one d with pooled tail chunks (K 320 = 40 chunks of 8: 5 rounds instead of
8), then the read; threadgroup 0 writes act and inject. Same operations in
the same order: act, gate, mixed and inject byte-equal to the fast sequence
(test_hc_tail_fused_bit_identical_to_sequence, 7 shapes); under
Q38_REF_KERNELS=linear or fuse the sequence runs. Isolated: 48-65 -> 30-35
us per site. In the graph with R2.7 (`profile-hctail`, AC): dispatches 2140 -> 1560,
normal-mode GPU 45.6 -> 44.0 ms/token, real short decode (1+3)
22.54 -> 22.91 tok/s (`speed-d52cb7c`, `speed-hctail`).

### R2.5 — HC down and inject in one multi-GEMV

Each HC read runs two F16 projections of the same normed rows: down
(10240 -> 320) and inject (10240 -> 4). The inject alone ran 4 output rows on
the whole GPU (~10 GB/s, 7.6 us). Both now go through the R1.4 multi-GEMV,
bit-identical to two single calls: 132 dispatches and barriers fewer per
token (2272 -> 2140). Measured (`profile-short-hcmulti`, battery):
hc_ffn.down+inject 2.02 -> 1.77 ms, total GPU groups unchanged within noise
(median 41.64 -> 41.47 ms). Kept for the dispatch count, not for speed.

### R2.4 — GDN scan: a simdgroup per four state rows

The gated delta rule keeps, per value head, a 128x128 state S and per token
does `S *= exp(g); u = S·k; delta = (v - u)·beta; S += delta·k; o = S·q`.
The oracle gives each thread one state row: it walks its 512-byte row in
device memory twice per token, and neighbouring threads touch rows 512 bytes
apart, so with each layer's 3 MB state cold in the graph the scan ran at
~58 GB/s. Now a simdgroup owns 4 rows of one head (as ds4's `gdn_scan_r4`):
each lane keeps 4 consecutive state values per row in registers for the
whole token loop (one coalesced read and write per dispatch), q and k are
loaded once for the 4 rows, and the two dots become per-lane sums + simd_sum.
One row per simdgroup was slower than the oracle; four were needed. The
input check of `q38c_check` (bits 1 and 4) is folded in. Tolerance
`gdn_scan_fast` = 2^-21 (max 4.3 u; the state's S is the running sum of the
magnitudes its roundings scale with). Chunked tokens stay bit-identical to
one-token steps. Profile short (`bench/qwen38_r2_20261029/profile-short-scan`,
battery): gdn.scan 3.70 -> 1.83 ms, GPU groups 43.6 -> 41.7 ms/token.

### R2.3 — F16 matvecs: hardware half -> float on every chunk

The hyper-connection (HC) matvecs are F16: "down" 10240 -> 320 and "up"
320 -> 10240, 6.5 MB each, at 96 sites per token (GDN, attention and FFN
blocks). R1.9c had moved the block scales to the GPU's half -> float
conversion (one instruction, the same bits for every finite pattern) but the
F16 matvec still rebuilt every whole-chunk weight bit by bit (branches +
clz). Now whole chunks convert in hardware too; nonfinite weights still set
error bit 1 (`test_f16_decode_matches_oracle` checks all 65536 patterns).
With one simdgroup per row (K < 2560) the kernel also packs 8 rows per
threadgroup. Same bits per row. Isolated (20 encodes): up 41.9 -> 27.1 us
(157 -> 243 GB/s), down 28.7 -> 22.1 us. Profile short (`profile-short-f16`):
GPU groups 45.2 -> 43.6 ms/token, gdn.hc_pre 4.07 -> 3.35 ms.

### R2.2 — exact fast index scores and block selection

Two more qsa.core stages were latency bound: one thread per block ran its
whole dependent chain alone (2048 threads at 8K, too few to hide memory
latency). Both are rewritten so the results stay the same **bit for bit**:
scores stage the query once per threadgroup and run the 4 head dots as
independent chains, each still added in increasing feature order and summed
over heads in the same order; rank and compact give each block one
simdgroup whose lanes count strided candidates and `simd_sum` the counts
(integers, so any order is exact). Warm isolated times at 8K (scratch bench,
after a GPU clock warm-up: the first cold numbers were inflated): scores
81 -> 64 us, select 364 -> 23 us per layer. Profile at 8K
(`profile-8k-index`): qsa.core 8.1 -> 4.1 ms, GPU groups 51.1 -> 47.1
ms/token. The test `test_index_scores_and_selection_bit_identical` compares
scores, IDs and error word byte for byte with the oracles.

### R2.1 — split-K decode attention ("flash decoding")

The first 8K profile (`bench/qwen38_r2_20261028/profile-8k-r1`) put 39.6 of
82.7 ms/token in `qsa.core`, against 2.2 ms on the short prompt: per layer
2.44 ms attention over the 2048 selected keys, 0.39 ms index scores, 0.43 ms
block selection. The oracle attention runs one threadgroup per head (24 on a
20-core GPU) and walks the keys in tiles of 16 with four barriers each.
Attention is a softmax-weighted average, and softmax can be computed in
pieces: each piece keeps its running max `m`, sum `l = Σ exp(s-m)` and
`acc = Σ exp(s-m)·v`; two pieces merge by rescaling with `exp(m_i - max)`.
So the new kernel gives each threadgroup one split of <= 32 keys (64 splits
at most) and one KV head; its 4 simdgroups share the K/V rows and own 3 query
heads each, a lane owns 8 features, and a small merge kernel combines the
partials in split order. The split layout depends on the key count only, so
every token, chunk and MTP row keeps the same order. Isolated at the 8K
shape: 2415 -> 120 us; profile at 8K: qsa.core 39.6 -> 8.1 ms, GPU groups
82.7 -> 51.1 ms/token (`profile-8k-attn`). Tolerance `attend_fast` = 2^-12,
set by an extreme case (logits ~ +-1000); realistic logits stay below 2^-19,
and the fast kernel is closer to FP64 than the oracle in every case.
`Q38_REF_KERNELS=attention` (or 1) keeps the oracle.

## R0/R1 — measure, then make Qwen3.8 Q2 decode fast (from 2026-10-17)

Plan: [R0-R1-plan.md](specs/qwen38/R0-R1-plan.md), spec:
[R0-R1-decode.md](specs/qwen38/R0-R1-decode.md). For this stage the human
chose to **build first and study afterwards**: agents implement each step
under the usual gates, and every step leaves a note here (concept, what
changed, the measured number) to be studied line by line later. Notes below
are newest first.

Correctness baseline on the unchanged code (`360acf0`): `make test-qwen38`,
`make test-qwen38-mtp test-qwen38-speculative`, `make test-all` and
`make qwen38-g3` all pass; G3 scaled logits error 2.24e-6 against the 1e-3
gate, argmax and ordered top-5 12/12.

### R1.9c — factored block dot (rejected) and the fp16 scale decode (shipped)

R1.9b's ablations showed the routed kernels are bound by the arithmetic that
dequantizes each weight, not by memory. Shipped: the fp16 block scales were
rebuilt bit by bit (branches + clz); the GPU's half->float conversion gives
the same bits for every finite pattern (half subnormals are FP32 normals, so
the FP32 flush never applies), so no fast kernel changes numerically.
Rejected (ruling R42): the factored Q2_K dot `dl·Σ(q·x) − ml·Σx` instead of
`Σ(dl·q − ml)·x`. It is not a reordering: with d = dmin, scale 1, min 3 and
codes 3 every weight is exactly 0 (S = 0, the oracle gives 0), yet the two
sums round apart and give up to 3.8e-6, so tolerance (b) cannot hold;
`test_q2_k_zero_weights_give_zero` keeps that case. IQ2_XXS factored gave no gain.
Bench (same session, alternated, battery): Q2_K down+reduce at 10 slots
75.4 → 64.8 µs (pilot, fp16 alone, not in runs.jsonl), IQ2_XXS gate+up 65.3 → 61.6, Q4_K 84.2 → 77.1.
Measured (controller, profile-t18 at f7bbeed, battery R38): GPU groups 44.99 ms/token (was 46.35),
moe 10.94 ms (down 3.35 ms at 92 GB/s, gate_up 3.28 ms at 124 GB/s); g3 pass (fp32 5.42e-7), ref
bit-identical, default max abs 5.19e-5, argmax 9/9.

### R1.9b — routed MoE kernels at bandwidth (target not reached)

Target: fused down+reduce and gate+up+SwiGLU (one simdgroup per output row) at >= 200 GB/s.
Bit-identical restructurings measured on the isolated bench (t=1 x 10 slots) did not move
them: slots in flight, rows per simdgroup sharing x, simdgroups per threadgroup, wider loads,
pooling the idle-lane tail chunks of the down dot (75.8 us vs 75.4 for the restored kernel,
a scratch run NOT recorded: no raw file; reverted). Only pooling
the gate+up tail chunks (K 2560 = 80 chunks: 16 + 16 fill one round of 32 lanes, each sum
shuffled back to its lane, same bits) stays: IQ2_XXS 67.7 -> 65.3 us, Q4_K 87.2 -> 84.2 us
(before: committed AC bench qwen38_r1_20261021; after: battery, qwen38_r1_20261028; no
same-session A/B, so within condition noise). Unrecorded scratch ablations (no raw file)
suggest why: gate_up IQ2_XXS falls to 36.9 us with the decode replaced by a constant,
down Q2_K to 52 us with the decode arithmetic dropped and 33 us with no weight read. The
kernels are bound by the exact FP32 per-value decode and the serial adds, not by DRAM;
reaching bandwidth needs a numerics ruling (factored block dot or FMA, with new tolerances).
Measured (controller, profile-t18): moe.down 3.83 -> 3.35 ms, moe.gate_up 3.75 -> 3.28 ms
(mostly the fp16 scale decode of R1.9c, not this step).

### R1.9a — MoE profile split

After the GDN fix the MoE block is the largest single profile group, but `moe` is one row: it
mixes the router/shared products, top-k routing, the gate+up experts, the shared expert and the
down+reduce. Like R1.8, Task 18-0 splits it in `moe.shared_in` (multi call), `moe.route`,
`moe.gate_up`, `moe.shared` (silu/mul/shared down/sigmoid) and `moe.down` (fused down+reduce or
its fallback), each with its weight bytes so the summary prints GB/s, plus a `moe rollup` line.
Profile mode only (a hook in `Q38FRoutedWork`): normal mode adds no dispatch, barrier or split.
Measured (controller, profile-t18-0 at 461d528): moe 11.95 ms/token = down 3.83, gate_up 3.75,
shared_in 2.01, route 1.28, shared 1.09; the rollup sums back to the old `moe` total.

### R1.8 — GDN hotspot

The GDN block was 60% of the token but one profile group, so Task 16-0 split it (profile only):
gdn.prep 23.2 ms/token (32.9%), proj 6.8, hc_pre 4.2, scan 3.7, out 3.0. QSA and the n-gram
path got no work (each under 5%). The prepare step L2-normalises the 32 q/k heads of 128
(q also by 1/sqrt(128)), copies the 6144 values and computes 48 gates (g = A·softplus(α+dt),
β = sigmoid(b)), and it ran on ONE GPU thread per token. `q38c_gdn_prepare_fast` gives each
head a simdgroup (lane-strided squares, then simd_sum: the order depends only on dk) and
strides the copy and gates over 1024 threads, with the input check folded in. Copy and gates
are bit-identical to the oracle; the norms differ only by summation order (tolerance 2^-20,
measured max 5.7e-7). Bench at the real dims, t = 1: 666.8 -> 19.4 µs, about 24.0 -> 0.7 ms
per token by estimate (`bench/qwen38_r1_20261026/gdn-prep`). The scan was not touched.
Idea from ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_gdn_front/gdn_prep; no code copied.
Measured (controller, quick closure t16, 8-token profile, AC): GPU 46.9 ms/token (Task 15
69.2, -32%; normal-mode 50.2 ms); gdn 42.2 → 18.8 ms, gdn.prep 23.2 → 0.33 ms; G3 scaled 5.4e-7;
`Q38_REF_KERNELS=1` bit-identical (`bench/qwen38_r1_20261026/profile-t16`).

### R1.7b — GPU argmax

Greedy decode needs one number per token, the id of the largest logit, yet every token copied
the whole 248320-float row (~1 MB) to the host and scanned it on the CPU. Now one threadgroup
of 1024 threads reduces the row on the GPU right after the LM head (after an explicit barrier,
the encoder is concurrent): each thread keeps the first maximum of its strided slice, then a
tree keeps the larger value and, on a tie, the smaller index, so the answer is the FIRST
maximum whatever the split. Non-finite logits never compete and set error bit 1 (the session
turns it into Q38_GPU_ERROR). The result is an integer, so the test is exact equality with
the CPU scan: random rows, ties across and inside a thread's slice, all-equal, -0/+0, sizes
that are not a multiple of 1024, NaN/Inf, and decode/chunked appends in the graph. Only plain
greedy reads just the id (`qwen38_gpu_forward_argmax`); `--logits`, MTP and the gate paths
still read logits. `Q38_REF_KERNELS=argmax` (or 1) keeps the logits readback and CPU scan.
Idea from ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_argmax; no code copied.
Measured (controller): not exercised by the teacher-forced quick-closure profile (no greedy
step); the greedy gain is measured by the Task 17 ladder.

### R1.7a — Session-wide weight claims

Every command used to declare (claim) every weight tensor, and each claim was checked
against all earlier ones: O(n^2) work and a buffer retain per tensor, per token, before the
GPU got anything. The weights are read-only and live as long as the session, so the session
now registers them once (`qwen38_gpu_session_pin_read`, sorted ranges with a running max of
the end address, one binary search per later claim). Semantics are unchanged: a WRITE over a
pinned READ is still refused, READ/READ overlap is allowed, no-copy aliases are still found by
address. Per token the graph now claims only its ~60 scratch buffers. R1.7a also adds one
`MTLResidencySet` (macOS 15+, probe `residency_set`) with the weight views, shared by the main
and MTP queues, intended to keep them resident across command buffers (not measured); it is
skipped, with a stderr line, unless views + 4 GiB fit 85% of the recommended working set, and
it logs its allocatedSize; weights only, no numeric effect. `Q38_REF_KERNELS=claim` (or 1) restores per-command claims and no
residency set. Claim cost per token with 1500 weights + 60 scratch (g0 micro-benchmark,
`Q38_CLAIM_BENCH=1`, ASan build): 1.57 ms -> 0.009 ms.
Measured (controller, quick closure t15, 8-token teacher-forced profile, AC): host encode
5.2 → 4.7 ms/token, wall decode 77.8 → 77.1 ms/token. On the real store the residency set is
skipped by its gate (42720 MiB + 4096 MiB > 85% of 53084 MiB). `Q38_REF_KERNELS=1` bit-identical;
G3 scaled 4.2e-7 (`bench/qwen38_r1_20261026/profile-t15`).

### R1.6b — Concurrent encoder

A serial Metal encoder runs dispatch N+1 only after N has finished, even when N+1 does not
need N's results. A concurrent encoder (`MTLDispatchTypeConcurrent`) lets them overlap and
orders only at explicit memory barriers, so every write → read (or read → write) pair needs
a barrier: a missing one is a race. The session now opens concurrent encoders; with
`Q38_REF_KERNELS=barrier` (or 1) it keeps the serial encoder and today's barriers. An audit of
every dispatch path showed only two kinds of removable barrier: check → op inside a C op
(the check now ORs into a second error word that no kernel reads, `finish()` ORs both; kept
when the op writes what the check reads: HC_ADD, CONV, GDN_SCAN) and the graph's duplicate
barrier after the MoE block. A failed check then no longer stops its op, so on a failed
command the error bits may differ from the serial ones (bit 1 still set; the status is the
contract). Fake graph, one decode token: 211 dispatches either way, barriers 207 → 165.
Logits and traces bit-identical over 3 runs per mode.
Measured (controller, quick closure, 8-token profile, AC): GPU 68.8 ms/token (Task 14b
75.6, -9%; normal-mode decode 62.3 ms); barriers 2270 → 1788, dispatches 2307. Default logits
bit-identical over 3 runs and to the serial Task 14b run; `Q38_REF_KERNELS=1` bit-identical;
G3 scaled 4.2e-7 (`bench/qwen38_r1_20261026/profile-t14a`).

### R1.6a — Parallel RMSNorm

RMSNorm: y = x / sqrt(sum(x²)/width + eps) * gain (GDN norm: then * sigmoid(z)).
`q38c_rms` used ONE thread per row: at one token each HC norm [1, 4, 2560] (per stream)
was 4 threads, each a serial pass over 2560 floats, 290 µs (61 calls = 17.7 ms in the t12
profile; 39 more inside the GDN block and PLE). `q38c_rms_fast`: one 256-thread threadgroup
per row, strided squares, `simd_sum`, the 8 simdgroup sums in order (order depends only on
the width), the oracle's expressions. At [t, 4, 2560] it is within 2.6e-7 of FP64, the oracle
1.2e-6; tolerance `rms_fast` 2^-18 (max 3.4e-6, MTP h-norm). The input check is folded in
(1 dispatch + 1 barrier instead of 2 + 2, same error word). All RMS and GDN norms use it;
`Q38_REF_KERNELS=fuse` (or 1) keeps check + `q38c_rms`. Idea: ds4 fc80bd6 (MIT)
`kernel_qwen4_hc_norm`, nothing copied. AC protocol bench
(`bench/qwen38_r1_20261026/rms`): HC norm at t = 1 286.6 → 25.9 µs (11.1x), t = 32 378.3 →
12.8 µs; small widths at t = 1 are faster too (GDN gated 61.5 → 4.5 µs). Per token, counting
100 HC-shaped norms (61 in the profile's rms groups + 36 GDN block + 3 PLE) and the small
widths: 32.0 → 3.3 ms (estimate from per-dispatch time, not end-to-end).
Measured (controller, quick closure, 8-token profile, AC): GPU 75.6 ms/token (Task 13b
105.7, -28%); rms groups 17.9 → 0.63 ms, gdn 56.3 → 44.4 ms (its 36 norms); dispatches 2307.
G3 scaled 4.2e-7; `Q38_REF_KERNELS=1` bit-identical (`bench/qwen38_r1_20261026/profile-t14b`).

### R1.5d — Parallel router top-k

The router turns 512 logits into p = softmax(z) and keeps the 10 largest as the
token's experts, weights p renormalized to sum 1. `q38f_route` did it in ONE GPU
thread per token (512 exp, a serial sum, 10 scans): 1227.3 µs per layer, 58.9 ms
per decoded token over 48 layers (36.7% of the 160.6 ms post-Task-12 token).
`q38_moe_route_fast` (`qwen38_moe_fused.metal`): one 256-thread threadgroup per
token. Max, exp, divide and each selection round (arg-max over larger p, then
smaller id = the oracle's strict `>`; chosen experts masked in threadgroup memory)
are exact in any order. Float addition is not, so the exp sum (thread 0, e = 0..511)
and the sum of the 10 weights keep the oracle's order. Ids, weights and probs are
bit-identical to `q38f_route` (5 kernel tests; graph layers 0–3);
`Q38_REF_KERNELS=moe` (or 1), E > 1024 or k > E dispatch the oracle. Idea: ds4
fc80bd6 (MIT) `kernel_qwen4_router_topk`, nothing copied. Protocol bench on AC
(`bench/qwen38_r1_20261023/moe-route`, ee67690, synthetic logits, µs median):

| tokens | q38f_route (p10/p90) | q38_moe_route_fast (p10/p90) |
|---|---|---|
| 1 | 1227.3 (1224.7/1229.6) | 23.9 (23.8/24.7) |
| 4 | 1230.6 (1228.5/1232.3) | 24.0 (23.9/24.0) |
| 32 | 1294.8 (1294.0/1295.4) | 24.6 (24.6/25.0) |
| 128 | 1310.6 (1310.5/1311.1) | 38.7 (38.7/38.7) |

t = 1, 48 layers: 58.9 → 1.15 ms per token (estimate); faster at 32/128 too: no threshold.

Measured (controller, full closure for Tasks 13+13b at 0f04548, AC): ladder
9.371 tok/s (p10 9.247, p90 9.470, n 20; Task 9 3.956, x2.37; 0.28x DwarfStar
R0). 8-token profile: GPU 105.7 ms/token (Task 12 160.6, -34%); the MoE group
70.7 → 12.9 ms (94.6 GB/s); dispatches 2578 → 2482. `make test-qwen38`, G3 and
`Q38_REF_KERNELS=1` bit-identical all pass (`bench/qwen38_r1_20261026/ladder-t13`,
`profile-t13`).

### R1.5c — Fused down + reduce

After the gate+up+SwiGLU, each routed expert projects its act row back to the
hidden size (down, 640 → 2560), and the token's output is
y = Σ_s w_s·down_s + σ(z)·shared, summed in slot order 0..k−1. Before, this
took two dispatches: an indexed down writing a [pairs, 2560] scratch (100 KiB
per token), then `q38f_reduce` reading it back. Now `q38_moe_down_reduce`
(in `qwen38_moe_fused.metal`) does both in one dispatch, after the shared
expert. One simdgroup per (token, column) runs the same `q38_quant_row` per
slot and accumulates with `q38f_reduce`'s expressions (contraction off in
both files), so y and routed are bit-identical to the unfused sequence
(kernel level, all four formats; graph, layers 0–3). Per MoE layer call, the
identity map (`q38f_iota`), the indexed down and `q38f_reduce` (3 dispatches,
3 barriers) become 1 dispatch and 1 barrier. `Q38_REF_KERNELS=moe` or `indexed`
restores the sequence. Bench `bench/qwen38_r1_20261021/moe-down/`
(512 experts, 10 slots):

| µs, fused / sequence | 1 token | 4 | 32 | 128 |
|---|---|---|---|---|
| Q2_K 640/768→2560 | 75.3 / 81.7 | 282.9 / 311.3 | 2205 / 2471 | 8877 / 9995 |
| MXFP4 (layer 48) | 94.4 / 101.4 | 354.5 / 383.3 | 2786 / 3027 | 11120 / 12124 |

Measured together with R1.5d (one checkpoint closure): see R1.5d.

### R1.5b — Fused gate+up+SwiGLU

An expert's MLP starts with h = SiLU(g)·u, where g = gate·x and u = up·x
(SwiGLU: the gate decides how much of u passes). R1.5a did this in five
dispatches: an indexed gate, an indexed up, CHECK, ACT, MUL, each one writing to
memory and followed by a barrier. Now `q38_moe_gate_up` (in
`qwen38_moe_fused.metal`) does it in one. For each (slot, row), one simdgroup
decodes the gate and up chunks, multiplies each x value it loads by both, and
applies q38c_activate's SILU formula. Each dot keeps `q38_quant_row`'s order,
so act is bit-identical to the five-dispatch sequence: at kernel level for
all four formats, and in the graph, where y is bit-identical too. gate/up are
no longer written. `Q38_REF_KERNELS=moe` (or `indexed`, whose oracle sequence
the fused kernel does not match) restores the five dispatches. Protocol bench
`bench/qwen38_r1_20261021/moe-gate-up/` (512 experts, 10 slots per token):

| µs, fused / 4 dispatches | 1 token | 4 | 32 | 128 |
|---|---|---|---|---|
| IQ2_XXS 2560→640 (gate+up) | 67.7 / 81.7 | 261.1 / 294.1 | 2088 / 2361 | 8279 / 9463 |
| Q4_K (layer 48) | 87.2 / 101.5 | 340.8 / 372.0 | 2704 / 2981 | 10656 / 11713 |

Measured (controller, quick closure for Tasks 11+12, 8-token profile, AC): GPU
160.6 ms/token (Task 10 248.6, -35%); the MoE group 159.4 → 70.7 ms (17.2 GB/s,
now 44% of the token); dispatches 2770 → 2578. G3 scaled 9.1e-7;
`Q38_REF_KERNELS=1` bit-identical (`bench/qwen38_r1_20261021/profile-t12`).

### R1.5a — Indexed experts

After R1.3 the routed experts were 64% of the decode token: 10 experts per
token, each a gate, up and down matvec through `q38_packed_linear_indexed`,
one thread per output row walking K serially (7.6 GB/s). Now each output gets
one 32-lane simdgroup: lane l decodes chunks l, l+32, ... of the row (an
IQ2_XXS group or Q4_K sub-block of 32 columns, a Q2_K scale group of 16, an
MXFP4 block of 32), sums each chunk in column order, then its chunks, then
`simd_sum`. The order depends on K and the format only. The row dot
(`q38_quant_row`, in `qwen38_quant_decode.h`) is shared with the fused MoE
kernels. Weights decode to the oracle's FP32 values, so only the order differs:
`tol = 2^-21` for all four formats (the GPU flushes the 16 subnormal MXFP4
weights, oracle and fast alike). `Q38_REF_KERNELS=indexed` restores the oracle.
Fast is faster at every row count, so there is no threshold. Protocol bench
`bench/qwen38_r1_20261021/matvec-indexed/` (512 experts, 10 slots per token):

| µs, fast / oracle | 1 token | 4 | 32 | 128 |
|---|---|---|---|---|
| gate IQ2_XXS 2560→640 | 37.2 / 704.9 | 141.4 / 1799.3 | 1162 / 13055 | 4643 / 51443 |
| down Q2_K 640/768→2560 | 78.1 / 543.6 | 307.3 / 1916.5 | 2465 / 14619 | 9885 / 58415 |
| gate Q4_K (layer 48) | 46.9 / 1069.6 | | 1463 / 17799 | |
| down MXFP4 (layer 48) | 95.4 / 372.5 | | 3008 / 10545 | |

Measured (controller): with Task 12, in one quick closure — see R1.5b
(`bench/qwen38_r1_20261021/profile-t12`).

### R1.4 — Multi-GEMV

Several projections often read the same x: QSA's five (qg, k, v, iq, ik), GDN's
QKV, Z, alpha, beta, and the router, shared gate/up and shared logit of the
MoE. Each was its own dispatch plus a barrier, a few µs of launch per call.
`q38_packed_encode_multi` encodes up to 8 of them as one dispatch per format:
the grid holds the sum of their n_out, a threadgroup finds its projection
from prefix offsets and runs the single kernel's row function, so every
output is bit-identical to the single call (order still K and format only).
x-side fields come from `p[0]`, only the tensor side from `p[j]`: indexing
every field dynamically made each short threadgroup ~5% slower. Graph: one
call and one barrier per site; profile group `qsa.proj` replaces the five
`qsa.*` projection groups. Synthetic, 20 per buffer, K 2560
([bench/qwen38_r1_20261020/matvec-multi](../bench/qwen38_r1_20261020/matvec-multi/README.md)):

| group, rows 1 / 4 (µs) | multi | single calls |
|---|---|---|
| QSA, 5 × Q8_0 | 139.1 / 220.2 | 147.7 / 231.9 |
| GDN, 2 Q8_0 + 2 F32 | 182.1 / 266.7 | 180.4 / 271.2 |
| shared expert, 2 F32 + 2 Q8_0 | 26.1 / 42.1 | 31.5 / 51.6 |

Measured (controller, quick closure, 8-token profile, AC): GPU 248.6 ms/token
(Task 9 248.4: no measurable gain), dispatches 2986 → 2770, barriers 3033 →
2733; the MoE (64%) hides it. G3 scaled 8.0e-7; `Q38_REF_KERNELS=1`
bit-identical; default logits equal to Task 9's (`bench/qwen38_r1_20261020`).

### R1.3 — F16 and F32 matvecs

The HC tensors are F16 (down 10240→320, inject 10240→4, up 320→10240, 49
layers each) and the router and ssm_alpha/beta are F32; the oracle gave
inject 4 threads for 10240 columns each, 0.04 GB/s. `q38_matvec_f16` and
`q38_matvec_f32` repeat the Q8_0 scheme with an 8-column chunk in place of a
block (one 16- or 32-byte load per lane), and add one thing: S simdgroups
share an output row, S = `dense_split(K)` = 1 below K 2560, 2, 4, then 8 from
K 10240. S depends on K alone, so the order of the sum (chunk, lane's chunks,
`simd_sum`, the S partials in order) still never depends on the rows or the
tiling. Every weight decodes exactly (F16 by hand, like the oracle; the
65,536 patterns are tested), so the products equal the oracle's.
`tol["f16"] = 2^-21` (max 4.2e-7, p99 1.7e-7, 255,206 outputs),
`tol["f32"] = 2^-20` (max 4.9e-7, p99 1.8e-7, 255,182). Evidence:
[bench/qwen38_r1_20261020/matvec-f16-f32](../bench/qwen38_r1_20261020/matvec-f16-f32/README.md).

| shape, 1 row (µs) | fast | oracle | speedup |
|---|---|---|---|
| F16 500 MiB, 20 per buffer | 1830, **287 GB/s (93%)** | 5903, 89 GB/s | 3.2x |
| F32 500 MiB, 20 per buffer | 1823, **288 GB/s (94%)** | 2649, 198 GB/s | 1.5x |
| HC down 320×10240, 200 per buffer | 24.5 | 2154.5 | 88x |
| HC inject 4×10240 | 5.2 | 2002.3 | 385x |
| HC up 10240×320 | 42.1 | 72.9 | 1.7x |
| router 512×2560 | 12.6 | 439.6 | 35x |

Fast is faster at every row count 1–1024 on HC down, router, up and inject
(narrowest: HC down at 1024 rows, 43.0 vs 58.2 ms). Alternatives tried
(exploration runs, not committed): S fixed at 1 / 2 / 4 / 8 on inject 20.6 /
11.5 / 6.3 / 4.2 µs, on HC down 27.8 / 22.4 / 21.5 / 22.3, on HC up 41.2 /
59.8 / 91.4 / 152 (so S grows with K only); F16 by hardware `float(half)`
(also exact on all patterns) 285 vs 290 GB/s and slower on the small shapes;
chunks of 4 or 16 columns with byte loads 184–200 GB/s on F16 (F32 stays at
~294 either way); rows of x per threadgroup T = 1/2/4/8 on HC down at 1/4/32
rows 19.0/67.0/514, 22.0/46.2/382, **24.5/36.6/349**, 30.0/46.8/333 µs.

In the graph: `q38_packed_encode` now runs the F16 and F32 kernels by default,
at any row count, like Q8_0 (one predicate: a format with a fast pipeline and
`linear` not selected). `Q38_REF_KERNELS=linear` restores the oracle bit for
bit. Tests: `test_default_selects_fast_f16`/`_f32`, and
`test_reference_env_selects_oracle` over the three formats. No Make target
compares `q38_packed_encode` with the CPU bit for bit (B-GPU's exact checks are
a zero row and a single impulse). One GPU-vs-GPU check compared formats: the
MoE padding test ran the Q2_K down tensor against the same values as F32 with
1e30 padding, equal only while both used the oracle's order; it now compares
two F32 runs (original vs poisoned padding), same intent.
Measured (controller, ladder, short, 3+20, AC): 3.956 tok/s (p10 3.952, p90
3.962; Task 8 1.377, x2.87), GPU 248 ms/token (Task 8 722); HC F16 now ~2.4%,
MoE 64%, GDN 22%. G3 scaled 8.0e-7; `Q38_REF_KERNELS=1` bit-identical
(`bench/qwen38_r1_20261020`).

### R1.2b — Q8_0 matvec by default in the graph

`q38_packed_encode` now runs `q38_matvec_q8_0` for every Q8_0 tensor, at any
row count (R1.2a measured it faster than the oracle from 1 to 1024 rows).
Both entry points share one helper, so validation is identical. Other formats,
the indexed path and the lookup keep the oracle. The oracle stays reachable:
`Q38_REF_KERNELS=linear` (or `1`, or a typo, ruling R18) restores it bit for
bit. So the default path is no longer bit-identical to `360acf0`. The gate is
G3, plus `Q38_REF_KERNELS=1` bit-identical to the oracle logits. Tests:
`test_default_selects_fast_q8_0`, `test_reference_env_selects_oracle`. No
Make target needed the reference: none compares `q38_packed_encode` with the
CPU bit for bit. B, F, G2 and compositions use the γ bound or a tolerance. The
exact checks are GPU against GPU (chunk, replay, MTP) or exact IDs, and they
pass on the fast path.
Measured (controller, ladder, short, 3+20, AC): 1.377 tok/s (p10 1.376, p90
1.377; R0 1.064, +29%), GPU 722 ms/token (R0 936); LM head 2.44 ms at 277 GB/s.
G3 scaled 1.7e-6; `Q38_REF_KERNELS=1` bit-identical (`bench/qwen38_r1_20261019`).

### R1.2a — Cooperative Q8_0 matvec kernel

Decode reads every weight once per token, so a matvec should stream bytes at
the 307 GB/s ceiling. The oracle gives one thread per output row and walks
K serially, decoding one element at a time. `q38_matvec_q8_0`
(`src/qwen38_matvec.metal`, reached only through `q38_packed_encode_fast`
until Task 8) gives one 32-lane simdgroup per output row. Lane l takes whole
34-byte blocks l, l+32, …, so the simdgroup reads 1088 contiguous bytes. It
sums each block in column order, then its blocks, then `simd_sum`. That order
depends only on K, never on rows or tiling. Each decoded block serves 4 rows
of x. w = d·code is exact in FP32, so every product equals the oracle's; only
the order of the sum differs. Against FP64 the fast kernel is about 10x closer
than the oracle (real tensors: 1.6e-8 vs 1.5e-7 of S = Σ|w·x|).
`tol["q8_0"] = 2^-21`: corpus max 4.6e-7, p99 2.6e-8, 257,323 outputs. The
fp16 scale decodes exactly on the 63,488 finite patterns; the 2,048 non-finite
ones set bit 1. Measured on synthetic data, median of 20 after 3 warm-ups, AC
power, thermal state nominal before and after. Raw evidence and conditions are
in [bench/qwen38_r1_20261019/matvec-q8_0](../bench/qwen38_r1_20261019/matvec-q8_0/README.md):

| tensor (K 2560) | fast, 1 encode per command buffer | fast, 20 per buffer | oracle |
|---|---|---|---|
| LM head 248320 rows, 644 MiB | 2518 µs, 269 GB/s | **2414 µs, 280 GB/s (91%)** | 12933 µs, 52 GB/s |
| attn_qkv 10240 rows, 26.6 MiB | 320 µs, 87 GB/s | 80.2 µs, 348 GB/s¹ | 625 µs, 45 GB/s |

¹ above the ceiling: back to back, part of the 26.6 MiB stays in the system
cache. The LM head, 24 times larger, is the honest number. A lone short
command buffer runs at an idle GPU clock, so it is slow and noisy; the decode
graph encodes about 3000 dispatches per buffer.

| attn_qkv, rows of x | 1 | 2 | 4 | 8 | 32 | 128 | 1024 |
|---|---|---|---|---|---|---|---|
| fast µs | 80.2 | 107.1 | 151.4 | 298.7 | 2128 | 10011 | 87942 |
| oracle µs | 633.5 | 1273 | 2311 | 4462 | 17019 | 66885 | 538161 |
| speedup | 7.9x | 11.9x | 15.3x | 14.9x | 8.0x | 6.7x | 6.1x |

Rows ≤ 32 are timed 20 per buffer, 128 and 1024 one per buffer. Fast is
faster at every row count, also on the other real shapes (2560×640,
2560×6144, 640×2560: 5.8x to 45x, in the same folder). Beyond 8 rows it is
compute-bound: whole-block lanes read x at a 128-byte stride.

Alternatives tried (exploration runs of 2026-10-18, not committed as evidence),
rows = 1, GB/s on 159 MiB / attn_qkv / ssm_out (K 6144):
- Simdgroups × output rows per simdgroup: **1×1: 287/339/405**; 2×1 280/317/388;
  4×1 253/305/360; 8×1 243/293/390; 1×2 277/314/349; 4×2 255/257/350;
  1×4 238/258/343; 8×4 222/266/337.
- Loads, 4×2 tile: bytes 272/315/351; packed_char4 + packed_float4
  272/297/338; ushort 264/314/350.
- Quarter block per lane, the layout of ds4 fc80bd6 (MIT,
  metal/dense.metal:kernel_mul_mv_q8_0_f32_impl): 280/270/290. At 128 rows
  25% faster, at 4–8 rows slower.
- Rows of x per threadgroup, µs at 1/4/8/128 rows: T=1 102/221/488/11817;
  T=2 115/175/361/10698; **T=4 105/180/305/9984**; T=8 104/219/312/10217;
  T=16 125/329/502/9906.
- x staged in threadgroup memory (33-float pitch): 167 µs vs 85 at one row,
  24 ms vs 10 at 128. Occupancy loss, rejected.
- Explicit `fma`: no gain (bound by x loads), so `fp contract(off)` stays.

### R1.1 — Reference-kernel selector and fast-kernel harness

Every R1 fast kernel is judged against the slow oracle it replaces (one thread
per output row, serial K loop), so the oracle stays selectable for good:
`Q38_REF_KERNELS=1` (or `all`, or a family list such as `linear,moe`) is read
once when a context is created; an unknown token selects every oracle, so a
typo can never silently run a fast kernel. `q38_packed_encode_fast` exists with
the same admission as `q38_packed_encode` and answers UNSUPPORTED for now; the
default path is untouched. `make test-qwen38-fast-kernels` runs oracle, fast
path and an FP64 product (from `q38_packed_decode_row`) on the real row slices
plus seeded shapes, with S = Σ|w·x| per output. A fast kernel must meet
(a) |y−r| ≤ γ(2K+8)·S + 2·ulp32(r), valid for any summation order, and
(b) |y_fast−y_oracle| / max(S, 2^-126) ≤ tol, tol the power of two above the
measured corpus maximum (`tests/qwen38_kernel_tolerances.json`, empty today).
Measured: 45 cases, 19,944 outputs, oracle against itself 0 bits apart, (a) at
most 1.1% of the bound, ~15 s per run; every fast case is "unsupported".

### R1.0 — Fenced G3 launches and picoforge-only runs

G3 (`tools/qwen38_g3.py`) maps the 46 GiB checkpoint four times back to back:
`forward_real`, the stock-Metal oracle, `ds4 --cpu`, the picoforge chat. A
Metal process's wired pages stay wired for ~1.5 s after it exits, and mapping
the checkpoint again inside that window froze the Mac twice (ruling R9); in
the R0 baseline G3 the stock-Metal process started under a second after
`forward_real` exited. Those four steps now start through the paired
harness's `Launcher` (release wait before and after, strict memory guard with
the R0.5a sleep gap, full-wake precheck), each in its own `launch-<name>`
directory with a 900 s timeout; clang and the synthetic-graph replay map
nothing and stay plain. The gate itself is untouched. The second change is
`qwen38_compare.py run --engines picoforge`: DwarfStar gets no samples
(`(0, 0)` in the schedule, `null` in the summary, `engines` in the manifest).
Every R1 task measures picoforge alone, 3 warm-ups + 20 measured on the short
frontier; DwarfStar's baseline stays the R0 one.

### R0.5 — Recorded decode baseline

Before any R1 kernel changes, decode is measured where it stands and against
DwarfStar on the same Mac and file, so the profile, not taste, orders R1 and
every step has a baseline. Decode reads each weight once per token: ceiling =
bytes ÷ 307 GB/s. Short frontier, paired, teacher-forced (9 + 128 IDs, one
process per sample, alternated, 3 warm-ups + 20 measured): picoforge 1.064
tok/s (p10 1.062, p90 1.066), DwarfStar `fc80bd6` 33.075 (p10 23.62: three
consecutive slow samples and one at 27.6, thermal nominal, kept), ratio 0.0322.
Profile, 935.5 ms/token: `gdn` 34.6%, `moe` 28.1%, HC F16 26.5% (`inject`,
n_out = 4, 0.04 GB/s), so the spec order R1.2 → R1.3 → R1.5 stands. 6104.78 MB
per token (ceiling ~50 tok/s), 6.5 GB/s = 2.1% of 307; DwarfStar ≈ 202 GB/s
(65.8%, estimated with picoforge's bytes); 2986 dispatches, floor 4.2 ms. At
8K only a profile (974.0 ms/token, prefill 1406 s, `qsa.core` 3.1 → 40.5 ms):
the paired 8K run is deferred (ruling R22) after three invalid attempts, all
kept. `oracle/oracle-logits.f32` is the reference for `Q38_REF_KERNELS=1`.
Evidence and limits: `bench/qwen38_r0_20261017/README.md`.

### R0.5a — Sleep-gap abort and full-wake precheck

The first 8K paired baseline ran while the Mac was in a DarkWake (lid closed):
it wakes for ~20 s maintenance slices every ~15 min with no Graphics, so
picoforge advanced only during those slices and the sample ran into the 3600 s
timeout without anything noticing. The observer's clock is CLOCK_MONOTONIC,
which on macOS keeps counting through sleep, so a sleep appears as one very
long gap between two memory-sample attempts. A sample spanning a sleep is not
a measurement. `run_observed.py` takes an optional guard key `gap_s` (absent
from the default guard, so other callers are unchanged): a larger gap SIGTERMs
then SIGKILLs the child with `stop_reason: "sleep gap"`, and `process.json`
always records `max_sample_gap_s`. The paired harness sets `gap_s=5`, and its
Launcher refuses any checkpoint launch unless Graphics is in `pmset -g
systemstate`, the lid is open (`ioreg` AppleClamshellState) and the Mac is on
AC; `run` checks the same before it creates its output directory, and every
conditions.json records the wake state. The number: the failed attempt's
sampling gaps reached 899.05 s; the longest gap in the 46 valid short-run
samples is 0.35 s, so 5 s separates the two by a wide margin.

Ruling R20 (user decision 2026-10-18): clamshell mode counts as awake. With the
lid closed, two external displays and AC, `ioreg` prints `AppleClamshellCausesSleep`
= No and `AppleClamshellState` = Yes, and the Mac does not sleep. The precheck
now reads `lid_ok` = lid open or `AppleClamshellCausesSleep` = No; the sleep-gap abort is unchanged.

### R0.4b — Paired picoforge/DwarfStar decode harness

Paired: both engines decode the same IDs on the same Mac and file, so the
ratio isolates the engine. Teacher-forced: after prefill to a frontier, 128
fixed IDs are appended one by one, so tok/s (forced IDs / decode seconds)
counts equal work. Alternated: engine blocks interleave, so heat or background
drift falls on both. `tools/eval/qwen38_compare.py`: `prepare` fixes the IDs of
the short frontier (frozen 9-ID coding prompt + 128 IDs of picoforge's greedy
continuation, refused if a stop ID appears) and of the 8K one (first 8063 + 128
IDs of `promessi_sposi.txt`), and requires picoforge and `ds4 --dump-tokens`
to tokenize them identically; `run` gives one observed process per sample for
both engines (ds4-bench measures one frontier per process); `summarize` gives
the median, nearest-sample p10/p90 and the ratio; `ssd-check` makes one try.
Lesson: mapping the checkpoint while the last Metal process's ~46 GiB were
still wired froze the Mac twice. Every launch now waits for release before
and after and runs under a strict memory guard. Smoke (short frontier, AC,
1 warm-up + 2 measured per engine, not the baseline): picoforge 1.064 tok/s,
DwarfStar 33.07 tok/s, ratio 0.032; pressure normal, swap 0. `ds4
--ssd-streaming` refuses Qwen3.8 (exit 1: SSD streaming not supported).

### R0.4a — Teacher-forced decode in the CLI

Teacher forcing means the next input token comes from a fixed list, not from
the model's own choice. Given the same IDs, picoforge and DwarfStar
(`ds4-bench --teacher-forced-decode`) do exactly the same work, so tok/s
compares equal work even where the two would sample differently. Each forced
ID is its own one-token append (one logits row, one argmax), as in
`ds4_session_eval`: a batched append would be prefill, not decode. New
`--teacher-forced N` (with `--tokens`; excludes `--mtp`, `--session`,
`--stop-after`, `--max-new`): the first count−N IDs are prefilled in chunks,
the last N appended one by one, no stop on EOS; `decode_s` spans the N
appends and argmaxes, `decode_tokens = N`, JSONL adds `mode`/`forced_tokens`.
Test: forcing an ordinary greedy run's own tokens reproduces its logits bit
for bit. Smoke, real checkpoint, slow baseline, on battery, one run: 4 forced
tokens after a 9-token prompt, 3.763 s → 1.06 tok/s (941 ms/token).

### R0.3 — Bytes, dispatches and barriers per profiled op

Decode reads each weight it uses once per token, so time ≥ bytes ÷ 307 GB/s;
launches set a second floor, time ≥ dispatches × 1.4 µs (Step 4.1). The
profile now fills both per group: `weight_bytes` from the real bindings, also
for composites (`gdn` its linears, `moe` router + shared + top_k experts of
gate/up/down, `ple` key/value + staged n-gram rows; `qsa.core` reads caches,
not weights: 0), and dispatches/barriers counted by an `NSProxy` around the
encoder, in profile mode only. `tools/eval/qwen38_profile_summary.py` makes
the per-op table and per-token totals. Also: a stale reused timestamp now
fails, warm-up repetitions write no rows, resolve left `wait_s`. Real
checkpoint, one decode token (mean of 2, on battery): 6.10 GB of weights →
floor 19.9 ms, measured 935.6 ms (6.5 GB/s, 2.1% of peak); 2977 dispatches →
floor 4.2 ms; 3024 barriers; sum of groups / normal `gpu_s` 1.0015. Logits
bitwise identical with and without `--profile`.

### R0.2 — Opt-in per-op GPU profile

The GPU stamps time only at encoder boundaries, so `--profile FILE.csv` gives
each group (embed, every `linear()`, each HC C op, `gdn`, `qsa.core`, `moe`,
`hc_add`, `final.*`, `lm_head`) its own compute encoder in the same command
buffer, with start/end timestamps in a counter sample buffer; GPU ticks are
calibrated with `sampleTimestamps:gpuTimestamp:` (1 ns per tick here). Metal
keeps encoder order and hazard tracking, so results are unchanged; but each
boundary costs a little and the gaps between encoders belong to no group, so
the sum of `gpu_us` is not `gpu_s`. New: `src/qwen38_profile.{h,m}`,
`q38g_session_split`, graph marks (no-op, no extra encoder, without a profile;
MTP appends are not profiled), `make test-qwen38-profile`. Real checkpoint,
coding prompt, one decode token per mode (n=1): logits bitwise identical;
`gpu_s` 931.7 ms normal, 934.2 ms profiled (+0.27%); sum of the 656 groups
932.1 ms (+0.04%). Tiny CLI graph, median of 20: +1.8% and +1.0%.

### R0.1 — Metal capability probe

We ask the live device what it supports instead of trusting headers: Metal 4
and residency sets are young, so each is called only after
`respondsToSelector:`/`@available`. What matters for R0.2 is the counter
sampling point: *stage* sampling reads the GPU clock at the start/end of a
whole encoder pass, *dispatch* sampling around every dispatch inside one
encoder (what per-op timing in a fused graph would need). New:
`tools/probe/qwen38_capabilities.m` and `make test-qwen38-r0-tools` (one lock
in the recipe, none in the test). Measured on this M5 Pro (macOS 27.0.1, AC,
thermal `nominal`; `bench/qwen38_r0_20261017/capabilities.json`): the only
counter set is `timestamp`, sampling works at the **stage** boundary only
(dispatch, draw, blit, tile false), so per-op times need separate encoders or
command buffers. `GPUStartTime/EndTime` are valid; Metal 4 family,
`MTL4CommandQueue` and residency sets are available; working set 55.66 GB,
max buffer 41.75 GB, unified memory.

## 2026-10-11 — accepted-route cost predictor, opt-in experiment

`auto-experts` forecasts verify cost from the last 32 accepted input routes,
then learns time versus distinct experts online. Rejected drafts contribute
to observed cost, not accepted history. CLI diagnostics, CPU/unit and greedy
integration tests, benchmark metrics and the local UI are included.

The six-prompt pilot gives 1.009x decode speed and 0.988x process speed versus
`auto`, with all selected outputs identical to greedy. The cost forecast is
less accurate than the existing estimate (2.459 vs 1.909 ms mean absolute
error). The 10% adoption threshold is not met; the default stays unchanged.
The [report](phase4-expert-experiment.md) retains every prompt, graph and the
audit trail for an interrupted run and a repeated prompt with invalid timers.
The benchmark runner now matches the engine's clock and rejects inconsistent
timing boundaries. `make test-all`, final targeted integration tests, timer
validation and CPU sanitizers passed. First-use latency remains separate work.

## 2026-10-11 — attention bounds and reproducible request timings

Attention score storage now follows the active context, with a GPU-capacity
check. Tests cover 2,048, 2,049 and 4,096 positions and compare the tiny MoE's
full 4,096-token routing trace with Transformers. Generation accepts
`--context N`; ordinary and speculative greedy agree at the cache boundary.

The request runner now keeps raw runs, warmups, shuffled configuration order,
hashes, peak process RSS and separate prefill/decode/process timers. No runtime
dependency was added. The old adaptive result was 2.3% ahead of fixed `k=2`,
which needs a stronger comparison before it can support a performance claim.

**Correction to step 4.8b below:** the old 4,096-token routing result crossed
a 2,048-entry attention buffer. Its 39% locality figure is invalid. The corrected
trace gives 29.8 distinct experts for eight tokens, or 58% of uniform routing.
Historical text is retained below; the corrected CSV and current protocol take
precedence.

The new ten-prompt pilot completed 150 measurements and 30 warmups with exact
greedy parity. Adaptive decode reached 1.427x versus no draft, fixed `k=2`
1.412x; full-process speedups were 1.131x and 1.099x. The first target forward
accounts for most of the gap, so separating first-use cost from warmed prefill
comes before another scheduler change. Process RSS does not capture the full
GPU memory footprint; see `BENCHMARKS.md` before interpreting that field.

## 2026-09-14 — Phase 3 merged; Phase 4 aimed at a gap

`make test-all` re-run: green. Repository published to
github.com/CCristiann/picoforge; Phase 3 merged into main.

Phase 4 is aimed at something that has not been built before. A literature
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

### Step 4.5a — reading a checkpoint the way the 30B ships

Qwen3-30B-A3B is 16 safetensors files and an index. Before it exists here,
every reader is held to the tiny MoE split into three files the Hub's way
(`tools/synth/shard_checkpoint.py`), and `make test-shards` checks them all
against the single-file model:

- the C loader maps every shard the index names, holds the index's tensor
  count to what the files contain, and sorts tensors by name so a lookup is a
  binary search (the 30B has ~19k tensors: binding by linear search would be
  1.8e8 string compares) and a name in two shards is caught. CPU logits from
  three shards: identical bits.
- the GPU refuses a sharded checkpoint: one no-copy buffer, capped at 41.75 GB.
- quantize.py reads shards and writes ONE file in name order: identical bytes
  from either input -- the path from the 61 GB download to a ~31 GB GPU file.
- the oracle reads a sharded checkpoint lazily (fp32 would be 122 GB), with a
  byte-bounded LRU cache; with 40 MB of cache on an 80 MB model its logits
  equal the eager load's exactly.

### Step 4.7b — the instrument for the one number still missing

What a verify step costs is known as a function of distinct experts; how many
distinct experts real consecutive tokens touch is a property of the trained
router. The instrument for it is built and verified, so the download can go
straight to the measurement:

- `--routing-trace TEXT MAX_TOKENS OUT.bin` feeds a text through a MoE model on
  the GPU, 32 tokens a pass, and records every token's top-k experts in every
  layer (a copy kernel after moe_route; routing never leaves the pass).
- `make test-routing-trace` holds the trace to transformers on the tiny MoE,
  router logits top-k'd in PyTorch: 46/46 tokens identical in both MoE layers,
  in probability order; the dense layer marked dense.
- `tools/eval/routing_overlap.py` reports distinct experts per window of
  w = 1..32 tokens against the uniform-routing expectation, and turns them into
  the estimated MoE cost of a verify step in decode steps, from the committed
  cost model.

With the 30B, the measurement is four commands:

    tools/venv/bin/hf download Qwen/Qwen3-30B-A3B --local-dir models/Qwen3-30B-A3B
    tools/venv/bin/python tools/quant/quantize.py models/Qwen3-30B-A3B q8_row
    ./picoforge models/Qwen3-30B-A3B-q8_row --routing-trace TEXT 4096 build/trace_30b.bin
    tools/venv/bin/python tools/eval/routing_overlap.py build/trace_30b.bin bench/routing_overlap_30b.csv

### A full-depth 30B-shaped model, and the UI

`tools/synth/make_moe_model.py` writes all 48 layers of Qwen3-30B-A3B's shapes
straight in q8_row, random weights: 31.2 GB, one file, inside the 41.75 GB
buffer cap, in 27 s. The GPU matches the CPU on it (2 tokens, relative 4.4e-6,
same argmax) -- the first integration check at the real shapes and depth.
*(Corrected: the first run of this check passed its token ids as one unsplit
zsh word, so the engine saw one token; 4.7e-6 was that 1-token figure. Rerun
with two tokens: 4.4e-6.)*
`--bench-moe-e2e`, `bench/moe_e2e_synth30b_m5pro.csv`, cache depth 512:

  decode       48.5 ms   20.6 tok/s   72 GB/s (23% of 307)   8.0 experts/layer
  verify 8     53.3 ms   1.10 x decode                        9.9 experts/layer
  verify 32    90.4 ms   1.86 x decode                       11.7 experts/layer

Decode is exact for the shapes. The verify rows are NOT the upper bounds the
bench's first comment claimed: random weights route every token to nearly the
same ~10 experts per layer (measured by the trace, printed in the last column),
where a trained router would spread them. The honest reading: at D ~ 10, a
verify of 8 costs 1.1 decodes; what D real text reaches is the missing number.

`make ui` gained two tabs: **Speculative** runs plain greedy and a drafter side
by side on the real engine, shades each loop step, underlines the accepted
drafts and checks the output against plain greedy; **Phase 4** draws the
measured results from the CSVs. Building it found an instrument bug: profile
group names contain commas and the CSV writer did not quote them, so every
reader split one column into three. The committed CSV was re-quoted in place
(the numbers are the original run's) and the writer fixed.

### Step 4.5b — the real checkpoint, and the chain of oracles on it

Qwen3-30B-A3B at revision ad44e77: 16 shards, 61.07 GB, all SHA256 sums equal
to the Hub's (`docs/weights-qwen3-30b-a3b.lock`). Its tokenizer.json is
byte-identical to Qwen3-0.6B's, so the drafter and the target share every id.
The engine binds all 18867 tensors, shape-checked against the config, in
0.46 s (sorted names, binary search).

Loading it exposed two summary lines that had only ever seen tied models: an
untied head was described as absent, and its 311 M parameters were
subtracted as a copy. A copy is now recognised by comparing bytes.

The chain, on the real weights:
- **transformers -> oracle, one layer at a time** (`tests/test_layer_real.py`):
  the whole model is 122 GB as fp32, but a decoder layer is a function of its
  input and its own weights. Layers 0, 24 and 47 in transformers'
  Qwen3MoeDecoderLayer (experts fused as it holds them) against the oracle's
  blocks: relative error 7.8e-7, 3.2e-6, 1.1e-6. Loading layer 25's weights
  into the oracle instead moves it to 0.88.
- **oracle -> C CPU, whole model** (`tests/test_forward.py`, lazy sharded
  oracle): 1, 5, 9 and 40 tokens, batch and incremental, worst relative error
  4.5e-6, KL 3.0e-10, argmax 40/40. "The capital of France is" -> " Paris".
  The scalar CPU runs it at 0.6-0.7 tok/s.

### Step 4.7c — Qwen3-30B-A3B on the GPU, and how its router really routes

`tools/quant/quantize.py` turned the 16 bf16 shards into one q8_row file:
31.2 GB, 37491 tensors, inside the 41.75 GB buffer cap. The GPU pass on it is
held to the CPU pass on the same file (`tests/test_gpu_vs_cpu.py`): 1, 5, 9
and 40 tokens, batch and incremental, worst relative error 1.7e-5, KL 1.9e-9,
argmax 40/40. "The capital of France is" -> " Paris" at 74.7%.

`--bench-moe-e2e`, `bench/moe_e2e_qwen3_30b_q8_m5pro.csv`, cache depth 512:

  decode       47.7 ms   21.0 tok/s   73 GB/s (24% of 307)    8.0 experts/layer
  verify 4     58.6 ms   1.23 x decode                        17.7
  verify 8     68.0 ms   1.43 x decode                        24.0
  verify 32   116.4 ms   2.44 x decode                        43.8

`--routing-trace` over the first 4096 tokens of `build/corpus_local.txt`, then
`tools/eval/routing_overlap.py`, `bench/routing_overlap_qwen3_30b_corpus.csv`:
**Superseded: see the October 11 correction above.**
distinct experts per MoE layer touched by w consecutive tokens, against
w independent uniformly routed tokens.

  w          1     2     4     8    16    32
  measured  8.0  10.8  14.9  20.3  26.7  33.6
  uniform   8.0  15.5  29.1  51.6  82.4 111.8
  ratio    1.00  0.70  0.51  0.39  0.32  0.30

**The trained router has strong temporal locality:** eight consecutive tokens
share their experts so much that they touch 39% of what uniform routing would.
With the measured cost of 22 us per distinct q8 expert, that is what keeps a
verify of 8 tokens at 1.43 decode steps instead of ~2.5.

(The trace first died in the tokenizer: the encoder will not truncate a text
longer than its buffer, so the whole corpus is now encoded and cut after.)

### Step 4.8c — speculative decoding on the real 30B

`tools/eval/bench_speculate.py`, `bench/spec_qwen3_30b_m5pro.csv`: Qwen3-30B-A3B
q8_row on the GPU, drafted by Qwen3-0.6B bf16, five chat prompts, 128 tokens,
median of 3 runs per configuration. All 105 runs emit exactly plain greedy's
tokens.

  prompt    plain      best fixed draft             lookup-4   draft 8
  rewrite   23.6 tok/s 37.2  (k=3, 1.58x, 78% acc)  1.11x      1.41x
  code      23.7       36.4  (k=6, 1.54x, 64%)      1.11x      1.43x
  list      23.5       35.3  (k=4, 1.50x, 73%)      0.89x      1.14x
  explain   23.4       32.9  (k=2, 1.41x, 75%)      0.91x      1.01x
  story     23.6       26.0  (k=2, 1.10x, 43%)      0.92x      0.58x

- **It works, and it is not free.** 1.4-1.6x on text the 0.6B can predict;
  1.1x on a story it cannot.
- **A fixed draft length is a trap.** The best k runs from 2 to 6 across five
  prompts, and the wrong one turns the story into a 0.58x slowdown: eight
  mostly rejected drafts cost 4.6 s of drafting against 3.8 s of verifying.
- **On a MoE, wrong drafts cost real time.** Prompt lookup, free on the dense
  0.6B, loses 8-11% here when it finds nothing, because a 5-token verify
  costs ~1.2 decode steps where on the dense model it cost ~1.0.

That is the case for choosing the draft length step by step from what the
machine and the text are doing -- expected accepted tokens against the
measured cost of drafting and of verifying n tokens.

### Step 4.8d — choosing the draft length from the machine and the text

`DRAFT=auto` (src/speculate.c): before every pass, the scheduler picks the
k that maximises expected tokens per second, from three running estimates --
draft acceptance (forgotten counts over a Beta(1,1) prior), drafter seconds
per token, and target seconds per pass size -- with k = 0 as plain decoding
and a one-draft probe after eight plain steps. Nothing is trained; on a MoE
the verify cost already contains the experts the text's routing touches,
because it is timed.

The first version lost to every fixed length (0.98x on the list prompt,
35 of 42 passes at k = 0). It charged the drafter's prompt prefill, and each
catch-up pass after plain steps, to the draft tokens that followed: drafting
looked like 17 ms a token instead of 9.5, so it drafted less, fell further
behind, and looked dearer still. The drafter now reads the prompt before the
loop and its cost is timed on ordinary passes. Moving that prefill first wrote
the drafter's logits over the target's and changed the first token -- 61
tokens where greedy emits 47 -- caught by comparing against greedy before any
number was taken, and fixed.

Same benchmark as 4.8c, re-run in full, `bench/spec_qwen3_30b_m5pro.csv`,
speedup over plain decoding, all 135 runs identical to greedy:

                 explain  code  list  rewrite  story   geomean  worst
  draft 2          1.48   1.54  1.40   1.55    1.13     1.41    1.13
  draft 3          1.29   1.51  1.48   1.64    0.97     1.36    0.97
  draft 4          1.28   1.54  1.70   1.53    0.86     1.35    0.86
  draft 8          1.06   1.50  1.25   1.48    0.58     1.11    0.58
  auto             1.43   1.49  1.57   1.67    1.11     1.44    1.11

**Without being told anything about the text, the scheduler beats every fixed
draft length on the geometric mean and keeps the worst case at 1.11x**, within
93-101% of the best length chosen after the fact for each prompt. Prompt
lookup, costly on a MoE when it finds nothing, goes from 0.98x to 1.03x under
the same scheduler.

**Next:** expert-aware drafting proper -- predicting the experts a draft will
touch from the routing of the tokens just accepted, and pricing drafts by their
predicted union rather than by the average verify cost; and the decode path
itself, where the 30B spends 47.7 ms a token at 24% of the machine's bandwidth.

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

**Next:** published perplexity numbers still need a comparable corpus
(WikiText-2, 733 KB test split) and a larger post-cutoff one. Phase 4 needs
Qwen3-30B-A3B: 16 bf16 shards, 61.07 GB.

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
