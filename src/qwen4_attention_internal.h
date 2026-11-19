/* Private attention test/core boundary. Not a second public attention ABI. */
#ifndef PICOFORGE_QWEN4_ATTENTION_INTERNAL_H
#define PICOFORGE_QWEN4_ATTENTION_INTERNAL_H
/* R2.8: threads per block of the fast index scores (one per head, h <= 8). */
#define QW4D_SCORE_LANES 8u
#ifndef __METAL_VERSION__
#include "qwen4_attention.h"
#include "qwen4_gpu.h"

/* All arrays below contiguous and disjoint. Caller owns capacities/lifetime.
 * Failure after writes makes output unusable. No normalization here: the CPU ops own it. */
Qwen4Status qw4d_rope(float *, uint32_t heads, uint32_t dim,
    uint32_t rotary_dim, uint32_t pos, float theta);
/* Immutable default-RoPE constant, shared by CPU and Metal parameter setup.
 * Requires rotary_dim>0, i<rotary_dim/2 and finite positive theta. */
float qw4d_rope_frequency(float theta, uint32_t i, uint32_t rotary_dim);
Qwen4Status qw4d_scores(const float *iq, const float *blocks,
    uint32_t heads, uint32_t dim, uint32_t count, float *scores);
/* Mean of exactly one complete raw-key block; RMS/rotation follow later. */
Qwen4Status qw4d_pool(const float *raw, uint32_t ratio, uint32_t dim, float *pooled);
/* Heap scratch and selected blocks each need min(count,budget) uint32 slots.
 * Equal FP32 scores choose smaller IDs. Output is chronological block IDs. */
Qwen4Status qw4d_select(const float *scores, uint32_t count, uint32_t budget,
    uint32_t *heap, uint32_t *selected);
/* One query, prepared Q[H,D], gate factors[H,D], K/V[visible,J,D].
 * Selected token IDs strictly increasing; scratch needs D+16 floats. */
Qwen4Status qw4d_attend(const Qwen4AttentionConfig *, const float *q,
    const float *gate_factor, const float *k, const float *v, uint32_t visible,
    const uint32_t *ids, uint32_t count, float *out, float *scratch);
/* Test/core boundary after normalization/activation (CPU ops) and RoPE (above). blocks
 * contains ONLY newly completed normalized/rotated blocks, in time order.
 * ik is raw and persists across appends. No approximation of missing C2. */
typedef struct {
    Qwen4F32 q, gate_factor, k, v, iq, ik, blocks;
} QW4DPrepared;
Qwen4Status qw4d_append_prepared(Qwen4AttentionTicket *, const QW4DPrepared *,
    Qwen4F32Mut out, Qwen4AttentionWorkspace *, Qwen4Error *);
/* Read-only canonical cache for the bounded test harness; caller must not
 * retain it beyond state lifetime. Published lengths come from state_info. */
const float *qw4d_cache_keys(const Qwen4AttentionState *);
const float *qw4d_cache_raw_index(const Qwen4AttentionState *);
const float *qw4d_cache_blocks(const Qwen4AttentionState *);
/* Valid after a nonempty successful public cpu call, until next computation. */
void qw4d_last_query(const Qwen4AttentionWorkspace *, const float **q,
    const float **gate, const float **iq, const float **scores, const uint32_t **ids);
/* Private contiguous stage API, no session state or submission. Caller owns
 * resources until completion and checks command status + error_word. Dense
 * FP32 spans; last slot is output, all spans disjoint, offset allowed.
 * ROPE x,out; POOL raw[N,D],out[D]; SCORES iq[H,D],blocks[N,D],out[N];
 * ATTEND q[H,D],gate[H,D],k/v[N,J,D],ids[M] uint32,out[H,D].
 * Attention sees only the explicit visible prefix N and sorted selected IDs.
 * Lifecycle belongs to the caller; no GPU-state publication API is implied. */
typedef struct QW4DGpu QW4DGpu;
typedef uint32_t QW4DUint;
#else
typedef uint QW4DUint;
#endif
enum { QW4D_ROPE, QW4D_POOL, QW4D_SCORES, QW4D_ATTEND, QW4D_GPU_OPS };
enum { QW4D_RANK = QW4D_GPU_OPS, QW4D_COMPACT, QW4D_TAIL, QW4D_COPY, QW4D_SPLIT, QW4D_MERGE,
       QW4D_SCORES_FAST, QW4D_SELECT_FAST, QW4D_ROPE_FAST,
       QW4D_ROPE_BATCH, QW4D_POOL_BATCH, QW4D_SCORES_BATCH, QW4D_SELECT_BATCH,
       QW4D_TAIL_BATCH, QW4D_SPLIT_BATCH, QW4D_MERGE_BATCH, QW4D_SPLIT_MMA, QW4D_SPLIT_MMA_BATCH, QW4D_MERGE_MMA_BATCH, QW4D_ALL_OPS };
/* P5.1 MMA split attention: head dim it is written for, most keys per split. */
#define QW4D_MMA_D 256u
#define QW4D_MMA_KEYS 40u
#define QW4D_MMA_GROUP 2u /* P5.2: R2.1 splits per MMA threadgroup (4: 8K prefill -0.4 s but decode +1 ms/token) */
/* R3.3 batched prefill: the wire record of the batch kernels (QW4DBatch in the
 * kernel), tokens per split-attention dispatch, and the append size from which
 * qw4d_layer_encode batches. */
typedef struct { QW4DUint pos0, step, x_stride, y_stride, ratio, budget, scores_stride, ids_stride, first, count; } QW4DBatchArgs;
#define QW4D_BATCH_SPLIT_TOKENS 32u
#define QW4D_BATCH_MIN_TOKENS 32u
typedef struct { QW4DUint h,j,d,n,m,r,pos; float theta; } QW4DGpuArgs;
#ifndef __METAL_VERSION__
_Static_assert(sizeof(QW4DGpuArgs)==32, "D Metal wire ABI");
Qwen4Status qw4d_gpu_create(void *device, void *library, QW4DGpu **, Qwen4Error *);
void qw4d_gpu_destroy(QW4DGpu *);
Qwen4Status qw4d_gpu_encode(QW4DGpu *, Qwen4GpuCommand, uint32_t op,
    const QW4DGpuArgs *, const Qwen4GpuSpan *, size_t count, Qwen4Error *);
/* R2.1 split attention: at most 64 splits of at least 32 keys, no empty split.
 * Depends on the key count m only, so the summation order does too. */
#define QW4D_MAX_SPLITS 64u
static inline __attribute__((unused)) void qw4d_attend_splits(uint32_t m, uint32_t *splits, uint32_t *keys) {
    uint32_t s=(m+31)/32; s=s<1 ? 1 : (s>QW4D_MAX_SPLITS ? QW4D_MAX_SPLITS : s);
    *keys=(m+s-1)/s; *splits=(m+*keys-1)/ *keys;
}
/* QW4D_ATTEND with the fast split kernel (+ merge when splits > 1) unless the
 * oracle is selected (QW4_REF_KERNELS family "attention", QW4_QSA_SCALAR or
 * qw4d_gpu_set_reference), the shape is outside it (d % 32, d > 256, more than
 * 16 query heads per KV head) or `part` is smaller than splits*h*(d+2) floats.
 * v = q, gate, k, v, ids, out as for qw4d_gpu_encode. */
Qwen4Status qw4d_attend_encode(QW4DGpu *, Qwen4GpuCommand, const QW4DGpuArgs *,
    const Qwen4GpuSpan v[6], Qwen4GpuSpan part, Qwen4Error *);
void qw4d_gpu_set_reference(QW4DGpu *, bool reference);
/* Select complete blocks and append the causal tail, entirely on device.
 * scores/keep need visible/ratio words; ids needs min(blocks,budget/ratio)*ratio
 * + visible%ratio words. Exact FP32 ties prefer smaller block IDs. All spans
 * disjoint, error word preserved; no host readback or resource allocation. */
Qwen4Status qw4d_select_encode(QW4DGpu *, Qwen4GpuCommand,
    uint32_t visible, uint32_t ratio, uint32_t budget, Qwen4GpuSpan scores,
    Qwen4GpuSpan keep, Qwen4GpuSpan ids, Qwen4Error *);
/* G2 layer composition on device, no ID readback: norms/sigmoid in batch,
 * then per token RoPE -> cache, block pool+norm+RoPE, scores, device
 * selection and attention. Projections and Wout stay with the caller (packed layer).
 * Cache rows [0, pos0) must hold earlier appends; rows pos0..pos0+T-1 and
 * completed blocks are written here and become history only if the whole
 * command succeeds (the caller's session decides). Views are dense, with
 * disjoint writable spans. On encode failure discard the command; after a
 * completion error invalidate all caches and reset/replay through G0. */
typedef struct {
    Qwen4GpuF32 qg, k, v, iq, ik;                /* [T,H,2,D] [T,J,D] [T,J,D] [T,Hi,Di] [T,Di] */
    Qwen4GpuF32 gamma_q, gamma_k, gamma_iq, gamma_ik;  /* [D] [D] [Di] [Di] */
    Qwen4GpuF32 key_cache, value_cache, index_cache, block_cache; /* [cap,J,D] x2, [cap,Di], [cap/r,Di] */
    Qwen4GpuF32 qn, qr, gate, kn, iqn, iqr, pooled, pooled_n, scores; /* scratch */
    Qwen4GpuSpan keep, ids;                      /* uint32 [cap/r], [budget+r-1] */
    Qwen4GpuF32 out;                             /* [T,H,D] gated attention */
    Qwen4GpuSpan part;   /* R2.1 split partials, QW4D_MAX_SPLITS*H*(D+2) floats; empty: oracle attention */
    /* R3.3 batched prefill scratch (all empty: the per-token loop): per token
     * scores and keep [T, cap/r], ids [T, budget+r-1], split partials for
     * QW4D_BATCH_SPLIT_TOKENS tokens, pooled blocks [T/r+1, 1, Di] x2. */
    Qwen4GpuSpan bscores, bkeep, bids, bpart;
    Qwen4GpuF32 bpooled, bpooled_n;
} QW4DGpuLayer;
typedef struct QW4CGpu QW4CGpu;
Qwen4Status qw4d_layer_encode(QW4DGpu *, QW4CGpu *, Qwen4GpuCommand,
    const Qwen4AttentionConfig *, uint32_t pos0, uint32_t tokens,
    const QW4DGpuLayer *, Qwen4Error *);
#endif
#endif
