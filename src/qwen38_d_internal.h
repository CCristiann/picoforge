/* Private D test/core boundary. Not a second public attention ABI. */
#ifndef PICOFORGE_QWEN38_D_INTERNAL_H
#define PICOFORGE_QWEN38_D_INTERNAL_H
/* R2.8: threads per block of the fast index scores (one per head, h <= 8). */
#define Q38D_SCORE_LANES 8u
#ifndef __METAL_VERSION__
#include "qwen38_attention.h"
#include "qwen38_gpu.h"

/* All arrays below contiguous and disjoint. Caller owns capacities/lifetime.
 * Failure after writes makes output unusable. No normalization here: C owns it. */
Qwen38Status q38d_rope(float *, uint32_t heads, uint32_t dim,
    uint32_t rotary_dim, uint32_t pos, float theta);
/* Immutable default-RoPE constant, shared by CPU and Metal parameter setup.
 * Requires rotary_dim>0, i<rotary_dim/2 and finite positive theta. */
float q38d_rope_frequency(float theta, uint32_t i, uint32_t rotary_dim);
Qwen38Status q38d_scores(const float *iq, const float *blocks,
    uint32_t heads, uint32_t dim, uint32_t count, float *scores);
/* Mean of exactly one complete raw-key block; RMS/rotation follow later. */
Qwen38Status q38d_pool(const float *raw, uint32_t ratio, uint32_t dim, float *pooled);
/* Heap scratch and selected blocks each need min(count,budget) uint32 slots.
 * Equal FP32 scores choose smaller IDs. Output is chronological block IDs. */
Qwen38Status q38d_select(const float *scores, uint32_t count, uint32_t budget,
    uint32_t *heap, uint32_t *selected);
/* One query, prepared Q[H,D], gate factors[H,D], K/V[visible,J,D].
 * Selected token IDs strictly increasing; scratch needs D+16 floats. */
Qwen38Status q38d_attend(const Qwen38AttentionConfig *, const float *q,
    const float *gate_factor, const float *k, const float *v, uint32_t visible,
    const uint32_t *ids, uint32_t count, float *out, float *scratch);
/* Test/core boundary after C normalization/activation and D RoPE. blocks
 * contains ONLY newly completed normalized/rotated blocks, in time order.
 * ik is raw and persists across appends. No approximation of missing C2. */
typedef struct {
    Qwen38F32 q, gate_factor, k, v, iq, ik, blocks;
} Q38DPrepared;
Qwen38Status q38d_append_prepared(Qwen38AttentionTicket *, const Q38DPrepared *,
    Qwen38F32Mut out, Qwen38AttentionWorkspace *, Qwen38Error *);
/* Read-only canonical cache for the bounded test harness; caller must not
 * retain it beyond state lifetime. Published lengths come from state_info. */
const float *q38d_cache_keys(const Qwen38AttentionState *);
const float *q38d_cache_raw_index(const Qwen38AttentionState *);
const float *q38d_cache_blocks(const Qwen38AttentionState *);
/* Valid after a nonempty successful public cpu call, until next computation. */
void q38d_last_query(const Qwen38AttentionWorkspace *, const float **q,
    const float **gate, const float **iq, const float **scores, const uint32_t **ids);
/* Private contiguous stage API, no session state or submission. Caller owns
 * resources until completion and checks command status + error_word. Dense
 * FP32 spans; last slot is output, all spans disjoint, offset allowed.
 * ROPE x,out; POOL raw[N,D],out[D]; SCORES iq[H,D],blocks[N,D],out[N];
 * ATTEND q[H,D],gate[H,D],k/v[N,J,D],ids[M] uint32,out[H,D].
 * Attention sees only the explicit visible prefix N and sorted selected IDs.
 * Lifecycle belongs to the caller; no GPU-state publication API is implied. */
typedef struct Q38DGpu Q38DGpu;
typedef uint32_t Q38DUint;
#else
typedef uint Q38DUint;
#endif
enum { Q38D_ROPE, Q38D_POOL, Q38D_SCORES, Q38D_ATTEND, Q38D_GPU_OPS };
enum { Q38D_RANK = Q38D_GPU_OPS, Q38D_COMPACT, Q38D_TAIL, Q38D_COPY, Q38D_SPLIT, Q38D_MERGE,
       Q38D_SCORES_FAST, Q38D_RANK_FAST, Q38D_COMPACT_FAST, Q38D_ROPE_FAST, Q38D_ALL_OPS };
typedef struct { Q38DUint h,j,d,n,m,r,pos; float theta; } Q38DGpuArgs;
#ifndef __METAL_VERSION__
_Static_assert(sizeof(Q38DGpuArgs)==32, "D Metal wire ABI");
Qwen38Status q38d_gpu_create(void *device, void *library, Q38DGpu **, Qwen38Error *);
void q38d_gpu_destroy(Q38DGpu *);
Qwen38Status q38d_gpu_encode(Q38DGpu *, Qwen38GpuCommand, uint32_t op,
    const Q38DGpuArgs *, const Qwen38GpuSpan *, size_t count, Qwen38Error *);
/* R2.1 split attention: at most 64 splits of at least 32 keys, no empty split.
 * Depends on the key count m only, so the summation order does too. */
#define Q38D_MAX_SPLITS 64u
static inline __attribute__((unused)) void q38d_attend_splits(uint32_t m, uint32_t *splits, uint32_t *keys) {
    uint32_t s=(m+31)/32; s=s<1 ? 1 : (s>Q38D_MAX_SPLITS ? Q38D_MAX_SPLITS : s);
    *keys=(m+s-1)/s; *splits=(m+*keys-1)/ *keys;
}
/* Q38D_ATTEND with the fast split kernel (+ merge when splits > 1) unless the
 * oracle is selected (Q38_REF_KERNELS family "attention", Q38_QSA_SCALAR or
 * q38d_gpu_set_reference), the shape is outside it (d % 32, d > 256, more than
 * 16 query heads per KV head) or `part` is smaller than splits*h*(d+2) floats.
 * v = q, gate, k, v, ids, out as for q38d_gpu_encode. */
Qwen38Status q38d_attend_encode(Q38DGpu *, Qwen38GpuCommand, const Q38DGpuArgs *,
    const Qwen38GpuSpan v[6], Qwen38GpuSpan part, Qwen38Error *);
void q38d_gpu_set_reference(Q38DGpu *, bool reference);
/* Select complete blocks and append the causal tail, entirely on device.
 * scores/keep need visible/ratio words; ids needs min(blocks,budget/ratio)*ratio
 * + visible%ratio words. Exact FP32 ties prefer smaller block IDs. All spans
 * disjoint, error word preserved; no host readback or resource allocation. */
Qwen38Status q38d_select_encode(Q38DGpu *, Qwen38GpuCommand,
    uint32_t visible, uint32_t ratio, uint32_t budget, Qwen38GpuSpan scores,
    Qwen38GpuSpan keep, Qwen38GpuSpan ids, Qwen38Error *);
/* G2 layer composition on device, no ID readback: C norms/sigmoid in batch,
 * then per token D RoPE -> cache, block pool+norm+RoPE, scores, device
 * selection and attention. Projections and Wout stay with the caller (B).
 * Cache rows [0, pos0) must hold earlier appends; rows pos0..pos0+T-1 and
 * completed blocks are written here and become history only if the whole
 * command succeeds (the caller's session decides). Views are dense, with
 * disjoint writable spans. On encode failure discard the command; after a
 * completion error invalidate all caches and reset/replay through G0. */
typedef struct {
    Qwen38GpuF32 qg, k, v, iq, ik;                /* [T,H,2,D] [T,J,D] [T,J,D] [T,Hi,Di] [T,Di] */
    Qwen38GpuF32 gamma_q, gamma_k, gamma_iq, gamma_ik;  /* [D] [D] [Di] [Di] */
    Qwen38GpuF32 key_cache, value_cache, index_cache, block_cache; /* [cap,J,D] x2, [cap,Di], [cap/r,Di] */
    Qwen38GpuF32 qn, qr, gate, kn, iqn, iqr, pooled, pooled_n, scores; /* scratch */
    Qwen38GpuSpan keep, ids;                      /* uint32 [cap/r], [budget+r-1] */
    Qwen38GpuF32 out;                             /* [T,H,D] gated attention */
    Qwen38GpuSpan part;   /* R2.1 split partials, Q38D_MAX_SPLITS*H*(D+2) floats; empty: oracle attention */
} Q38DGpuLayer;
typedef struct Q38CGpu Q38CGpu;
Qwen38Status q38d_layer_encode(Q38DGpu *, Q38CGpu *, Qwen38GpuCommand,
    const Qwen38AttentionConfig *, uint32_t pos0, uint32_t tokens,
    const Q38DGpuLayer *, Qwen38Error *);
#endif
#endif
