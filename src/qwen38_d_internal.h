/* Private D test/core boundary. Not a second public attention ABI. */
#ifndef PICOFORGE_QWEN38_D_INTERNAL_H
#define PICOFORGE_QWEN38_D_INTERNAL_H
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
 * Selection/lifecycle remain CPU; no GPU-state publication API is implied. */
typedef struct Q38DGpu Q38DGpu;
typedef uint32_t Q38DUint;
#else
typedef uint Q38DUint;
#endif
enum { Q38D_ROPE, Q38D_POOL, Q38D_SCORES, Q38D_ATTEND, Q38D_GPU_OPS };
typedef struct { Q38DUint h,j,d,n,m,r,pos; float theta; } Q38DGpuArgs;
#ifndef __METAL_VERSION__
_Static_assert(sizeof(Q38DGpuArgs)==32, "D Metal wire ABI");
Qwen38Status q38d_gpu_create(void *device, void *library, Q38DGpu **, Qwen38Error *);
void q38d_gpu_destroy(Q38DGpu *);
Qwen38Status q38d_gpu_encode(Q38DGpu *, Qwen38GpuCommand, uint32_t op,
    const Q38DGpuArgs *, const Qwen38GpuSpan *, size_t count, Qwen38Error *);
#endif
#endif
