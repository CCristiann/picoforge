/* Private D test/core boundary. Not a second public attention ABI. */
#ifndef PICOFORGE_QWEN38_D_INTERNAL_H
#define PICOFORGE_QWEN38_D_INTERNAL_H
#include "qwen38_attention.h"

/* All arrays below contiguous and disjoint. Caller owns capacities/lifetime.
 * Failure after writes makes output unusable. No normalization here: C owns it. */
Qwen38Status q38d_rope(float *, uint32_t heads, uint32_t dim,
    uint32_t rotary_dim, uint32_t pos, float theta);
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
#endif
