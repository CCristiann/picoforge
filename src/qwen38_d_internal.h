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
/* Heap scratch and selected blocks each need min(count,budget) uint32 slots.
 * Equal FP32 scores choose smaller IDs. Output is chronological block IDs. */
Qwen38Status q38d_select(const float *scores, uint32_t count, uint32_t budget,
    uint32_t *heap, uint32_t *selected);
#endif
