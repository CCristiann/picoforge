/* Contract v1: declarations only. See docs/specs/qwen38/integration.md. */
#ifndef PICOFORGE_QWEN38_TYPES_H
#define PICOFORGE_QWEN38_TYPES_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QWEN38_CONTRACT_VERSION 1u
#define QWEN38_MAX_RANK 4u
typedef enum {
    Q38_OK = 0, Q38_INVALID_ARGUMENT, Q38_MISSING, Q38_TYPE, Q38_RANGE,
    Q38_SHAPE, Q38_STRIDE, Q38_FORMAT, Q38_UNSUPPORTED, Q38_OVERFLOW,
    Q38_CAPACITY, Q38_OOM, Q38_OWNER, Q38_BUSY, Q38_STALE, Q38_BAD_STATE,
    Q38_NONFINITE, Q38_IO, Q38_EOF, Q38_CANCELLED, Q38_GPU_ERROR
} Qwen38Status;
typedef struct {
    Qwen38Status code;
    int os_errno;
    uint64_t file_offset, row;
    size_t requested_bytes, completed_bytes;
    char subject[128], detail[256];
} Qwen38Error;
typedef struct { const uint8_t *data; size_t len; } Qwen38Bytes;
typedef struct { const unsigned char *data; size_t bytes; } Qwen38CpuSpan;
typedef struct { void *data; size_t bytes; } Qwen38Memory;

/* All strides/capacities below are BYTES, dimensions are element counts.
 * Last axis contiguous; outer axes positive, non-overlapping. No broadcast. */
typedef struct {
    const float *data;
    size_t capacity_bytes, rank, dim[QWEN38_MAX_RANK];
    size_t stride_bytes[QWEN38_MAX_RANK];
} Qwen38F32;
typedef struct {
    float *data;
    size_t capacity_bytes, rank, dim[QWEN38_MAX_RANK];
    size_t stride_bytes[QWEN38_MAX_RANK];
} Qwen38F32Mut;
typedef struct {
    uint32_t *data;
    size_t capacity_bytes, rows, cols, row_stride_bytes;
} Qwen38U32Mut;
typedef struct {
    const void *model;             /* stable identity token, never dereferenced */
    uint32_t layer;
    uint64_t epoch;
} Qwen38StateKey;
typedef struct {
    Qwen38StateKey key;
    uint64_t next_pos;
    bool valid, busy;
} Qwen38StateInfo;
typedef struct {
    size_t metadata_bytes, payload_bytes, alignment;
} Qwen38StateSize;

typedef struct { size_t channels, kernel, dilation; bool apply_silu; } Qwen38ConvShape;
typedef struct {
    size_t key_heads, value_heads, key_dim, value_dim;
    float rms_eps, l2_eps;
} Qwen38GdnShape;
typedef struct { size_t streams, hidden_size, low_rank; float rms_eps; } Qwen38HcShape;
typedef struct {
    uint32_t heads, kv_heads, head_dim, index_heads, index_kv_heads, index_dim;
    uint32_t ratio, token_budget, rotary_dim;
    float rms_eps, rope_theta;
} Qwen38AttentionConfig;           /* v1 cache and arithmetic: FP32 only */
typedef struct {
    uint32_t hidden_size, experts, top_k, intermediate, shared_intermediate;
    uint32_t down_physical;
    bool norm_topk_prob;
} Qwen38MoeConfig;
typedef struct {
    uint32_t vocab_size, eos_id, layer, order, heads_per_order, head_dim;
    uint32_t hidden_size, streams, conv_kernel;
    float rms_eps;
    uint64_t row_count;
    const uint64_t *multipliers, *head_offsets, *head_vocab_sizes;
    size_t multiplier_count, head_count;
} Qwen38NgramConfig;               /* arrays borrowed from immutable A config */
#endif
