/* Contract v1: declarations only. */
#ifndef PICOFORGE_QWEN4_TYPES_H
#define PICOFORGE_QWEN4_TYPES_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QWEN4_CONTRACT_VERSION 1u
#define QWEN4_MAX_RANK 4u
/* The longest context a graph can hold (the model allows 262144). Kernel indices
 * stay within 32 bits up to here; memory, about 57 KB of state per token on
 * Qwen3.8-Flash-Next, is the practical limit. */
#define QWEN4_MAX_CONTEXT 131072u
typedef enum {
    QW4_OK = 0, QW4_INVALID_ARGUMENT, QW4_MISSING, QW4_TYPE, QW4_RANGE,
    QW4_SHAPE, QW4_STRIDE, QW4_FORMAT, QW4_UNSUPPORTED, QW4_OVERFLOW,
    QW4_CAPACITY, QW4_OOM, QW4_OWNER, QW4_BUSY, QW4_STALE, QW4_BAD_STATE,
    QW4_NONFINITE, QW4_IO, QW4_EOF, QW4_CANCELLED, QW4_GPU_ERROR
} Qwen4Status;
typedef struct {
    Qwen4Status code;
    int os_errno;
    uint64_t file_offset, row;
    size_t requested_bytes, completed_bytes;
    char subject[128], detail[256];
} Qwen4Error;
typedef struct { const uint8_t *data; size_t len; } Qwen4Bytes;
typedef struct { const unsigned char *data; size_t bytes; } Qwen4CpuSpan;
typedef struct { void *data; size_t bytes; } Qwen4Memory;

/* All strides/capacities below are BYTES, dimensions are element counts.
 * Last axis contiguous; outer axes positive, non-overlapping. No broadcast. */
typedef struct {
    const float *data;
    size_t capacity_bytes, rank, dim[QWEN4_MAX_RANK];
    size_t stride_bytes[QWEN4_MAX_RANK];
} Qwen4F32;
typedef struct {
    float *data;
    size_t capacity_bytes, rank, dim[QWEN4_MAX_RANK];
    size_t stride_bytes[QWEN4_MAX_RANK];
} Qwen4F32Mut;
typedef struct {
    uint32_t *data;
    size_t capacity_bytes, rows, cols, row_stride_bytes;
} Qwen4U32Mut;
typedef struct {
    const void *model;             /* stable identity token, never dereferenced */
    uint32_t layer;
    uint64_t epoch;
} Qwen4StateKey;
typedef struct {
    Qwen4StateKey key;
    uint64_t next_pos;
    bool valid, busy;
} Qwen4StateInfo;
typedef struct {
    size_t metadata_bytes, payload_bytes, alignment;
} Qwen4StateSize;

typedef struct { size_t channels, kernel, dilation; bool apply_silu; } Qwen4ConvShape;
typedef struct {
    size_t key_heads, value_heads, key_dim, value_dim;
    float rms_eps, l2_eps;
    /* Output gate of the gated norm: false = sigmoid (Flash-Next sets it), true =
     * SiLU (upstream's default when output_gate_type is unset). */
    bool silu_gate;
} Qwen4GdnShape;
typedef struct { size_t streams, hidden_size, low_rank; float rms_eps; } Qwen4HcShape;
typedef struct {
    uint32_t heads, kv_heads, head_dim, index_heads, index_kv_heads, index_dim;
    uint32_t ratio, token_budget, rotary_dim;
    float rms_eps, rope_theta;
} Qwen4AttentionConfig;           /* v1 cache and arithmetic: FP32 only */
typedef struct {
    uint32_t hidden_size, experts, top_k, intermediate, shared_intermediate;
    uint32_t down_physical;
    bool norm_topk_prob;
    uint32_t trunk_type[3];   /* R5.3: GGUF types of the trunk's routed gate, up, down (Q2 or Q4 profile) */
} Qwen4MoeConfig;
typedef struct {
    uint32_t vocab_size, eos_id, layer, order, heads_per_order, head_dim;
    uint32_t hidden_size, streams, conv_kernel;
    float rms_eps;
    uint64_t row_count;
    const uint64_t *multipliers, *head_offsets, *head_vocab_sizes;
    size_t multiplier_count, head_count;
} Qwen4NgramConfig;               /* arrays borrowed from immutable A config */
#endif
