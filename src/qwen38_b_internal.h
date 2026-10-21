/* Private B helpers; the public ABI remains in qwen38_packed.h. */
#ifndef PICOFORGE_QWEN38_B_INTERNAL_H
#define PICOFORGE_QWEN38_B_INTERNAL_H
#include "qwen38_packed.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define Q38_B_INLINE static inline __attribute__((unused))
_Static_assert(sizeof(float) == 4, "packed activations require binary32 storage");

/* Private wire record, mirrored explicitly in qwen38_packed.metal. */
typedef struct {
    uint32_t type, k_storage, k_logical, n_out, rows, reserved;
    uint64_t row_bytes, weight_offset, weight_end, x_stride, y_stride, x_bytes, y_bytes;
} Q38PackedArgs;
_Static_assert(sizeof(Q38PackedArgs) == 80 && offsetof(Q38PackedArgs,row_bytes) == 24 &&
               offsetof(Q38PackedArgs,y_bytes) == 72, "packed Metal argument layout");

typedef struct {
    Q38PackedArgs base;            /* weight_offset = tensor start, weight_end = tensor end */
    uint64_t expert_bytes;
    uint32_t experts, x_divisor;
} Q38PackedIndexedArgs;
_Static_assert(sizeof(Q38PackedIndexedArgs) == 96 && offsetof(Q38PackedIndexedArgs,expert_bytes) == 80,
               "indexed Metal argument layout");

Q38_B_INLINE Qwen38Status q38_b_result(Qwen38Error *err, Qwen38Status code,
                                       const char *detail) {
    if (err) {
        memset(err, 0, sizeof *err);
        err->code = code;
        if (code != Q38_OK) {
            snprintf(err->subject, sizeof err->subject, "packed weights");
            snprintf(err->detail, sizeof err->detail, "%s", detail);
        }
    }
    return code;
}
Q38_B_INLINE bool q38_b_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}
Q38_B_INLINE bool q38_b_address(const void *p, size_t bytes) {
    return p && bytes <= UINTPTR_MAX - (uintptr_t)p;
}
Q38_B_INLINE bool q38_b_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t)a < (uintptr_t)b + nb &&
                      (uintptr_t)b < (uintptr_t)a + na;
}
Q38_B_INLINE bool q38_b_same_layout(const Q38PackedLayout *a, const Q38PackedLayout *b) {
    return a->gguf_type == b->gguf_type && a->k_storage == b->k_storage &&
        a->k_logical == b->k_logical && a->n_out == b->n_out &&
        a->n_experts == b->n_experts && a->row_bytes == b->row_bytes &&
        a->expert_bytes == b->expert_bytes && a->tensor_bytes == b->tensor_bytes;
}
/* Nonfatal admission before the existing registry's fatal unknown-ID path. */
Q38_B_INLINE Qwen38Status q38_b_format(uint32_t type, const GgufFormat **format) {
    if (type == 0 || type == 1 || type == 8 || type == 10 || type == 12 || type == 16 || type == 30 || type == 39) {
        *format = gguf_format(type); return Q38_OK;
    }
    return Q38_FORMAT;
}
/* Common rank-2 activation contract, usable by both CPU and GPU admission.
 * Returns only the reachable extent, not the caller's entire allocation. */
Q38_B_INLINE Qwen38Status q38_b_matrix(size_t rank, const size_t dim[4],
        const size_t stride[4], size_t capacity, size_t *extent) {
    *extent = 0;
    if (rank != 2 || !dim[1]) return Q38_SHAPE;
    size_t width;
    if (!q38_b_mul(dim[1], sizeof(float), &width)) return Q38_OVERFLOW;
    if (stride[1] != sizeof(float) || stride[0] % sizeof(float) || stride[0] < width)
        return Q38_STRIDE;
    if (dim[0]) {
        size_t prefix;
        if (!q38_b_mul(dim[0] - 1, stride[0], &prefix) || prefix > SIZE_MAX - width)
            return Q38_OVERFLOW;
        *extent = prefix + width;
    }
    return *extent > capacity ? Q38_CAPACITY : Q38_OK;
}
/* R1 kernel choice (qwen38_ref.h): override the flags read at creation. */
void q38_packed_gpu_set_reference(Q38PackedGpu *, bool linear, bool indexed);
/* Same contract and admission as q38_packed_encode, always the fast kernel:
 * Q38_UNSUPPORTED if the format has no fast kernel yet. */
Qwen38Status q38_packed_encode_fast(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
/* Same contract and admission as q38_packed_encode_indexed, always the
 * indexed fast kernel (R1.5): Q38_UNSUPPORTED if the format has none. */
Qwen38Status q38_packed_encode_indexed_fast(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    Qwen38GpuSpan expert_ids, size_t x_divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
/* R1.5b fused gate+up+SwiGLU of the routed experts (qwen38_moe_fused.metal):
 * act[p, n] = silu(gate_e[n] . x) * (up_e[n] . x), e = ids[p], x row
 * p / x_divisor, bit-identical to the unfused fast sequence (indexed fast gate
 * and up, q38c_activate SILU, q38f_product). gate and up are each admitted as
 * q38_packed_encode_indexed with act as their output. Q38_UNSUPPORTED, nothing
 * encoded: gate and up formats differ, the format has no fused kernel, or the
 * indexed oracle is selected (its sequence is not the fused kernel's): the
 * caller encodes the unfused sequence. No barrier after the dispatch. */
Qwen38Status q38_packed_encode_gate_up(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *gate, const Q38PackedLayout *gate_layout,
    const GgufBinding *up, const Q38PackedLayout *up_layout, Qwen38GpuSpan expert_ids,
    size_t x_divisor, Qwen38GpuF32 x, Qwen38GpuF32 act, Qwen38Error *);
/* R1.4 multi-GEMV: count (1..Q38_MULTI_MAX) projections of the same x, each
 * with q38_packed_encode's admission (expert 0). One dispatch per format;
 * every y[i] is bit-identical to q38_packed_encode_fast(bindings[i], x, y[i]).
 * Under Q38_REF_KERNELS=linear, and for a format with no fast kernel, it
 * encodes q38_packed_encode per projection. No barrier between or after the
 * dispatches: the caller adds one after the call. */
enum { Q38_MULTI_MAX = 8 };
Qwen38Status q38_packed_encode_multi(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *const *bindings,
    const Q38PackedLayout *layouts, size_t count, Qwen38GpuF32 x,
    const Qwen38GpuF32 *y, Qwen38Error *);
#undef Q38_B_INLINE
#endif
