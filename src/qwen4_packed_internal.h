/* Private helpers of the packed (quantized) layer; the public ABI remains in qwen4_packed.h. */
#ifndef PICOFORGE_QWEN4_PACKED_INTERNAL_H
#define PICOFORGE_QWEN4_PACKED_INTERNAL_H
#include "qwen4_packed.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define QW4_B_INLINE static inline __attribute__((unused))
_Static_assert(sizeof(float) == 4, "packed activations require binary32 storage");

/* Private wire record, mirrored explicitly in qwen4_packed.metal. */
typedef struct {
    uint32_t type, k_storage, k_logical, n_out, rows, reserved;
    uint64_t row_bytes, weight_offset, weight_end, x_stride, y_stride, x_bytes, y_bytes;
} QW4PackedArgs;
_Static_assert(sizeof(QW4PackedArgs) == 80 && offsetof(QW4PackedArgs,row_bytes) == 24 &&
               offsetof(QW4PackedArgs,y_bytes) == 72, "packed Metal argument layout");

typedef struct {
    QW4PackedArgs base;            /* weight_offset = tensor start, weight_end = tensor end */
    uint64_t expert_bytes;
    uint32_t experts, x_divisor;
} QW4PackedIndexedArgs;
_Static_assert(sizeof(QW4PackedIndexedArgs) == 96 && offsetof(QW4PackedIndexedArgs,expert_bytes) == 80,
               "indexed Metal argument layout");

QW4_B_INLINE Qwen4Status qw4_b_result(Qwen4Error *err, Qwen4Status code,
                                       const char *detail) {
    if (err) {
        memset(err, 0, sizeof *err);
        err->code = code;
        if (code != QW4_OK) {
            snprintf(err->subject, sizeof err->subject, "packed weights");
            snprintf(err->detail, sizeof err->detail, "%s", detail);
        }
    }
    return code;
}
QW4_B_INLINE bool qw4_b_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}
QW4_B_INLINE bool qw4_b_address(const void *p, size_t bytes) {
    return p && bytes <= UINTPTR_MAX - (uintptr_t)p;
}
QW4_B_INLINE bool qw4_b_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t)a < (uintptr_t)b + nb &&
                      (uintptr_t)b < (uintptr_t)a + na;
}
QW4_B_INLINE bool qw4_b_same_layout(const QW4PackedLayout *a, const QW4PackedLayout *b) {
    return a->gguf_type == b->gguf_type && a->k_storage == b->k_storage &&
        a->k_logical == b->k_logical && a->n_out == b->n_out &&
        a->n_experts == b->n_experts && a->row_bytes == b->row_bytes &&
        a->expert_bytes == b->expert_bytes && a->tensor_bytes == b->tensor_bytes;
}
/* Nonfatal admission before the existing registry's fatal unknown-ID path. */
QW4_B_INLINE Qwen4Status qw4_b_format(uint32_t type, const GgufFormat **format) {
    if (type == 0 || type == 1 || type == 8 || type == 10 || type == 12 || type == 16 || type == 30 || type == 39) {
        *format = gguf_format(type); return QW4_OK;
    }
    return QW4_FORMAT;
}
/* Common rank-2 activation contract, usable by both CPU and GPU admission.
 * Returns only the reachable extent, not the caller's entire allocation. */
QW4_B_INLINE Qwen4Status qw4_b_matrix(size_t rank, const size_t dim[4],
        const size_t stride[4], size_t capacity, size_t *extent) {
    *extent = 0;
    if (rank != 2 || !dim[1]) return QW4_SHAPE;
    size_t width;
    if (!qw4_b_mul(dim[1], sizeof(float), &width)) return QW4_OVERFLOW;
    if (stride[1] != sizeof(float) || stride[0] % sizeof(float) || stride[0] < width)
        return QW4_STRIDE;
    if (dim[0]) {
        size_t prefix;
        if (!qw4_b_mul(dim[0] - 1, stride[0], &prefix) || prefix > SIZE_MAX - width)
            return QW4_OVERFLOW;
        *extent = prefix + width;
    }
    return *extent > capacity ? QW4_CAPACITY : QW4_OK;
}
/* R1 kernel choice (qwen4_ref.h): override the flags read at creation. */
void qw4_packed_gpu_set_reference(QW4PackedGpu *, bool linear, bool indexed);
/* Same contract and admission as qw4_packed_encode, always the fast kernel:
 * QW4_UNSUPPORTED if the format has no fast kernel yet. */
Qwen4Status qw4_packed_encode_fast(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const QW4PackedLayout *,
    size_t expert, Qwen4GpuF32 x, Qwen4GpuF32 y, Qwen4Error *);
/* Same contract and admission as qw4_packed_encode_indexed, always the
 * indexed fast kernel (R1.5): QW4_UNSUPPORTED if the format has none. */
Qwen4Status qw4_packed_encode_indexed_fast(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const QW4PackedLayout *,
    Qwen4GpuSpan expert_ids, size_t x_divisor, Qwen4GpuF32 x, Qwen4GpuF32 y, Qwen4Error *);
/* R1.5b fused gate+up+SwiGLU of the routed experts (qwen4_moe_fused.metal):
 * act[p, n] = silu(gate_e[n] . x) * (up_e[n] . x), e = ids[p], x row
 * p / x_divisor, bit-identical to the unfused fast sequence (indexed fast gate
 * and up, qw4c_activate SILU, qw4f_product). gate and up are each admitted as
 * qw4_packed_encode_indexed with act as their output. QW4_UNSUPPORTED, nothing
 * encoded: gate and up formats differ, the format has no fused kernel, or the
 * indexed oracle is selected (its sequence is not the fused kernel's): the
 * caller encodes the unfused sequence. No barrier after the dispatch. */
Qwen4Status qw4_packed_encode_gate_up(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *gate, const QW4PackedLayout *gate_layout,
    const GgufBinding *up, const QW4PackedLayout *up_layout, Qwen4GpuSpan expert_ids,
    size_t x_divisor, Qwen4GpuF32 x, Qwen4GpuF32 act, Qwen4Error *);
/* R1.5c fused routed down + weighted reduce (qwen4_moe_fused.metal), the
 * graph's IDX down (divisor 1) then qw4f_reduce (identity map) in one pass:
 * routed[t, n] = sum over s = 0..k-1 of scores[t, s] (down_e[n] . act[t*k+s]),
 * e = ids[t*k+s], and y[t, n] = routed[t, n] + sigmoid[t] shared[t, n], both
 * bit-identical to that fast sequence; the [pairs, n] down scratch is never
 * written. act [t*k, k_logical] and y [t, n_out] are admitted as
 * qw4_packed_encode's x and y (pitches allowed); scores [t, k], shared and
 * routed [t, n_out] and sigmoid [t, 1] must be dense; routed and y alias no
 * other operand. QW4_UNSUPPORTED, nothing encoded: the format has no fused
 * kernel or the indexed oracle is selected: the caller encodes the unfused
 * sequence. No barrier after the dispatch. */
/* Decode III: hc_residual [t, h, n] and hc_inject [t, h] (buffer NULL: none) ask the
 * parallel-slot kernel to add y into the residual (HC_ADD, qw4c_hc_add_fast's
 * expression); *hc_added (when non-NULL) tells whether it did. */
typedef struct { Qwen4GpuF32 scores, shared, sigmoid, routed, hc_residual, hc_inject; bool *hc_added; } QW4DownReduceViews;
typedef struct { QW4PackedIndexedArgs down; uint32_t tokens, slots; } QW4DownReduceArgs;
_Static_assert(sizeof(QW4DownReduceArgs) == 104, "down+reduce Metal argument layout"); /* DownReduceArgs */
Qwen4Status qw4_packed_encode_down_reduce(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const QW4PackedLayout *down_layout,
    Qwen4GpuSpan expert_ids, Qwen4GpuF32 act, const QW4DownReduceViews *, Qwen4GpuF32 y, Qwen4Error *);
/* R2.6 fused HC read tail (qwen4_matvec.metal): HC_LOW (silu(low/h)), the
 * F16 up projection, HC_READ and HC_INJECT (2 sigmoid(il/h)) of one HC site in
 * one dispatch, act, gate, mixed and inject bit-identical to that fast
 * sequence (qw4c_gpu_encode ops and qw4_packed_encode_fast). up is admitted as
 * qw4_packed_encode with x = act [t, K] and y = gate [t, h*E]; low [t, K],
 * normed [t, h*E], mixed [t, E], il and inject [t, h] are rank 2 with unit
 * column strides; an empty il (no buffer) skips the inject. No output aliases
 * an operand. QW4_UNSUPPORTED, nothing encoded: not F16, K split over several
 * simdgroups or above 1024, h above 4, or the linear or fuse oracle selected:
 * the caller encodes the sequence. No barrier after the dispatch. */
typedef struct { QW4PackedArgs up; uint32_t streams, width, inject, reserved; uint64_t low, normed, mixed, il, out; } QW4HcTailArgs;
_Static_assert(sizeof(QW4HcTailArgs) == 136, "HC tail Metal argument layout"); /* HcTailArgs */
Qwen4Status qw4_packed_encode_hc_tail(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *up, const QW4PackedLayout *up_layout, size_t streams,
    Qwen4GpuF32 low, Qwen4GpuF32 normed, Qwen4GpuF32 il, Qwen4GpuF32 act, Qwen4GpuF32 gate,
    Qwen4GpuF32 mixed, Qwen4GpuF32 inject, Qwen4Error *);
/* Decode III: the HC read's RMSNorm folded into its F16 down projection (and the
 * optional inject, n_out rows each, K = streams * width): low = down . N,
 * il = inject . N and normed = N, N = rmsnorm(r) * gain per stream (eps), from
 * r [t, K] in one dispatch instead of RMS + multi-GEMV. gain is [width] or
 * [streams, width]. The sums are reordered (tolerance in the fast-kernel tests).
 * QW4_UNSUPPORTED, nothing encoded: not F16, GEMM rows, streams above 4, width
 * not a multiple of 8, or the exact, linear or fuse family selected: the caller
 * encodes RMS + the multi-GEMV. No barrier after the dispatch. */
typedef struct { QW4PackedArgs down, inject; uint32_t streams, width, n_inject, gain_rank; float eps; uint32_t pad[3]; } QW4HcDownArgs;
_Static_assert(sizeof(QW4HcDownArgs) == 192, "HC down Metal argument layout"); /* HcDownArgs */
/* Decode III: qw4_packed_encode of a Q8_0 linear (y [t, n_out]) followed by HC_ADD
 * into residual [t, h, n_out] with inject [t, h], in one dispatch (bit-identical
 * to the linear then qw4c_hc_add_fast). QW4_UNSUPPORTED, nothing encoded: not
 * Q8_0, GEMM rows, the exact, linear or fuse family, or views not dense. */
Qwen4Status qw4_packed_encode_add(QW4PackedGpu *, const Qwen4GpuCommand *, const GpuWeightStore *,
    const GgufBinding *, const QW4PackedLayout *, Qwen4GpuF32 x, Qwen4GpuF32 y, Qwen4GpuF32 residual,
    Qwen4GpuF32 inject, Qwen4Error *);
bool qw4_packed_add_ready(const QW4PackedGpu *, uint32_t gguf_type, size_t rows, size_t n_out); /* selection only */
bool qw4_packed_hc_down_rms_ready(const QW4PackedGpu *, uint32_t gguf_type, size_t rows, size_t k, size_t streams); /* selection only */
Qwen4Status qw4_packed_encode_hc_down_rms(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const QW4PackedLayout *down_layout,
    const GgufBinding *inject, const QW4PackedLayout *inject_layout, size_t streams, float eps,
    Qwen4GpuF32 r, Qwen4GpuF32 gain, Qwen4GpuF32 normed, Qwen4GpuF32 low, Qwen4GpuF32 il, Qwen4Error *);
/* R3.1: true when a linear of this GGUF type over this many rows of x runs the
 * prefill GEMM (then the HC tail answers UNSUPPORTED and the caller, which
 * opens profile groups before encoding, takes the sequence directly). */
bool qw4_packed_gemm_rows(const QW4PackedGpu *, uint32_t gguf_type, size_t rows);
/* R3.2 grouped MoE for prefill: the (token, slot) pairs grouped by expert in
 * `group` (qw4_packed_group_bytes(experts, pairs) bytes, caller scratch), then
 * one GEMM per expert. Each call groups the ids again. gate_up writes act as
 * qw4_packed_encode_gate_up (x row = pair / divisor); down writes y[pair] as qw4_packed_encode_indexed with divisor 1. QW4_UNSUPPORTED,
 * nothing encoded: below 256 pairs, a format without a grouped kernel, or
 * `indexed` / `gemm` selected. Not bit-identical to the per-pair kernels
 * (order of the sums); rows are bit-independent of their place. */
bool qw4_packed_grouped_rows(const QW4PackedGpu *, uint32_t gguf_type, size_t pairs);
size_t qw4_packed_group_bytes(size_t experts, size_t pairs);
Qwen4Status qw4_packed_encode_grouped_gate_up(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *gate, const QW4PackedLayout *, const GgufBinding *up,
    const QW4PackedLayout *, Qwen4GpuSpan ids, size_t divisor, Qwen4GpuF32 x, Qwen4GpuF32 act,
    Qwen4GpuSpan group, Qwen4Error *);
Qwen4Status qw4_packed_encode_grouped_down(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const QW4PackedLayout *, Qwen4GpuSpan ids,
    Qwen4GpuF32 act, Qwen4GpuF32 y, Qwen4GpuSpan group, Qwen4Error *);
/* R1.4 multi-GEMV: count (1..QW4_MULTI_MAX) projections of the same x, each
 * with qw4_packed_encode's admission (expert 0). One dispatch per format;
 * every y[i] is bit-identical to qw4_packed_encode_fast(bindings[i], x, y[i]).
 * Under QW4_REF_KERNELS=linear, and for a format with no fast kernel, it
 * encodes qw4_packed_encode per projection. No barrier between or after the
 * dispatches: the caller adds one after the call. */
enum { QW4_MULTI_MAX = 8 };
Qwen4Status qw4_packed_encode_multi(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *const *bindings,
    const QW4PackedLayout *layouts, size_t count, Qwen4GpuF32 x,
    const Qwen4GpuF32 *y, Qwen4Error *);
#undef QW4_B_INLINE
#endif
