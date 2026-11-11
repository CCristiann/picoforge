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
/* Decode III: the same with the router folded in (IQ2_XXS table-decode kernel only):
 * expert_ids, weights and probs are WRITTEN from logits [t, pitch] as
 * q38_moe_route_simd would (t tokens, e experts, k slots, norm). Q38_UNSUPPORTED,
 * nothing encoded, whenever that kernel does not apply; the caller then routes first. */
typedef struct { Qwen38GpuF32 logits, probs, weights; uint32_t t, e, k, norm, pitch; } Q38GateUpRoute;
Qwen38Status q38_packed_encode_gate_up_routed(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *gate, const Q38PackedLayout *gate_layout,
    const GgufBinding *up, const Q38PackedLayout *up_layout, Qwen38GpuSpan expert_ids,
    size_t x_divisor, Qwen38GpuF32 x, Qwen38GpuF32 act, const Q38GateUpRoute *, Qwen38Error *);
/* R1.5c fused routed down + weighted reduce (qwen38_moe_fused.metal), the
 * graph's IDX down (divisor 1) then q38f_reduce (identity map) in one pass:
 * routed[t, n] = sum over s = 0..k-1 of scores[t, s] (down_e[n] . act[t*k+s]),
 * e = ids[t*k+s], and y[t, n] = routed[t, n] + sigmoid[t] shared[t, n], both
 * bit-identical to that fast sequence; the [pairs, n] down scratch is never
 * written. act [t*k, k_logical] and y [t, n_out] are admitted as
 * q38_packed_encode's x and y (pitches allowed); scores [t, k], shared and
 * routed [t, n_out] and sigmoid [t, 1] must be dense; routed and y alias no
 * other operand. Q38_UNSUPPORTED, nothing encoded: the format has no fused
 * kernel or the indexed oracle is selected: the caller encodes the unfused
 * sequence. No barrier after the dispatch. */
/* Decode III: hc_residual [t, h, n] and hc_inject [t, h] (buffer NULL: none) ask the
 * parallel-slot kernel to add y into the residual (HC_ADD, q38c_hc_add_fast's
 * expression); *hc_added (when non-NULL) tells whether it did. */
typedef struct { Qwen38GpuF32 scores, shared, sigmoid, routed, hc_residual, hc_inject; bool *hc_added; } Q38DownReduceViews;
typedef struct { Q38PackedIndexedArgs down; uint32_t tokens, slots; } Q38DownReduceArgs;
_Static_assert(sizeof(Q38DownReduceArgs) == 104, "down+reduce Metal argument layout"); /* DownReduceArgs */
Qwen38Status q38_packed_encode_down_reduce(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const Q38PackedLayout *down_layout,
    Qwen38GpuSpan expert_ids, Qwen38GpuF32 act, const Q38DownReduceViews *, Qwen38GpuF32 y, Qwen38Error *);
/* R2.6 fused HC read tail (qwen38_matvec.metal): HC_LOW (silu(low/h)), the
 * F16 up projection, HC_READ and HC_INJECT (2 sigmoid(il/h)) of one HC site in
 * one dispatch, act, gate, mixed and inject bit-identical to that fast
 * sequence (q38c_gpu_encode ops and q38_packed_encode_fast). up is admitted as
 * q38_packed_encode with x = act [t, K] and y = gate [t, h*E]; low [t, K],
 * normed [t, h*E], mixed [t, E], il and inject [t, h] are rank 2 with unit
 * column strides; an empty il (no buffer) skips the inject. No output aliases
 * an operand. Q38_UNSUPPORTED, nothing encoded: not F16, K split over several
 * simdgroups or above 1024, h above 4, or the linear or fuse oracle selected:
 * the caller encodes the sequence. No barrier after the dispatch. */
typedef struct { Q38PackedArgs up; uint32_t streams, width, inject, reserved; uint64_t low, normed, mixed, il, out; } Q38HcTailArgs;
_Static_assert(sizeof(Q38HcTailArgs) == 136, "HC tail Metal argument layout"); /* HcTailArgs */
Qwen38Status q38_packed_encode_hc_tail(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *up, const Q38PackedLayout *up_layout, size_t streams,
    Qwen38GpuF32 low, Qwen38GpuF32 normed, Qwen38GpuF32 il, Qwen38GpuF32 act, Qwen38GpuF32 gate,
    Qwen38GpuF32 mixed, Qwen38GpuF32 inject, Qwen38Error *);
/* Decode III: the HC read's RMSNorm folded into its F16 down projection (and the
 * optional inject, n_out rows each, K = streams * width): low = down . N,
 * il = inject . N and normed = N, N = rmsnorm(r) * gain per stream (eps), from
 * r [t, K] in one dispatch instead of RMS + multi-GEMV. gain is [width] or
 * [streams, width]. The sums are reordered (tolerance in the fast-kernel tests).
 * Q38_UNSUPPORTED, nothing encoded: not F16, GEMM rows, streams above 4, width
 * not a multiple of 8, or the exact, linear or fuse family selected: the caller
 * encodes RMS + the multi-GEMV. No barrier after the dispatch. */
typedef struct { Q38PackedArgs down, inject; uint32_t streams, width, n_inject, gain_rank; float eps; uint32_t pad[3]; } Q38HcDownArgs;
_Static_assert(sizeof(Q38HcDownArgs) == 192, "HC down Metal argument layout"); /* HcDownArgs */
/* Decode III: q38_packed_encode of a Q8_0 linear (y [t, n_out]) followed by HC_ADD
 * into residual [t, h, n_out] with inject [t, h], in one dispatch (bit-identical
 * to the linear then q38c_hc_add_fast). Q38_UNSUPPORTED, nothing encoded: not
 * Q8_0, GEMM rows, the exact, linear or fuse family, or views not dense. */
Qwen38Status q38_packed_encode_add(Q38PackedGpu *, const Qwen38GpuCommand *, const GpuWeightStore *,
    const GgufBinding *, const Q38PackedLayout *, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38GpuF32 residual,
    Qwen38GpuF32 inject, Qwen38Error *);
/* Decode III: the shared expert's Q8_0 down, y = W . (silu(gate) * up), plus
 * ssig = sigmoid(slogit) [t, 1], in one dispatch (bit-identical to q38f_shared_act
 * then the Q8_0 linear). Q38_UNSUPPORTED: not Q8_0, K % 32, GEMM rows, the exact,
 * linear or fuse family, or views not dense. */
Qwen38Status q38_packed_encode_swiglu(Q38PackedGpu *, const Qwen38GpuCommand *, const GpuWeightStore *,
    const GgufBinding *, const Q38PackedLayout *, Qwen38GpuF32 gate, Qwen38GpuF32 up, Qwen38GpuF32 y,
    Qwen38GpuF32 slogit, Qwen38GpuF32 ssig, Qwen38Error *);
bool q38_packed_add_ready(const Q38PackedGpu *, uint32_t gguf_type, size_t rows, size_t n_out); /* selection only */
bool q38_packed_hc_down_rms_ready(const Q38PackedGpu *, uint32_t gguf_type, size_t rows, size_t k, size_t streams); /* selection only */
Qwen38Status q38_packed_encode_hc_down_rms(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const Q38PackedLayout *down_layout,
    const GgufBinding *inject, const Q38PackedLayout *inject_layout, size_t streams, float eps,
    Qwen38GpuF32 r, Qwen38GpuF32 gain, Qwen38GpuF32 normed, Qwen38GpuF32 low, Qwen38GpuF32 il, Qwen38Error *);
/* R3.1: true when a linear of this GGUF type over this many rows of x runs the
 * prefill GEMM (then the HC tail answers UNSUPPORTED and the caller, which
 * opens profile groups before encoding, takes the sequence directly). */
bool q38_packed_gemm_rows(const Q38PackedGpu *, uint32_t gguf_type, size_t rows);
/* R3.2 grouped MoE for prefill: the (token, slot) pairs grouped by expert in
 * `group` (q38_packed_group_bytes(experts, pairs) bytes, caller scratch), then
 * one GEMM per expert. Each call groups the ids again. gate_up writes act as
 * q38_packed_encode_gate_up (x row = pair / divisor); down writes y[pair] as q38_packed_encode_indexed with divisor 1. Q38_UNSUPPORTED,
 * nothing encoded: below 256 pairs, a format without a grouped kernel, or
 * `indexed` / `gemm` selected. Not bit-identical to the per-pair kernels
 * (order of the sums); rows are bit-independent of their place. */
bool q38_packed_grouped_rows(const Q38PackedGpu *, uint32_t gguf_type, size_t pairs);
size_t q38_packed_group_bytes(size_t experts, size_t pairs);
Qwen38Status q38_packed_encode_grouped_gate_up(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *gate, const Q38PackedLayout *, const GgufBinding *up,
    const Q38PackedLayout *, Qwen38GpuSpan ids, size_t divisor, Qwen38GpuF32 x, Qwen38GpuF32 act,
    Qwen38GpuSpan group, Qwen38Error *);
Qwen38Status q38_packed_encode_grouped_down(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *down, const Q38PackedLayout *, Qwen38GpuSpan ids,
    Qwen38GpuF32 act, Qwen38GpuF32 y, Qwen38GpuSpan group, Qwen38Error *);
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
