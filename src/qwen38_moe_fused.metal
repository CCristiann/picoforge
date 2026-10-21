/* Fused MoE kernels of the R1 fast path (R1.5b): gate + up + SwiGLU of the
 * routed experts in one dispatch. FP32 activations and accumulators; no
 * fast-math, no FMA contraction (also -ffp-contract=off in the build). */
#include <metal_stdlib>
#include "qwen38_quant_decode.h"
using namespace metal;
#pragma METAL fp contract(off)

/* Wire records mirrored from qwen38_b_internal.h (Q38PackedArgs,
 * Q38PackedIndexedArgs), as in qwen38_matvec.metal. */
struct PackedArgs {
    uint type, k_storage, k_logical, n_out, rows, reserved;
    ulong row_bytes, weight_offset, weight_end, x_stride, y_stride, x_bytes, y_bytes;
};
struct IndexedArgs { PackedArgs base; ulong expert_bytes; uint experts, x_divisor; };
struct GateUpArgs { IndexedArgs gate, up; };
static_assert(sizeof(GateUpArgs) == 192, "gate/up wire record size");

/* q38c_activate's SILU (qwen38_recurrent.metal) with its scales p1 = p2 = 1:
 * the same sigmoid and the same FP32 operations, so silu(g) has its bits. */
static float q38_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float z = exp(x); return z / (1.0f + z);
}
static float q38_silu(float g) {
    float s = g / 1.0f;
    s = s * q38_sigmoid(s);
    s *= 1.0f;
    return s;
}

/* act[p, n] = silu(gate_e[n] . x) * (up_e[n] . x), e = ids[p], x row
 * p / x_divisor: the unfused sequence IDX gate, IDX up, ACT SILU, MUL in one
 * pass. One simdgroup per output (p, n), as many along n per threadgroup as
 * the host dispatches (read from the dispatch, no constant shared with the
 * host). Both dots come from q38_quant_row_pair, whose order is
 * q38_quant_row's (the indexed fast kernels'): act is bit-identical to the
 * unfused fast sequence and, like it, independent of the number of tokens or
 * slots, the slot and the expert id. Bounds as the indexed kernels: an expert
 * id past the tensor, or a gate/up row, x row or act element outside its
 * extent, sets bit 2 and that element is not written; a nonfinite g, u,
 * silu(g) or act sets bit 1. (Idea of one fused mid kernel per routed format:
 * ds4 fc80bd6, MIT, metal/qwen4.metal: kernel_qwen4_moe_mid.) */
template <typename F>
static inline void gate_up(device const uchar *gw, device const uchar *uw, device const float *x, device float *act,
        device atomic_uint *error, constant GateUpArgs &a, device const uint *ids, uint2 group, uint sg, uint sgs,
        uint lane) {
    constant PackedArgs &p = a.gate.base, &q = a.up.base;
    const ulong n = ulong(group.x) * sgs + sg;
    const uint m = group.y;
    if (n >= p.n_out || m >= p.rows) return; /* uniform over the simdgroup, as every return below */
    const uint expert = ids[m];
    const ulong grow = p.weight_offset + ulong(expert) * a.gate.expert_bytes + n * p.row_bytes;
    const ulong urow = q.weight_offset + ulong(expert) * a.up.expert_bytes + n * q.row_bytes;
    const ulong xi = ulong(m / a.gate.x_divisor) * (p.x_stride / 4), yi = ulong(m) * (p.y_stride / 4) + n;
    if (expert >= a.gate.experts || expert >= a.up.experts || grow > p.weight_end || p.row_bytes > p.weight_end - grow ||
        urow > q.weight_end || q.row_bytes > q.weight_end - urow ||
        xi > p.x_bytes / 4 || p.k_logical > p.x_bytes / 4 - xi || yi >= p.y_bytes / 4) {
        if (!lane) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    float g, u;
    q38_quant_row_pair<F>(gw + grow, uw + urow, x + xi, p.k_logical, lane, g, u);
    if (lane) return;
    const float s = q38_silu(g), h = s * u;
    act[yi] = h;
    if (!isfinite(g) || !isfinite(u) || !isfinite(s) || !isfinite(h))
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}
#define GATE_UP_KERNEL(name, F) \
kernel void name(device const uchar *gw [[buffer(0)]], device const uchar *uw [[buffer(1)]], \
        device const float *x [[buffer(2)]], device float *act [[buffer(3)]], \
        device atomic_uint *error [[buffer(4)]], constant GateUpArgs &a [[buffer(5)]], \
        device const uint *ids [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], \
        uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    gate_up<F>(gw, uw, x, act, error, a, ids, group, sg, sgs, lane); \
}
GATE_UP_KERNEL(q38_moe_gate_up_iq2_xxs, Q38Iq2Xxs)
GATE_UP_KERNEL(q38_moe_gate_up_q2_k, Q38Q2K)
GATE_UP_KERNEL(q38_moe_gate_up_q4_k, Q38Q4K)
GATE_UP_KERNEL(q38_moe_gate_up_mxfp4, Q38Mxfp4)
