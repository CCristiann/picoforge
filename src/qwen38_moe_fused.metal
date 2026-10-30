/* Fused MoE kernels of the R1 fast path: gate + up + SwiGLU of the routed
 * experts in one dispatch (R1.5b), their down projection and the weighted
 * reduce in another (R1.5c). FP32 activations and accumulators; no
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
 * host). Both dots run q38_quant_chunk in q38_quant_row's order (the
 * indexed fast kernels', tail chunks pooled below): act is bit-identical to the
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
    /* R1.9b: the tail chunks of the gate and up rows (chunks % 32 each; K 2560
     * with 32-column chunks: 16 + 16 of 80) share one round of the 32 lanes, and
     * each sum goes back to the lane that adds it last, as q38_quant_row would.
     * Each x chunk is loaded once per row (twice per lane, unlike the former
     * q38_quant_row_pair): measured not slower, x re-reads are not the limit. */
    const uint chunks = (p.k_logical + F::CHUNK - 1) / F::CHUNK, head = chunks / 32 * 32, tail = chunks - head;
    float acc[2] = {0.0f, 0.0f}, pooled[2];
    device const uchar *row[2] = {gw + grow, uw + urow};
    for (uint c = lane; c < head; c += 32)
        for (uint r = 0; r < 2; r++) acc[r] += q38_quant_chunk<F>(row[r], x + xi, p.k_logical, c);
    for (uint j = 0; j < 2; j++) {
        const uint i = 32 * j + lane;
        pooled[j] = i < 2 * tail ? q38_quant_chunk<F>(i < tail ? row[0] : row[1], x + xi, p.k_logical, head + i % tail) : 0.0f;
    }
    for (uint r = 0; r < 2; r++) {
        const uint i = r * tail + lane;
        const float v0 = simd_shuffle(pooled[0], ushort(i % 32)), v1 = simd_shuffle(pooled[1], ushort(i % 32));
        if (lane < tail) acc[r] += i < 32 ? v0 : v1;
    }
    const float g = simd_sum(acc[0]), u = simd_sum(acc[1]);
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

/* R1.5c: the routed down projection and q38f_reduce in one pass.
 * DownReduceArgs mirrors Q38DownReduceArgs (qwen38_b_internal.h): the down
 * projection as one indexed call over the tokens x slots pairs (x = act, its
 * y extent and pitch are y's), then the token and slot counts. */
struct DownReduceArgs { IndexedArgs down; uint tokens, slots; };
/* R2.7: output rows per simdgroup, mirrored by Q38_DOWN_REDUCE_ROWS (qwen38_packed_gpu.m). */
#define DOWN_ROWS 4
static_assert(sizeof(DownReduceArgs) == 104, "down+reduce wire record size");
/* y[t, n] = sum_s scores[t, s] (down_e[n] . act[t*k+s]) + sigmoid[t] shared[t, n],
 * e = ids[t*k+s], and routed[t, n] = the sum before the shared term: IDX down
 * (divisor 1) then q38f_reduce with the identity map, without writing the
 * [pairs, n] down scratch. One simdgroup per DOWN_ROWS outputs (t, n..n+3)
 * since R2.7; each down dot is q38_quant_row's (the indexed fast kernel's
 * order, through q38_quant_rows), accumulated in slot order
 * 0..k-1 with q38f_reduce's expressions (contraction off in both files), so
 * y and routed are bit-identical to that sequence and independent of the
 * number of tokens, n's position and the expert ids. scores, shared, sigmoid
 * and routed are dense [t, k], [t, n], [t], [t, n], their extents checked by
 * the host. An expert id past the tensor, a down row, act row or y element
 * outside its extent sets bit 2 and (t, n) is not written; a nonfinite dot
 * or y sets bit 1. (Idea of a fused down + weighted reduce: ds4 fc80bd6, MIT,
 * metal/qwen4.metal: kernel_qwen4_moe_down, kernel_qwen4_moe_reduce.) */
template <typename F>
static inline void down_reduce(device const uchar *w, device const float *act, device const float *shared,
        device const float *sigmoid, device const float *scores, device const uint *ids, device float *routed,
        device float *y, device atomic_uint *error, constant DownReduceArgs &a, uint2 group, uint sg, uint sgs,
        uint lane) {
    constant PackedArgs &p = a.down.base;
    const ulong n0 = (ulong(group.x) * sgs + sg) * DOWN_ROWS;
    const uint t = group.y;
    if (n0 >= p.n_out || t >= a.tokens) return; /* uniform over the simdgroup, as every return below */
    const uint nr = uint(min(ulong(DOWN_ROWS), p.n_out - n0));
    const ulong yi = ulong(t) * (p.y_stride / 4) + n0, ri = ulong(t) * p.n_out + n0;
    if (yi + nr > p.y_bytes / 4) {
        if (!lane) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    float sum[DOWN_ROWS], d[DOWN_ROWS];
    for (uint r = 0; r < DOWN_ROWS; r++) sum[r] = 0.0f;
    bool finite = true;
    for (uint s = 0; s < a.slots; s++) {
        const uint pair = t * a.slots + s, expert = ids[pair];
        const ulong row = p.weight_offset + ulong(expert) * a.down.expert_bytes + n0 * p.row_bytes;
        const ulong last = row + (nr - 1) * p.row_bytes, xi = ulong(pair) * (p.x_stride / 4);
        if (expert >= a.down.experts || last > p.weight_end || p.row_bytes > p.weight_end - last ||
            xi > p.x_bytes / 4 || p.k_logical > p.x_bytes / 4 - xi) {
            if (!lane) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
            return;
        }
        q38_quant_rows<F, DOWN_ROWS>(w + row, p.row_bytes, nr, act + xi, p.k_logical, lane, d);
        for (uint r = 0; r < nr; r++) {
            finite = finite && isfinite(d[r]);
            sum[r] += scores[pair] * d[r];
        }
    }
    if (lane) return;
    for (uint r = 0; r < nr; r++) {
        routed[ri + r] = sum[r];
        const float v = sum[r] + sigmoid[t] * shared[ri + r];
        if (!finite || !isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        y[yi + r] = v;
    }
}
#define DOWN_REDUCE_KERNEL(name, F) \
kernel void name(device const uchar *w [[buffer(0)]], device const float *act [[buffer(1)]], \
        device const float *shared [[buffer(2)]], device const float *sigmoid [[buffer(3)]], \
        device const float *scores [[buffer(4)]], device const uint *ids [[buffer(5)]], \
        device float *routed [[buffer(6)]], device float *y [[buffer(7)]], device atomic_uint *error [[buffer(8)]], \
        constant DownReduceArgs &a [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], \
        uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    down_reduce<F>(w, act, shared, sigmoid, scores, ids, routed, y, error, a, group, sg, sgs, lane); \
}
DOWN_REDUCE_KERNEL(q38_moe_down_reduce_iq2_xxs, Q38Iq2Xxs)
DOWN_REDUCE_KERNEL(q38_moe_down_reduce_q2_k, Q38Q2K)
DOWN_REDUCE_KERNEL(q38_moe_down_reduce_q4_k, Q38Q4K)
DOWN_REDUCE_KERNEL(q38_moe_down_reduce_mxfp4, Q38Mxfp4)

/* R1.5d router softmax + top-k, one threadgroup per token: bit-identical to
 * q38f_route (qwen38_moe.metal) in ids, weights and probs, which ran ONE thread
 * per token (512 exp, a serial sum, then k scans of all experts: 59 ms per
 * decode token). What is parallel is exact in any order: the maximum (only
 * z - maximum is used, and exp(+-0) = 1), exp and the divide per element, and
 * each selection round, an arg-max over (larger p, then smaller id) -- the
 * oracle's strict > scan in ascending id. What is NOT reordered: the exp sum
 * (thread 0, e = 0..E-1, the order of the oracle: a tree sum would change the
 * bits of every probability) and `selected` (serial in s). A chosen expert is
 * masked with -1 (below every p >= 0) in threadgroup memory by the thread that
 * owns it, so no used-id rescan and no barrier for the mask. The threadgroup
 * width is the host's (a multiple of 32, at most 1024). (Idea of a parallel
 * router top-k: ds4 fc80bd6, MIT, metal/qwen4.metal: kernel_qwen4_router_topk.)
 * Capacity Q38F_ROUTE_MAX_EXPERTS mirrors qwen38_f_internal.h; beyond it, or
 * k > e, the host keeps q38f_route and this kernel only sets bit 2. */
#define Q38F_ROUTE_MAX_EXPERTS 1024
struct RouteArgs { uint t, e, k, norm, pitch; };
kernel void q38_moe_route_fast(device const float *logits [[buffer(0)]], device float *probs [[buffer(1)]],
        device uint *ids [[buffer(2)]], device float *weights [[buffer(3)]], constant RouteArgs &a [[buffer(4)]],
        device atomic_uint *error [[buffer(5)]], uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]], uint tn [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
        uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float work[Q38F_ROUTE_MAX_EXPERTS];  /* p, then p with the chosen experts at -1 */
    threadgroup float sg_value[2][32];               /* per-simdgroup (max z | best p) */
    threadgroup uint sg_flag[2][32];                 /* per-simdgroup (nonfinite | best id) */
    threadgroup float sum_all;
    if (token >= a.t) return;
    if (a.e == 0 || a.e > Q38F_ROUTE_MAX_EXPERTS || a.k == 0 || a.k > a.e) {
        if (!tid) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    device const float *z = logits + ulong(token) * a.pitch;
    device float *p = probs + ulong(token) * a.e;
    device uint *id = ids + ulong(token) * a.k;
    device float *w = weights + ulong(token) * a.k;
    float local = -INFINITY; uint bad = 0;
    for (uint e = tid; e < a.e; e += tn) { float v = z[e]; bad |= !isfinite(v); local = v > local ? v : local; }
    float group = simd_max(local); uint group_bad = simd_max(bad);
    if (!lane) { sg_value[0][sg] = group; sg_flag[0][sg] = group_bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -INFINITY; uint any_bad = 0;
    for (uint g = 0; g < nsg; g++) { maximum = max(maximum, sg_value[0][g]); any_bad |= sg_flag[0][g]; }
    if (any_bad) { /* the oracle: bit 1, fallback IDs and zero weights, probs untouched */
        if (!tid) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        for (uint s = tid; s < a.k; s += tn) { id[s] = s; w[s] = 0.0f; }
        return;
    }
    for (uint e = tid; e < a.e; e += tn) work[e] = exp(z[e] - maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) { float sum = 0.0f; for (uint e = 0; e < a.e; e++) sum += work[e]; sum_all = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum = sum_all;
    for (uint e = tid; e < a.e; e += tn) { float v = work[e] / sum; work[e] = v; p[e] = v; }
    float selected = 0.0f;
    for (uint s = 0; s < a.k; s++) {
        float best = -2.0f; uint at = 0xffffffffu; /* this thread's elements ascend: first strict > = smallest id */
        for (uint e = tid; e < a.e; e += tn) if (work[e] > best) { best = work[e]; at = e; }
        float top = simd_max(best);
        uint first = simd_min(best == top ? at : 0xffffffffu), slot = s & 1; /* two slots: no barrier before reuse */
        if (!lane) { sg_value[slot][sg] = top; sg_flag[slot][sg] = first; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value = -3.0f; uint chosen = 0xffffffffu;
        for (uint g = 0; g < nsg; g++) {
            float v = sg_value[slot][g]; uint i = sg_flag[slot][g];
            if (v > value || (v == value && i < chosen)) { value = v; chosen = i; }
        }
        if (chosen % tn == tid) work[chosen] = -1.0f; /* only the owner touches its element */
        if (!tid) { id[s] = chosen; w[s] = value; }
        selected += value; /* serial in s, as the oracle (every thread tracks the same sum) */
    }
    if (a.norm && !tid) for (uint s = 0; s < a.k; s++) w[s] /= selected;
}
