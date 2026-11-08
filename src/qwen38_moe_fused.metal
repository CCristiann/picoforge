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

/* D1.2: the shared expert's three element-wise steps in one dispatch, bit-identical to the
 * sequence ACT SILU (sgate -> sact), MUL (sact * sup -> sact), ACT SIGMOID (slogit -> ssig):
 * q38_silu and q38_sigmoid are q38c_activate's expressions, the product is q38f_product's.
 * Thread i < count handles one shared-intermediate element, the next `tokens` threads one
 * token's sigmoid. A nonfinite sgate, silu, sup, product, slogit or sigmoid sets bit 1; the
 * outputs are then unspecified (the sequence stops at its first failing op). One dispatch and
 * one barrier instead of three of each, 48 times per token. */
struct SharedActArgs { uint count, tokens; };
kernel void q38f_shared_act(device const float *sgate [[buffer(0)]], device const float *sup [[buffer(1)]],
        device float *sact [[buffer(2)]], device const float *slogit [[buffer(3)]], device float *ssig [[buffer(4)]],
        device atomic_uint *error [[buffer(5)]], constant SharedActArgs &a [[buffer(6)]],
        uint i [[thread_position_in_grid]]) {
    if (i < a.count) {
        const float g = sgate[i], u = sup[i];
        const float y = q38_silu(g), v = y * u;
        if (!isfinite(g) || !isfinite(y) || !isfinite(u) || !isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        sact[i] = v;
    } else if (i - a.count < a.tokens) {
        const uint j = i - a.count;
        const float x = slogit[j];
        float y = q38_sigmoid(x / 1.0f);
        y *= 1.0f;
        if (!isfinite(x) || !isfinite(y)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        ssig[j] = y;
    }
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

/* ---- R3.2 grouped MoE for prefill. The kernels above read an expert's rows
 * once per (token, slot) pair, so a 1024-token chunk read each active expert
 * ~20 times. Here the pairs are grouped by expert (counting sort) and each
 * expert runs one GEMM over its rows: its weights are decoded once per tile.
 * Group buffer g (uint): counts[E], cursor[E], offsets[E+1], tiles[E+1],
 * order[pairs]; expert e owns order[offsets[e] .. offsets[e+1]) and the GEMM
 * row tiles tiles[e] .. tiles[e+1]. The order inside an expert depends on
 * atomics, but a GEMM row's bits do not depend on its place in the tile, so
 * the outputs are deterministic. */
struct GroupArgs { uint pairs, experts, tile, reserved; };
kernel void q38_moe_group_clear(device uint *g [[buffer(0)]], constant GroupArgs &a [[buffer(1)]],
        uint e [[thread_position_in_grid]]) {
    if (e < a.experts) g[e] = 0;
}
/* An expert id past the tensor sets bit 2 and its pair joins no group. */
kernel void q38_moe_group_count(device const uint *ids [[buffer(0)]], device atomic_uint *g [[buffer(1)]],
        device atomic_uint *error [[buffer(2)]], constant GroupArgs &a [[buffer(3)]], uint q [[thread_position_in_grid]]) {
    if (q >= a.pairs) return;
    const uint e = ids[q];
    if (e >= a.experts) { atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return; }
    atomic_fetch_add_explicit(g + e, 1u, memory_order_relaxed);
}
/* One thread: E is 512, a few microseconds. */
kernel void q38_moe_group_scan(device uint *g [[buffer(0)]], constant GroupArgs &a [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
    if (i) return;
    const uint E = a.experts;
    device uint *cursor = g + E, *offsets = g + 2 * E, *tiles = g + 3 * E + 1;
    uint o = 0, t = 0;
    for (uint e = 0; e < E; e++) {
        offsets[e] = o; tiles[e] = t; cursor[e] = o;
        o += g[e]; t += (g[e] + a.tile - 1) / a.tile;
    }
    offsets[E] = o; tiles[E] = t;
}
kernel void q38_moe_group_scatter(device const uint *ids [[buffer(0)]], device atomic_uint *g [[buffer(1)]],
        device uint *order [[buffer(2)]], constant GroupArgs &a [[buffer(3)]], uint q [[thread_position_in_grid]]) {
    if (q >= a.pairs) return;
    const uint e = ids[q];
    if (e < a.experts) order[atomic_fetch_add_explicit(g + a.experts + e, 1u, memory_order_relaxed)] = q;
}

/* Grouped GEMM tile: GG_M rows of one expert x GG_N outputs, K in GG_K steps,
 * 4 simdgroups in a 2 x 2 grid, each 16 rows x 32 outputs. P4.2: 32 x 64 instead
 * of R3.3's 64 x 32, same work per simdgroup: a chunk routes ~20 rows to each of
 * the 512 experts, so a 64-row tile multiplied ~70% padding rows (isolated
 * kernel, 10240 pairs: gate+up 39.5 -> 22.9 ms, down 25.7 -> 16.0 ms). A row's
 * bits do not depend on the tile shape (byte-equal outputs). PAIR: gate and up together, act = silu(g) * u as
 * the fused kernel (same q38_silu, same checks); else one matrix (down).
 * x row of pair q: q / x_divisor; y row: q. Weights decode exactly (F), so
 * only the order of the sums differs from the indexed kernels. */
#define GG_M 32
#define GG_N 64
#define GG_K 32
#define GG_THREADS 128
template <typename F, bool PAIR>
static inline void grouped(device const uchar *gw, device const uchar *uw, device const float *x, device float *y,
        device atomic_uint *error, constant IndexedArgs &ga, constant IndexedArgs &ua, device const uint *g,
        uint2 group, uint tid, uint sg, threadgroup float *buf) {
    /* x and weight tiles during the K loop, then the same 16 KB stage the results. */
    threadgroup float *xs = buf, *ws = buf + GG_M * GG_K, *cs = buf;
    constant PackedArgs &p = ga.base, &q = ua.base;
    const uint E = ga.experts;
    device const uint *offsets = g + 2 * E, *tiles = g + 3 * E + 1, *order = g + 4 * E + 2;
    const uint ti = group.y;
    if (ti >= tiles[E]) return; /* uniform over the threadgroup, as every return below */
    uint lo = 0, hi = E; /* the expert whose tile range holds ti: last e with tiles[e] <= ti */
    while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (tiles[mid] <= ti) lo = mid; else hi = mid; }
    const uint e = lo, r0 = offsets[e] + (ti - tiles[e]) * GG_M, mt = min(uint(GG_M), offsets[e + 1] - r0);
    const ulong n0 = ulong(group.x) * GG_N;
    if (n0 >= p.n_out) return;
    const uint nt = uint(min(ulong(GG_N), p.n_out - n0));
    const ulong gbase = p.weight_offset + ulong(e) * ga.expert_bytes, ubase = q.weight_offset + ulong(e) * ua.expert_bytes;
    const ulong gl = gbase + (n0 + nt - 1) * p.row_bytes, ul = ubase + (n0 + nt - 1) * q.row_bytes;
    if (e >= ga.experts || e >= ua.experts || gl > p.weight_end || p.row_bytes > p.weight_end - gl ||
        (PAIR && (ul > q.weight_end || q.row_bytes > q.weight_end - ul))) {
        if (!tid) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    const uint sr = (sg & 1) * 16, sc = (sg >> 1) * 32, M = PAIR ? 2 : 1;
    simdgroup_float8x8 acc[2][2][4];
    for (uint m = 0; m < 2; m++)
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 4; j++) acc[m][i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    constexpr uint CPS = GG_K / F::CHUNK; /* chunks per K step: 1, or 2 for Q2_K */
    for (uint k0 = 0; k0 < p.k_logical; k0 += GG_K) {
        const uint kc = min(uint(GG_K), p.k_logical - k0);
        for (uint i = tid; i < GG_M * GG_K; i += GG_THREADS) {
            const uint r = i / GG_K, c = i % GG_K;
            float v = 0.0f;
            if (r < mt && c < kc) {
                const ulong xi = ulong(order[r0 + r] / ga.x_divisor) * (p.x_stride / 4);
                if (xi <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xi) v = x[xi + k0 + c];
                else if (!c) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
            }
            xs[i] = v;
        }
        for (uint job = tid; job < M * GG_N * CPS; job += GG_THREADS) {
            const uint m = job / (GG_N * CPS), r = job / CPS % GG_N, j = job % CPS;
            threadgroup float *dst = ws + (m * GG_N + r) * GG_K + j * F::CHUNK;
            float w[F::CHUNK];
            if (r < nt) F::decode((m ? uw + ubase + (n0 + r) * q.row_bytes : gw + gbase + (n0 + r) * p.row_bytes),
                                  k0 / F::CHUNK + j, w);
            for (uint i = 0; i < F::CHUNK; i++) dst[i] = r < nt && j * F::CHUNK + i < kc ? w[i] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < GG_K; kk += 8) {
            simdgroup_float8x8 a[2], b[4];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], xs + (sr + i * 8) * GG_K + kk, GG_K);
            for (uint m = 0; m < M; m++) {
                for (uint j = 0; j < 4; j++)
                    simdgroup_load(b[j], ws + (m * GG_N + sc + j * 8) * GG_K + kk, GG_K, ulong2(0, 0), true);
                for (uint i = 0; i < 2; i++)
                    for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[m][i][j], a[i], b[j], acc[m][i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint m = 0; m < M; m++)
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_store(acc[m][i][j], cs + m * GG_M * GG_N + (sr + i * 8) * GG_N + sc + j * 8, GG_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < GG_M * GG_N; i += GG_THREADS) {
        const uint r = i / GG_N, c = i % GG_N;
        if (r >= mt || c >= nt) continue;
        const ulong yi = ulong(order[r0 + r]) * (p.y_stride / 4) + n0 + c;
        if (yi >= p.y_bytes / 4) { atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); continue; }
        const float gv = cs[i];
        if (PAIR) {
            const float u = cs[GG_M * GG_N + i], s = q38_silu(gv), h = s * u;
            y[yi] = h;
            if (!isfinite(gv) || !isfinite(u) || !isfinite(s) || !isfinite(h))
                atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        } else {
            y[yi] = gv;
            if (!isfinite(gv)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        }
    }
}
#define GROUPED_KERNEL(name, F, PAIR) \
kernel void name(device const uchar *gw [[buffer(0)]], device const uchar *uw [[buffer(1)]], \
        device const float *x [[buffer(2)]], device float *y [[buffer(3)]], device atomic_uint *error [[buffer(4)]], \
        constant IndexedArgs &ga [[buffer(5)]], constant IndexedArgs &ua [[buffer(6)]], device const uint *g [[buffer(7)]], \
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
        uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float buf[GG_M * GG_K + 2 * GG_N * GG_K]; /* >= cs: 2*GG_M*GG_N */ \
    grouped<F, PAIR>(gw, uw, x, y, error, ga, ua, g, group, tid, sg, buf); \
}
GROUPED_KERNEL(q38_moe_grouped_gate_up_iq2_xxs, Q38Iq2Xxs, true)
GROUPED_KERNEL(q38_moe_grouped_gate_up_q2_k, Q38Q2K, true)
GROUPED_KERNEL(q38_moe_grouped_gate_up_q4_k, Q38Q4K, true)
GROUPED_KERNEL(q38_moe_grouped_gate_up_mxfp4, Q38Mxfp4, true)
GROUPED_KERNEL(q38_moe_grouped_down_iq2_xxs, Q38Iq2Xxs, false)
GROUPED_KERNEL(q38_moe_grouped_down_q2_k, Q38Q2K, false)
GROUPED_KERNEL(q38_moe_grouped_down_q4_k, Q38Q4K, false)
GROUPED_KERNEL(q38_moe_grouped_down_mxfp4, Q38Mxfp4, false)

/* ---- R3.5 grouped MoE on the Neural Accelerators: grouped<F, PAIR> above with
 * half operands (expert weights decoded exactly by F, then rounded to half; x
 * rounded to half) and matmul2d with FP32 accumulation. Reduced precision:
 * its own gate ("nax_half" in tests/qwen38_kernel_tolerances.json), the FP32
 * grouped GEMM is its oracle, prefill only (PICOFORGE_PREFILL_PRECISION).
 * Tile: GN_M rows of one expert (the host cuts the groups' row tiles at GN_M
 * in this mode, at GG_M otherwise) x GN_N outputs, K in 32-wide steps (the matmul2d k; the K tail is
 * zero-padded here, the op would truncate it). Gate and up accumulate in two
 * cooperative tensors with one element layout, so act = silu(g) * u is formed
 * in registers and one FP32 tile goes through threadgroup memory, which
 * reuses the staging area. Idea from DwarfStar fc80bd6 (MIT),
 * metal/qwen4.metal kernel_qwen4_moe_mm_mid_nax_t. */
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define GN_M 64
#define GN_N 64
#define GN_K 32
using GroupedNax = mpp::tensor_ops::matmul2d<
    mpp::tensor_ops::matmul2d_descriptor(GN_M, GN_N, GN_K, false, true, false,
                                         mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
    execution_simdgroups<4>>;
template <typename F, bool PAIR>
static inline void grouped_nax(device const uchar *gw, device const uchar *uw, device const float *x, device float *y,
        device atomic_uint *error, constant IndexedArgs &ga, constant IndexedArgs &ua, device const uint *g,
        uint2 group, uint tid, threadgroup float *smem) {
    constant PackedArgs &p = ga.base, &q = ua.base;
    const uint E = ga.experts;
    device const uint *offsets = g + 2 * E, *tiles = g + 3 * E + 1, *order = g + 4 * E + 2;
    const uint ti = group.y;
    if (ti >= tiles[E]) return; /* uniform over the threadgroup, as every return below */
    uint lo = 0, hi = E;
    while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (tiles[mid] <= ti) lo = mid; else hi = mid; }
    const uint e = lo, r0 = offsets[e] + (ti - tiles[e]) * GN_M, mt = min(uint(GN_M), offsets[e + 1] - r0);
    const ulong n0 = ulong(group.x) * GN_N;
    if (n0 >= p.n_out) return;
    const uint nt = uint(min(ulong(GN_N), p.n_out - n0));
    const ulong gbase = p.weight_offset + ulong(e) * ga.expert_bytes, ubase = q.weight_offset + ulong(e) * ua.expert_bytes;
    const ulong gl = gbase + (n0 + nt - 1) * p.row_bytes, ul = ubase + (n0 + nt - 1) * q.row_bytes;
    if (e >= ga.experts || e >= ua.experts || gl > p.weight_end || p.row_bytes > p.weight_end - gl ||
        (PAIR && (ul > q.weight_end || q.row_bytes > q.weight_end - ul))) {
        if (!tid) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    constexpr uint M = PAIR ? 2 : 1;
    threadgroup half *xs = (threadgroup half *)smem, *ws = xs + GN_M * GN_K; /* ws: gate rows, then up rows */
    auto tx = tensor(xs, dextents<int32_t, 2>(GN_K, GN_M));
    auto tg = tensor(ws, dextents<int32_t, 2>(GN_K, GN_N));
    auto tu = tensor(ws + GN_N * GN_K, dextents<int32_t, 2>(GN_K, GN_N));
    GroupedNax mm;
    auto cg = mm.get_destination_cooperative_tensor<decltype(tx), decltype(tg), float>();
    auto cu = mm.get_destination_cooperative_tensor<decltype(tx), decltype(tu), float>();
    for (uint16_t i = 0; i < cg.get_capacity(); i++) if (cg.is_valid_element(i)) { cg[i] = 0.0f; cu[i] = 0.0f; }
    /* Each staging thread owns the x rows it gathers for the whole K loop. */
    constexpr uint CPS = GN_K / F::CHUNK;
    for (uint k0 = 0; k0 < p.k_logical; k0 += GN_K) {
        const uint kc = min(uint(GN_K), p.k_logical - k0);
        for (uint i = tid; i < GN_M * GN_K / 4; i += GG_THREADS) {
            const uint r = i / (GN_K / 4), c = (i % (GN_K / 4)) * 4;
            half4 v = half4(0.0h);
            if (r < mt) {
                const ulong xi = ulong(order[r0 + r] / ga.x_divisor) * (p.x_stride / 4);
                if (xi <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xi) {
                    for (uint j = 0; j < 4; j++) v[j] = c + j < kc ? half(x[xi + k0 + c + j]) : 0.0h;
                } else if (!c) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
            }
            *(threadgroup half4 *)(xs + r * GN_K + c) = v;
        }
        for (uint job = tid; job < M * GN_N * CPS; job += GG_THREADS) {
            const uint m = job / (GN_N * CPS), r = job / CPS % GN_N, j = job % CPS;
            threadgroup half *dst = ws + (m * GN_N + r) * GN_K + j * F::CHUNK;
            float w[F::CHUNK];
            if (r < nt) F::decode((m ? uw + ubase + (n0 + r) * q.row_bytes : gw + gbase + (n0 + r) * p.row_bytes),
                                  k0 / F::CHUNK + j, w);
            for (uint i = 0; i < F::CHUNK; i++) dst[i] = r < nt && j * F::CHUNK + i < kc ? half(w[i]) : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sx = tx.slice(0, 0);
        auto sg = tg.slice(0, 0);
        mm.run(sx, sg, cg);
        if (PAIR) {
            auto su = tu.slice(0, 0);
            mm.run(sx, su, cu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* act in registers (gate and up share one layout), with grouped's checks;
     * a bad element is flagged once per thread. Elements of padded rows or
     * columns are zeros (silu(0) * 0) and are never stored below. */
    bool bad = false;
    if (PAIR) {
        for (uint16_t i = 0; i < cg.get_capacity(); i++) {
            if (!cg.is_valid_element(i)) continue;
            const float gv = cg[i], u = cu[i], sv = q38_silu(gv), h = sv * u;
            bad = bad || !isfinite(gv) || !isfinite(u) || !isfinite(sv) || !isfinite(h);
            cg[i] = h;
        }
    }
    auto tc = tensor(smem, dextents<int32_t, 2>(GN_N, GN_M));
    auto sc = tc.slice(0, 0);
    cg.store(sc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < GN_M * GN_N; i += GG_THREADS) {
        const uint r = i / GN_N, c = i % GN_N;
        if (r >= mt || c >= nt) continue;
        const ulong yi = ulong(order[r0 + r]) * (p.y_stride / 4) + n0 + c;
        if (yi >= p.y_bytes / 4) { atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); continue; }
        const float v = smem[i];
        y[yi] = v;
        if (!isfinite(v)) bad = true;
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}
#define GROUPED_NAX_KERNEL(name, F, PAIR) \
kernel void name(device const uchar *gw [[buffer(0)]], device const uchar *uw [[buffer(1)]], \
        device const float *x [[buffer(2)]], device float *y [[buffer(3)]], device atomic_uint *error [[buffer(4)]], \
        constant IndexedArgs &ga [[buffer(5)]], constant IndexedArgs &ua [[buffer(6)]], device const uint *g [[buffer(7)]], \
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float smem[GN_M * GN_N]; /* 16 KB: staging (x 4 KB + 2 x W 8 KB), then the FP32 tile */ \
    grouped_nax<F, PAIR>(gw, uw, x, y, error, ga, ua, g, group, tid, smem); \
}
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_gate_up_iq2_xxs, Q38Iq2Xxs, true)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_gate_up_q2_k, Q38Q2K, true)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_gate_up_q4_k, Q38Q4K, true)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_gate_up_mxfp4, Q38Mxfp4, true)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_down_iq2_xxs, Q38Iq2Xxs, false)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_down_q2_k, Q38Q2K, false)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_down_q4_k, Q38Q4K, false)
GROUPED_NAX_KERNEL(q38_moe_grouped_nax_down_mxfp4, Q38Mxfp4, false)
