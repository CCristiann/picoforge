/* Cooperative matvec kernels of the R1 fast path: y[m,n] = sum_k w[n,k] x[m,k].
 * FP32 activations and accumulators; no fast-math, no FMA contraction. */
#include <metal_stdlib>
#include "qwen38_quant_decode.h"
using namespace metal;
#pragma METAL fp contract(off)

/* Same wire record as q38_packed_linear (mirrored from qwen38_b_internal.h). */
struct PackedArgs {
    uint type, k_storage, k_logical, n_out, rows, reserved;
    ulong row_bytes, weight_offset, weight_end, x_stride, y_stride, x_bytes, y_bytes;
};
static_assert(sizeof(PackedArgs) == 80, "host/Metal wire record size");

/* Rows of x per threadgroup, mirrored by Q38_Q8_T in qwen38_packed_gpu.m.
 * Measured (R1.2a in docs/devlog.md): one 32-lane simdgroup per threadgroup
 * and one output row per simdgroup beat 2-8 simdgroups and 2-4 rows, and 4
 * rows of x beat 1, 2, 8 and 16 up to 8 rows. */
#define Q8_T 4

/* Q8_0 block = fp16 scale d + 32 int8 codes (34 bytes). w = d*code is exact
 * in FP32 (11 x 8 significant bits), so every product w*x is bit-identical to
 * the oracle's and only the summation order differs from it.
 * Lane l takes whole blocks l, l+32, ... (the simdgroup reads 32 consecutive
 * blocks, 1088 contiguous bytes), sums each block in column order, adds the
 * block sums in block order, and simd_sum joins the 32 lanes. That order
 * depends only on K: never on how many rows of x or w the dispatch has, nor
 * on where a row sits in the tile, so chunk, decode and MTP agree bit for bit.
 * Each decoded block is reused for the tile's Q8_T rows of x.
 * p: the x side (rows, x_stride, x_bytes, k_logical); o: output row n's
 * tensor (n_out, weights, y). A single call passes its record twice. */
template <bool DEC>
static inline void q8_0_matvec(device const uchar *weights, device const float *x, device float *y,
        device atomic_uint *error, constant PackedArgs &p, constant PackedArgs &o, ulong n, uint tile, uint lane) {
    const ulong m0 = ulong(tile) * Q8_T;
    if (n >= o.n_out || m0 >= p.rows) return;
    const uint mt = uint(min(ulong(Q8_T), p.rows - m0));
    /* Bounds as the oracle, once per row: a weight row, x row or y element
     * outside its extent sets bit 2 and the simdgroup reads and writes
     * nothing. Tile slots past the last row of x repeat row m0, unused. */
    const ulong row = o.weight_offset + n * o.row_bytes;
    bool inside = row <= o.weight_end && o.row_bytes <= o.weight_end - row;
    ulong xi[Q8_T];
    for (uint t = 0; t < Q8_T; t++) {
        const ulong m = m0 + (t < mt ? t : 0);
        xi[t] = m * (p.x_stride / 4);
        inside = inside && xi[t] <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xi[t] &&
                 m * (o.y_stride / 4) + n < o.y_bytes / 4;
    }
    if (!inside) {
        if (!lane) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    float acc[Q8_T];
    for (uint t = 0; t < Q8_T; t++) acc[t] = 0.0f;
    const uint blocks = (p.k_logical + 31) / 32;
    /* Decode III (the _dec kernels): 4 lanes per block, 8 blocks per round
     * (lane = 4*ix + it owns codes 8it..8it+7 of blocks ix, ix+8, ...), the
     * codes read as 2-byte pairs, s = sum(code * x) with FMAs and acc += d * s:
     * the block's 32 byte loads and 32 scale multiplies are gone (decode
     * 35.32 -> 34.20 ms/token in an A/B). Each row's order still depends on K
     * alone, for any number of rows of x, so decode and MTP verify agree; it
     * differs from the oracle's exact d*code products (tolerance linear_q8_dec).
     * Needs K % 32 == 0 (else the R1.2a loop below); x is read as float4 when
     * its rows are 16-byte aligned, as floats otherwise (same values, same
     * bits). Lane map idea: ds4 fc80bd6, MIT, qwen4_row_dot. */
    if (DEC && !(p.k_logical & 31)) {
        const bool x4 = !(ulong(x) & 15) && !(p.x_stride & 15); /* load width only: same values, same bits */
        const uint ix = lane / 4, it = lane % 4;
        for (uint b = ix; b < blocks; b += 8) {
            device const uchar *q = weights + row + ulong(b) * 34;
            const float d = float(*(device const half *)q);
            device const ushort *c = (device const ushort *)(q + 2) + 4 * it; /* blocks are 2-byte aligned */
            float code[8];
            for (uint i = 0; i < 4; i++) {
                const char2 v = as_type<char2>(c[i]);
                code[2 * i] = float(v.x); code[2 * i + 1] = float(v.y);
            }
            for (uint t = 0; t < Q8_T; t++) {
                if (t >= mt) break; /* uniform */
                device const float *xs = x + xi[t] + b * 32 + 8 * it;
                float4 x0, x1;
                if (x4) { x0 = ((device const float4 *)xs)[0]; x1 = ((device const float4 *)xs)[1]; }
                else { x0 = float4(xs[0], xs[1], xs[2], xs[3]); x1 = float4(xs[4], xs[5], xs[6], xs[7]); }
                float sm = code[0] * x0.x;
                sm = fma(code[1], x0.y, sm); sm = fma(code[2], x0.z, sm); sm = fma(code[3], x0.w, sm);
                sm = fma(code[4], x1.x, sm); sm = fma(code[5], x1.y, sm); sm = fma(code[6], x1.z, sm);
                sm = fma(code[7], x1.w, sm);
                acc[t] = fma(d, sm, acc[t]);
            }
        }
    } else
    for (uint b = lane; b < blocks; b += 32) {
        /* Only the last block may be partial: its columns past k_logical are
         * left out of the sum and never read from x. */
        const uint cols = min(32u, p.k_logical - b * 32);
        device const uchar *q = weights + row + ulong(b) * 34;
        const float d = q38_fp16(q);
        float w[32];
        for (uint i = 0; i < 32; i++) w[i] = d * float(as_type<char>(q[2 + i]));
        for (uint t = 0; t < Q8_T; t++) {
            if (t >= mt) break; /* uniform: decode (one row of x) skips the empty slots */
            device const float *xb = x + xi[t] + b * 32;
            float s = 0.0f;
            if (cols == 32)
                for (uint i = 0; i < 32; i++) s += w[i] * xb[i];
            else
                for (uint i = 0; i < cols; i++) s += w[i] * xb[i];
            acc[t] += s;
        }
    }
    /* Finiteness on the output (bit 1): a NaN or Inf weight or input reaches
     * the sum through + and *, so it shows up here as in the oracle. */
    for (uint t = 0; t < Q8_T; t++) {
        const float v = simd_sum(acc[t]);
        if (lane || t >= mt) continue;
        y[(m0 + t) * (o.y_stride / 4) + n] = v;
        if (!isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
kernel void q38_matvec_q8_0(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    q8_0_matvec<false>(weights, x, y, error, p, p, group.x, group.y, lane);
}
kernel void q38_matvec_q8_0_dec(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    q8_0_matvec<true>(weights, x, y, error, p, p, group.x, group.y, lane);
}

/* Dense formats (F16, F32): rows of x per threadgroup (mirrored by Q38_DENSE_T
 * in qwen38_packed_gpu.m), columns per lane chunk, most simdgroups per row. */
#define DENSE_T 4
#define DENSE_C 8
#define DENSE_S 8

/* Simdgroups that share one output row, from K alone (mirrored by
 * q38_dense_split in qwen38_packed_gpu.m): it doubles while every lane keeps
 * at least 5 chunks, up to 8. HC inject (n_out 4, K 10240) would otherwise
 * run 4 simdgroups on the whole GPU: measured 20.6 us with 1, 4.2 with 8
 * (R1.3 in docs/devlog.md). K 320 (HC up) stays at 1, where a split is slower. */
static inline uint dense_split(uint k) {
    uint s = 1;
    while (s < DENSE_S && k >= 2u * s * 32u * DENSE_C * 5u) s *= 2;
    return s;
}

/* One chunk of up to DENSE_C weights as FP32, each bit-identical to the
 * oracle's packed_value(): F32 is its bits; F16 goes through the GPU's
 * half -> float (R1.9c; whole chunks too since R2.3, which made the HC up
 * matvec 43 -> 27 us): the same bits for every finite pattern (subnormals
 * included, which Metal arithmetic would flush; test_f16_decode_matches_oracle). A whole chunk of a 16-byte
 * aligned row is one 16- or 32-byte load; a tail chunk or an unaligned row is
 * read byte by byte. The values, so the products, are the same either way. */
template <uint TYPE>
static inline void dense_chunk(device const uchar *q, bool whole, thread float *w, uint cols) {
    if (TYPE == 1 && whole) {
        const uint4 v = *(device const uint4 *)q;
        const half4 a = as_type<half4>(v.xy), b = as_type<half4>(v.zw);
        for (uint i = 0; i < 4; i++) { w[i] = float(a[i]); w[4 + i] = float(b[i]); }
    } else if (TYPE == 1) {
        for (uint i = 0; i < cols; i++) w[i] = q38_fp16(q + 2 * i);
    } else if (whole) {
        const float4 a = ((device const float4 *)q)[0], b = ((device const float4 *)q)[1];
        for (uint i = 0; i < 4; i++) { w[i] = a[i]; w[4 + i] = b[i]; }
    } else {
        for (uint i = 0; i < cols; i++)
            w[i] = as_type<float>(uint(q[4 * i]) | (uint(q[4 * i + 1]) << 8) |
                                  (uint(q[4 * i + 2]) << 16) | (uint(q[4 * i + 3]) << 24));
    }
}

/* F16 and F32 matvec: the Q8_0 scheme with a chunk of DENSE_C columns in
 * place of a block, over S = dense_split(K) simdgroups. Lane g = 32*sg + lane
 * takes chunks g, g+32S, ... (the threadgroup reads 32S consecutive chunks),
 * sums each chunk in column order and the chunk sums in chunk order;
 * simd_sum joins the 32 lanes of a simdgroup and the S partial sums are added
 * in simdgroup order. Every weight decodes exactly, so the products equal
 * the oracle's; only the order of the sum differs, and it depends only on K.
 * p and o as in q8_0_matvec. */
template <uint TYPE>
static inline void dense_matvec(device const uchar *weights, device const float *x, device float *y,
        device atomic_uint *error, constant PackedArgs &p, constant PackedArgs &o, ulong n, uint tile,
        uint lane, uint sg, uint sgs, threadgroup float *part) {
    const uint E = TYPE == 1 ? 2 : 4, S = dense_split(p.k_logical); /* E: bytes per element */
    /* R2.3: with one simdgroup per row (S = 1, K < 2560: HC up, K 320) the single
     * kernels pack sgs rows per threadgroup (row group*sgs + sg): with the
     * hardware fp16 decode, 32-thread threadgroups gave 32.4 us, 8 rows 27.1 us
     * (HC up, 20 encodes). A row's arithmetic is unchanged. */
    if (S == 1 && sgs > 1) { n = n * sgs + sg; sg = 0; sgs = 1; }
    const ulong m0 = ulong(tile) * DENSE_T;
    if (n >= o.n_out || m0 >= p.rows) return; /* uniform over the threadgroup, as every return below */
    const uint mt = uint(min(ulong(DENSE_T), p.rows - m0));
    const ulong row = o.weight_offset + n * o.row_bytes;
    bool inside = sgs == S && row <= o.weight_end && o.row_bytes <= o.weight_end - row;
    ulong xi[DENSE_T];
    for (uint t = 0; t < DENSE_T; t++) {
        const ulong m = m0 + (t < mt ? t : 0);
        xi[t] = m * (p.x_stride / 4);
        inside = inside && xi[t] <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xi[t] &&
                 m * (o.y_stride / 4) + n < o.y_bytes / 4;
    }
    if (!inside) { /* also a dispatch whose simdgroups per row differ from dense_split(K) */
        if (!lane && !sg) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    const bool aligned = (row & 15) == 0; /* chunks are 16 (F16) or 32 (F32) bytes */
    float acc[DENSE_T];
    for (uint t = 0; t < DENSE_T; t++) acc[t] = 0.0f;
    const uint chunks = (p.k_logical + DENSE_C - 1) / DENSE_C;
    for (uint c = sg * 32 + lane; c < chunks; c += 32 * S) {
        /* Only the last chunk may be partial: its columns past k_logical are
         * read neither from the weights nor from x. */
        const uint cols = min(uint(DENSE_C), p.k_logical - c * DENSE_C);
        float w[DENSE_C];
        dense_chunk<TYPE>(weights + row + ulong(c) * DENSE_C * E, aligned && cols == DENSE_C, w, cols);
        for (uint t = 0; t < DENSE_T; t++) {
            if (t >= mt) break; /* uniform: decode (one row of x) skips the empty slots */
            device const float *xb = x + xi[t] + c * DENSE_C;
            float s = 0.0f;
            if (cols == DENSE_C)
                for (uint i = 0; i < DENSE_C; i++) s += w[i] * xb[i];
            else
                for (uint i = 0; i < cols; i++) s += w[i] * xb[i];
            acc[t] += s;
        }
    }
    float total[DENSE_T];
    for (uint t = 0; t < DENSE_T; t++) total[t] = simd_sum(acc[t]);
    if (S > 1) {
        if (!lane)
            for (uint t = 0; t < DENSE_T; t++) part[sg * DENSE_T + t] = total[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg) return;
        for (uint t = 0; t < DENSE_T; t++) {
            total[t] = part[t];
            for (uint i = 1; i < S; i++) total[t] += part[i * DENSE_T + t];
        }
    }
    if (lane) return;
    /* Finiteness on the output (bit 1), as in the Q8_0 kernel. */
    for (uint t = 0; t < mt; t++) {
        y[(m0 + t) * (o.y_stride / 4) + n] = total[t];
        if (!isfinite(total[t])) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
kernel void q38_matvec_f16(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float part[DENSE_S * DENSE_T];
    dense_matvec<1>(weights, x, y, error, p, p, group.x, group.y, lane, sg, sgs, part);
}
kernel void q38_matvec_f32(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float part[DENSE_S * DENSE_T];
    dense_matvec<0>(weights, x, y, error, p, p, group.x, group.y, lane, sg, sgs, part);
}

/* Multi-GEMV (R1.4): up to MULTI_MAX projections of the same x, one format,
 * in one dispatch (idea: ds4 fc80bd6, MIT, metal/qwen4.metal:
 * kernel_qwen4_multi_gemv). Threadgroup column g of the grid belongs to
 * projection j with first[j] <= g < first[j] + p[j].n_out and computes its
 * local row g - first[j] with the single kernel's own row function above, so
 * every output is bit-identical to a single call: the order inside a row
 * still depends only on K and the format. Fixed slots (no argument buffer):
 * weights in 0..7, outputs in 8..15, so Metal tracks every buffer as usual;
 * the host binds unused slots to the error word, never read here.
 * The x side comes from p[0] (all projections share x), only the tensor side
 * from p[j], and j without a loop: a dynamic index on every field, or a
 * walk, made each short threadgroup ~5% slower (R1.4 in docs/devlog.md). */
#define MULTI_MAX 8
struct MultiArgs { uint count, first[MULTI_MAX]; };
#define MULTI_SLOT(i) device const uchar *w##i [[buffer(i)]], device float *y##i [[buffer(8 + i)]]
#define MULTI_PARAMS MULTI_SLOT(0), MULTI_SLOT(1), MULTI_SLOT(2), MULTI_SLOT(3), MULTI_SLOT(4), \
    MULTI_SLOT(5), MULTI_SLOT(6), MULTI_SLOT(7), device const float *x [[buffer(16)]], \
    device atomic_uint *error [[buffer(17)]], constant PackedArgs *p [[buffer(18)]], \
    constant MultiArgs &m [[buffer(19)]], uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]
#define MULTI_CASE(i) case i: w = w##i; y = y##i; break;
/* Projection of grid column g (uniform over the threadgroup) and its buffers. */
#define MULTI_PICK \
    uint j = 0; \
    for (uint i = 1; i < MULTI_MAX; i++) j += i < m.count && group.x >= m.first[i]; \
    device const uchar *w = w0; device float *y = y0; \
    switch (j) { MULTI_CASE(1) MULTI_CASE(2) MULTI_CASE(3) MULTI_CASE(4) MULTI_CASE(5) MULTI_CASE(6) MULTI_CASE(7) }
kernel void q38_matvec_multi_q8_0(MULTI_PARAMS) {
    MULTI_PICK
    q8_0_matvec<false>(w, x, y, error, p[0], p[j], group.x - m.first[j], group.y, lane);
}
kernel void q38_matvec_multi_q8_0_dec(MULTI_PARAMS) {
    MULTI_PICK
    q8_0_matvec<true>(w, x, y, error, p[0], p[j], group.x - m.first[j], group.y, lane);
}
kernel void q38_matvec_multi_f16(MULTI_PARAMS, uint sg [[simdgroup_index_in_threadgroup]],
        uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float part[DENSE_S * DENSE_T];
    MULTI_PICK
    dense_matvec<1>(w, x, y, error, p[0], p[j], group.x - m.first[j], group.y, lane, sg, sgs, part);
}
kernel void q38_matvec_multi_f32(MULTI_PARAMS, uint sg [[simdgroup_index_in_threadgroup]],
        uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float part[DENSE_S * DENSE_T];
    MULTI_PICK
    dense_matvec<0>(w, x, y, error, p[0], p[j], group.x - m.first[j], group.y, lane, sg, sgs, part);
}

/* Indexed expert matvecs (R1.5): y[p, n] = row n of expert ids[p] . x[p / x_divisor],
 * the contract of q38_packed_linear_indexed. One simdgroup per output
 * element (p, n), IDX_SG of them along n per threadgroup (mirrored by
 * Q38_IDX_SG in qwen38_packed_gpu.m); the dot is q38_quant_row<F> of
 * qwen38_quant_decode.h, whose order depends only on K and the format, so
 * the bits of (p, n) depend neither on the number of slots or tokens nor on
 * the slot, the expert id or the x divisor. Bounds as the oracle: an expert
 * id past the tensor, or a weight row, x row or y element outside its
 * extent, sets bit 2 and that element is not written; a nonfinite output
 * sets bit 1. (Idea of one cooperative kernel per routed format: ds4
 * fc80bd6, MIT, metal/qwen4.metal: kernel_qwen4_moe_mid / moe_down.) */
struct IndexedArgs { PackedArgs base; ulong expert_bytes; uint experts, x_divisor; };
static_assert(sizeof(IndexedArgs) == 96, "indexed wire record size");
#define IDX_SG 4
template <typename F>
static inline void indexed_matvec(device const uchar *weights, device const float *x, device float *y,
        device atomic_uint *error, constant IndexedArgs &a, device const uint *ids, uint2 group, uint sg,
        uint lane) {
    constant PackedArgs &p = a.base;
    const ulong n = ulong(group.x) * IDX_SG + sg;
    const uint m = group.y;
    if (n >= p.n_out || m >= p.rows) return; /* uniform over the simdgroup, as every return below */
    const uint expert = ids[m];
    const ulong row = p.weight_offset + ulong(expert) * a.expert_bytes + n * p.row_bytes;
    const ulong xi = ulong(m / a.x_divisor) * (p.x_stride / 4), yi = ulong(m) * (p.y_stride / 4) + n;
    if (expert >= a.experts || row > p.weight_end || p.row_bytes > p.weight_end - row ||
        xi > p.x_bytes / 4 || p.k_logical > p.x_bytes / 4 - xi || yi >= p.y_bytes / 4) {
        if (!lane) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    const float v = q38_quant_row<F>(weights + row, x + xi, p.k_logical, lane);
    if (lane) return;
    y[yi] = v;
    if (!isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}
#define INDEXED_KERNEL(name, F) \
kernel void name(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]], \
        device float *y [[buffer(2)]], device atomic_uint *error [[buffer(3)]], \
        constant IndexedArgs &a [[buffer(4)]], device const uint *ids [[buffer(5)]], \
        uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    indexed_matvec<F>(weights, x, y, error, a, ids, group, sg, lane); \
}
INDEXED_KERNEL(q38_matvec_indexed_iq2_xxs, Q38Iq2Xxs)
INDEXED_KERNEL(q38_matvec_indexed_q2_k, Q38Q2K)
INDEXED_KERNEL(q38_matvec_indexed_q4_k, Q38Q4K)
INDEXED_KERNEL(q38_matvec_indexed_mxfp4, Q38Mxfp4)

/* R2.6 fused HC read tail: HC_LOW, the F16 up projection, HC_READ and
 * HC_INJECT of one HC site in one dispatch, bit-identical to that fast
 * sequence (q38c_activate SILU then SIGMOID x2, the dense_matvec rows above,
 * q38c_hc_read):
 *   act[r] = silu(low[r] / h)           (every threadgroup, in threadgroup memory)
 *   gate[k*E + d] = up row k*E + d . act (the h rows of one d, one simdgroup)
 *   mixed[d] = sum_k normed[k*E + d] sigmoid(gate[k*E + d]) / h
 *   inject[j] = 2 sigmoid(il[j] / h)    (threadgroup 0; il absent: no inject)
 * Each up row keeps dense_matvec's order at S = 1: lane l sums chunks l, l+32,
 * ... of DENSE_C columns, each in column order, then simd_sum. The h rows'
 * tail chunks (chunks % 32 each; K 320: 8 of 40) share rounds of the 32 lanes
 * and each sum goes back to the lane that adds it last, as in the fused
 * gate+up, so the 4 rows of HC up take 5 rounds instead of 8. act and gate
 * are written as in the sequence (threadgroup 0 writes act). A nonfinite
 * low, act, il, inject, gate, normed or sum sets bit 1; outputs are then
 * unspecified (the sequence's checks stop at the first failing op). Wire
 * record HcTailArgs mirrors Q38HcTailArgs (qwen38_b_internal.h); row pitches
 * in floats. (Idea of one up + gate + mix kernel per HC read: ds4 fc80bd6,
 * MIT, metal/qwen4.metal: kernel_qwen4_hc_gate_mix.) */
#define HC_TAIL_MAX_K 1024
#define HC_TAIL_MAX_H 4
struct HcTailArgs { PackedArgs up; uint streams, width, inject, reserved; ulong low, normed, mixed, il, out; };
static_assert(sizeof(HcTailArgs) == 136, "HC tail wire record size");
/* q38c_activate's sigmoid (qwen38_recurrent.metal): the same FP32 operations. */
static float hc_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float z = exp(x); return z / (1.0f + z);
}
static inline float hc_chunk(device const uchar *row, bool aligned, threadgroup const float *xs, uint k, uint c) {
    const uint cols = min(uint(DENSE_C), k - c * DENSE_C);
    float w[DENSE_C];
    dense_chunk<1>(row + ulong(c) * DENSE_C * 2, aligned && cols == DENSE_C, w, cols);
    threadgroup const float *xb = xs + c * DENSE_C;
    float s = 0.0f;
    if (cols == DENSE_C)
        for (uint i = 0; i < DENSE_C; i++) s += w[i] * xb[i];
    else
        for (uint i = 0; i < cols; i++) s += w[i] * xb[i];
    return s;
}
kernel void q38_hc_tail_f16(device const uchar *weights [[buffer(0)]], device const float *low [[buffer(1)]],
        device const float *normed [[buffer(2)]], device const float *il [[buffer(3)]],
        device float *act [[buffer(4)]], device float *gate [[buffer(5)]], device float *mixed [[buffer(6)]],
        device float *inject [[buffer(7)]], device atomic_uint *error [[buffer(8)]],
        constant HcTailArgs &a [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
        uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float xs[HC_TAIL_MAX_K];
    const uint t = group.y, k = a.up.k_logical, h = a.streams, E = a.width, tid = sg * 32 + lane;
    bool bad = false;
    for (uint r = tid; r < k; r += sgs * 32) { /* HC_LOW: p1 = h, p2 = 1 */
        const float v = low[t * a.low + r];
        float x = v / float(h), y = x * hc_sigmoid(x);
        y *= float(1u);
        bad = bad || !isfinite(v) || !isfinite(y);
        xs[r] = y;
        if (!group.x) act[t * (a.up.x_stride / 4) + r] = y;
    }
    if (!group.x && a.inject && tid < h) { /* HC_INJECT: p1 = h, p2 = 2 */
        const float v = il[t * a.il + tid];
        float y = hc_sigmoid(v / float(h));
        y *= float(2u);
        bad = bad || !isfinite(v) || !isfinite(y);
        inject[t * a.out + tid] = y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = group.x * sgs + sg;
    if (d < E) { /* uniform over the simdgroup */
        const uint chunks = (k + DENSE_C - 1) / DENSE_C, head = chunks / 32 * 32, tail = chunks - head;
        device const uchar *row[HC_TAIL_MAX_H];
        bool aligned[HC_TAIL_MAX_H];
        float acc[HC_TAIL_MAX_H], pooled[HC_TAIL_MAX_H];
        for (uint r = 0; r < HC_TAIL_MAX_H; r++) {
            const ulong at = a.up.weight_offset + ulong(r < h ? r * E + d : d) * a.up.row_bytes;
            row[r] = weights + at; aligned[r] = (at & 15) == 0; acc[r] = 0.0f;
        }
        for (uint c = lane; c < head; c += 32)
            for (uint r = 0; r < h; r++) acc[r] += hc_chunk(row[r], aligned[r], xs, k, c);
        for (uint j = 0; j < HC_TAIL_MAX_H; j++) {
            const uint i = 32 * j + lane, r = tail ? i / tail : 0;
            pooled[j] = i < h * tail ? hc_chunk(row[r], aligned[r], xs, k, head + i % tail) : 0.0f;
        }
        float sum = 0.0f;
        for (uint r = 0; r < h; r++) {
            const uint i = r * tail + lane;
            float v = 0.0f;
            for (uint j = 0; j < HC_TAIL_MAX_H; j++) { /* every lane shuffles: uniform control */
                const float s = simd_shuffle(pooled[j], ushort(i % 32));
                if (i / 32 == j) v = s;
            }
            if (lane < tail) acc[r] += v;
            const float g = simd_sum(acc[r]);
            if (!lane) { /* HC_READ, k = r in ascending order */
                const float n = normed[t * a.normed + r * E + d];
                gate[t * (a.up.y_stride / 4) + r * E + d] = g;
                sum += n * hc_sigmoid(g);
                bad = bad || !isfinite(g) || !isfinite(n);
            }
        }
        if (!lane) {
            bad = bad || !isfinite(sum);
            mixed[t * a.mixed + d] = sum / float(h);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

/* Decode III: q38_hc_tail_f16 with a lane map for K 320. There one simdgroup's
 * 32 lanes split each up row into 8-column chunks, ~1.25 chunk per lane and
 * row, and every row paid a 5-step simd_sum: 155 GB/s in the decode graph.
 * Here lanes 8s..8s+7 own stream s (h <= 4): lane i sums chunks i, i+8, ...
 * of row s*E + d with 16-byte loads (8 lanes read 128 contiguous bytes), then
 * a 3-step xor tree; lane 0 gathers the h gates and does HC_READ in ascending
 * stream order as before. Same prologue (HC_LOW into threadgroup memory, act
 * and inject written by threadgroup 0), same outputs and error bits; only the
 * order of each gate's sum changes (tolerance in the fast-kernel tests).
 * Needs k % 8 == 0 and 16-byte aligned rows (host-checked; else the R2.6
 * kernel). Lane map idea: ds4 fc80bd6, MIT, metal/qwen4.metal:
 * kernel_qwen4_hc_gate_mix. Q38_REF_KERNELS=exact keeps q38_hc_tail_f16. */
kernel void q38_hc_tail8_f16(device const uchar *weights [[buffer(0)]], device const float *low [[buffer(1)]],
        device const float *normed [[buffer(2)]], device const float *il [[buffer(3)]],
        device float *act [[buffer(4)]], device float *gate [[buffer(5)]], device float *mixed [[buffer(6)]],
        device float *inject [[buffer(7)]], device atomic_uint *error [[buffer(8)]],
        constant HcTailArgs &a [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
        uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float xs[HC_TAIL_MAX_K];
    const uint t = group.y, k = a.up.k_logical, h = a.streams, E = a.width, tid = sg * 32 + lane;
    bool bad = false;
    for (uint r = tid; r < k; r += sgs * 32) { /* HC_LOW: p1 = h, p2 = 1 */
        const float v = low[t * a.low + r];
        float x = v / float(h), y = x * hc_sigmoid(x);
        y *= float(1u);
        bad = bad || !isfinite(v) || !isfinite(y);
        xs[r] = y;
        if (!group.x) act[t * (a.up.x_stride / 4) + r] = y;
    }
    if (!group.x && a.inject && tid < h) { /* HC_INJECT: p1 = h, p2 = 2 */
        const float v = il[t * a.il + tid];
        float y = hc_sigmoid(v / float(h));
        y *= float(2u);
        bad = bad || !isfinite(v) || !isfinite(y);
        inject[t * a.out + tid] = y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = group.x * sgs + sg;
    if (d < E) { /* uniform over the simdgroup */
        const uint st = lane / 8, li = lane % 8, chunks = k / 8;
        float acc = 0.0f;
        if (st < h) {
            device const half4 *row = (device const half4 *)(weights + a.up.weight_offset + ulong(st * E + d) * a.up.row_bytes);
            for (uint c = li; c < chunks; c += 8) {
                const half4 w0 = row[2 * c], w1 = row[2 * c + 1];
                threadgroup const float *xb = xs + 8 * c;
                float sm = 0.0f;
                for (uint i = 0; i < 4; i++) sm += float(w0[i]) * xb[i];
                for (uint i = 0; i < 4; i++) sm += float(w1[i]) * xb[4 + i];
                acc += sm;
            }
        }
        acc += simd_shuffle_xor(acc, ushort(1));
        acc += simd_shuffle_xor(acc, ushort(2));
        acc += simd_shuffle_xor(acc, ushort(4));
        float g[HC_TAIL_MAX_H];
        for (uint r = 0; r < HC_TAIL_MAX_H; r++) g[r] = simd_shuffle(acc, ushort(8 * r));
        if (!lane) {
            float sum = 0.0f;
            for (uint r = 0; r < h; r++) { /* HC_READ, k = r in ascending order */
                const float n = normed[t * a.normed + r * E + d];
                gate[t * (a.up.y_stride / 4) + r * E + d] = g[r];
                sum += n * hc_sigmoid(g[r]);
                bad = bad || !isfinite(g[r]) || !isfinite(n);
            }
            bad = bad || !isfinite(sum);
            mixed[t * a.mixed + d] = sum / float(h);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

/* R3.1 prefill GEMM: Y[m,n] = sum_k W[n,k] X[m,k] for many rows of x at once.
 * The matvecs above reread the whole matrix every Q8_T/DENSE_T rows, so a
 * 1024-row chunk read each weight ~256 times. Here a threadgroup of 4
 * simdgroups owns a GEMM_M x GEMM_N tile of Y and walks K in GEMM_K steps:
 * it decodes GEMM_N weight rows to FP32 into threadgroup memory once and
 * reuses them for GEMM_M rows of x, with 8x8 FP32 simdgroup matrix products.
 * Every weight still decodes exactly (as the matvecs), so only the order of
 * the sum changes; it depends on K alone, never on M or on the row's place
 * in the tile (padding adds +0 products). Used from GEMM_MIN_ROWS rows of x
 * (qwen38_packed_gpu.m); below, decode and MTP keep the matvecs bit for bit. */
#define GEMM_M 32
#define GEMM_N 64
#define GEMM_K 32
#define GEMM_THREADS 128

/* Weight (n, c) of a dense row as FP32: F16 through the hardware conversion
 * on a 2-byte aligned row (exact, subnormals included, as dense_chunk), else
 * byte by byte; F32 is its bits. */
template <uint TYPE>
static inline float gemm_dense(device const uchar *row, bool aligned, uint c) {
    if (TYPE == 1) return aligned ? float(as_type<half>(((device const ushort *)row)[c])) : q38_fp16(row + 2 * c);
    device const uchar *q = row + 4 * c;
    return as_type<float>(uint(q[0]) | (uint(q[1]) << 8) | (uint(q[2]) << 16) | (uint(q[3]) << 24));
}

template <uint TYPE>
static inline void gemm(device const uchar *weights, device const float *x, device float *y,
        device atomic_uint *error, constant PackedArgs &p, uint2 group, uint tid, uint sg,
        threadgroup float *xs, threadgroup float *ws, threadgroup float *cs) {
    const ulong n0 = ulong(group.x) * GEMM_N, m0 = ulong(group.y) * GEMM_M;
    if (n0 >= p.n_out || m0 >= p.rows) return; /* uniform over the threadgroup, as every return below */
    const uint nt = uint(min(ulong(GEMM_N), p.n_out - n0)), mt = uint(min(ulong(GEMM_M), p.rows - m0));
    /* Bounds as the matvecs, on the tile's last weight row, x row and y
     * element: offsets grow with n and m, so the rest lie inside too. */
    const ulong last = p.weight_offset + (n0 + nt - 1) * p.row_bytes, xl = (m0 + mt - 1) * (p.x_stride / 4);
    const bool inside = last <= p.weight_end && p.row_bytes <= p.weight_end - last &&
        xl <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xl &&
        (m0 + mt - 1) * (p.y_stride / 4) + n0 + nt - 1 < p.y_bytes / 4;
    if (!inside) {
        if (!tid) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    /* Simdgroup sg: rows (sg/2)*16 .. +16, columns (sg%2)*32 .. +32 of the tile. */
    const uint sr = (sg / 2) * 16, sc = (sg % 2) * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k_logical; k0 += GEMM_K) {
        /* Columns past k_logical are read neither from x nor from the weights:
         * zeros in the tile. Rows past mt / nt are zeros as well. */
        const uint kc = min(uint(GEMM_K), p.k_logical - k0);
        for (uint i = tid; i < GEMM_M * GEMM_K; i += GEMM_THREADS) {
            const uint r = i / GEMM_K, c = i % GEMM_K;
            xs[i] = r < mt && c < kc ? x[(m0 + r) * (p.x_stride / 4) + k0 + c] : 0.0f;
        }
        if (TYPE == 8) { /* one Q8_0 block per weight row; two threads per block */
            const uint r = tid / 2, h = (tid % 2) * 16;
            device const uchar *q = weights + p.weight_offset + (n0 + r) * p.row_bytes + ulong(k0 / 32) * 34;
            const float d = r < nt ? q38_fp16(q) : 0.0f;
            for (uint i = 0; i < 16; i++) {
                const uint c = h + i;
                ws[r * GEMM_K + c] = r < nt && c < kc ? d * float(as_type<char>(q[2 + c])) : 0.0f;
            }
        } else {
            for (uint i = tid; i < GEMM_N * GEMM_K; i += GEMM_THREADS) {
                const uint r = i / GEMM_K, c = i % GEMM_K;
                device const uchar *row = weights + p.weight_offset + (n0 + r) * p.row_bytes;
                ws[i] = r < nt && c < kc ? gemm_dense<TYPE>(row, (p.weight_offset + (n0 + r) * p.row_bytes) % 2 == 0, k0 + c) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < GEMM_K; kk += 8) {
            simdgroup_float8x8 a[2], b[4];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], xs + (sr + i * 8) * GEMM_K + kk, GEMM_K);
            /* ws holds W rows (n major); loaded transposed it is the k x n operand. */
            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], ws + (sc + j * 8) * GEMM_K + kk, GEMM_K, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], cs + (sr + i * 8) * GEMM_N + sc + j * 8, GEMM_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* Finiteness on the output (bit 1), as in the matvecs. */
    for (uint i = tid; i < GEMM_M * GEMM_N; i += GEMM_THREADS) {
        const uint r = i / GEMM_N, c = i % GEMM_N;
        if (r >= mt || c >= nt) continue;
        const float v = cs[i];
        y[(m0 + r) * (p.y_stride / 4) + n0 + c] = v;
        if (!isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
#define GEMM_KERNEL(name, TYPE) \
kernel void name(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]], \
        device float *y [[buffer(2)]], device atomic_uint *error [[buffer(3)]], \
        constant PackedArgs &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], \
        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float xs[GEMM_M * GEMM_K], ws[GEMM_N * GEMM_K], cs[GEMM_M * GEMM_N]; \
    gemm<TYPE>(weights, x, y, error, p, group, tid, sg, xs, ws, cs); \
}
GEMM_KERNEL(q38_gemm_q8_0, 8)
GEMM_KERNEL(q38_gemm_f16, 1)
GEMM_KERNEL(q38_gemm_f32, 0)

/* ---- R3.5 reduced-precision prefill GEMM on the Neural Accelerators (Metal 4
 * TensorOps, matmul2d). Same product as gemm<TYPE> above, but both operands
 * are staged as half (weights decoded exactly, then rounded to half; x rounded
 * to half) and multiplied by the tensor units with FP32 accumulation. It is a
 * different numeric mode, not a reordering: the input rounding (2^-11
 * relative per operand) is the dominant error, so it has its own gate
 * (tests/qwen38_kernel_tolerances.json, "nax_half") and the FP32 GEMM stays
 * its oracle. Only the prefill uses it (PICOFORGE_PREFILL_PRECISION, from
 * GEMM_MIN_ROWS rows); decode never does. Layout idea from DwarfStar fc80bd6
 * (MIT), metal/qwen4.metal kernel_qwen4_moe_mm_mid_nax_t; checked by
 * tools/probe/tensorops_gemm_probe. matmul2d constraints: its k (NAX_K, 32)
 * is a fixed step and a shorter K is not padded by the op, so the K tail is
 * zero-padded here; M and N are multiples of 16. |x| above 65504 becomes inf
 * in half and trips the finiteness bit (error 1): the FP32 path is then the
 * way out. */
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define NAX_M 64
#define NAX_N 64
#define NAX_K 32
#define NAX_THREADS 128
using NaxMatmul = mpp::tensor_ops::matmul2d<
    mpp::tensor_ops::matmul2d_descriptor(NAX_M, NAX_N, NAX_K, false, true, false,
                                         mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
    execution_simdgroups<4>>;

template <uint TYPE>
static inline void gemm_nax(device const uchar *weights, device const float *x, device float *y,
        device atomic_uint *error, constant PackedArgs &p, uint2 group, uint tid,
        threadgroup half *xs, threadgroup half *ws, threadgroup float *cs) {
    const ulong n0 = ulong(group.x) * NAX_N, m0 = ulong(group.y) * NAX_M;
    if (n0 >= p.n_out || m0 >= p.rows) return; /* uniform over the threadgroup */
    const uint nt = uint(min(ulong(NAX_N), p.n_out - n0)), mt = uint(min(ulong(NAX_M), p.rows - m0));
    const ulong last = p.weight_offset + (n0 + nt - 1) * p.row_bytes, xl = (m0 + mt - 1) * (p.x_stride / 4);
    const bool inside = last <= p.weight_end && p.row_bytes <= p.weight_end - last &&
        xl <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xl &&
        (m0 + mt - 1) * (p.y_stride / 4) + n0 + nt - 1 < p.y_bytes / 4;
    if (!inside) {
        if (!tid) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    /* left: x tile, k contiguous, one row of x per column; right: W rows
     * (n major) read transposed; destination extents (N, M) as cs[m][n]. */
    auto tx = tensor(xs, dextents<int32_t, 2>(NAX_K, NAX_M));
    auto tw = tensor(ws, dextents<int32_t, 2>(NAX_K, NAX_N));
    NaxMatmul mm;
    /* float4 loads of x need 16-byte rows and a 16-byte aligned binding; the host sets
     * p.reserved = 1 for the NAX dispatch when the x offset is aligned (else 0). */
    const bool vec = p.reserved == 1 && p.x_stride % 16 == 0;
    auto acc = mm.get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (uint k0 = 0; k0 < p.k_logical; k0 += NAX_K) {
        const uint kc = min(uint(NAX_K), p.k_logical - k0);
        if (vec && kc == NAX_K) { /* 4 columns per item: float4 in, half4 out */
            for (uint i = tid; i < NAX_M * NAX_K / 4; i += NAX_THREADS) {
                const uint r = i / (NAX_K / 4), c = (i % (NAX_K / 4)) * 4;
                const float4 v = r < mt ? *(device const float4 *)(x + (m0 + r) * (p.x_stride / 4) + k0 + c) : float4(0.0f);
                *(threadgroup half4 *)(xs + r * NAX_K + c) = half4(v);
            }
        } else {
            for (uint i = tid; i < NAX_M * NAX_K; i += NAX_THREADS) {
                const uint r = i / NAX_K, c = i % NAX_K;
                xs[i] = r < mt && c < kc ? half(x[(m0 + r) * (p.x_stride / 4) + k0 + c]) : 0.0h;
            }
        }
        if (TYPE == 8) { /* one Q8_0 block per weight row; two threads per block */
            const uint r = tid / 2, h = (tid % 2) * 16;
            device const uchar *q = weights + p.weight_offset + (n0 + r) * p.row_bytes + ulong(k0 / 32) * 34;
            const float d = r < nt ? q38_fp16(q) : 0.0f;
            for (uint i = 0; i < 16; i++) {
                const uint c = h + i;
                ws[r * NAX_K + c] = r < nt && c < kc ? half(d * float(as_type<char>(q[2 + c]))) : 0.0h;
            }
        } else {
            for (uint i = tid; i < NAX_N * NAX_K; i += NAX_THREADS) {
                const uint r = i / NAX_K, c = i % NAX_K;
                device const uchar *row = weights + p.weight_offset + (n0 + r) * p.row_bytes;
                ws[i] = r < nt && c < kc ? half(gemm_dense<TYPE>(row, (p.weight_offset + (n0 + r) * p.row_bytes) % 2 == 0, k0 + c)) : 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sx = tx.slice(0, 0);
        auto sw = tw.slice(0, 0);
        mm.run(sx, sw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tc = tensor(cs, dextents<int32_t, 2>(NAX_N, NAX_M));
    auto sc = tc.slice(0, 0);
    acc.store(sc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < NAX_M * NAX_N; i += NAX_THREADS) {
        const uint r = i / NAX_N, c = i % NAX_N;
        if (r >= mt || c >= nt) continue;
        const float v = cs[i];
        y[(m0 + r) * (p.y_stride / 4) + n0 + c] = v;
        if (!isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
#define GEMM_NAX_KERNEL(name, TYPE) \
kernel void name(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]], \
        device float *y [[buffer(2)]], device atomic_uint *error [[buffer(3)]], \
        constant PackedArgs &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], \
        uint tid [[thread_index_in_threadgroup]]) { \
    /* The FP32 result tile reuses the staging area after the K loop (16 KB, not 24): two \
     * threadgroups fit per core. */ \
    threadgroup float smem[NAX_M * NAX_N]; \
    threadgroup half *xs = (threadgroup half *)smem, *ws = xs + NAX_M * NAX_K; \
    gemm_nax<TYPE>(weights, x, y, error, p, group, tid, xs, ws, smem); \
}
GEMM_NAX_KERNEL(q38_gemm_nax_q8_0, 8)
GEMM_NAX_KERNEL(q38_gemm_nax_f16, 1)
GEMM_NAX_KERNEL(q38_gemm_nax_f32, 0)
