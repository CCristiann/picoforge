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
    q8_0_matvec(weights, x, y, error, p, p, group.x, group.y, lane);
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
 * oracle's packed_value(): F32 is its bits, F16 is q38_fp16_bits (exact on
 * subnormals, which Metal arithmetic would flush). A whole chunk of a 16-byte
 * aligned row is one 16- or 32-byte load; a tail chunk or an unaligned row is
 * read byte by byte. The values, so the products, are the same either way. */
template <uint TYPE>
static inline void dense_chunk(device const uchar *q, bool whole, thread float *w, uint cols) {
    if (TYPE == 1 && whole) {
        const uint4 v = *(device const uint4 *)q;
        for (uint i = 0; i < 4; i++) {
            w[2 * i] = q38_fp16_bits(v[i] & 0xffffu);
            w[2 * i + 1] = q38_fp16_bits(v[i] >> 16);
        }
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

