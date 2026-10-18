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
 * Each decoded block is reused for the tile's Q8_T rows of x. */
kernel void q38_matvec_q8_0(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const ulong n = group.x, m0 = ulong(group.y) * Q8_T;
    if (n >= p.n_out || m0 >= p.rows) return;
    const uint mt = uint(min(ulong(Q8_T), p.rows - m0));
    /* Bounds as the oracle, once per row: a weight row, x row or y element
     * outside its extent sets bit 2 and the simdgroup reads and writes
     * nothing. Tile slots past the last row of x repeat row m0, unused. */
    const ulong row = p.weight_offset + n * p.row_bytes;
    bool inside = row <= p.weight_end && p.row_bytes <= p.weight_end - row;
    ulong xi[Q8_T];
    for (uint t = 0; t < Q8_T; t++) {
        const ulong m = m0 + (t < mt ? t : 0);
        xi[t] = m * (p.x_stride / 4);
        inside = inside && xi[t] <= p.x_bytes / 4 && p.k_logical <= p.x_bytes / 4 - xi[t] &&
                 m * (p.y_stride / 4) + n < p.y_bytes / 4;
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
        y[(m0 + t) * (p.y_stride / 4) + n] = v;
        if (!isfinite(v)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
