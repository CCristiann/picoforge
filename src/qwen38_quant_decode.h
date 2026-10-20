/* Decode helpers shared by the R1 fast kernels (ruling R3). The oracle in
 * qwen38_packed.metal keeps its own copy and stays untouched. Metal only. */
#ifndef PICOFORGE_QWEN38_QUANT_DECODE_H
#define PICOFORGE_QWEN38_QUANT_DECODE_H
#include <metal_stdlib>

/* fp16 bits -> FP32, exact for all 65536 patterns: signed zero, subnormals,
 * Inf and NaN payloads. The bits are built by hand like the oracle's
 * half_value(), because Metal arithmetic flushes subnormals. A subnormal
 * fraction f is f * 2^-24, a normal FP32: shift its top bit to bit 10 (the
 * implicit one) and lower the exponent by the shift; clz replaces the
 * oracle's loop and gives the same bits (test_q8_0_scale_decode_matches_oracle). */
static inline float q38_fp16_bits(uint h) {
    uint sign = (h & 0x8000u) << 16, exponent = (h >> 10) & 31u, fraction = h & 1023u;
    if (!exponent) {
        if (!fraction) return as_type<float>(sign);
        uint shift = metal::clz(fraction) - 21u;
        return as_type<float>(sign | ((113u - shift) << 23) | (((fraction << shift) & 1023u) << 13));
    }
    return as_type<float>(sign | ((exponent == 31u ? 255u : exponent + 112u) << 23) | (fraction << 13));
}
static inline float q38_fp16(device const uchar *p) { return q38_fp16_bits(uint(p[0]) | (uint(p[1]) << 8)); }

/* ---- Block formats of the routed experts (R1.5) ----
 * Each format decodes one CHUNK of consecutive columns of a row into FP32,
 * every value bit-identical to the oracle's packed_value() (same FP32
 * operations in the same order, contraction off in every including file),
 * and q38_quant_row<F> is the cooperative dot of one row with one x row:
 * lane l takes chunks l, l+32, ..., sums each chunk in column order and its
 * chunks in chunk order, and simd_sum joins the 32 lanes. That order is a
 * function of K and the format only, never of the rows, slots or expert of
 * the dispatch. The indexed matvecs call it today; the fused MoE kernels of
 * Tasks 12-13 call the same function, so their outputs stay bit-identical. */
static inline uint q38_le32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
static constant ulong q38_iq2_grid[256] = {
#include "qwen38_iq2_grid.inc"
};
/* IQ2_XXS (66-byte blocks of 256): fp16 d, then 8 groups of 32 columns, each
 * 4 grid indices (8 magnitudes each) + a word of 4x7 sign bits (the 8th by
 * even parity) and a 4-bit scale. Chunk = one group. */
struct Q38Iq2Xxs {
    static constant constexpr uint CHUNK = 32;
    static inline void decode(device const uchar *row, uint c, thread float *w) {
        device const uchar *b = row + ulong(c / 8) * 66, *g = b + 2 + (c % 8) * 8;
        const uint bits = q38_le32(g + 4);
        const float db = (q38_fp16(b) * float(2 * (bits >> 28) + 1)) * 0.125f;
        for (uint s = 0; s < 4; s++) {
            uint signs = (bits >> (7 * s)) & 127u;
            signs |= (metal::popcount(signs) & 1u) << 7;
            const ulong grid = q38_iq2_grid[g[s]];
            for (uint i = 0; i < 8; i++) {
                const float v = db * float(uint(grid >> (8 * i)) & 255u);
                w[8 * s + i] = signs & (1u << i) ? -v : v;
            }
        }
    }
};
/* Q2_K (84-byte blocks of 256): 16 scale bytes (low nibble scale, high
 * nibble min, 16 columns each), 64 bytes of 2-bit codes (two halves of 128
 * columns, four bit planes of 32), fp16 d and dmin at 80 and 82. Chunk = the
 * 16 columns of one scale byte. */
struct Q38Q2K {
    static constant constexpr uint CHUNK = 16;
    static inline void decode(device const uchar *row, uint c, thread float *w) {
        device const uchar *b = row + ulong(c / 16) * 84;
        const uint j = c % 16, scale = b[j], shift = 2 * ((j % 8) / 2);
        const float dl = q38_fp16(b + 80) * float(scale & 15u), ml = q38_fp16(b + 82) * float(scale >> 4);
        device const uchar *q = b + 16 + (j / 8) * 32 + (j % 2) * 16;
        for (uint i = 0; i < 16; i++) w[i] = dl * float((q[i] >> shift) & 3u) - ml;
    }
};
/* Dot of one packed row with x[0..k): columns past k are read neither from
 * the weights' sum nor from x (a partial last chunk sums only its first
 * columns). Every lane returns the same total. */
template <typename F>
static inline float q38_quant_row(device const uchar *row, device const float *x, uint k, uint lane) {
    float acc = 0.0f;
    const uint chunks = (k + F::CHUNK - 1) / F::CHUNK;
    for (uint c = lane; c < chunks; c += 32) {
        const uint cols = metal::min(F::CHUNK, k - c * F::CHUNK);
        float w[F::CHUNK];
        F::decode(row, c, w);
        device const float *xb = x + c * F::CHUNK;
        float s = 0.0f;
        if (cols == F::CHUNK)
            for (uint i = 0; i < F::CHUNK; i++) s += w[i] * xb[i];
        else
            for (uint i = 0; i < cols; i++) s += w[i] * xb[i];
        acc += s;
    }
    return metal::simd_sum(acc);
}

#endif
