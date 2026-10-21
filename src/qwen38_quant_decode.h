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
/* Q4_K (144-byte blocks of 256): fp16 d and dmin, 12 bytes of 6-bit scales
 * and mins for 8 sub-blocks of 32 columns, 128 bytes of 4-bit codes (sub-blocks
 * 2j and 2j+1 share 32 bytes: low and high nibbles). Chunk = one sub-block. */
struct Q38Q4K {
    static constant constexpr uint CHUNK = 32;
    static inline void decode(device const uchar *row, uint c, thread float *w) {
        device const uchar *b = row + ulong(c / 8) * 144, *sc = b + 4;
        const uint g = c % 8;
        const uint scale = g < 4 ? sc[g] & 63u : (sc[g + 4] & 15u) | ((sc[g - 4] & 192u) >> 2);
        const uint minimum = g < 4 ? sc[g + 4] & 63u : (sc[g + 4] >> 4) | ((sc[g] & 192u) >> 2);
        const float dl = q38_fp16(b) * float(scale), ml = q38_fp16(b + 2) * float(minimum);
        device const uchar *q = b + 16 + (g / 2) * 32;
        const uint shift = (g % 2) * 4;
        for (uint i = 0; i < 32; i++) w[i] = dl * float((q[i] >> shift) & 15u) - ml;
    }
};
/* One MXFP4 value: E2M1 code times 2^(e-127), built as exact FP32 bits as the
 * oracle does (Metal arithmetic flushes subnormals; a power of two times an
 * E2M1 value needs no rounding). Magnitudes 1..7 = 0.5, 1, 1.5, 2, 3, 4, 6:
 * exponent field 126 + m/2, fraction 0.5 for the odd m above 1. e = 255 is
 * the escape: Inf, or NaN for magnitude 0. */
static inline float q38_mxfp4_value(uint e, uint code) {
    const uint sign = (code & 8u) << 28, magnitude = code & 7u;
    if (e == 255) return as_type<float>(sign | 0x7f800000u | (magnitude ? 0u : 0x400000u));
    if (!magnitude) return as_type<float>(sign);
    const uint fraction = magnitude > 1 ? (magnitude & 1u) << 22 : 0u;
    const int exponent = int(126 + magnitude / 2) + int(e) - 127;
    const uint bits = exponent <= 0 ? (0x800000u | fraction) >> uint(1 - exponent) :
        exponent >= 255 ? 0x7f800000u : (uint(exponent) << 23) | fraction;
    return as_type<float>(sign | bits);
}
/* MXFP4 (17-byte blocks of 32): the E8M0 exponent byte, then 16 bytes of codes
 * (column i in the low nibble of byte i, column 16+i in the high one).
 * Chunk = one block. A block whose every value is normal and finite (e in
 * 2..252) takes a branch-free form of the same bits (95.7 vs 140.5 us on the
 * down shape at 10 slots); the others go through q38_mxfp4_value. */
struct Q38Mxfp4 {
    static constant constexpr uint CHUNK = 32;
    static inline void decode(device const uchar *row, uint c, thread float *w) {
        device const uchar *b = row + ulong(c) * 17;
        const uint e = b[0];
        if (e >= 2 && e <= 252) { /* every code normal and finite: exponent field e - 1 + m/2 */
            for (uint i = 0; i < 32; i++) {
                const uint code = (b[1 + i % 16] >> (i / 16 * 4)) & 15u, m = code & 7u;
                const uint bits = m ? ((e - 1 + m / 2) << 23) | (m > 1 ? (m & 1u) << 22 : 0u) : 0u;
                w[i] = as_type<float>(((code & 8u) << 28) | bits);
            }
            return;
        }
        for (uint i = 0; i < 16; i++) {
            const uint byte = b[1 + i];
            w[i] = q38_mxfp4_value(e, byte & 15u);
            w[16 + i] = q38_mxfp4_value(e, byte >> 4);
        }
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
/* Two rows of one format against the same x (an expert's gate and up rows,
 * R1.5b): each x chunk is loaded once for both, and each accumulator follows
 * exactly q38_quant_row's order, so *a and *b are bit-identical to two
 * q38_quant_row calls (test_fused_gate_up_bit_identical_to_sequence). */
template <typename F>
static inline void q38_quant_row_pair(device const uchar *ra, device const uchar *rb, device const float *x,
        uint k, uint lane, thread float &a, thread float &b) {
    float acc_a = 0.0f, acc_b = 0.0f;
    const uint chunks = (k + F::CHUNK - 1) / F::CHUNK;
    for (uint c = lane; c < chunks; c += 32) {
        const uint cols = metal::min(F::CHUNK, k - c * F::CHUNK);
        float wa[F::CHUNK], wb[F::CHUNK];
        F::decode(ra, c, wa);
        F::decode(rb, c, wb);
        device const float *xb = x + c * F::CHUNK;
        float sa = 0.0f, sb = 0.0f;
        if (cols == F::CHUNK)
            for (uint i = 0; i < F::CHUNK; i++) { const float v = xb[i]; sa += wa[i] * v; sb += wb[i] * v; }
        else
            for (uint i = 0; i < cols; i++) { const float v = xb[i]; sa += wa[i] * v; sb += wb[i] * v; }
        acc_a += sa;
        acc_b += sb;
    }
    a = metal::simd_sum(acc_a);
    b = metal::simd_sum(acc_b);
}

#endif
