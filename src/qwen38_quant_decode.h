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

#endif
