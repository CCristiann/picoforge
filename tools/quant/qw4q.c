/* qw4q.c — picoforge weight encoders for the routed experts.
 *
 * Compat mode calls DwarfStar's quantizer (third_party/ds4_quants.c, unchanged,
 * MIT): its bytes are the published Q2 file's, and only the unchanged
 * translation unit reproduces them (third_party/README.md). picoforge's own
 * part is the O1 refinement below, which starts from the compat block, and the
 * decoders used to measure it.
 *
 * IQ2_XXS block (66 bytes per 256 weights): f16 d, then per 32-weight
 * sub-block two u32: four 8-bit grid indices, and four 7-bit sign masks with a
 * 4-bit scale l in the top nibble. Decoded weight = d*(2l+1)/8 * grid byte *
 * sign; grid bytes are 8, 25, 43 and the eighth sign is the parity of the
 * other seven.
 */
#include "qw4q.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define QK 256

#include "third_party/quants.h"

/* The decoder's grid (also used by the Metal kernels): 256 patterns of 8 bytes. */
static const uint64_t dec_grid[256] = {
#include "qwen4_iq2_grid.inc"
};

/* ---- f16 and rounding helpers (same bit tricks as GGML) ---- */

static float from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t to_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

static uint16_t f32_to_f16(float f) {
    float base = (fabsf(f) * 0x1.0p+112f) * 0x1.0p-110f;
    const uint32_t w = to_bits(f), shl1_w = w + w, sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t out = to_bits(base);
    const uint32_t nonsign = ((out >> 13) & 0x00007C00u) + (out & 0x00000FFFu);
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16, sign = w & 0x80000000u, two_w = w + w;
    const float normalized = from_bits((two_w >> 4) + (0xE0u << 23)) * 0x1.0p-112f;
    const float denormalized = from_bits((two_w >> 17) | (126u << 23)) - 0.5f;
    return from_bits(sign | (two_w < (1u << 27) ? to_bits(denormalized) : to_bits(normalized)));
}

static void iq2_block_decode(const uint8_t *y, float *out) {
    const float d = f16_to_f32((uint16_t)(y[0] | y[1] << 8));
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a0, a1;
        memcpy(&a0, y + 2 + 8 * ib, 4);
        memcpy(&a1, y + 6 + 8 * ib, 4);
        const float db = d * (float)(2 * (a1 >> 28) + 1) * 0.125f;
        for (int k = 0; k < 4; k++) {
            uint32_t s = (a1 >> (7 * k)) & 127;
            s |= (uint32_t)(__builtin_popcount(s) & 1) << 7;
            const uint64_t g = dec_grid[(a0 >> (8 * k)) & 255];
            for (int i = 0; i < 8; i++) {
                float v = db * (float)(uint8_t)(g >> (8 * i));
                out[32 * ib + 8 * k + i] = (s >> i & 1) ? -v : v;
            }
        }
    }
}

/* ---- O1: refinement against the decoder's real levels ----
 * The compat search models the three magnitudes as 1, 3, 5, but the decoder's
 * grid bytes 8, 25, 43 mean 1, 3.125, 5.375 (times d*(2l+1)/8). Starting from
 * the compat block, two rounds of: best pattern and sign mask per 8-weight group
 * at the decoded scale (all 256 patterns), then the best f16 d (41 candidates,
 * +-20%) with the best 4-bit l per sub-block in closed form. The result is kept
 * only if the encoder's own weighted error, under the real decoder, goes down. */

typedef struct { float d; int l[8]; int idx[32]; uint8_t sign[32]; } Iq2Block;   /* sign: 8 bits, even parity */

static uint8_t grid_byte(int idx, int i) { return (uint8_t)(dec_grid[idx] >> (8 * i)); }

static void iq2_unpack(const uint8_t *y, Iq2Block *b) {
    b->d = f16_to_f32((uint16_t)(y[0] | y[1] << 8));
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a0, a1;
        memcpy(&a0, y + 2 + 8 * ib, 4);
        memcpy(&a1, y + 6 + 8 * ib, 4);
        b->l[ib] = (int)(a1 >> 28);
        for (int k = 0; k < 4; k++) {
            uint8_t s = (a1 >> (7 * k)) & 127;
            b->sign[4 * ib + k] = (uint8_t)(s | ((__builtin_popcount(s) & 1) << 7));
            b->idx[4 * ib + k] = (a0 >> (8 * k)) & 255;
        }
    }
}

static void iq2_pack(const Iq2Block *b, uint8_t *y) {
    uint16_t hd = f32_to_f16(b->d);
    memcpy(y, &hd, 2);
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a0 = 0, a1 = (uint32_t)b->l[ib] << 28;
        for (int k = 0; k < 4; k++) {
            a0 |= (uint32_t)b->idx[4 * ib + k] << (8 * k);
            a1 |= (uint32_t)(b->sign[4 * ib + k] & 127) << (7 * k);
        }
        memcpy(y + 2 + 8 * ib, &a0, 4);
        memcpy(y + 6 + 8 * ib, &a1, 4);
    }
}

static double iq2_error(const uint8_t *y, const float *x, const float *w) {
    float q[QK];
    iq2_block_decode(y, q);
    double e = 0;
    for (int i = 0; i < QK; i++) e += (double)w[i] * ((double)x[i] - q[i]) * ((double)x[i] - q[i]);
    return e;
}

/* Best pattern and even-parity sign mask for 8 weights at decoded step s
 * (value = s * grid byte). With an odd number of negatives one element must
 * take the wrong sign: each of the eight is tried. */
static void best_group(const float *x, const float *w, float s, int *idx, uint8_t *sign) {
    uint8_t neg = 0;
    for (int i = 0; i < 8; i++) if (x[i] < 0) neg |= (uint8_t)(1u << i);
    const int options = (__builtin_popcount(neg) & 1) ? 8 : 1;
    double best = INFINITY;
    for (int f = 0; f < options; f++) {
        const uint8_t mask = options == 1 ? neg : (uint8_t)(neg ^ (1u << f));
        float a[8];
        for (int i = 0; i < 8; i++) a[i] = (mask >> i & 1) ? -x[i] : x[i];
        for (int k = 0; k < 256; k++) {
            double e = 0;
            for (int i = 0; i < 8; i++) { double r = a[i] - (double)s * grid_byte(k, i); e += w[i] * r * r; }
            if (e < best) { best = e; *idx = k; *sign = mask; }
        }
    }
}

/* Best 4-bit l of sub-block ib for a given d, patterns fixed: E = A - 2*db*B + db^2*C. */
static int best_l(const Iq2Block *b, int ib, float d, const float *x, const float *w, double *err) {
    double A = 0, B = 0, C = 0;
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < 8; i++) {
            const int j = 32 * ib + 8 * k + i;
            const double m = grid_byte(b->idx[4 * ib + k], i) * ((b->sign[4 * ib + k] >> i & 1) ? -1.0 : 1.0);
            A += w[j] * (double)x[j] * x[j]; B += w[j] * (double)x[j] * m; C += w[j] * m * m;
        }
    int bl = 0;
    double be = INFINITY;
    for (int l = 0; l < 16; l++) {
        const double db = (double)d * (2 * l + 1) * 0.125;
        const double e = A - 2 * db * B + db * db * C;
        if (e < be) { be = e; bl = l; }
    }
    *err = be;
    return bl;
}

static void iq2_refine(uint8_t *y, const float *x, const float *qw) {
    Iq2Block b;
    iq2_unpack(y, &b);
    if (b.d == 0) return;
    float w[QK], sumx2 = 0;
    for (int i = 0; i < QK; i++) sumx2 += x[i] * x[i];
    const float sigma2 = sumx2 / QK;
    for (int i = 0; i < QK; i++) w[i] = qw[i] * sqrtf(sigma2 + x[i] * x[i]);
    for (int round = 0; round < 2; round++) {
        for (int ib = 0; ib < 8; ib++) {
            const float s = b.d * (float)(2 * b.l[ib] + 1) * 0.125f;
            for (int k = 0; k < 4; k++)
                best_group(x + 32 * ib + 8 * k, w + 32 * ib + 8 * k, s, &b.idx[4 * ib + k], &b.sign[4 * ib + k]);
        }
        const float d0 = b.d;
        double best = INFINITY;
        for (int c = 0; c <= 40; c++) {
            const float d = f16_to_f32(f32_to_f16(d0 * (0.8f + 0.01f * (float)c)));
            if (d <= 0) continue;
            int ls[8];
            double e = 0, ei;
            for (int ib = 0; ib < 8; ib++) { ls[ib] = best_l(&b, ib, d, x, w, &ei); e += ei; }
            if (e < best) { best = e; b.d = d; memcpy(b.l, ls, sizeof ls); }
        }
    }
    uint8_t cand[66];
    iq2_pack(&b, cand);
    if (iq2_error(cand, x, w) < iq2_error(y, x, w)) memcpy(y, cand, 66);
}

/* ---- Q2_K (84 bytes per 256 weights): 16 sub-blocks of 16 with 4-bit scale and
 * 4-bit min, both multiples of the f16 d / dmin at bytes 80 and 82; 2-bit codes
 * at bytes 16..79. Decoded weight = d*scale*code - dmin*min. ---- */

/* Compat encoder: the unchanged DwarfStar translation unit (third_party/README.md says why). */

static void q2k_block_decode(const uint8_t *y, float *out) {
    const float d = f16_to_f32((uint16_t)(y[80] | y[81] << 8)), dmin = f16_to_f32((uint16_t)(y[82] | y[83] << 8));
    for (int j = 0; j < QK; j++) {
        const uint8_t sc = y[j / 16];
        const int code = (y[16 + j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        out[j] = (d * (float)(sc & 15)) * (float)code - dmin * (float)(sc >> 4);
    }
}

/* ---- public API ---- */

size_t qw4q_row_bytes(int type, int64_t ncols) {
    if (ncols <= 0 || ncols % QK) return 0;
    if (type == QW4Q_IQ2_XXS) return (size_t)(ncols / QK) * 66;
    if (type == QW4Q_Q2_K) return (size_t)(ncols / QK) * 84;
    return 0;
}

int qw4q_quantize_iq2_xxs(const float *src, int64_t nrows, int64_t ncols, const float *importance,
                          qw4q_mode mode, uint8_t *dst) {
    if (!src || !importance || !dst || nrows < 0 || !qw4q_row_bytes(QW4Q_IQ2_XXS, ncols)) return -1;
    if (mode != QW4Q_COMPAT && mode != QW4Q_REFINE) return -1;
    if (ds4q_quantize_chunk(DS4Q_TYPE_IQ2_XXS, src, dst, 0, nrows, ncols, importance) !=
        (size_t)nrows * qw4q_row_bytes(QW4Q_IQ2_XXS, ncols)) return -1;
    if (mode == QW4Q_REFINE) {
        const int64_t blocks = ncols / QK;
        for (int64_t r = 0; r < nrows; r++)
            for (int64_t b = 0; b < blocks; b++)
                iq2_refine(dst + (r * blocks + b) * 66, src + r * ncols + b * QK, importance + b * QK);
    }
    return 0;
}

int qw4q_quantize_q2_k(const float *src, int64_t nrows, int64_t ncols, const float *importance, uint8_t *dst) {
    if (!src || !importance || !dst || nrows < 0 || !qw4q_row_bytes(QW4Q_Q2_K, ncols)) return -1;
    if (ds4q_quantize_chunk(DS4Q_TYPE_Q2_K, src, dst, 0, nrows, ncols, importance) != (size_t)nrows * qw4q_row_bytes(QW4Q_Q2_K, ncols)) return -1;
    return 0;
}

int qw4q_dequantize(int type, const uint8_t *src, int64_t nrows, int64_t ncols, float *dst) {
    if (!src || !dst || nrows < 0 || !qw4q_row_bytes(type, ncols)) return -1;
    const size_t bytes = type == QW4Q_IQ2_XXS ? 66 : 84;
    for (int64_t i = 0; i < nrows * (ncols / QK); i++) {
        if (type == QW4Q_IQ2_XXS) iq2_block_decode(src + i * bytes, dst + i * QK);
        else q2k_block_decode(src + i * bytes, dst + i * QK);
    }
    return 0;
}
