/* qw4q.c — picoforge weight encoders for the routed experts.
 *
 * The compat encoders follow the search procedures of the GGML/llama.cpp
 * quantizers as extracted by DwarfStar (antirez/ds4 b3f5018,
 * gguf-tools/quants.c), operation for operation, so the bytes match: our own
 * Q2 file is the test. Both are MIT licensed:
 *   Copyright (c) 2023-2026 The ggml authors
 *   Copyright (c) 2026 The ds4.c authors
 * The refinement (QW4Q_REFINE) and the code layout are picoforge's.
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
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define QK 256
#define GROUP_EPS 1e-15f
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

static int nearest_int(float v) {   /* round half to even through the f32 mantissa */
    float t = v + 12582912.f;
    int i;
    memcpy(&i, &t, 4);
    return (i & 0x007fffff) - 0x00400000;
}

/* ---- IQ2_XXS tables: encoder levels 0..2 per weight, derived from dec_grid ---- */

enum { IQ2_MAP = 43692, IQ2_SHELLS = 2 };

static struct {
    uint64_t grid[256];   /* per pattern, int8 values 2l+1 in {1,3,5}: the encoder's model */
    int *map;             /* packed 2-bit tuple -> grid index, or -(offset+1) into neighbours */
    uint16_t *neighbours; /* per missing tuple: count, then the indices of its two nearest shells */
} iq2;
static pthread_once_t iq2_once = PTHREAD_ONCE_INIT;

static int level_of_byte(uint8_t b) { return b == 8 ? 0 : b == 25 ? 1 : 2; }

static int cmp_dist(const void *a, const void *b) {
    const int *l = a, *r = b;
    return l[0] != r[0] ? (l[0] < r[0] ? -1 : 1) : (l[1] < r[1] ? -1 : l[1] > r[1]);
}

/* Grid patterns sorted by distance (then index) from the tuple `u`. */
static void sorted_by_distance(int u, int dist[512]) {
    for (int j = 0; j < 256; j++) {
        const int8_t *pg = (const int8_t *)&iq2.grid[j];
        int d2 = 0;
        for (int k = 0; k < 8; k++) {
            int p = 2 * ((u >> (2 * k)) & 3) + 1;
            d2 += (pg[k] - p) * (pg[k] - p);
        }
        dist[2 * j] = d2; dist[2 * j + 1] = j;
    }
    qsort(dist, 256, 2 * sizeof(int), cmp_dist);
}

/* Number of patterns within the first IQ2_SHELLS distinct distances. */
static int shell_count(const int dist[512]) {
    int n = 0, d2 = dist[0], have = 1;
    for (int j = 0; j < 256; j++) {
        if (dist[2 * j] > d2) {
            if (have == IQ2_SHELLS) break;
            d2 = dist[2 * j]; have++;
        }
        n++;
    }
    return n;
}

static void iq2_init(void) {
    iq2.map = malloc(IQ2_MAP * sizeof *iq2.map);
    int *dist = malloc(512 * sizeof *dist);
    if (!iq2.map || !dist) abort();
    for (int i = 0; i < IQ2_MAP; i++) iq2.map[i] = -1;
    for (int k = 0; k < 256; k++) {
        int key = 0;
        for (int i = 0; i < 8; i++) {
            int l = level_of_byte((uint8_t)(dec_grid[k] >> (8 * i)));
            ((int8_t *)&iq2.grid[k])[i] = (int8_t)(2 * l + 1);
            key |= l << (2 * i);
        }
        iq2.map[key] = k;
    }
    size_t total = 0;
    for (int u = 0; u < IQ2_MAP; u++)
        if (iq2.map[u] < 0) { sorted_by_distance(u, dist); total += 1 + (size_t)shell_count(dist); }
    iq2.neighbours = malloc(total * sizeof *iq2.neighbours);
    if (!iq2.neighbours) abort();
    int at = 0;
    for (int u = 0; u < IQ2_MAP; u++) {
        if (iq2.map[u] >= 0) continue;
        sorted_by_distance(u, dist);
        int n = shell_count(dist);
        iq2.map[u] = -(at + 1);
        iq2.neighbours[at++] = (uint16_t)n;
        for (int j = 0; j < n; j++) iq2.neighbours[at++] = (uint16_t)dist[2 * j + 1];
    }
    free(dist);
}

/* Best allowed pattern among the neighbours of a missing tuple; writes its levels to L. */
static int best_neighbour(const uint16_t *nbs, const float *xval, const float *weight, float scale, uint8_t *L) {
    float best = FLT_MAX;
    int index = -1;
    for (int j = 1; j <= nbs[0]; j++) {
        const int8_t *pg = (const int8_t *)&iq2.grid[nbs[j]];
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best) { best = d2; index = nbs[j]; }
    }
    const int8_t *pg = (const int8_t *)&iq2.grid[index];
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return index;
}

/* Weighted scale for levels 0..nmax of non-negative x (GGML make_qp_quants). */
static float qp_scale(int n, int nmax, const float *x, uint8_t *L, const float *w) {
    float max = 0;
    for (int i = 0; i < n; i++) max = MAX(max, x[i]);
    if (max < GROUP_EPS) { memset(L, 0, (size_t)n); return 0.0f; }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = (uint8_t)nearest_int(iscale * x[i]);
    float scale = 1 / iscale, best_mse = 0;
    for (int i = 0; i < n; i++) { float diff = x[i] - scale * L[i]; best_mse += w[i] * diff * diff; }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max, scale_is = 1 / iscale_is, mse = 0;
        for (int i = 0; i < n; i++) {
            int l = MIN(nmax, nearest_int(iscale_is * x[i]));
            float diff = x[i] - scale_is * l;
            mse += w[i] * diff * diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = MIN(nmax, nearest_int(iscale * x[i]));
        L[i] = (uint8_t)l;
        sumlx += w[i] * x[i] * l; suml2 += w[i] * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {   /* coordinate descent on single levels */
        int changed = 0;
        for (int i = 0; i < n; i++) {
            float slx = sumlx - w[i] * x[i] * L[i], sl2 = suml2 - w[i] * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int nl = MIN(nmax, nearest_int(x[i] * sl2 / slx));
                if (nl != L[i]) {
                    slx += w[i] * x[i] * nl; sl2 += w[i] * nl * nl;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) { L[i] = (uint8_t)nl; sumlx = slx; suml2 = sl2; changed++; }
                }
            }
        }
        if (!changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

/* Absolute values of one 8-weight group with an even number of negatives: if odd,
 * the least important element (smallest w*x^2) is represented with the wrong sign. */
static uint8_t group_signs(const float *xb, const float *weight, float *xval) {
    int nflip = 0;
    uint8_t s = 0;
    for (int i = 0; i < 8; i++) {
        if (xb[i] >= 0) xval[i] = xb[i];
        else { xval[i] = -xb[i]; nflip++; s |= (uint8_t)(1u << i); }
    }
    if (nflip % 2) {
        int imin = 0;
        float min = weight[0] * xb[0] * xb[0];
        for (int i = 1; i < 8; i++) {
            float ax = weight[i] * xb[i] * xb[i];
            if (ax < min) { min = ax; imin = i; }
        }
        xval[imin] = -xval[imin];
        s ^= (uint8_t)(1u << imin);
    }
    return s & 127;
}

/* Levels 0..2 for 8 values at inverse scale id; snapped to an allowed pattern. */
static int snap_group(const float *xval, const float *waux, float id, float scale, uint8_t *L, int *index) {
    uint16_t u = 0;
    for (int i = 0; i < 8; i++) {
        int l = MAX(0, MIN(2, nearest_int(0.5f * (id * xval[i] - 1))));
        L[i] = (uint8_t)l;
        u |= (uint16_t)(l << (2 * i));
    }
    *index = iq2.map[u];
    if (*index < 0) *index = best_neighbour(iq2.neighbours - iq2.map[u] - 1, xval, waux, scale, L);
    return *index;
}

/* Scale and levels of one 32-weight sub-block. Returns the scale (0 = empty). */
static float iq2_subblock(const float *xval, const float *weight, const float *waux, uint8_t *L) {
    float max = xval[0];
    for (int i = 1; i < 32; i++) max = MAX(max, xval[i]);
    if (max < GROUP_EPS) { memset(L, 0, 32); return 0; }
    enum { KMAXQ = 3 };   /* level values 1, 3, 5: three levels */
    float scale = qp_scale(32, KMAXQ + 1, xval, L, weight);
    float eff_max = scale * KMAXQ;
    if (eff_max <= 0) { memset(L, 0, 32); return 0; }
    float best = 0;
    uint8_t Laux[32];
    for (int is = -6; is <= 6; is++) {   /* 13 candidate scales around the qp estimate */
        float id = (2 * KMAXQ - 1 + is * 0.1f) / eff_max, this_scale = 1 / id;
        int index;
        for (int k = 0; k < 4; k++) snap_group(xval + 8 * k, waux + 8 * k, id, this_scale, Laux + 8 * k, &index);
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 32; i++) {
            float q = 2 * Laux[i] + 1;
            sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q;
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
            scale = sumqx / sumq2; best = scale * sumqx;
            memcpy(L, Laux, 32);
        }
    }
    return scale;
}

static void iq2_block_compat(const float *x, uint8_t *y, const float *qw) {
    uint32_t q2[16] = {0};
    float scales[8], weight[32], waux[32], xval[32];
    uint8_t L[32], signs[4];
    float sumx2 = 0;
    for (int i = 0; i < QK; i++) sumx2 += x[i] * x[i];
    const float sigma2 = sumx2 / QK;
    float max_scale = 0;
    uint16_t hd = f32_to_f16(0.0f);
    memcpy(y, &hd, 2);
    for (int ib = 0; ib < 8; ib++) {
        const float *xb = x + 32 * ib, *w = qw + 32 * ib;
        for (int i = 0; i < 32; i++) { weight[i] = w[i] * sqrtf(sigma2 + xb[i] * xb[i]); waux[i] = sqrtf(weight[i]); }
        for (int k = 0; k < 4; k++) signs[k] = group_signs(xb + 8 * k, weight + 8 * k, xval + 8 * k);
        float scale = iq2_subblock(xval, weight, waux, L);
        if (scale > 0) {   /* re-snap at the final scale, then the closed-form scale of those levels */
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                int index;
                snap_group(xval + 8 * k, waux + 8 * k, id, scale, L + 8 * k, &index);
                const int8_t *pg = (const int8_t *)&iq2.grid[index];
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; i++) {
                float q = 2 * L[i] + 1;
                sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }
        if (scale < 0) { scale = -scale; for (int k = 0; k < 4; k++) signs[k] = (~signs[k]) & 127; }
        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            q2[2 * ib] |= (uint32_t)iq2.map[u] << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)signs[k] << (7 * k);
        }
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) { memset(y + 2, 0, 64); return; }
    float d = max_scale / 31;
    hd = f32_to_f16(d);
    memcpy(y, &hd, 2);
    float id = 1 / d;
    for (int ib = 0; ib < 8; ib++) {
        int l = MAX(0, MIN(15, nearest_int(0.5f * (id * scales[ib] - 1))));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    memcpy(y + 2, q2, 64);
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

/* ---- Q2_K (84 bytes per 256 weights): 16 sub-blocks of 16 with 4-bit scale and
 * 4-bit min, both multiples of the f16 d / dmin at bytes 80 and 82; 2-bit codes
 * at bytes 16..79. Decoded weight = d*scale*code - dmin*min. ---- */

/* Compat encoder: the unchanged DwarfStar translation unit (third_party/README.md says why). */
#include "third_party/quants.h"   /* DwarfStar's Q2_K encoder, verbatim: see third_party/README.md */

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
    if (mode != QW4Q_COMPAT) return -1;
    pthread_once(&iq2_once, iq2_init);
    const int64_t blocks = ncols / QK;
    for (int64_t r = 0; r < nrows; r++)
        for (int64_t b = 0; b < blocks; b++)
            iq2_block_compat(src + r * ncols + b * QK, dst + (r * blocks + b) * 66, importance + b * QK);
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
