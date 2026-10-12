/* QSA CPU core, FP32 and no fast-math. See integration.md and spec D. */
#include "qwen38_d_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

Qwen38Status q38d_rope(float *x, uint32_t heads, uint32_t dim,
                      uint32_t rotary, uint32_t pos, float theta) {
    if (!x || !heads || !dim || !rotary || rotary % 2 || rotary > dim)
        return Q38_SHAPE;
    if (!isfinite(theta) || theta <= 0) return Q38_RANGE;
    for (uint32_t h = 0; h < heads; ++h) {
        float *row = x + (size_t)h * dim;
        for (uint32_t i = 0; i < dim; ++i)
            if (!isfinite(row[i])) return Q38_NONFINITE;
        for (uint32_t i = 0; i < rotary / 2; ++i) {
            /* Match default HF frequency construction: reciprocal pow,
             * then F32 position multiplication, not a half-angle variant. */
            float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rotary);
            float angle = (float)pos * inv;
            float c = cosf(angle), s = sinf(angle);
            float a = row[i], b = row[i + rotary / 2];
            row[i] = a * c - b * s;
            row[i + rotary / 2] = b * c + a * s;
            if (!isfinite(row[i]) || !isfinite(row[i + rotary / 2]))
                return Q38_NONFINITE;
        }
    }
    return Q38_OK;
}

Qwen38Status q38d_scores(const float *iq, const float *blocks,
                        uint32_t heads, uint32_t dim, uint32_t count, float *scores) {
    if (!heads || !dim || !iq || (count && (!blocks || !scores))) return Q38_SHAPE;
    for (uint32_t b = 0; b < count; ++b) {
        float score = 0;
        for (uint32_t h = 0; h < heads; ++h) {
            float dot = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                float q = iq[(size_t)h * dim + d], k = blocks[(size_t)b * dim + d];
                if (!isfinite(q) || !isfinite(k)) return Q38_NONFINITE;
                dot += q * k;
            }
            if (!isfinite(dot)) return Q38_NONFINITE;
            score += fmaxf(dot, 0);
        }
        if (!isfinite(score)) return Q38_NONFINITE;
        scores[b] = score; /* Unscaled, unlike HF's positive 1/sqrt(Di). */
    }
    return Q38_OK;
}

static bool worse(const float *s, uint32_t a, uint32_t b) {
    return s[a] < s[b] || (s[a] == s[b] && a > b);
}

static void sift(const float *s, uint32_t *heap, uint32_t n, uint32_t root) {
    while (root < n / 2) {
        uint32_t child = 2 * root + 1;
        if (child + 1 < n && worse(s, heap[child + 1], heap[child])) ++child;
        if (!worse(s, heap[child], heap[root])) break;
        uint32_t swap = heap[root]; heap[root] = heap[child]; heap[child] = swap;
        root = child;
    }
}

static int id_order(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

Qwen38Status q38d_select(const float *scores, uint32_t count, uint32_t budget,
                        uint32_t *heap, uint32_t *selected) {
    if (!budget || (count && (!scores || !heap || !selected))) return Q38_SHAPE;
    /* Preflight every visible score. Never inspect future/poisoned slots. */
    for (uint32_t i = 0; i < count; ++i)
        if (!isfinite(scores[i])) return Q38_NONFINITE;
    uint32_t n = count < budget ? count : budget;
    if (!n) return Q38_OK;
    for (uint32_t i = 0; i < n; ++i) heap[i] = i;
    for (uint32_t i = n / 2; i > 0; --i) sift(scores, heap, n, i - 1);
    for (uint32_t i = n; i < count; ++i)
        if (worse(scores, heap[0], i)) { heap[0] = i; sift(scores, heap, n, 0); }
    memcpy(selected, heap, (size_t)n * sizeof(*heap));
    qsort(selected, n, sizeof(*selected), id_order);
    return Q38_OK;
}
