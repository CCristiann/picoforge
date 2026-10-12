/* F v1: routing on caller-owned projected logits. Packed products belong to B. */
#include "qwen38_moe.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct { uintptr_t first, end; } MoeSpan;

static Qwen38Status moe_error(Qwen38Error *err, Qwen38Status code, const char *detail) {
    if (err) {
        memset(err, 0, sizeof *err);
        err->code = code;
        if (code != Q38_OK) {
            snprintf(err->subject, sizeof err->subject, "qwen38_moe");
            snprintf(err->detail, sizeof err->detail, "%s", detail);
        }
    }
    return code;
}

static Qwen38Status moe_config(const Qwen38MoeConfig *c, Qwen38Error *err) {
    if (!c) return moe_error(err, Q38_INVALID_ARGUMENT, "null config");
    if (!c->hidden_size || !c->experts || !c->top_k || c->top_k > c->experts ||
        !c->intermediate || !c->shared_intermediate || c->down_physical < c->intermediate)
        return moe_error(err, Q38_RANGE, "invalid MoE dimensions");
    return Q38_OK;
}

/* Bounding spans include row padding: aliasing through a gap is rejected too.
 * Empty batches validate shape/stride but neither inspect nor require data. */
static Qwen38Status moe_span(const void *data, size_t cap, size_t rows, size_t cols,
                            size_t pitch, MoeSpan *span, Qwen38Error *err) {
    span->first = span->end = 0;
    if (cols > SIZE_MAX / 4) return moe_error(err, Q38_OVERFLOW, "row bytes overflow");
    size_t row_bytes = cols * 4;
    if (pitch % 4 || pitch < row_bytes)
        return moe_error(err, Q38_STRIDE, "row stride must cover contiguous features");
    if (!rows) return Q38_OK;
    if (rows - 1 > (SIZE_MAX - row_bytes) / pitch)
        return moe_error(err, Q38_OVERFLOW, "view extent overflow");
    size_t bytes = (rows - 1) * pitch + row_bytes;
    if (cap < bytes) return moe_error(err, Q38_CAPACITY, "view capacity too small");
    if (!data) return moe_error(err, Q38_INVALID_ARGUMENT, "null nonempty view");
    uintptr_t first = (uintptr_t)data;
    if (first % 4) return moe_error(err, Q38_STRIDE, "unaligned view");
    if (bytes > UINTPTR_MAX - first)
        return moe_error(err, Q38_OVERFLOW, "address extent overflow");
    *span = (MoeSpan){first, first + bytes};
    return Q38_OK;
}

static Qwen38Status moe_f32(Qwen38F32 v, size_t rows, size_t cols,
                           MoeSpan *span, Qwen38Error *err) {
    if (v.rank != 2 || v.dim[0] != rows || v.dim[1] != cols)
        return moe_error(err, Q38_SHAPE, "expected rank-2 token-major view");
    if (v.stride_bytes[1] != sizeof(float))
        return moe_error(err, Q38_STRIDE, "features must be contiguous");
    return moe_span(v.data, v.capacity_bytes, rows, cols, v.stride_bytes[0], span, err);
}

static Qwen38F32 moe_const(Qwen38F32Mut v) {
    Qwen38F32 out = {.data = v.data, .capacity_bytes = v.capacity_bytes, .rank = v.rank};
    memcpy(out.dim, v.dim, sizeof out.dim);
    memcpy(out.stride_bytes, v.stride_bytes, sizeof out.stride_bytes);
    return out;
}

static Qwen38Status moe_disjoint(const MoeSpan *spans, size_t n, Qwen38Error *err) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < i; j++)
            if (spans[i].first < spans[j].end && spans[j].first < spans[i].end)
                return moe_error(err, Q38_INVALID_ARGUMENT, "overlapping views");
    return Q38_OK;
}

Qwen38Status qwen38_moe_route(const Qwen38MoeConfig *c, Qwen38F32 logits,
                            Qwen38MoeRoutes routes, Qwen38F32Mut probabilities,
                            Qwen38Error *err) {
    Qwen38Status st = moe_config(c, err);
    if (st != Q38_OK) return st;
    const size_t t_count = logits.dim[0], e_count = c->experts, k = c->top_k;
    MoeSpan spans[4];
    st = moe_f32(logits, t_count, e_count, &spans[0], err);
    if (st != Q38_OK) return st;
    st = moe_f32(moe_const(probabilities), t_count, e_count, &spans[1], err);
    if (st != Q38_OK) return st;
    st = moe_f32(moe_const(routes.weights), t_count, k, &spans[2], err);
    if (st != Q38_OK) return st;
    if (routes.ids.rows != t_count || routes.ids.cols != k)
        return moe_error(err, Q38_SHAPE, "route ID shape mismatch");
    st = moe_span(routes.ids.data, routes.ids.capacity_bytes, t_count, k,
                  routes.ids.row_stride_bytes, &spans[3], err);
    if (st != Q38_OK) return st;
    st = moe_disjoint(spans, 4, err);
    if (st != Q38_OK) return st;
    /* Preflight all input rows so NaN/Inf cannot publish a partial route. */
    for (size_t t = 0; t < t_count; t++) {
        const float *z = logits.data + t * (logits.stride_bytes[0] / 4);
        for (size_t e = 0; e < e_count; e++) {
            if (!isfinite(z[e])) {
                st = moe_error(err, Q38_NONFINITE, "nonfinite router logit");
                if (err) err->row = t;
                return st;
            }
        }
    }
    for (size_t t = 0; t < t_count; t++) {
        const float *z = logits.data + t * (logits.stride_bytes[0] / 4);
        float *p = probabilities.data + t * (probabilities.stride_bytes[0] / 4);
        float *w = routes.weights.data + t * (routes.weights.stride_bytes[0] / 4);
        uint32_t *ids = routes.ids.data + t * (routes.ids.row_stride_bytes / 4);
        float maximum = z[0], sum = 0.0f;
        for (size_t e = 1; e < e_count; e++) if (z[e] > maximum) maximum = z[e];
        for (size_t e = 0; e < e_count; e++) {
            /* Finite extreme inputs can give -Inf here; exp(-Inf)=0 is correct. */
            p[e] = expf(z[e] - maximum);
            sum += p[e];
        }
        for (size_t e = 0; e < e_count; e++) p[e] /= sum;
        float selected_sum = 0.0f;
        for (size_t slot = 0; slot < k; slot++) {
            size_t best = 0;
            bool found = false;
            for (size_t e = 0; e < e_count; e++) {
                bool used = false;
                for (size_t j = 0; j < slot; j++) used |= ids[j] == e;
                if (!used && (!found || p[e] > p[best])) { best = e; found = true; }
            }
            ids[slot] = (uint32_t)best; /* ascending scan + strict > resolves ties */
            w[slot] = p[best];
            selected_sum += w[slot];
        }
        if (c->norm_topk_prob)
            for (size_t slot = 0; slot < k; slot++) w[slot] /= selected_sum;
    }
    return moe_error(err, Q38_OK, "");
}
