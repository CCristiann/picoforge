/* Native C11 packed rows. No file I/O, allocation, upstream decoder or GPU. */
#include "qwen38_b_internal.h"
#pragma STDC FP_CONTRACT OFF

Qwen38Status q38_packed_layout_init(const GgufTensor *t, size_t logical,
                                   Q38PackedLayout *out, Qwen38Error *err) {
    if (out) memset(out, 0, sizeof *out);
    if (!t || !out) return q38_b_result(err, Q38_INVALID_ARGUMENT, "null descriptor/output");
    if (t->rank < 1 || t->rank > 3)
        return q38_b_result(err, Q38_SHAPE, "expected rank 1, 2 or 3");
    const GgufFormat *format = NULL;
    Qwen38Status status = q38_b_format(t->type, &format);
    if (status) return q38_b_result(err, status, "format not executable");
    Q38PackedLayout layout = { .gguf_type = t->type, .k_storage = t->shape[0],
        .k_logical = logical, .n_out = t->rank > 1 ? t->shape[1] : 1,
        .n_experts = t->rank > 2 ? t->shape[2] : 1 };
    if (!logical || logical > layout.k_storage || !layout.n_out || !layout.n_experts)
        return q38_b_result(err, Q38_SHAPE, "empty or incompatible dimensions");
    if (layout.k_storage % format->block)
        return q38_b_result(err, Q38_FORMAT, "incomplete physical block");
    if (!q38_b_mul(layout.k_storage / format->block, format->bytes, &layout.row_bytes) ||
        !q38_b_mul(layout.row_bytes, layout.n_out, &layout.expert_bytes) ||
        !q38_b_mul(layout.expert_bytes, layout.n_experts, &layout.tensor_bytes) ||
        t->file_offset > SIZE_MAX - layout.tensor_bytes)
        return q38_b_result(err, Q38_OVERFLOW, "tensor extent overflow");
    if (t->bytes != layout.tensor_bytes)
        return q38_b_result(err, Q38_FORMAT, "descriptor bytes differ from physical shape");
    *out = layout;
    return q38_b_result(err, Q38_OK, "");
}

static Qwen38Status packed_view(const Q38PackedCpuView *w, size_t expert) {
    if (!w) return Q38_INVALID_ARGUMENT;
    const Q38PackedLayout *l = &w->layout;
    GgufTensor tensor = {.type = l->gguf_type, .rank = 3,
        .shape = {l->k_storage, l->n_out, l->n_experts, 1}, .bytes = l->tensor_bytes};
    Q38PackedLayout expected;
    Qwen38Status status = q38_packed_layout_init(&tensor, l->k_logical, &expected, NULL);
    if (status) return status;
    if (!q38_b_same_layout(l, &expected)) return Q38_FORMAT;
    if (expert >= l->n_experts) return Q38_RANGE;
    if (w->span.bytes < l->tensor_bytes) return Q38_CAPACITY;
    if (!q38_b_address(w->span.data, l->tensor_bytes)) return Q38_INVALID_ARGUMENT;
    return Q38_OK;
}

static uint16_t little16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}
/* Exact widening, including signed zero and subnormals; no _Float16 ABI. */
static float half_value(const unsigned char *p) {
    uint32_t h = little16(p), sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 31u, fraction = h & 1023u, bits;
    if (!exponent && fraction) {
        exponent = 113;
        while (!(fraction & 1024u)) { fraction <<= 1; exponent--; }
        bits = sign | (exponent << 23) | ((fraction & 1023u) << 13);
    } else if (!exponent) bits = sign;
    else bits = sign | ((exponent == 31 ? 255u : exponent + 112u) << 23) | (fraction << 13);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}
static float packed_value(const unsigned char *row, size_t column, uint32_t type) {
    if (type == 0 || type == 30) {
        uint32_t bits;
        if (type == 30) bits = (uint32_t)little16(row + 2 * column) << 16;
        else {
            const unsigned char *p = row + 4 * column;
            bits = (uint32_t)little16(p) | (uint32_t)little16(p + 2) << 16;
        }
        float value;
        memcpy(&value, &bits, sizeof value);
        return value;
    }
    if (type == 1) return half_value(row + 2 * column);
    if (type == 8) {
        const unsigned char *q8 = row + column / 32 * 34;
        int code = q8[2 + column % 32];
        if (code >= 128) code -= 256; /* No implementation-defined signed cast. */
        return half_value(q8) * (float)code;
    }
    const unsigned char *block = row + column / 256 * 84;
    size_t lane = column % 256;
    unsigned scale = block[lane / 16];
    unsigned code = (block[16 + lane / 128 * 32 + lane % 32] >> (2 * (lane % 128 / 32))) & 3u;
    return (half_value(block + 80) * (float)(scale & 15u)) * (float)code -
           half_value(block + 82) * (float)(scale >> 4);
}

Qwen38Status q38_packed_decode_row(const Q38PackedCpuView *w, size_t expert,
        size_t row, float *out, size_t count, Qwen38Error *err) {
    Qwen38Status status = packed_view(w, expert);
    if (status) return q38_b_result(err, status, "invalid weight view");
    size_t bytes;
    if (row >= w->layout.n_out) return q38_b_result(err, Q38_RANGE, "row outside tensor");
    if (!q38_b_mul(w->layout.k_storage, sizeof(float), &bytes))
        return q38_b_result(err, Q38_OVERFLOW, "decoded row extent overflow");
    if (count < w->layout.k_storage) return q38_b_result(err, Q38_CAPACITY, "short output row");
    if (!q38_b_address(out, bytes) || (uintptr_t)out % _Alignof(float) ||
        q38_b_overlap(w->span.data, w->layout.tensor_bytes, out, bytes))
        return q38_b_result(err, Q38_INVALID_ARGUMENT, "invalid or aliased output");
    const unsigned char *data = w->span.data + expert * w->layout.expert_bytes + row * w->layout.row_bytes;
    for (size_t k = 0; k < w->layout.k_storage; k++) {
        float value = packed_value(data, k, w->layout.gguf_type);
        if (!isfinite(value)) return q38_b_result(err, Q38_NONFINITE, "nonfinite packed value");
        out[k] = value;
    }
    return q38_b_result(err, Q38_OK, "");
}

Qwen38Status q38_packed_matmul(const Q38PackedCpuView *w, size_t expert,
        Qwen38F32 x, Qwen38F32Mut y, Qwen38Error *err) {
    Qwen38Status status = packed_view(w, expert);
    if (status) return q38_b_result(err, status, "invalid weight view");
    size_t xbytes, ybytes;
    status = q38_b_matrix(x.rank, x.dim, x.stride_bytes, x.capacity_bytes, &xbytes);
    if (status) return q38_b_result(err, status, "invalid input matrix");
    status = q38_b_matrix(y.rank, y.dim, y.stride_bytes, y.capacity_bytes, &ybytes);
    if (status) return q38_b_result(err, status, "invalid output matrix");
    if (x.dim[0] != y.dim[0] || x.dim[1] != w->layout.k_logical || y.dim[1] != w->layout.n_out)
        return q38_b_result(err, Q38_SHAPE, "incompatible matrix dimensions");
    if (!x.dim[0]) return q38_b_result(err, Q38_OK, "");
    if (!q38_b_address(x.data, xbytes) || !q38_b_address(y.data, ybytes) ||
        (uintptr_t)x.data % _Alignof(float) || (uintptr_t)y.data % _Alignof(float))
        return q38_b_result(err, Q38_INVALID_ARGUMENT, "invalid activation address");
    if (q38_b_overlap(x.data, xbytes, y.data, ybytes) ||
        q38_b_overlap(w->span.data, w->layout.tensor_bytes, x.data, xbytes) ||
        q38_b_overlap(w->span.data, w->layout.tensor_bytes, y.data, ybytes))
        return q38_b_result(err, Q38_INVALID_ARGUMENT, "aliased matrices");
    for (size_t m = 0; m < x.dim[0]; m++) {
        const float *input = x.data + m * (x.stride_bytes[0] / sizeof(float));
        float *output = y.data + m * (y.stride_bytes[0] / sizeof(float));
        for (size_t n = 0; n < w->layout.n_out; n++) {
            const unsigned char *row = w->span.data + expert * w->layout.expert_bytes + n * w->layout.row_bytes;
            float sum = 0;
            for (size_t k = 0; k < w->layout.k_logical; k++) {
                float value = packed_value(row, k, w->layout.gguf_type);
                float product = value * input[k];
                sum += product;
                if (!isfinite(value) || !isfinite(input[k]) || !isfinite(product) || !isfinite(sum))
                    return q38_b_result(err, Q38_NONFINITE, "nonfinite input or arithmetic");
            }
            output[n] = sum;
        }
    }
    return q38_b_result(err, Q38_OK, "");
}
