/* CPU safety harness and bounded binary differential-test transport. */
#include "qwen38_packed.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void expect(Qwen38Status actual, Qwen38Status wanted, const Qwen38Error *err) {
    if (actual != wanted || err->code != wanted) {
        fprintf(stderr, "packed status %d, expected %d: %s\n", actual, wanted, err->detail);
        abort();
    }
}
static void q2_checks(void) {
    unsigned char raw[1 + 84 * 12] = {0}; /* Unaligned weight base is intentional. */
    for (size_t b = 0; b < 12; b++) {
        unsigned char *p = raw + 1 + b * 84;
        memset(p, 0x23, 16); memset(p + 16, 0xe4, 64);
        p[81] = 0x3c; p[83] = 0x38; /* d=1, dmin=.5 */
    }
    GgufTensor tensor = {.type=10, .rank=3, .shape={768, 2, 2}, .bytes=sizeof raw - 1};
    Q38PackedCpuView view = {.span={raw + 1, sizeof raw - 1}};
    Qwen38Error err;
    expect(q38_packed_layout_init(&tensor, 640, &view.layout, &err), Q38_OK, &err);
    assert(view.layout.row_bytes == 252 && view.layout.expert_bytes == 504);
    float out[769];
    for (size_t i=0; i<769; i++) out[i] = 123;
    expect(q38_packed_decode_row(&view, 1, 1, out, 768, &err), Q38_OK, &err);
    const float expected[] = {-1, 2, 5, 8};
    for (size_t i=0; i<768; i++) assert(out[i] == expected[i % 128 / 32]);
    assert(out[768] == 123);
    out[0] = 123;
    expect(q38_packed_decode_row(&view, 2, 0, out, 768, &err), Q38_RANGE, &err);
    expect(q38_packed_decode_row(&view, 0, 2, out, 768, &err), Q38_RANGE, &err);
    expect(q38_packed_decode_row(&view, 0, 0, out, 767, &err), Q38_CAPACITY, &err);
    view.span.bytes--;
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_CAPACITY, &err);
    view.span.bytes++;
    view.layout.row_bytes++;
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_FORMAT, &err);
    view.layout.row_bytes--;
    expect(q38_packed_decode_row(&view, 0, 0, NULL, 768, &err), Q38_INVALID_ARGUMENT, &err);
    expect(q38_packed_decode_row(&view, 0, 0, (float *)(void *)raw, 768, &err), Q38_INVALID_ARGUMENT, &err);
    assert(out[0] == 123); /* All structural errors precede every output write. */
    raw[1 + 81] = 0x7c; /* +Inf half scale. */
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_NONFINITE, &err);
    Q38PackedLayout bad;
    tensor.rank = 4;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_SHAPE, &err);
    assert(!bad.tensor_bytes);
    tensor.rank = 3; tensor.shape[0] = 640;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_FORMAT, &err);
    tensor.shape[0] = 768; tensor.shape[1] = SIZE_MAX;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_OVERFLOW, &err);
    tensor.type = 12;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_UNSUPPORTED, &err);
    tensor.type = UINT32_MAX;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_FORMAT, &err);
    puts("PASS: Q2_K layout, decode, bounds, padding and error preflight");
}

static size_t argument(const char *text) {
    assert(*text && strspn(text, "0123456789") == strlen(text));
    errno = 0;
    unsigned long long value = strtoull(text, NULL, 10);
    assert(!errno && value <= SIZE_MAX);
    return (size_t)value;
}
static int transport(int argc, char **argv) {
    assert(argc == 4);
    size_t type = argument(argv[1]), columns = argument(argv[2]), rows = argument(argv[3]);
    const size_t limit = 32u * 1024u * 1024u;
    assert(type <= UINT32_MAX && columns && rows && columns <= limit / 4 && rows <= limit / 4 / columns);
    /* Registry only called for these known ordinary formats. */
    assert(type == 0 || type == 1 || type == 8 || type == 10 || type == 16 || type == 30);
    const GgufFormat *format = gguf_format((uint32_t)type);
    assert(columns % format->block == 0);
    size_t bytes = columns / format->block * format->bytes * rows;
    assert(bytes <= limit);
    unsigned char *raw = malloc(bytes);
    float *out = malloc(columns * rows * sizeof *out);
    assert(raw && out);
    assert(fread(raw, 1, bytes, stdin) == bytes && fgetc(stdin) == EOF && !ferror(stdin));
    GgufTensor tensor = {.type=(uint32_t)type, .rank=2, .shape={columns,rows}, .bytes=bytes};
    Q38PackedCpuView view = {.span={raw,bytes}};
    Qwen38Error err;
    Qwen38Status status = q38_packed_layout_init(&tensor, columns, &view.layout, &err);
    for (size_t row=0; status == Q38_OK && row<rows; row++)
        status = q38_packed_decode_row(&view, 0, row, out + row * columns, columns, &err);
    if (status != Q38_OK) fprintf(stderr, "qwen38 native: %d %s\n", status, err.detail);
    else assert(fwrite(out, sizeof *out, columns * rows, stdout) == columns * rows);
    free(raw); free(out);
    return (int)status;
}
int main(int argc, char **argv) {
    if (argc > 1) return transport(argc, argv);
    q2_checks();
    return 0;
}
