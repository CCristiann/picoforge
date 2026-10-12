/* Bounded binary fixture transport; compiled with ASan/UBSan. CPU only. */
#include "qwen38_d_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void read_items(void *p, size_t size, size_t n) {
    assert(fread(p, size, n, stdin) == n);
}
static void write_items(const void *p, size_t size, size_t n) {
    assert(fwrite(p, size, n, stdout) == n);
}
static float *read_floats(size_t n) {
    assert(n <= 4000000);
    float *p = malloc((n ? n : 1) * sizeof(float));
    assert(p); read_items(p, sizeof(float), n); return p;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    uint32_t dims[4];
    read_items(dims, sizeof(uint32_t), 4);
    for (size_t i = 0; i < 4; ++i) assert(dims[i] <= 32768);
    Qwen38Status status;
    if (!strcmp(argv[1], "rope")) {
        float theta;
        read_items(&theta, sizeof(float), 1);
        size_t n = (size_t)dims[0] * dims[1];
        float *x = read_floats(n);
        status = q38d_rope(x, dims[0], dims[1], dims[2], dims[3], theta);
        write_items(&status, sizeof(status), 1); write_items(x, sizeof(float), n); free(x);
    } else if (!strcmp(argv[1], "scores")) {
        float *q = read_floats((size_t)dims[0] * dims[1]);
        float *k = read_floats((size_t)dims[2] * dims[1]);
        float *s = calloc(dims[2] ? dims[2] : 1, sizeof(float)); assert(s);
        status = q38d_scores(q, k, dims[0], dims[1], dims[2], s);
        write_items(&status, sizeof(status), 1); write_items(s, sizeof(float), dims[2]);
        free(q); free(k); free(s);
    } else {
        assert(!strcmp(argv[1], "select"));
        float *s = read_floats(dims[0]);
        uint32_t n = dims[0] < dims[1] ? dims[0] : dims[1];
        uint32_t *heap = calloc(n ? n : 1, sizeof(uint32_t));
        uint32_t *ids = calloc(n ? n : 1, sizeof(uint32_t)); assert(heap && ids);
        status = q38d_select(s, dims[0], dims[1], heap, ids);
        write_items(&status, sizeof(status), 1); write_items(ids, sizeof(uint32_t), n);
        free(s); free(heap); free(ids);
    }
    assert(getchar() == EOF);
    return 0;
}
