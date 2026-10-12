#include "gguf.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const unsigned char bytes[] = {1, 2, 3, 4, 5, 6, 7, 8};
    GgufCursor c = {bytes, sizeof bytes, 0};
    if (argc == 2) {
        if (!strcmp(argv[1], "truncated")) (void)gguf_bytes(&c, 9);
        if (!strcmp(argv[1], "add")) (void)gguf_add(SIZE_MAX, 1);
        if (!strcmp(argv[1], "mul")) (void)gguf_mul(SIZE_MAX, 2);
        if (!strcmp(argv[1], "array")) gguf_skip_value(&c, 9);
        if (!strcmp(argv[1], "type")) gguf_skip_value(&c, 13);
        if (!strcmp(argv[1], "string")) { size_t n; (void)gguf_string(&c, &n); }
        return 2; /* rejection must exit via die(), never reach here */
    }
    assert(gguf_uint(&c, 8) == UINT64_C(0x0807060504030201));
    assert(c.pos == 8);
    assert(gguf_add(12, 30) == 42 && gguf_mul(6, 7) == 42);
    const unsigned char array[] = {4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0};
    c = (GgufCursor){array, sizeof array, 0};
    gguf_skip_value(&c, 9);
    assert(c.pos == sizeof array);
    puts("PASS: bounded GGUF integer and metadata reads");
    return 0;
}
