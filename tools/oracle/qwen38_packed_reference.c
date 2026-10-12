/* Test-only adapter to unmodified DwarfStar fc80bd695da76ee14bcb86820bf900a5cdfda806.
 * The build supplies DS4_REFERENCE_SOURCE; no decoder is copied into picoforge.
 * Upstream: https://github.com/antirez/ds4 (MIT; its LICENSE accompanies builds).
 * This program consumes small packed matrices, never opens a model or uses GPU.
 */
#define DS4_NO_GPU
#include DS4_REFERENCE_SOURCE

static int reference_error(const char *message) {
    fprintf(stderr, "qwen38 reference: %s\n", message);
    return 2;
}

static bool reference_uint(const char *text, uint64_t *value) {
    if (!*text || strspn(text, "0123456789") != strlen(text)) return false;
    char *end;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end) return false;
    *value = (uint64_t)parsed;
    return true;
}

int main(int argc, char **argv) {
    _Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
                   "reference protocol requires IEEE binary32");
    const uint32_t endian = 1;
    if (*(const unsigned char *)&endian != 1)
        return reference_error("reference protocol requires little endian");
    uint64_t type, columns, rows;
    if (argc != 4 || !reference_uint(argv[1], &type) ||
        !reference_uint(argv[2], &columns) || !reference_uint(argv[3], &rows))
        return reference_error("usage: reference GGUF_TYPE PHYSICAL_COLUMNS ROWS");

    uint64_t block, block_bytes;
    switch (type) {
        case 0: block = 1; block_bytes = 4; break;
        case 1: case 30: block = 1; block_bytes = 2; break;
        case 8: block = 32; block_bytes = 34; break;
        case 10: block = 256; block_bytes = 84; break;
        case 16: block = 256; block_bytes = 66; break;
        default: return reference_error("unsupported ordinary-generation format");
    }
    /* Bound products before allocating; even hostile dimensions cannot wrap. */
    const uint64_t limit = 32u * 1024u * 1024u;
    if (!columns || !rows || columns % block || columns > limit / sizeof(float) ||
        rows > limit / sizeof(float) / columns)
        return reference_error("invalid shape or output exceeds 32 MiB");
    const uint64_t row_bytes = columns / block * block_bytes;
    if (rows > limit / row_bytes)
        return reference_error("packed input exceeds 32 MiB");
    const size_t bytes = (size_t)(rows * row_bytes);
    unsigned char *packed = malloc(bytes);
    float *decoded = malloc((size_t)columns * sizeof(float));
    if (!packed || !decoded) {
        free(packed); free(decoded);
        return reference_error("allocation failed");
    }
    if (fread(packed, 1, bytes, stdin) != bytes || fgetc(stdin) != EOF || ferror(stdin)) {
        free(packed); free(decoded);
        return reference_error("input must contain exactly the declared packed bytes");
    }
    const ds4_model model = {.fd = -1, .map = packed, .size = bytes,
                             .file_size = bytes, .ngram_fd = -1};
    const ds4_tensor tensor = {.ndim = 2, .dim = {columns, rows},
                              .type = (uint32_t)type, .elements = columns * rows,
                              .bytes = bytes};
    int result = 0;
    for (uint64_t row = 0; row < rows; row++) {
        qwen4_ref_row(&model, &tensor, row, decoded);
        if (fwrite(decoded, sizeof(float), (size_t)columns, stdout) != columns) {
            result = reference_error("output write failed");
            break;
        }
    }
    if (fflush(stdout)) result = reference_error("output flush failed");
    free(decoded); free(packed);
    return result;
}
