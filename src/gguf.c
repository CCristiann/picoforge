/* A deliberately narrow qwen4exp GGUF v3 loader. No tensor payload is read,
 * converted, or exposed to Metal here. File contents must remain immutable. */
#include "picoforge.h"
#include "gguf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

const GgufFormat *gguf_format(uint32_t id) {
    static const GgufFormat formats[] = {
        {0, "F32", 1, 4}, {1, "F16", 1, 2}, {8, "Q8_0", 32, 34},
        {10, "Q2_K", 256, 84}, {12, "Q4_K", 256, 144},
        {16, "IQ2_XXS", 256, 66}, {30, "BF16", 1, 2}, {39, "MXFP4", 32, 17}
    };
    for (size_t i = 0; i < sizeof formats / sizeof *formats; i++)
        if (formats[i].id == id) return &formats[i];
    die("gguf: unsupported tensor format %u", id);
    return NULL;
}

/* Both descriptor types begin with their name array. */
static int by_name(const void *a, const void *b) { return strcmp(a, b); }
static int by_offset(const void *a, const void *b) {
    size_t x = ((const GgufTensor *)a)->file_offset, y = ((const GgufTensor *)b)->file_offset;
    return (x > y) - (x < y);
}

void gguf_open(const char *path, Gguf *g) {
    memset(g, 0, sizeof *g);
    g->fd = open(path, O_RDONLY);
    struct stat st;
    if (g->fd < 0 || fstat(g->fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 24)
        die("gguf: cannot open a complete regular file: %s", path);
    g->file_size = (size_t)st.st_size;
    g->header_size = g->file_size < 128 * 1024 * 1024 ? g->file_size : 128 * 1024 * 1024;
    g->header = mmap(NULL, g->header_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->header == MAP_FAILED) die("gguf: header mapping failed");
    GgufCursor c = {g->header, g->header_size, 0};
    if (memcmp(gguf_bytes(&c, 4), "GGUF", 4) || gguf_uint(&c, 4) != 3)
        die("gguf: expected little-endian GGUF v3");
    g->tensor_count = (size_t)gguf_uint(&c, 8);
    g->metadata_count = (size_t)gguf_uint(&c, 8);
    if (!g->tensor_count || g->tensor_count > 100000 || g->metadata_count > 100000)
        die("gguf: invalid descriptor counts");
    g->tensors = calloc(g->tensor_count, sizeof *g->tensors);
    g->metadata = calloc(g->metadata_count + 1, sizeof *g->metadata);
    if (!g->tensors || !g->metadata) die("gguf: descriptor allocation failed");
    size_t alignment = 32;
    bool architecture = false;
    for (size_t i = 0; i < g->metadata_count; i++) {
        GgufMetadata *m = &g->metadata[i];
        gguf_name(&c, m->name, sizeof m->name);
        m->type = (uint32_t)gguf_uint(&c, 4);
        m->offset = c.pos;
        gguf_skip_value(&c, m->type);
        m->bytes = c.pos - m->offset;
        GgufCursor value = {c.data, c.pos, m->offset};
        if (!strcmp(m->name, "general.architecture")) {
            if (m->type != 8) die("gguf: architecture must be a string");
            size_t n;
            const unsigned char *s = gguf_string(&value, &n);
            architecture = n == 8 && !memcmp(s, "qwen4exp", 8);
        } else if (!strcmp(m->name, "general.alignment")) {
            if (m->type != 4) die("gguf: alignment must be uint32");
            alignment = (size_t)gguf_uint(&value, 4);
        }
    }
    qsort(g->metadata, g->metadata_count, sizeof *g->metadata, by_name);
    for (size_t i = 1; i < g->metadata_count; i++)
        if (!strcmp(g->metadata[i-1].name, g->metadata[i].name)) die("gguf: duplicate metadata key");
    if (!architecture) die("gguf: expected qwen4exp architecture");
    if (!alignment || (alignment & (alignment - 1)) || alignment > 128 * 1024 * 1024)
        die("gguf: invalid alignment");
    for (size_t i = 0; i < g->tensor_count; i++) {
        GgufTensor *t = &g->tensors[i];
        gguf_name(&c, t->name, sizeof t->name);
        t->rank = (uint32_t)gguf_uint(&c, 4);
        if (!t->rank || t->rank > 4) die("gguf: invalid tensor rank");
        for (uint32_t d = 0; d < t->rank; d++) {
            t->shape[d] = (size_t)gguf_uint(&c, 8);
            if (!t->shape[d]) die("gguf: zero tensor dimension");
        }
        t->type = (uint32_t)gguf_uint(&c, 4);
        t->file_offset = (size_t)gguf_uint(&c, 8); /* relative until header is complete */
        const GgufFormat *f = gguf_format(t->type);
        if (t->shape[0] % f->block || t->file_offset % alignment)
            die("gguf: unaligned tensor row or offset: %s", t->name);
        t->bytes = gguf_mul(t->shape[0] / f->block, f->bytes);
        for (uint32_t d = 1; d < t->rank; d++) t->bytes = gguf_mul(t->bytes, t->shape[d]);
    }
    g->data_offset = gguf_add(c.pos, alignment - 1) & ~(alignment - 1);
    (void)gguf_bytes(&c, g->data_offset - c.pos); /* also require alignment padding */
    qsort(g->tensors, g->tensor_count, sizeof *g->tensors, by_name);
    for (size_t i = 1; i < g->tensor_count; i++)
        if (!strcmp(g->tensors[i-1].name, g->tensors[i].name)) die("gguf: duplicate tensor name");
    qsort(g->tensors, g->tensor_count, sizeof *g->tensors, by_offset);
    size_t end = g->data_offset;
    for (size_t i = 0; i < g->tensor_count; i++) {
        GgufTensor *t = &g->tensors[i];
        t->file_offset = gguf_add(g->data_offset, t->file_offset);
        if (t->file_offset < end || t->file_offset > g->file_size || t->bytes > g->file_size - t->file_offset)
            die("gguf: overlapping or out-of-file tensor: %s", t->name);
        end = t->file_offset + t->bytes;
    }
}

void gguf_close(Gguf *g) {
    if (g->header) { munmap(g->header, g->header_size); close(g->fd); }
    free(g->tensors);
    free(g->metadata);
    memset(g, 0, sizeof *g);
    g->fd = -1;
}

void gguf_inspect(const Gguf *g) {
    printf("{\"architecture\":\"qwen4exp\",\"payload_verified\":false,"
           "\"actual_file_bytes\":%zu,\"data_offset\":%zu,\"metadata_count\":%zu,\"tensors\":[\n",
           g->file_size, g->data_offset, g->metadata_count);
    for (size_t i = 0; i < g->tensor_count; i++) {
        const GgufTensor *t = &g->tensors[i];
        printf("%s{\"name\":\"%s\",\"type_id\":%u,\"format\":\"%s\",\"shape\":[",
               i ? ",\n" : "", t->name, t->type, gguf_format(t->type)->name);
        for (uint32_t d = 0; d < t->rank; d++) printf("%s%zu", d ? "," : "", t->shape[d]);
        printf("],\"file_offset\":%zu,\"size_bytes\":%zu}", t->file_offset, t->bytes);
    }
    puts("\n]}");
}
