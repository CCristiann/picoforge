/* Little-endian reads are checked before advancing; no unaligned casts. */
#include "picoforge.h"
#include "gguf.h"

_Static_assert(sizeof(size_t) == sizeof(uint64_t), "GGUF requires a 64-bit host");

size_t gguf_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) die("gguf: offset overflow");
    return a + b;
}

size_t gguf_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a) die("gguf: size overflow");
    return a * b;
}

const unsigned char *gguf_bytes(GgufCursor *c, size_t n) {
    if (c->pos > c->size || n > c->size - c->pos)
        die("gguf: truncated or oversized header");
    const unsigned char *p = c->data + c->pos;
    c->pos += n;
    return p;
}

uint64_t gguf_uint(GgufCursor *c, size_t width) {
    if (width != 1 && width != 2 && width != 4 && width != 8)
        die("gguf: invalid integer width");
    const unsigned char *p = gguf_bytes(c, width);
    uint64_t value = 0;
    for (size_t i = 0; i < width; i++) value |= (uint64_t)p[i] << (8 * i);
    return value;
}

const unsigned char *gguf_string(GgufCursor *c, size_t *length) {
    uint64_t n = gguf_uint(c, 8);
    if (n > 16 * 1024 * 1024) die("gguf: oversized string");
    *length = (size_t)n;
    return gguf_bytes(c, *length);
}

void gguf_name(GgufCursor *c, char *out, size_t capacity) {
    size_t n;
    const unsigned char *p = gguf_string(c, &n);
    if (!n || n >= capacity) die("gguf: invalid name length");
    /* The selected architecture uses ASCII identifiers. Tokenizer strings
     * are unrestricted byte strings and never pass through this function. */
    for (size_t i = 0; i < n; i++)
        if (!((p[i] >= 'a' && p[i] <= 'z') || (p[i] >= 'A' && p[i] <= 'Z') ||
              (p[i] >= '0' && p[i] <= '9') || p[i] == '_' || p[i] == '.' || p[i] == '-'))
            die("gguf: unsupported identifier byte");
    memcpy(out, p, n);
    out[n] = 0;
}

void gguf_skip_value(GgufCursor *c, uint32_t type) {
    static const size_t widths[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type >= sizeof widths / sizeof *widths) die("gguf: unsupported metadata type");
    if (type == 8) {
        size_t n;
        (void)gguf_string(c, &n);
    } else if (type == 9) {
        uint32_t element = (uint32_t)gguf_uint(c, 4);
        uint64_t count = gguf_uint(c, 8);
        if (element >= sizeof widths / sizeof *widths || element == 9 || count > 2000000)
            die("gguf: invalid metadata array");
        for (uint64_t i = 0; i < count; i++) gguf_skip_value(c, element);
    } else if (type == 7) {
        if (gguf_uint(c, 1) > 1) die("gguf: invalid boolean");
    } else {
        (void)gguf_bytes(c, widths[type]);
    }
}
