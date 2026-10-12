/* Private to A. Public contracts remain owned by the coordinator. */
#ifndef QWEN38_A_INTERNAL_H
#define QWEN38_A_INTERNAL_H
#include "qwen38_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Qwen38Config {
    Qwen38ConfigView view;
    Qwen38Layer *layers;
    Qwen38NgramConfig ngram;
    uint64_t *multipliers, *offsets, *vocabs;
    uint32_t interval, inner, ple_divisor;
};
static inline Qwen38Status q38a_error(Qwen38Error *e, Qwen38Status s,
                                     const char *subject, const char *detail) {
    if (e) {
        memset(e, 0, sizeof *e);
        e->code = s;
        snprintf(e->subject, sizeof e->subject, "%s", subject ? subject : "");
        snprintf(e->detail, sizeof e->detail, "%s", detail ? detail : "");
    }
    return s;
}
/* Unlike the CLI reader these metadata helpers never abort the process. */
Qwen38Status q38a_meta(const Gguf *, const char *, uint32_t, Qwen38Bytes *, Qwen38Error *);
Qwen38Status q38a_array(const Gguf *, const char *, uint32_t, Qwen38Bytes *, size_t *, Qwen38Error *);
static inline uint64_t q38a_le(const uint8_t *p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8*i);
    return v;
}
static inline bool q38a_mul(size_t a, size_t b, size_t *r) {
    if (a && b > SIZE_MAX/a) return false;
    *r = a*b;
    return true;
}
#endif
