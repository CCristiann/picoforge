/* Private to A. Public contracts remain owned by the coordinator. */
#ifndef QWEN38_A_INTERNAL_H
#define QWEN38_A_INTERNAL_H
#include "qwen38_text.h"
#include "picoforge.h"
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
/* One shared BPE implementation. Qwen3 retains its original split profile. */
Qwen38Status q38a_bpe_load(Tokenizer *, const Qwen38Bytes *, size_t,
    const Qwen38Bytes *, size_t, Qwen38Error *);
int q38a_bpe_segment(const Tokenizer *, const char *, int, int *, int *, int *, int *,
    unsigned (*category)(unsigned));
unsigned q38a_category(unsigned);
/* Return 0 for incomplete, -1 for malformed, otherwise sequence width. */
static inline int q38a_utf8(const uint8_t *p, size_t n, unsigned *cp) {
    if (!n) return 0;
    unsigned c = p[0]; size_t k = c < 128 ? 1 : c >= 194 && c <= 223 ? 2 :
        c >= 224 && c <= 239 ? 3 : c >= 240 && c <= 244 ? 4 : 0;
    if (!k) return -1;
    for (size_t i = 1; i < k && i < n; ++i)
        if (p[i] < 128 || p[i] > 191) return -1;
    if (n > 1 && ((c == 224 && p[1] < 160) || (c == 237 && p[1] >= 160) ||
                  (c == 240 && p[1] < 144) || (c == 244 && p[1] >= 144))) return -1;
    if (n < k) return 0;
    unsigned v = c & (k == 1 ? 127u : (1u << (7-k))-1);
    for (size_t i = 1; i < k; ++i) v = (v << 6) | (p[i]&63u);
    *cp = v; return (int)k;
}
Qwen38Status q38a_nfc(Qwen38Bytes, uint8_t **, size_t *, Qwen38Error *);
bool q38a_template_matches(Qwen38Bytes);
struct Qwen38Tokenizer {
    Tokenizer bpe;
    uint32_t vocab_size;
    bool skip[64];
    int32_t eos, pad;
};
#define Q38A_TEXT_LIMIT (4u*1024u*1024u) /* <256 MiB total per encode */
static inline bool q38a_string(Qwen38Bytes *rest, Qwen38Bytes *out) {
    if (rest->len < 8) return false;
    uint64_t n = q38a_le(rest->data, 8);
    if (n > rest->len-8) return false;
    *out = (Qwen38Bytes){rest->data+8, (size_t)n};
    rest->data += 8+n; rest->len -= 8+(size_t)n;
    return true;
}
#endif
