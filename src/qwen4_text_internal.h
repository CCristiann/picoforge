/* Private helpers of the text layer (config, tokenizer, chat template); the public contracts are in qwen4_text.h. */
#ifndef PICOFORGE_QWEN4_TEXT_INTERNAL_H
#define PICOFORGE_QWEN4_TEXT_INTERNAL_H
#include "qwen4_text.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The header gate compiles private headers as translation
 * units too. Keep header-only helpers without unused-function diagnostics. */
#if defined(__clang__) || defined(__GNUC__)
#define QW4A_INLINE static inline __attribute__((unused))
#else
#define QW4A_INLINE static inline
#endif

struct Qwen4Config {
    Qwen4ConfigView view;
    Qwen4Layer *layers;
    Qwen4NgramConfig ngram;
    uint64_t *multipliers, *offsets, *vocabs;
    uint32_t inner, ple_divisor;
    bool has_ple;                 /* qwen4exp.ple.layers present: one PLE site */
};
QW4A_INLINE Qwen4Status qw4a_error(Qwen4Error *e, Qwen4Status s,
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
Qwen4Status qw4a_meta(const Gguf *, const char *, uint32_t, Qwen4Bytes *, Qwen4Error *);
Qwen4Status qw4a_array(const Gguf *, const char *, uint32_t, Qwen4Bytes *, size_t *, Qwen4Error *);
QW4A_INLINE uint64_t qw4a_le(const uint8_t *p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8*i);
    return v;
}
QW4A_INLINE bool qw4a_mul(size_t a, size_t b, size_t *r) {
    if (a && b > SIZE_MAX/a) return false;
    *r = a*b;
    return true;
}
/* ---- byte-level BPE tables (qwen4_bpe.c) ----
 * Every token is the byte string it stands for; the GPT-2 alphabet used by
 * the file format is undone at load time. */

/* An added token: matched in the raw text before the pre-tokenizer and before
 * BPE, because no merge rule can build one and none must ever be allowed to. */
typedef struct { int id; int len; unsigned char text[64]; } SpecialToken;

typedef struct {
    unsigned char *blob;        /* every token's bytes, concatenated        */
    int           *offset;      /* [n_tokens + 1] indices into blob         */
    int            n_tokens;
    int            n_merges;

    /* byte string -> id. Open addressing, slot holds id + 1 (0 = empty). */
    int           *str_slot;
    unsigned       str_mask;

    /* (left_id << 32 | right_id) -> merge rank. rank -1 marks an empty slot. */
    uint64_t      *merge_key;
    int           *merge_rank;
    unsigned       merge_mask;

    SpecialToken   specials[64];
    int            n_specials;
} Tokenizer;

int  tokenizer_find(const Tokenizer *t, const unsigned char *b, int n); /* -1 absent */
int  tokenizer_rank(const Tokenizer *t, int left, int right);          /* -1 absent */
void tokenizer_free(Tokenizer *t);

/* Unicode tables (qwen4_unicode.c): sorted codepoint ranges for the general
 * categories L, N and M, and the NFC composition data. */
typedef struct { unsigned lo, hi; } CodepointRange;
typedef struct { unsigned a, b, cp; } Composition;
typedef struct { unsigned cp; unsigned char ccc; } CombiningClass;

Qwen4Status qw4a_bpe_load(Tokenizer *, const Qwen4Bytes *, size_t,
    const Qwen4Bytes *, size_t, Qwen4Error *);
int qw4a_bpe_segment(const Tokenizer *, const char *, int, int *, int *, int *, int *,
    unsigned (*category)(unsigned));
unsigned qw4a_category(unsigned);
/* Return 0 for incomplete, -1 for malformed, otherwise sequence width. */
QW4A_INLINE int qw4a_utf8(const uint8_t *p, size_t n, unsigned *cp) {
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
Qwen4Status qw4a_nfc(Qwen4Bytes, uint8_t **, size_t *, Qwen4Error *);
bool qw4a_template_matches(Qwen4Bytes);
struct Qwen4Tokenizer {
    Tokenizer bpe;
    uint32_t vocab_size;
    bool skip[64];
    int32_t eos, pad;
};
#define QW4A_TEXT_LIMIT (4u*1024u*1024u) /* <256 MiB total per encode */
QW4A_INLINE bool qw4a_string(Qwen4Bytes *rest, Qwen4Bytes *out) {
    if (rest->len < 8) return false;
    uint64_t n = qw4a_le(rest->data, 8);
    if (n > rest->len-8) return false;
    *out = (Qwen4Bytes){rest->data+8, (size_t)n};
    rest->data += 8+n; rest->len -= 8+(size_t)n;
    return true;
}
#undef QW4A_INLINE
#endif
