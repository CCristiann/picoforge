/* normalize.c — Unicode NFC, because the tokenizer's config asks for it.
 *
 * Why this file exists at all: "é" has two encodings. U+00E9 is the composed
 * one; U+0065 U+0301 is "e" followed by a combining acute. They look
 * identical, compare unequal, and tokenise to completely different ids.
 * HuggingFace runs NFC before the pre-tokenizer, so skipping it means every
 * accented prompt can diverge — and macOS in particular hands out decomposed
 * text, so this is not a theoretical case for a Mac-only engine.
 *
 * NFC is three passes (UAX #15):
 *   1. decompose everything canonically, recursively;
 *   2. sort each run of combining marks by combining class, stably;
 *   3. recompose greedily onto the last starter.
 *
 * Hangul is handled by arithmetic in steps 1 and 3 rather than by table:
 * its 11172 syllables decompose into jamo by division and compose back by
 * multiplication.
 */
#include "picoforge.h"

#include <stdlib.h>

#define SBASE  0xAC00
#define LBASE  0x1100
#define VBASE  0x1161
#define TBASE  0x11A7
#define LCOUNT 19
#define VCOUNT 21
#define TCOUNT 28
#define NCOUNT (VCOUNT * TCOUNT)     /* 588  */
#define SCOUNT (LCOUNT * NCOUNT)     /* 11172 */

static int ccc_of(unsigned cp) {
    int lo = 0, hi = unicode_ccc_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (cp < unicode_ccc[mid].cp)      hi = mid - 1;
        else if (cp > unicode_ccc[mid].cp) lo = mid + 1;
        else return unicode_ccc[mid].ccc;
    }
    return 0;
}

static const Decomposition *decomp_of(unsigned cp) {
    int lo = 0, hi = unicode_decomp_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (cp < unicode_decomp[mid].cp)      hi = mid - 1;
        else if (cp > unicode_decomp[mid].cp) lo = mid + 1;
        else return &unicode_decomp[mid];
    }
    return NULL;
}

static int compose_pair(unsigned a, unsigned b) {
    /* Hangul L + V, then LV + T. Arithmetic, not tabulated. */
    if (a >= LBASE && a < LBASE + LCOUNT && b >= VBASE && b < VBASE + VCOUNT)
        return (int)(SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT);
    if (a >= SBASE && a < SBASE + SCOUNT && (a - SBASE) % TCOUNT == 0
        && b > TBASE && b < TBASE + TCOUNT)
        return (int)(a + (b - TBASE));

    int lo = 0, hi = unicode_compose_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        const Composition *c = &unicode_compose[mid];
        if (a < c->a || (a == c->a && b < c->b))      hi = mid - 1;
        else if (a > c->a || (a == c->a && b > c->b)) lo = mid + 1;
        else return (int)c->cp;
    }
    return -1;
}

/* Full canonical decomposition of one codepoint. At most 4 codepoints come
 * out (Hangul LVT is 3; the deepest canonical chain is 2). */
static int decompose(unsigned cp, unsigned *out) {
    if (cp >= SBASE && cp < SBASE + SCOUNT) {
        unsigned s = cp - SBASE;
        out[0] = LBASE + s / NCOUNT;
        out[1] = VBASE + (s % NCOUNT) / TCOUNT;
        if (s % TCOUNT) { out[2] = TBASE + s % TCOUNT; return 3; }
        return 2;
    }

    const Decomposition *d = decomp_of(cp);
    if (!d) { out[0] = cp; return 1; }

    /* Recursive: a canonical decomposition's parts may decompose further. */
    unsigned tmp[4];
    int n = 0;
    int k = decompose(d->a, tmp);
    for (int i = 0; i < k; i++) out[n++] = tmp[i];
    if (d->b) {
        k = decompose(d->b, tmp);
        for (int i = 0; i < k; i++) out[n++] = tmp[i];
    }
    return n;
}

static const char *utf8_decode(const char *p, const char *end, unsigned *cp) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80)              { *cp = c; return p + 1; }
    if ((c & 0xE0) == 0xC0 && p + 1 < end) {
        *cp = ((unsigned)(c & 0x1F) << 6) | (unsigned)(p[1] & 0x3F);
        return p + 2;
    }
    if ((c & 0xF0) == 0xE0 && p + 2 < end) {
        *cp = ((unsigned)(c & 0x0F) << 12) | ((unsigned)(p[1] & 0x3F) << 6)
            | (unsigned)(p[2] & 0x3F);
        return p + 3;
    }
    if ((c & 0xF8) == 0xF0 && p + 3 < end) {
        *cp = ((unsigned)(c & 0x07) << 18) | ((unsigned)(p[1] & 0x3F) << 12)
            | ((unsigned)(p[2] & 0x3F) << 6) | (unsigned)(p[3] & 0x3F);
        return p + 4;
    }
    die("normalize: invalid UTF-8 at byte 0x%02X", c);
    return p;                                          /* unreachable */
}

static int utf8_encode(unsigned cp, char *out) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));
                        out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12));
                        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

int nfc_normalize(const char *in, int len, char *out, int cap) {
    if (len == 0) { if (cap > 0) out[0] = '\0'; return 0; }

    /* One byte cannot become more than one codepoint, and one codepoint
     * decomposes into at most four. */
    unsigned *cps = malloc((size_t)len * 4 * sizeof *cps);
    if (!cps) die("out of memory normalising %d bytes", len);

    int n = 0;
    for (const char *p = in, *end = in + len; p < end;) {
        unsigned cp;
        p = utf8_decode(p, end, &cp);
        n += decompose(cp, cps + n);
    }

    /* Canonical ordering: an insertion sort over runs of combining marks.
     * It must be STABLE — marks of equal class keep their order, because
     * their order is meaningful — which is exactly what stopping at
     * ccc[j-1] > c, rather than >=, gives. A starter has class 0 and so
     * always stops the loop: runs never merge across one. */
    for (int i = 1; i < n; i++) {
        int c = ccc_of(cps[i]);
        if (c == 0) continue;
        int j = i;
        while (j > 0 && ccc_of(cps[j - 1]) > c) {
            unsigned t = cps[j - 1]; cps[j - 1] = cps[j]; cps[j] = t;
            j--;
        }
    }

    /* Composition. A character composes onto the last starter only if it is
     * not "blocked": nothing between them may have a combining class greater
     * than or equal to its own. prev_ccc == -1 means it sits immediately
     * after the starter, which is how two starters (Hangul L + V) compose. */
    int m = 0, last_starter = -1, prev_ccc = -1;
    for (int i = 0; i < n; i++) {
        unsigned cp = cps[i];
        int cc = ccc_of(cp);

        if (last_starter >= 0 && (prev_ccc == -1 || prev_ccc < cc)) {
            int composed = compose_pair(cps[last_starter], cp);
            if (composed >= 0) {
                cps[last_starter] = (unsigned)composed;
                continue;            /* absorbed: it blocks nothing */
            }
        }
        if (cc == 0) { last_starter = m; prev_ccc = -1; }
        else prev_ccc = cc;
        cps[m++] = cp;
    }

    int outlen = 0;
    for (int i = 0; i < m; i++) {
        if (outlen + 4 >= cap) die("normalize: output exceeds %d bytes", cap);
        outlen += utf8_encode(cps[i], out + outlen);
    }
    out[outlen] = '\0';

    free(cps);
    return outlen;
}
