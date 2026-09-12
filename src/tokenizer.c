/* tokenizer.c — byte-level BPE: the vocabulary and the merge table.
 *
 * Two files describe it. vocab.json maps a token's spelling to its id, and
 * merges.txt lists the merge rules in priority order, best first. Neither
 * spells tokens in bytes: they use the 256-character alphabet GPT-2 invented
 * so that every byte, control codes included, has a printable stand-in that
 * survives a text file. ' ' is written 'Ġ', newline is 'Ċ'.
 *
 * That mapping is a bijection, so we undo it at load time and keep every
 * token as the byte string it actually means. The encoder never sees Unicode
 * again — which removes a whole class of bug, because BPE compares and
 * concatenates strings, and doing that on UTF-8 by codepoint instead of by
 * byte is wrong in ways that only show up on non-ASCII input.
 */
#include "picoforge.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

/* ----------------------------------------------------------- alphabet */

/* GPT-2's bytes_to_unicode, inverted: codepoint -> the byte it stands for.
 *
 * Bytes that are already printable ASCII or printable Latin-1 stand for
 * themselves; the other 68 are pushed above the BMP-free zone at 256, in
 * increasing byte order. Hence 0x20 -> U+0120 'Ġ' and 0x0A -> U+010A 'Ċ',
 * and hence exactly 324 possible codepoints. */
#define ALPHABET_CP 324

static void build_alphabet(int *cp_to_byte) {
    for (int i = 0; i < ALPHABET_CP; i++) cp_to_byte[i] = -1;

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172)
                      || (b >= 174 && b <= 255);
        int cp = printable ? b : 256 + extra;
        if (!printable) extra++;
        cp_to_byte[cp] = b;
    }
}

/* One UTF-8 sequence -> codepoint. Rejects overlong forms and truncation:
 * a tokenizer that quietly accepts malformed UTF-8 produces token ids that
 * cannot be decoded back, and the failure surfaces much later as garbled
 * output rather than as a parse error here. */
static const char *utf8_next(const char *p, int *cp) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80)        { *cp = c;            return p + 1; }
    if ((c & 0xE0) == 0xC0) {
        if ((p[1] & 0xC0) != 0x80) die("tokenizer: truncated UTF-8");
        *cp = ((c & 0x1F) << 6) | (p[1] & 0x3F);
        if (*cp < 0x80) die("tokenizer: overlong UTF-8");
        return p + 2;
    }
    if ((c & 0xF0) == 0xE0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80)
            die("tokenizer: truncated UTF-8");
        *cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        if (*cp < 0x800) die("tokenizer: overlong UTF-8");
        return p + 3;
    }
    if ((c & 0xF8) == 0xF0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80)
            die("tokenizer: truncated UTF-8");
        *cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12)
            | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        if (*cp < 0x10000) die("tokenizer: overlong UTF-8");
        return p + 4;
    }
    die("tokenizer: invalid UTF-8 lead byte 0x%02X", c);
    return p;                                          /* unreachable */
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    die("tokenizer: bad \\u escape in vocab.json");
    return 0;
}

/* Read one JSON string and write the BYTES it denotes. Handles the escapes
 * vocab.json actually uses (1666 of its keys contain a quote or backslash)
 * and maps every resulting codepoint back through the alphabet.
 *
 * A codepoint outside the alphabet aborts. That is not pedantry: it means
 * the file is not byte-level BPE, and continuing would silently build a
 * vocabulary that cannot represent what the model was trained on. */
static const char *read_token(const char *p, unsigned char *out, int cap, int *len,
                              const int *cp_to_byte) {
    p = json_expect(p, '"');
    int n = 0;
    while (*p && *p != '"') {
        int cp;
        if (*p == '\\') {
            p++;
            switch (*p) {
            case '"': cp = '"';  p++; break;
            case '\\': cp = '\\'; p++; break;
            case '/': cp = '/';  p++; break;
            case 'b': cp = '\b'; p++; break;
            case 'f': cp = '\f'; p++; break;
            case 'n': cp = '\n'; p++; break;
            case 'r': cp = '\r'; p++; break;
            case 't': cp = '\t'; p++; break;
            case 'u':
                cp = (hexval(p[1]) << 12) | (hexval(p[2]) << 8)
                   | (hexval(p[3]) << 4) | hexval(p[4]);
                p += 5;
                break;
            default: die("tokenizer: unknown escape \\%c in vocab.json", *p);
                     return p;
            }
        } else {
            p = utf8_next(p, &cp);
        }

        if (cp < 0 || cp >= ALPHABET_CP || cp_to_byte[cp] < 0)
            die("tokenizer: codepoint U+%04X is not in the byte-level alphabet", cp);
        if (n >= cap) die("tokenizer: token longer than %d bytes", cap);
        out[n++] = (unsigned char)cp_to_byte[cp];
    }
    if (*p != '"') die("tokenizer: unterminated string in vocab.json");
    *len = n;
    return p + 1;
}

/* ------------------------------------------------------------- hashing */

static unsigned fnv1a(const unsigned char *s, int n) {
    unsigned h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= s[i]; h *= 16777619u; }
    return h;
}

/* splitmix64's finalizer: cheap, and it spreads the low bits of a packed
 * (left, right) pair, which a plain xor-fold would leave clustered. */
static unsigned hash64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return (unsigned)((x ^ (x >> 31)) & 0xFFFFFFFFu);
}

static unsigned pow2_at_least(int n) {
    unsigned cap = 16;
    while (cap < (unsigned)n * 2u) cap <<= 1;   /* keep the load factor < 0.5 */
    return cap;
}

int tokenizer_find(const Tokenizer *t, const unsigned char *b, int n) {
    unsigned i = fnv1a(b, n) & t->str_mask;
    for (;;) {
        int slot = t->str_slot[i];
        if (slot == 0) return -1;
        int id = slot - 1;
        int len = t->offset[id + 1] - t->offset[id];
        if (len == n && memcmp(t->blob + t->offset[id], b, (size_t)n) == 0) return id;
        i = (i + 1) & t->str_mask;
    }
}

static void str_insert(Tokenizer *t, int id) {
    int len = t->offset[id + 1] - t->offset[id];
    unsigned i = fnv1a(t->blob + t->offset[id], len) & t->str_mask;
    while (t->str_slot[i] != 0) i = (i + 1) & t->str_mask;
    t->str_slot[i] = id + 1;
}

int tokenizer_rank(const Tokenizer *t, int left, int right) {
    uint64_t key = ((uint64_t)(unsigned)left << 32) | (unsigned)right;
    unsigned i = hash64(key) & t->merge_mask;
    for (;;) {
        if (t->merge_rank[i] < 0) return -1;
        if (t->merge_key[i] == key) return t->merge_rank[i];
        i = (i + 1) & t->merge_mask;
    }
}

static void merge_insert(Tokenizer *t, int left, int right, int rank) {
    uint64_t key = ((uint64_t)(unsigned)left << 32) | (unsigned)right;
    unsigned i = hash64(key) & t->merge_mask;
    while (t->merge_rank[i] >= 0) {
        if (t->merge_key[i] == key) return;   /* first rule wins: it ranks better */
        i = (i + 1) & t->merge_mask;
    }
    t->merge_key[i] = key;
    t->merge_rank[i] = rank;
}

/* ------------------------------------------------------------- loading */

/* Decode a run of alphabet characters (plain UTF-8, as merges.txt writes
 * them) into the bytes they stand for. */
static int decode_alphabet(const char *p, const char *end, unsigned char *out,
                           int cap, const int *cp_to_byte) {
    int n = 0;
    while (p < end) {
        int cp;
        p = utf8_next(p, &cp);
        if (cp < 0 || cp >= ALPHABET_CP || cp_to_byte[cp] < 0)
            die("merges.txt: codepoint U+%04X is not in the byte-level alphabet", cp);
        if (n >= cap) die("merges.txt: token longer than %d bytes", cap);
        out[n++] = (unsigned char)cp_to_byte[cp];
    }
    return n;
}

static const char *entry_start(const char *p) {
    while (isspace((unsigned char)*p) || *p == ',') p++;
    return p;
}

static void load_vocab(const char *model_dir, Tokenizer *t, const int *cp_to_byte) {
    char path[1024];
    snprintf(path, sizeof path, "%s/vocab.json", model_dir);
    char *json = slurp(path, NULL);

    /* Three walks over 2.7 MB rather than one clever one. The ids decide
     * where each token lives in the blob, and they only stop arriving once
     * the file ends, so lengths must all be known before anything is placed.
     * Counting is cheap; guessing would not be. */
    int n = 0;
    const char *p = json_expect(json, '{');
    while (*(p = entry_start(p)) && *p != '}') {
        p = json_expect(json_skip(p), ':');
        p = json_skip(p);
        n++;
    }
    if (n == 0) die("vocab.json contains no entries");

    int *len = calloc((size_t)n, sizeof *len);
    if (!len) die("out of memory for %d token lengths", n);

    unsigned char buf[512];
    p = json_expect(json, '{');
    while (*(p = entry_start(p)) && *p != '}') {
        int tl;
        p = read_token(p, buf, (int)sizeof buf, &tl, cp_to_byte);
        p = json_expect(p, ':');
        char *end;
        long id = strtol(p, &end, 10);
        if (end == p) die("vocab.json: entry without an integer id");
        p = end;
        /* Dense ids are an assumption the blob layout depends on, so it is
         * checked rather than trusted. */
        if (id < 0 || id >= n)
            die("vocab.json: id %ld outside [0, %d) — ids are not dense", id, n);
        if (len[id] != 0) die("vocab.json: id %ld appears twice", id);
        if (tl == 0) die("vocab.json: id %ld has an empty spelling", id);
        len[id] = tl;
    }

    t->n_tokens = n;
    t->offset = malloc((size_t)(n + 1) * sizeof *t->offset);
    if (!t->offset) die("out of memory for %d token offsets", n);
    t->offset[0] = 0;
    for (int i = 0; i < n; i++) t->offset[i + 1] = t->offset[i] + len[i];

    t->blob = malloc((size_t)t->offset[n]);
    if (!t->blob) die("out of memory for %d bytes of token text", t->offset[n]);

    p = json_expect(json, '{');
    while (*(p = entry_start(p)) && *p != '}') {
        int tl;
        p = read_token(p, buf, (int)sizeof buf, &tl, cp_to_byte);
        p = json_expect(p, ':');
        char *end;
        long id = strtol(p, &end, 10);
        p = end;
        memcpy(t->blob + t->offset[id], buf, (size_t)tl);
    }

    t->str_mask = pow2_at_least(n) - 1;
    t->str_slot = calloc((size_t)t->str_mask + 1, sizeof *t->str_slot);
    if (!t->str_slot) die("out of memory for the vocabulary hash table");
    for (int id = 0; id < n; id++) str_insert(t, id);

    free(len);
    free(json);
}

static void load_merges(const char *model_dir, Tokenizer *t, const int *cp_to_byte) {
    char path[1024];
    snprintf(path, sizeof path, "%s/merges.txt", model_dir);
    size_t flen;
    char *txt = slurp(path, &flen);

    int lines = 0;
    for (size_t i = 0; i < flen; i++) if (txt[i] == '\n') lines++;

    t->merge_mask = pow2_at_least(lines) - 1;
    t->merge_key  = calloc((size_t)t->merge_mask + 1, sizeof *t->merge_key);
    t->merge_rank = malloc(((size_t)t->merge_mask + 1) * sizeof *t->merge_rank);
    if (!t->merge_key || !t->merge_rank) die("out of memory for the merge table");
    for (unsigned i = 0; i <= t->merge_mask; i++) t->merge_rank[i] = -1;

    unsigned char lb[512], rb[512];
    int rank = 0;
    bool first_line = true;
    char *line = txt;
    while (line < txt + flen) {
        char *nl = memchr(line, '\n', (size_t)(txt + flen - line));
        char *end = nl ? nl : txt + flen;

        /* The header is POSITIONAL: line one, and only if it announces a
         * version. Treating every line that starts with '#' as a comment
         * silently drops 96 real rules in this very file — "# #", "## ##",
         * "# include", "#### ####" are among the most frequent merges in
         * code and Markdown. merges.txt has no escaping, so the marker
         * collides with legitimate data and position is the only safe rule. */
        bool header = first_line && (size_t)(end - line) >= 8
                   && memcmp(line, "#version", 8) == 0;
        first_line = false;

        if (end > line && !header) {
            /* Split on the first literal space. Safe by construction: the
             * space BYTE is spelled 'Ġ' in this file, never as a space, so
             * the only real space is the separator. */
            char *sp = memchr(line, ' ', (size_t)(end - line));
            if (!sp) die("merges.txt line %d has no separator", rank + 1);

            int ln = decode_alphabet(line, sp, lb, (int)sizeof lb, cp_to_byte);
            int rn = decode_alphabet(sp + 1, end, rb, (int)sizeof rb, cp_to_byte);
            int li = tokenizer_find(t, lb, ln);
            int ri = tokenizer_find(t, rb, rn);
            if (li < 0 || ri < 0)
                die("merges.txt rule %d refers to a token absent from vocab.json", rank);
            merge_insert(t, li, ri, rank);
            rank++;
        }
        if (!nl) break;
        line = nl + 1;
    }
    t->n_merges = rank;
    free(txt);
}

/* Defined further down, next to the encoder it serves. */
static void load_specials(const char *model_dir, Tokenizer *t);

void tokenizer_load(const char *model_dir, Tokenizer *t) {
    int cp_to_byte[ALPHABET_CP];
    build_alphabet(cp_to_byte);

    memset(t, 0, sizeof *t);
    load_vocab(model_dir, t, cp_to_byte);
    load_merges(model_dir, t, cp_to_byte);
    load_specials(model_dir, t);

    /* Byte-level BPE's founding promise: every one of the 256 bytes is itself
     * a token, so no input can fail to encode. Checked, not assumed — a
     * vocabulary missing one byte would tokenise almost everything and then
     * fail on one rare input, long after this function returned. */
    for (int b = 0; b < 256; b++) {
        unsigned char one = (unsigned char)b;
        if (tokenizer_find(t, &one, 1) < 0)
            die("tokenizer: byte 0x%02X has no single-byte token", b);
    }
}

void tokenizer_free(Tokenizer *t) {
    free(t->blob); free(t->offset); free(t->str_slot);
    free(t->merge_key); free(t->merge_rank);
    memset(t, 0, sizeof *t);
}

void tokenizer_summary(const Tokenizer *t) {
    printf("\n=== tokenizer ===\n");
    printf("vocabulary            : %d tokens, %d bytes of text\n",
           t->n_tokens, t->offset[t->n_tokens]);
    printf("merge rules           : %d\n", t->n_merges);
    printf("single-byte tokens    : all 256 present (nothing is untokenisable)\n");

    int longest = 0, which = 0;
    for (int i = 0; i < t->n_tokens; i++) {
        int len = t->offset[i + 1] - t->offset[i];
        if (len > longest) { longest = len; which = i; }
    }
    printf("longest token         : id %d, %d bytes\n", which, longest);

    const unsigned char probe[] = " the";
    printf("probe \" the\"          : id %d\n", tokenizer_find(t, probe, 4));
    printf("probe rank(\" \",\" \")   : %d (rule 0 is the best-ranked merge)\n",
           tokenizer_rank(t, tokenizer_find(t, (const unsigned char *)" ", 1),
                             tokenizer_find(t, (const unsigned char *)" ", 1)));
    printf("added tokens          : %d (ids %d..%d, matched before the regex)\n",
           t->n_specials, t->specials[0].id, t->specials[t->n_specials - 1].id);
}

/* --------------------------------------------------- codepoint classes */

static bool in_ranges(const CodepointRange *r, int n, int cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if ((unsigned)cp < r[mid].lo)      hi = mid - 1;
        else if ((unsigned)cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool is_letter(int cp) {
    return in_ranges(unicode_letters, unicode_letters_count, cp);
}
static bool is_number(int cp) {
    return in_ranges(unicode_numbers, unicode_numbers_count, cp);
}

/* \s in the regex dialect HuggingFace uses is \p{White_Space}: a short and
 * very stable list, so it is spelled out rather than generated. */
static bool is_space(int cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0
        || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028
        || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static bool is_nl(int cp) { return cp == '\r' || cp == '\n'; }

static const char *cp_at(const char *p, const char *end, int *cp) {
    if (p >= end) { *cp = -1; return p; }
    return utf8_next(p, cp);
}

/* ------------------------------------------------------ pre-tokenizer */

/* Length in bytes of the pre-token starting at p. This hand-codes Qwen3's
 * split pattern:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]?\p{L}+
 *   | \p{N}
 *   |  ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 *
 * Alternatives are tried in order and the FIRST that matches wins — regex
 * alternation is ordered, not longest-match, and getting that backwards
 * changes the segmentation of ordinary text.
 *
 * Why any of this exists: merges are forbidden from crossing these
 * boundaries. Without the split, BPE would happily learn a single token for
 * "the cat", and every word's tokenisation would depend on its neighbour. */
static int pretoken_len(const char *p, const char *end) {
    int cp;

    /* 1. contractions, case-insensitively */
    if (*p == '\'' && p + 1 < end) {
        int a = tolower((unsigned char)p[1]);
        int b = (p + 2 < end) ? tolower((unsigned char)p[2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e')
                                   || (a == 'l' && b == 'l')) return 3;
    }

    /* 2. one optional non-letter/non-digit/non-newline, then letters.
     *
     * This is the alternative that attaches a leading space to a word, which
     * is why " the" is one token and "the" is another. No backtracking is
     * needed: the prefix only matches when p is not a letter, in which case
     * dropping it would leave \p{L}+ with nothing at p either. */
    {
        const char *q = p;
        const char *after = cp_at(q, end, &cp);
        if (cp >= 0 && !is_nl(cp) && !is_letter(cp) && !is_number(cp)) q = after;

        const char *r = q;
        int letters = 0;
        for (;;) {
            const char *nx = cp_at(r, end, &cp);
            if (cp < 0 || !is_letter(cp)) break;
            r = nx;
            letters++;
        }
        if (letters > 0) return (int)(r - p);
    }

    /* 3. exactly ONE digit. Not \p{N}+ — Qwen splits numbers per digit, so
     * "2026" is four tokens. Deliberate: it keeps arithmetic tractable. */
    {
        const char *q = cp_at(p, end, &cp);
        if (cp >= 0 && is_number(cp)) return (int)(q - p);
    }

    /* 4. optional space, then punctuation/symbols, then trailing newlines */
    {
        const char *q = p;
        if (q < end && *q == ' ') q++;
        const char *r = q;
        int n = 0;
        for (;;) {
            const char *nx = cp_at(r, end, &cp);
            if (cp < 0 || is_space(cp) || is_letter(cp) || is_number(cp)) break;
            r = nx;
            n++;
        }
        if (n > 0) {
            while (r < end && (*r == '\r' || *r == '\n')) r++;
            return (int)(r - p);
        }
    }

    /* The whitespace run, measured once for alternatives 5-7. */
    const char *ws_end = p, *last_nl_end = NULL, *last_ws_start = NULL;
    for (const char *r = p;;) {
        const char *nx = cp_at(r, end, &cp);
        if (cp < 0 || !is_space(cp)) { ws_end = r; break; }
        last_ws_start = r;
        if (is_nl(cp)) last_nl_end = nx;
        r = nx;
    }

    /* 5. \s*[\r\n]+ : greedy, so it runs through the LAST newline in the run */
    if (last_nl_end) return (int)(last_nl_end - p);

    /* 6. \s+(?!\S) : a whitespace run that is not followed by a non-space.
     * Greedy \s+ then backtracking means: the whole run if it reaches the end
     * of the text, otherwise all but its last character — which is what hands
     * the final space to the word that follows it. */
    if (ws_end > p) {
        if (ws_end == end) return (int)(ws_end - p);
        if (last_ws_start > p) return (int)(last_ws_start - p);
    }

    /* 7. \s+ */
    if (ws_end > p) return (int)(ws_end - p);

    die("pre-tokenizer: no alternative matched — the pattern should be total");
    return 0;                                          /* unreachable */
}

/* ----------------------------------------------------------------- BPE */

/* Merge one pre-token's bytes into ids, always taking the best-ranked pair.
 *
 * The rank order is not an optimisation, it is the definition. Ranks record
 * the order in which merges were learned during the tokenizer's training,
 * most frequent first. Applying them in any other order yields a different
 * but entirely plausible segmentation: the same text, different ids, ids the
 * model never saw in that arrangement. Nothing errors; the output just gets
 * quietly worse. That is why tokenizer bugs are the most dangerous kind.
 *
 * O(n^2) in the chunk length, which is fine because chunks are words. */
static int bpe_chunk(const Tokenizer *t, const unsigned char *b, int n,
                     int *start, int *len, int *id, int *out, int cap) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        start[m] = i;
        len[m] = 1;
        id[m] = tokenizer_find(t, b + i, 1);   /* all 256 exist; checked at load */
        m++;
    }

    for (;;) {
        int best = -1, best_rank = 0;
        for (int i = 0; i + 1 < m; i++) {
            int r = tokenizer_rank(t, id[i], id[i + 1]);
            if (r >= 0 && (best < 0 || r < best_rank)) { best = i; best_rank = r; }
        }
        if (best < 0) break;

        int merged_len = len[best] + len[best + 1];
        int merged = tokenizer_find(t, b + start[best], merged_len);
        /* A merge rule exists for this pair, so its result must be a token.
         * If it is not, the two files disagree with each other. */
        if (merged < 0)
            die("bpe: rule %d merges a pair whose result is not in the vocabulary",
                best_rank);

        len[best] = merged_len;
        id[best] = merged;
        memmove(&start[best + 1], &start[best + 2], (size_t)(m - best - 2) * sizeof *start);
        memmove(&len[best + 1],   &len[best + 2],   (size_t)(m - best - 2) * sizeof *len);
        memmove(&id[best + 1],    &id[best + 2],    (size_t)(m - best - 2) * sizeof *id);
        m--;
    }

    if (m > cap) die("bpe: %d tokens do not fit in %d slots", m, cap);
    memcpy(out, id, (size_t)m * sizeof *out);
    return m;
}

/* ------------------------------------------------------ added tokens */

/* A JSON string kept as raw bytes. Unlike vocab.json's keys, an added
 * token's content is literal text, not byte-level spelling: it is matched
 * against the input as-is. */
static const char *read_raw(const char *p, unsigned char *out, int cap, int *len) {
    p = json_expect(p, '"');
    int n = 0;
    while (*p && *p != '"') {
        unsigned char c;
        if (*p == '\\') {
            p++;
            switch (*p) {
            case '"':  c = '"';  break;  case '\\': c = '\\'; break;
            case '/':  c = '/';  break;  case 'b':  c = '\b'; break;
            case 'f':  c = '\f'; break;  case 'n':  c = '\n'; break;
            case 'r':  c = '\r'; break;  case 't':  c = '\t'; break;
            default: die("tokenizer_config.json: unsupported escape"); return p;
            }
            p++;
        } else {
            c = (unsigned char)*p++;          /* UTF-8 passes through as bytes */
        }
        if (n >= cap) die("tokenizer_config.json: added token longer than %d bytes", cap);
        out[n++] = c;
    }
    if (*p != '"') die("tokenizer_config.json: unterminated string");
    *len = n;
    return p + 1;
}

static void load_specials(const char *model_dir, Tokenizer *t) {
    char path[1024];
    snprintf(path, sizeof path, "%s/tokenizer_config.json", model_dir);
    char *json = slurp(path, NULL);

    static const char key[] = "\"added_tokens_decoder\"";
    const char *p = strstr(json, key);
    if (!p) die("tokenizer_config.json has no added_tokens_decoder");
    p = json_expect(json_expect(p + sizeof key - 1, ':'), '{');

    while (*(p = entry_start(p)) && *p != '}') {
        char idbuf[32];
        p = json_string(p, idbuf, sizeof idbuf);     /* the id, as a key */
        p = json_expect(p, ':');

        if ((size_t)t->n_specials >= sizeof t->specials / sizeof t->specials[0])
            die("tokenizer: more than %zu added tokens",
                sizeof t->specials / sizeof t->specials[0]);
        SpecialToken *s = &t->specials[t->n_specials];
        s->id = (int)strtol(idbuf, NULL, 10);
        s->len = 0;

        p = json_expect(p, '{');
        while (*(p = entry_start(p)) && *p != '}') {
            char field[32];
            p = json_string(p, field, sizeof field);
            p = json_expect(p, ':');
            if (strcmp(field, "content") == 0)
                p = read_raw(p, s->text, (int)sizeof s->text, &s->len);
            else
                p = json_skip(p);
        }
        p++;                                          /* past the entry's '}' */

        if (s->len == 0) die("added token %d has no content", s->id);
        t->n_specials++;
    }
    free(json);
}

/* --------------------------------------------------- encode and decode */

int tokenizer_encode(const Tokenizer *t, const char *text, int len, int *out, int cap) {
    /* tokenizer.json declares an NFC normalizer, and it runs before anything
     * else. Skipping it makes "e" + U+0301 tokenise differently from "e",
     * which is the same text to a reader and a different prompt to the model.
     * NFC is idempotent, so normalising already-normal input costs a copy. */
    int ncap = len * 4 + 8;
    char *norm = malloc((size_t)ncap);
    if (!norm) die("out of memory normalising %d bytes", len);
    len = nfc_normalize(text, len, norm, ncap);
    text = norm;

    /* One scratch allocation for the whole call: a pre-token can be as long
     * as the input (a single unbroken run of whitespace, say). */
    int *start = malloc((size_t)len * sizeof *start);
    int *slen  = malloc((size_t)len * sizeof *slen);
    int *sid   = malloc((size_t)len * sizeof *sid);
    if (!start || !slen || !sid) die("out of memory tokenising %d bytes", len);

    int n = 0;
    const char *p = text, *end = text + len;
    while (p < end) {
        /* Added tokens cut the text BEFORE the regex ever runs, and that
         * ordering is the whole point. Checking for them only at pre-token
         * boundaries does not work: the pattern reaches a boundary having
         * already swallowed " <|" as space-plus-punctuation, and the special
         * token is never seen. So find the earliest occurrence anywhere
         * ahead, tokenise the text up to it as its own segment, emit the
         * token, and resume after it.
         *
         * The visible consequence is that the space before <|im_start|>
         * becomes a token of its own rather than merging rightwards. */
        const char *hit = NULL;
        int hit_id = -1, hit_len = 0;

        for (int i = 0; i < t->n_specials; i++) {
            const SpecialToken *s = &t->specials[i];
            for (const char *q = p; q + s->len <= end; q++) {
                if (memcmp(q, s->text, (size_t)s->len) != 0) continue;
                /* Earliest wins; at equal positions the longer token wins, so
                 * a special that prefixes another cannot shadow it. */
                if (!hit || q < hit || (q == hit && s->len > hit_len)) {
                    hit = q;
                    hit_id = s->id;
                    hit_len = s->len;
                }
                break;
            }
        }

        const char *seg_end = hit ? hit : end;
        while (p < seg_end) {
            int plen = pretoken_len(p, seg_end);
            n += bpe_chunk(t, (const unsigned char *)p, plen, start, slen, sid,
                           out + n, cap - n);
            p += plen;
        }
        if (hit) {
            if (n >= cap) die("tokenizer: more than %d tokens", cap);
            out[n++] = hit_id;
            p = hit + hit_len;
        }
    }

    free(start); free(slen); free(sid); free(norm);
    return n;
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int cap) {
    int len = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char *bytes = NULL;
        int blen = 0;

        if (ids[i] >= 0 && ids[i] < t->n_tokens) {
            bytes = t->blob + t->offset[ids[i]];
            blen = t->offset[ids[i] + 1] - t->offset[ids[i]];
        } else {
            for (int k = 0; k < t->n_specials; k++)
                if (t->specials[k].id == ids[i]) {
                    bytes = t->specials[k].text;
                    blen = t->specials[k].len;
                    break;
                }
            if (!bytes) die("decode: id %d is in no table", ids[i]);
        }

        if (len + blen >= cap) die("decode: output longer than %d bytes", cap);
        memcpy(out + len, bytes, (size_t)blen);
        len += blen;
    }
    out[len] = '\0';
    return len;
}
