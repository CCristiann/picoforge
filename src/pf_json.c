/* pf_json.c — see pf_json.h. Sections: arena · parser · writer · access and builders. */
#include "pf_json.h"
#include "common.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------ arena
 * Blocks of at least 64 KiB, each allocation 16-byte aligned; nothing is freed
 * on its own: the whole arena goes when the request is answered. */
struct PfArenaBlock { PfArenaBlock *next; size_t used, cap; _Alignas(16) unsigned char data[]; };

void *pf_alloc(PfArena *a, size_t n) {
    n = (n+15) & ~(size_t)15;
    PfArenaBlock *b = a->head;
    if (!b || b->cap-b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        b = calloc(1, sizeof *b+cap);
        if (!b) die("out of memory");
        b->cap = cap; b->next = a->head; a->head = b;
    }
    void *p = b->data+b->used; b->used += n;
    return p;   /* calloc'd and never reused: already zero */
}
char *pf_strdup(PfArena *a, const char *s, size_t n) {
    char *p = pf_alloc(a, n+1);
    memcpy(p, s, n); p[n] = 0;
    return p;
}
void pf_arena_free(PfArena *a) {
    for (PfArenaBlock *b = a->head, *next; b; b = next) { next = b->next; free(b); }
    a->head = NULL;
}

/* ------------------------------------------------------------ parser
 * Recursive descent over RFC 8259. Arrays and objects are first collected in
 * a growable scratch list, then copied into the arena at their final size. */
typedef struct { PfArena *a; const char *start, *p, *end; char *err; size_t errlen; int depth; } Parser;

static bool bad(Parser *ps, const char *what) {
    if (ps->err && !ps->err[0]) snprintf(ps->err, ps->errlen, "%s at byte %td", what, ps->p-ps->start);
    return false;
}
static void space(Parser *ps) { while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++; }
static bool lit(Parser *ps, const char *word) {
    size_t n = strlen(word);
    if ((size_t)(ps->end-ps->p) < n || memcmp(ps->p, word, n)) return false;
    ps->p += n; return true;
}
static int hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i]; v <<= 4;
        if (c >= '0' && c <= '9') v |= c-'0';
        else if (c >= 'a' && c <= 'f') v |= c-'a'+10;
        else if (c >= 'A' && c <= 'F') v |= c-'A'+10;
        else return -1;
    }
    return v;
}
/* A string body after the opening quote; escapes decoded, raw bytes kept
 * (the template checks UTF-8 itself). */
static bool string(Parser *ps, Qwen4Bytes *out) {
    Buf b = {0};
    for (;;) {
        if (ps->p >= ps->end) { buf_free(&b); return bad(ps, "unterminated string"); }
        unsigned char c = (unsigned char)*ps->p++;
        if (c == '"') break;
        if (c < 0x20) { buf_free(&b); return bad(ps, "control character in a string"); }
        if (c != '\\') { buf_add(&b, &c, 1); continue; }
        if (ps->p >= ps->end) { buf_free(&b); return bad(ps, "bad escape"); }
        char e = *ps->p++; const char *simple = "\"\\/bfnrt", *value = "\"\\/\b\f\n\r\t", *at = strchr(simple, e);
        if (e && at) { buf_add(&b, value+(at-simple), 1); continue; }
        if (e != 'u' || ps->end-ps->p < 4) { buf_free(&b); return bad(ps, "bad escape"); }
        long cp = hex4(ps->p); ps->p += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && ps->end-ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
            int lo = hex4(ps->p+2);
            if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp-0xD800) << 10) + (lo-0xDC00); ps->p += 6; }
        }
        if (cp < 0) { buf_free(&b); return bad(ps, "bad \\u escape"); }
        utf8_put(&b, (unsigned long)cp);   /* a lone surrogate becomes '?' */
    }
    out->data = (const uint8_t *)pf_strdup(ps->a, b.p ? b.p : "", b.n); out->len = b.n;
    buf_free(&b);
    return true;
}
static bool number(Parser *ps, Qwen4Json *v) {
    const char *s = ps->p; bool real = false;
    if (ps->p < ps->end && *ps->p == '-') ps->p++;
    while (ps->p < ps->end && strchr("0123456789.eE+-", *ps->p)) { if (strchr(".eE", *ps->p)) real = true; ps->p++; }
    char tmp[64]; size_t n = (size_t)(ps->p-s);
    if (!n || n >= sizeof tmp) return bad(ps, "bad number");
    memcpy(tmp, s, n); tmp[n] = 0;
    char *end; errno = 0;
    if (!real) {
        long long i = strtoll(tmp, &end, 10);
        if (!*end && !errno) { *v = (Qwen4Json){.kind = QW4_JSON_INT, .as.integer = i}; return true; }
    }
    double d = strtod(tmp, &end);
    if (*end || !isfinite(d)) return bad(ps, "bad number");
    *v = (Qwen4Json){.kind = QW4_JSON_FLOAT, .as.number = d};
    return true;
}
static bool value(Parser *ps, Qwen4Json *v);
static bool list(Parser *ps, Qwen4Json *v, bool object) {
    if (++ps->depth > 128) return bad(ps, "nesting too deep");
    size_t n = 0, cap = 8; Qwen4Json *items = malloc(cap*sizeof *items); Qwen4Bytes *keys = malloc(cap*sizeof *keys);
    if (!items || !keys) die("out of memory");
    bool ok = true; char close = object ? '}' : ']';
    space(ps);
    if (ps->p < ps->end && *ps->p == close) ps->p++;
    else for (;;) {
        if (n == cap) {
            cap *= 2; items = realloc(items, cap*sizeof *items); keys = realloc(keys, cap*sizeof *keys);
            if (!items || !keys) die("out of memory");
        }
        space(ps);
        if (object) {
            if (ps->p >= ps->end || *ps->p++ != '"' || !string(ps, &keys[n])) { ok = bad(ps, "expected a key"); break; }
            space(ps);
            if (ps->p >= ps->end || *ps->p++ != ':') { ok = bad(ps, "expected ':'"); break; }
        }
        if (!value(ps, &items[n])) { ok = false; break; }
        n++; space(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == close) { ps->p++; break; }
        ok = bad(ps, object ? "expected ',' or '}'" : "expected ',' or ']'"); break;
    }
    if (ok && object) {
        Qwen4JsonMember *m = pf_alloc(ps->a, (n ? n : 1)*sizeof *m);
        for (size_t i = 0; i < n; i++) {
            Qwen4Json *boxed = pf_alloc(ps->a, sizeof *boxed); *boxed = items[i];
            m[i] = (Qwen4JsonMember){keys[i], boxed};
        }
        *v = (Qwen4Json){.kind = QW4_JSON_OBJECT, .as.object = {m, n}};
    } else if (ok) {
        Qwen4Json *a = pf_alloc(ps->a, (n ? n : 1)*sizeof *a);
        memcpy(a, items, n*sizeof *a);
        *v = (Qwen4Json){.kind = QW4_JSON_ARRAY, .as.array = {a, n}};
    }
    free(items); free(keys); ps->depth--;
    return ok;
}
static bool value(Parser *ps, Qwen4Json *v) {
    space(ps);
    if (ps->p >= ps->end) return bad(ps, "unexpected end");
    char c = *ps->p;
    if (c == '{' || c == '[') { ps->p++; return list(ps, v, c == '{'); }
    if (c == '"') { ps->p++; v->kind = QW4_JSON_STRING; return string(ps, &v->as.string); }
    if (lit(ps, "true")) { *v = (Qwen4Json){.kind = QW4_JSON_BOOL, .as.boolean = true}; return true; }
    if (lit(ps, "false")) { *v = (Qwen4Json){.kind = QW4_JSON_BOOL, .as.boolean = false}; return true; }
    if (lit(ps, "null")) { *v = (Qwen4Json){.kind = QW4_JSON_NULL}; return true; }
    if (c == '-' || (c >= '0' && c <= '9')) return number(ps, v);
    return bad(ps, "unexpected character");
}
Qwen4Json *pf_json_parse(PfArena *a, const char *text, size_t n, char *err, size_t errlen) {
    Parser ps = {.a = a, .start = text, .p = text, .end = text+n, .err = err, .errlen = errlen};
    if (err && errlen) err[0] = 0;
    Qwen4Json *v = pf_alloc(a, sizeof *v);
    if (!value(&ps, v)) return NULL;
    space(&ps);
    if (ps.p != ps.end) { bad(&ps, "trailing characters"); return NULL; }
    return v;
}

/* ------------------------------------------------------------ writer */
static void write_string(Buf *b, Qwen4Bytes s) {
    char *tmp = strndup((const char *)s.data ? (const char *)s.data : "", s.len);
    if (!tmp) die("out of memory");
    buf_str(b, "\"");
    if (strlen(tmp) == s.len) json_escape(b, tmp);
    else for (size_t i = 0; i < s.len; i++) {        /* an embedded NUL: escape byte by byte */
        char one[2] = {tmp[i], 0};
        if (!tmp[i]) buf_str(b, "\\u0000"); else json_escape(b, one);
    }
    buf_str(b, "\"");
    free(tmp);
}
void pf_json_write(Buf *b, const Qwen4Json *v) {
    if (!v) { buf_str(b, "null"); return; }
    switch (v->kind) {
    case QW4_JSON_NULL: buf_str(b, "null"); break;
    case QW4_JSON_BOOL: buf_str(b, v->as.boolean ? "true" : "false"); break;
    case QW4_JSON_INT: buf_fmt(b, "%lld", (long long)v->as.integer); break;
    case QW4_JSON_FLOAT: buf_fmt(b, "%.17g", v->as.number); break;
    case QW4_JSON_STRING: write_string(b, v->as.string); break;
    case QW4_JSON_ARRAY:
        buf_str(b, "[");
        for (size_t i = 0; i < v->as.array.count; i++) { if (i) buf_str(b, ","); pf_json_write(b, v->as.array.items+i); }
        buf_str(b, "]"); break;
    case QW4_JSON_OBJECT:
        buf_str(b, "{");
        for (size_t i = 0; i < v->as.object.count; i++) {
            if (i) buf_str(b, ",");
            write_string(b, v->as.object.items[i].key); buf_str(b, ":"); pf_json_write(b, v->as.object.items[i].value);
        }
        buf_str(b, "}"); break;
    }
}

/* ------------------------------------------------------------ access and builders */
const Qwen4Json *pf_json_get(const Qwen4Json *o, const char *key) {
    if (!o || o->kind != QW4_JSON_OBJECT) return NULL;
    size_t n = strlen(key);
    for (size_t i = 0; i < o->as.object.count; i++) {
        Qwen4Bytes k = o->as.object.items[i].key;
        if (k.len == n && !memcmp(k.data, key, n)) return o->as.object.items[i].value;
    }
    return NULL;
}
bool pf_json_is(const Qwen4Json *v, const char *s) {
    return v && v->kind == QW4_JSON_STRING && v->as.string.len == strlen(s) && !memcmp(v->as.string.data, s, v->as.string.len);
}
const char *pf_json_str(PfArena *a, const Qwen4Json *o, const char *key) {
    const Qwen4Json *v = pf_json_get(o, key);
    return v && v->kind == QW4_JSON_STRING ? pf_strdup(a, (const char *)v->as.string.data, v->as.string.len) : NULL;
}
double pf_json_num(const Qwen4Json *o, const char *key, double fallback) {
    const Qwen4Json *v = pf_json_get(o, key);
    if (v && v->kind == QW4_JSON_INT) return (double)v->as.integer;
    if (v && v->kind == QW4_JSON_FLOAT) return v->as.number;
    return fallback;
}
Qwen4Json pf_jstr(PfArena *a, const char *s, size_t n) {
    return (Qwen4Json){.kind = QW4_JSON_STRING, .as.string = {(const uint8_t *)pf_strdup(a, s ? s : "", s ? n : 0), s ? n : 0}};
}
Qwen4Json pf_jobject(PfArena *a, size_t cap) {
    return (Qwen4Json){.kind = QW4_JSON_OBJECT, .as.object = {pf_alloc(a, (cap ? cap : 1)*sizeof(Qwen4JsonMember)), 0}};
}
Qwen4Json pf_jarray(PfArena *a, size_t cap) {
    return (Qwen4Json){.kind = QW4_JSON_ARRAY, .as.array = {pf_alloc(a, (cap ? cap : 1)*sizeof(Qwen4Json)), 0}};
}
void pf_jset(PfArena *a, Qwen4Json *o, const char *key, Qwen4Json v) {
    Qwen4Json *boxed = pf_alloc(a, sizeof *boxed); *boxed = v;
    Qwen4JsonMember *m = (Qwen4JsonMember *)o->as.object.items;   /* built by pf_jobject: ours to fill */
    size_t n = strlen(key);
    m[o->as.object.count++] = (Qwen4JsonMember){{(const uint8_t *)pf_strdup(a, key, n), n}, boxed};
}
void pf_jpush(Qwen4Json *arr, Qwen4Json v) { ((Qwen4Json *)arr->as.array.items)[arr->as.array.count++] = v; }
