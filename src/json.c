/* json.c — the smallest JSON scanner three other files can share.
 *
 * Still not a JSON parser. It is four primitives that let a caller who knows
 * the shape of its file walk it: skip a value of any type, read a bare
 * string, read an array of integers, and assert the next character.
 *
 * It started inside safetensors.c. config.c, safetensors.c and tokenizer.c
 * all need it now, and three private copies is the reliable way to make three
 * subtly different parsers.
 */
#include "picoforge.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------- JSON
 * Step 1.1 looked up keys it already knew. Here we must ENUMERATE 311 keys
 * we do not know, which needs one primitive we did not have: skipping a
 * complete JSON value, whatever its type, including nesting and escapes.
 */
const char *json_skip(const char *p) {
    while (isspace((unsigned char)*p)) p++;
    if (*p == '"') {                                   /* string */
        p++;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        if (*p != '"') die("json: unterminated string in header");
        return p + 1;
    }
    if (*p == '{' || *p == '[') {                      /* object or array */
        char open = *p, close = (*p == '{') ? '}' : ']';
        int depth = 0;
        while (*p) {
            if (*p == '"') { p = json_skip(p); continue; }  /* braces inside
                                                               strings must
                                                               not count */
            if (*p == open) depth++;
            else if (*p == close && --depth == 0) return p + 1;
            p++;
        }
        die("json: unbalanced '%c' in header", open);
    }
    while (*p && *p != ',' && *p != '}' && *p != ']'   /* number or literal */
           && !isspace((unsigned char)*p)) p++;
    return p;
}

const char *json_string(const char *p, char *out, size_t cap) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != '"') die("json: expected a string in header");
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') die("json: escapes in names are not supported");
        if (i + 1 >= cap) die("json: name longer than %zu chars", cap - 1);
        out[i++] = *p++;
    }
    if (*p != '"') die("json: unterminated name in header");
    out[i] = '\0';
    return p + 1;
}

const char *json_ints(const char *p, long *out, int cap, int *n_out) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != '[') die("json: expected an array in header");
    p++;
    int n = 0;
    for (;;) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (*p == ']') { p++; break; }
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) die("json: malformed integer in header array");
        if (n >= cap) die("json: array longer than %d entries", cap);
        out[n++] = v;
        p = end;
    }
    *n_out = n;
    return p;
}

const char *json_expect(const char *p, char c) {
    while (isspace((unsigned char)*p)) p++;
    if (*p != c) die("json: expected '%c' in header, found '%c'", c, *p);
    return p + 1;
}

