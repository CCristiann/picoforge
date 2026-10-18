/* Q38_REF_KERNELS: which GPU kernel families must run their oracle (R1).
 * Every fast kernel is validated against a slow, obviously correct oracle; the
 * oracle stays selectable so a drift can be bisected to "this kernel or not"
 * and the starting code (360acf0) stays reproducible bit for bit.
 *   unset, "" or "0"  -> fast paths where they exist
 *   "1" or "all"      -> every oracle
 *   "linear,moe", ... -> only the listed families: linear, indexed, moe,
 *                        fuse, barrier, claim, argmax (exact, case-sensitive)
 * Ruling R18: a token that is not a family (typo, "2", empty) selects EVERY
 * oracle: a mistyped request for the reference must fail toward the reference,
 * never silently run a fast kernel. Read when contexts are created, never per
 * token, so one context never mixes choices mid-session. */
#ifndef PICOFORGE_QWEN38_REF_H
#define PICOFORGE_QWEN38_REF_H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* True iff token p[0..n) names `name` exactly (no prefixes). */
static inline bool q38_ref_token(const char *p, size_t n, const char *name) {
    return strlen(name) == n && !strncmp(p, name, n);
}
/* Pure parser, separate from getenv so tests can feed any value. */
static inline bool q38_ref_parse(const char *value, const char *family) {
    static const char *const known[] = {"linear", "indexed", "moe", "fuse",
                                        "barrier", "claim", "argmax"};
    if (!value || !*value || !strcmp(value, "0")) return false;
    if (!strcmp(value, "1") || !strcmp(value, "all")) return true;
    bool listed = false;
    for (const char *p = value;; p++) {
        const char *comma = strchr(p, ',');
        size_t n = comma ? (size_t)(comma - p) : strlen(p);
        bool valid = false;
        for (size_t i = 0; i < sizeof known / sizeof *known; i++)
            valid = valid || q38_ref_token(p, n, known[i]);
        if (!valid) return true;
        listed = listed || q38_ref_token(p, n, family);
        if (!comma) return listed;
        p = comma;
    }
}
static inline bool q38_ref_kernels(const char *family) {
    return q38_ref_parse(getenv("Q38_REF_KERNELS"), family);
}
#endif
