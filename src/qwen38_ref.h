/* Q38_REF_KERNELS: which GPU kernel families must run their oracle (R1).
 * Every fast kernel is validated against a slow, obviously correct oracle; the
 * oracle stays selectable so a drift can be bisected to "this kernel or not"
 * and the starting code (360acf0) stays reproducible bit for bit.
 *   unset, "" or "0"  -> fast paths where they exist
 *   "1" or "all"      -> every oracle
 *   "linear,moe", ... -> only the listed families: linear, indexed, moe,
 *                        fuse, barrier, claim, argmax, attention, gemm,
 *                        exact (exact names, case-sensitive)
 * Family "exact" (Decode III) is not an oracle: it keeps the bit-identical
 * fast kernels in place of the ones that reorder FP32 sums (validated
 * against a measured tolerance), so the pre-Decode-III bits stay reproducible.
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
                                        "barrier", "claim", "argmax", "attention", "gemm", "exact"};
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
/* unused: test-qwen38-contracts compiles this header alone, as in qwen38_b_internal.h. */
static inline __attribute__((unused)) bool q38_ref_kernels(const char *family) {
    return q38_ref_parse(getenv("Q38_REF_KERNELS"), family);
}
/* R3.5 prefill precision: PICOFORGE_PREFILL_PRECISION=fp32|half (the CLI's
 * --prefill-precision sets it). "half" runs the prefill GEMMs on the Neural
 * Accelerators with half operands and FP32 accumulation; decode is never
 * affected. Unset or empty: Q38_PREFILL_HALF_DEFAULT. Any other value reads as
 * fp32 (the oracle), never as half; the CLI rejects it. Q38_REF_KERNELS=gemm
 * (or 1) wins over half: no GEMM, so no reduced-precision GEMM either. */
#define Q38_PREFILL_HALF_DEFAULT 0
static inline bool q38_prefill_half_parse(const char *value) {
    if (!value || !*value) return Q38_PREFILL_HALF_DEFAULT;
    return !strcmp(value, "half");
}
static inline __attribute__((unused)) bool q38_prefill_half(void) {
    return q38_prefill_half_parse(getenv("PICOFORGE_PREFILL_PRECISION"));
}
#endif
