/* common.h — the few helpers every part of the engine shares.
 *
 * Deliberately tiny: the model-specific types live in the qwen38_*.h headers,
 * the GGUF reader in gguf.h. What is left here is "fail loudly" and a JSON
 * scanner just big enough for chat and tokenizer metadata.
 */
#ifndef PICOFORGE_COMMON_H
#define PICOFORGE_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Fail loudly and immediately. Printf-style, prefixed, exit(1).
 * There is no error-recovery story in this engine on purpose: an
 * inference engine that limps on after a malformed model file produces
 * plausible garbage, which is the one failure mode we cannot detect. */
void  die(const char *fmt, ...);

/* Read a whole file into a NUL-terminated heap buffer. Caller frees. */
char *slurp(const char *path, size_t *len_out);

/* JSON: enough to walk a file whose shape you already know. Every one of
 * these dies rather than returning an error: a malformed model file is not a
 * condition an inference engine can sensibly continue from. */
const char *json_skip(const char *p);                 /* past one whole value  */
const char *json_expect(const char *p, char c);       /* assert next char      */
const char *json_string(const char *p, char *out, size_t cap);   /* no escapes */
const char *json_ints(const char *p, long *out, int cap, int *n_out);

#endif
