/* common.h — the few helpers every part of the engine shares.
 *
 * Deliberately tiny: the model-specific types live in the qwen4_*.h headers,
 * the GGUF reader in gguf.h. What is left here is "fail loudly" and reading a
 * whole file.
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

#endif
