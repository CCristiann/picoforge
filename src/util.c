/* util.c — the two things every translation unit needs.
 *
 * These lived in config.c out of convenience. They are not config's
 * business, and the test binaries need them without dragging the JSON
 * scanner along, so they move here.
 */
#include "picoforge.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("picoforge: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

char *slurp(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    if (fseek(f, 0, SEEK_END) != 0) die("cannot seek %s", path);
    long n = ftell(f);
    if (n < 0) die("cannot size %s", path);
    rewind(f);

    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory reading %s (%ld bytes)", path, n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("short read on %s", path);
    buf[n] = '\0';                     /* so every str* below is safe */
    fclose(f);

    if (len_out) *len_out = (size_t)n;
    return buf;
}

