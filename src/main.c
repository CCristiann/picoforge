/* main.c — picoforge entry point.
 *
 *   picoforge [-m FILE.gguf] [-p TEXT] [-n N] [--mtp] ...   chat or one answer
 *   picoforge bench [same options]                           repeated decode timing
 *   picoforge --qwen38 FILE.gguf [flags]                     long form, every measurement flag
 *   picoforge --inspect-gguf FILE.gguf                       tensor inventory
 *
 * The engine runs one model family at a time. Everything model-specific
 * lives in src/qwen38_*; this file only decides which entry point to call.
 */
#include "common.h"
#include "gguf.h"
#include "qwen38_cli.h"

#include <stdio.h>

static int usage(void) {
    fputs("usage: picoforge [-m FILE.gguf] [-p TEXT] [-n N] [-c CTX] [--mtp] [--thinking]\n"
          "       picoforge bench [options]\n"
          "       picoforge --qwen38 FILE.gguf [flags]\n"
          "       picoforge --inspect-gguf FILE.gguf\n"
          "The model path may also come from $PICOFORGE_MODEL.\n", stderr);
    return 2;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--qwen38")) return qwen38_cli(argc, argv);
    if (qwen38_short_form(argc, argv)) return qwen38_short(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "bench")) return qwen38_bench(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--inspect-gguf")) {
        if (argc != 3) die("usage: picoforge --inspect-gguf FILE");
        Gguf g;
        gguf_open(argv[2], &g);
        gguf_inspect(&g);
        gguf_close(&g);
        return 0;
    }
    return usage();
}
