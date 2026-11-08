/* main.c — picoforge entry point.
 *
 *   picoforge [-m MODEL] [-p TEXT] [-n N] [--mtp] ...      chat or one answer
 *   picoforge bench [same options]                           repeated decode timing
 *   picoforge models                                         the models -m can name
 *   picoforge --qwen38 FILE.gguf [flags]                     long form, every measurement flag
 *   picoforge --inspect-gguf FILE.gguf                       tensor inventory
 *
 * The engine runs one model family at a time. Everything model-specific
 * lives in src/qwen38_*; this file only decides which entry point to call.
 */
#include "common.h"
#include "gguf.h"
#include "qwen38_cli.h"
#include "model_select.h"

#include <stdio.h>

static int usage(void) {
    fputs("usage: picoforge [-m MODEL] [-p TEXT] [-n N] [-c CTX] [--mtp] [--thinking]\n"
          "       picoforge bench [options]\n"
          "       picoforge models\n"
          "       picoforge --qwen38 FILE.gguf [flags]\n"
          "       picoforge --inspect-gguf FILE.gguf\n"
          "MODEL is a .gguf path or a short name from `picoforge models` (e.g. q4); without -m,\n"
          "$PICOFORGE_MODEL, the only model found, or a choice on the terminal.\n", stderr);
    return 2;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) return usage();
    if (argc == 2 && !strcmp(argv[1], "models")) return model_list(argv[0]);
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
