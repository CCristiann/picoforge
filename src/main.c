/* main.c — pf, the picoforge command.
 *
 *   pf [-m MODEL] [-p TEXT] [-n N] [--mtp] ...      chat or one answer
 *   pf bench [same options]  (or pf-bench)           repeated decode timing
 *   pf models                                         the models -m can name
 *   pf --qwen4 FILE.gguf [flags]                     long form, every measurement flag
 *   pf --inspect-gguf FILE.gguf                       tensor inventory
 *
 * The engine runs one model family at a time. Everything model-specific
 * lives in src/qwen4_*; this file only decides which entry point to call.
 */
#include "common.h"
#include "gguf.h"
#include "qwen4_cli.h"
#include "model_select.h"

#include <stdio.h>

static int usage(FILE *f) {
    fputs("usage: pf [-m MODEL] [-p TEXT] [-n N] [-c CTX] [--think] [options]\n"
          "       pf models                 the models -m can name\n"
          "       pf bench [options]        repeated runs, median and p10/p90 (also: pf-bench)\n"
          "       pf --qwen4 FILE [flags]   every measurement flag (pf --qwen4 FILE --help)\n"
          "       pf --inspect-gguf FILE    the tensors of a GGUF file\n"
          "MODEL is a .gguf path or a name from `pf models`; without -m, $PICOFORGE_MODEL,\n"
          "the only model found, or a choice on the terminal. Without -p, the chat (/help inside).\n", f);
    return f == stdout ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) return usage(stdout);
    if (argc == 2 && !strcmp(argv[1], "models")) return model_list(argv[0]);
    if (argc > 1 && !strcmp(argv[1], "--qwen4")) return qwen4_cli(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "bench")) return qwen4_bench(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--inspect-gguf")) {
        if (argc != 3) die("usage: pf --inspect-gguf FILE");
        Gguf g;
        gguf_open(argv[2], &g);
        gguf_inspect(&g);
        gguf_close(&g);
        return 0;
    }
    if (argc == 1 || argv[1][0] == '-') return qwen4_short(argc, argv);   /* the chat or one answer */
    return usage(stderr);
}
