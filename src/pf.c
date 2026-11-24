/* pf.c — pf and pf-bench, the picoforge commands (pf-bench is this program under another name).
 *
 *   pf [-m MODEL] [-p TEXT] [-n N] [-c CTX] ...   the chat, or one answer with -p
 *   pf bench [options]  (or pf-bench)             repeated runs, median and p10/p90
 *   pf models                                     the models -m can name
 *   pf --qwen4 FILE.gguf [flags]                  the long form, every measurement flag
 *   pf --inspect-gguf FILE.gguf                   the tensors of a GGUF file
 *
 * Everything model-specific lives in src/qwen4_*: this file only turns the short
 * forms into the long one (qw4_cli in src/qwen4_run.m).
 */
#include "common.h"
#include "gguf.h"
#include "pf_models.h"
#include "qwen4_run.h"

#include <stdio.h>
#include <stdlib.h>

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
static bool has(int n, char **opts, const char *name) {
    for (int i = 0; i < n; i++) if (!strcmp(opts[i], name)) return true;
    return false;
}

/* The ds4-style short form (`./ds4 -p ...`): -m MODEL, -p TEXT, -c N, -n N, every other option as
 * in the long form. Without -m the model comes from pf_models_pick. No option at all: the chat. */
static int short_form(const char *argv0, int n, char **opts) {
    const char *model = NULL;
    char **out = calloc((size_t)n+3, sizeof *out);
    if (!out) die("out of memory");
    int k = 3;
    for (int i = 0; i < n; i++) {
        const char *o = opts[i];
        const char *as = !strcmp(o, "-p") ? "--chat" : !strcmp(o, "-c") ? "--context" : !strcmp(o, "-n") ? "--max-new" : NULL;
        if ((as || !strcmp(o, "-m")) && i+1 == n) die("missing value for %s", o);
        if (!strcmp(o, "-m")) model = opts[++i];
        else if (as) { out[k++] = (char *)as; out[k++] = opts[++i]; }
        else out[k++] = opts[i];
    }
    char *path = pf_models_pick(model, argv0);
    out[0] = (char *)argv0; out[1] = "--qwen4"; out[2] = path;
    int rc = qw4_cli(k, out);
    free(path); free(out);
    return rc;
}

/* pf bench / pf-bench: the docs/BENCHMARKS.md protocol (3 warm-up, 20 measured, 128 new tokens)
 * on a fixed prompt unless one is given. The text is discarded, the summary goes to stdout. The
 * options come after the defaults, so they win. */
static int bench(const char *argv0, int n, char **opts) {
    if (has(n, opts, "-h") || has(n, opts, "--help")) {
        puts("usage: pf-bench [-m MODEL] [-p TEXT] [-n N] [options]   (also: pf bench ...)\n"
             "  3 warm-up and 20 measured runs of one answer (128 new tokens, a fixed prompt unless -p),\n"
             "  greedy, no MTP unless --mtp; prints the median and p10/p90 of prefill and decode tok/s and TTFT.\n"
             "  --warmup N --repeat N   other counts      --bench-csv FILE   every run as a CSV row\n"
             "  --tokens ID,... --teacher-forced N        decode fixed tokens; the rest as pf --qwen4 FILE --help");
        return 0;
    }
    bool prompt = false;
    static const char *const prompts[] = {"-p", "--chat", "--chat-file", "--complete", "--complete-file", "--tokens"};
    for (size_t i = 0; i < sizeof prompts/sizeof *prompts; i++) prompt |= has(n, opts, prompts[i]);
    bool forced = has(n, opts, "--teacher-forced");   /* teacher forcing fixes its own length: no -n */
    const char *defaults[] = {"--bench", "--warmup", "3", "--repeat", "20", "-n", "128"};
    int nd = (int)(sizeof defaults/sizeof *defaults) - (forced ? 2 : 0);
    char **all = calloc((size_t)(nd+2+n), sizeof *all);
    if (!all) die("out of memory");
    int k = 0;
    for (int i = 0; i < nd; i++) all[k++] = (char *)defaults[i];
    if (!prompt) { all[k++] = "-p"; all[k++] = "Write a long story about a lighthouse keeper."; }
    for (int i = 0; i < n; i++) all[k++] = opts[i];
    int rc = short_form(argv0, k, all);
    free(all);
    return rc;
}

int main(int argc, char **argv) {
    const char *slash = strrchr(argv[0], '/'), *self = slash ? slash+1 : argv[0];
    if (!strcmp(self, "pf-bench")) return bench(argv[0], argc-1, argv+1);
    if (argc == 1 || (argv[1][0] == '-' && strcmp(argv[1], "-h") && strcmp(argv[1], "--help") &&
                      strcmp(argv[1], "--qwen4") && strcmp(argv[1], "--inspect-gguf")))
        return short_form(argv[0], argc-1, argv+1);            /* the chat or one answer */
    const char *cmd = argv[1];
    if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help")) return usage(stdout);
    if (!strcmp(cmd, "models") && argc == 2) return pf_models_list(argv[0]);
    if (!strcmp(cmd, "bench")) return bench(argv[0], argc-2, argv+2);
    if (!strcmp(cmd, "--qwen4")) return qw4_cli(argc, argv);
    if (!strcmp(cmd, "--inspect-gguf") && argc == 3) {
        Gguf g;
        gguf_open(argv[2], &g);
        gguf_inspect(&g);
        gguf_close(&g);
        return 0;
    }
    return usage(stderr);
}
