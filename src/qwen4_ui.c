/* qwen4_ui.c — what the chat shows on a terminal: a summary of the model and the session at start
 * (and on /info), and one line of numbers after each reply. Only for a terminal: through a pipe the
 * chat keeps its plain lines, which the tests read (qwen4_repl.c). Colors are the 256-color palette
 * of pf-agent's status line, so the two programs look alike. */
#include "qwen4_run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LABEL "\x1b[38;5;245m"
#define VALUE "\x1b[38;5;252m"
#define ACCENT "\x1b[1;38;5;81m"
#define OFF "\x1b[0m"

/* 412, 8.2K, 100K: the counts of pf-agent's footer; a context of 8192 or 131072 tokens is 8K, 128K. */
static void count(char *out, size_t cap, size_t n) {
    if (n < 1000) snprintf(out, cap, "%zu", n);
    else if (n % 1024 == 0) snprintf(out, cap, "%zuK", n / 1024);
    else if (n < 100000) snprintf(out, cap, "%.1fK", (double)n / 1000.0);
    else snprintf(out, cap, "%zuK", n / 1000);
}
static const GgufTensor *tensor(const Gguf *g, const char *name) {
    for (size_t i = 0; i < g->tensor_count; i++) if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}
static const char *format(const GgufTensor *t) {
    const GgufFormat *f = t ? gguf_format(t->type) : NULL;
    return f ? f->name : "?";
}
/* The expert storage of layer 0, which every trunk layer shares: "512 experts per layer in
 * IQ2_XXS/Q2_K", or for a picoforge mixed file "512 experts per layer: 486 in IQ2_XXS/Q2_K, 26 in
 * Q4_K/MXFP4" (the hot ones, tools/quant/build_mixed.py). */
static void experts(const Gguf *g, char *out, size_t cap) {
    const GgufTensor *gate = tensor(g, "blk.0.ffn_gate_exps.weight"), *down = tensor(g, "blk.0.ffn_down_exps.weight");
    const GgufTensor *hot = tensor(g, "blk.0.ffn_gate_exps_hot.weight"), *hot_down = tensor(g, "blk.0.ffn_down_exps_hot.weight");
    if (!gate || !down || gate->rank < 3) { snprintf(out, cap, "experts ?"); return; }
    size_t cold = gate->shape[2], hot_n = hot && hot->rank >= 3 ? hot->shape[2] : 0;
    if (!hot_n) snprintf(out, cap, "%zu experts per layer in %s/%s", cold, format(gate), format(down));
    else snprintf(out, cap, "%zu experts per layer: %zu in %s/%s, %zu in %s/%s", cold + hot_n, cold, format(gate),
                  format(down), hot_n, format(hot), format(hot_down));
}
static void row(FILE *f, bool color, const char *label, const char *value) {
    fprintf(f, color ? " " LABEL "%-9s" VALUE " %s" OFF "\n" : " %-9s %s\n", label, value);
}

void qw4_ui_summary(FILE *f, const QW4Run *r, const char *path, bool thinking, double load_s, bool color) {
    const char *slash = strrchr(path, '/'), *file = slash ? slash + 1 : path;
    size_t stem = strlen(file) > 5 && !strcmp(file + strlen(file) - 5, ".gguf") ? strlen(file) - 5 : strlen(file);
    char line[512], ex[192], ctx[16];
    fprintf(f, color ? "\n " ACCENT "picoforge" OFF LABEL " · " OFF ACCENT "%.*s" OFF "\n" : "\n picoforge · %.*s\n", (int)stem, file);
    experts(&r->g, ex, sizeof ex);
    snprintf(line, sizeof line, "%.1f GiB · %u layers · %s", (double)r->g.file_size / 1073741824.0, r->c->trunk_blocks, ex);
    row(f, color, "model", line);
    uint64_t used, budget; qw4_run_memory(r, &used, &budget);
    snprintf(line, sizeof line, "%.1f of %.1f GiB in use · %s · loaded in %.1f s", (double)used / 1073741824.0,
             (double)budget / 1073741824.0, r->pool ? "routed experts streamed from the SSD" : "all weights resident", load_s);
    row(f, color, "GPU", line);
    const char *precision = getenv("PICOFORGE_PREFILL_PRECISION");
    count(ctx, sizeof ctx, r->capacity);
    snprintf(line, sizeof line, "context %s · MTP %s · thinking %s · power %d%% · prefill %s, chunk %u", ctx,
             r->mtp ? "on" : "off", thinking ? "on" : "off", r->power, precision && *precision ? precision : "fp32",
             r->chunk < r->capacity ? r->chunk : r->capacity);
    row(f, color, "session", line);
    if (r->sampler.temperature > 0)
    {
        int n = snprintf(line, sizeof line, "temperature %.2g · top-k %u · top-p %.2g", r->sampler.temperature,
                         r->sampler.top_k, r->sampler.top_p);
        if (r->sampler.min_p > 0 && n > 0) snprintf(line + n, sizeof line - (size_t)n, " · min-p %.2g", r->sampler.min_p);
    }
    else snprintf(line, sizeof line, "greedy (temperature 0)");
    row(f, color, "sampling", line);
    fputs(color ? " " LABEL "/help commands · /info this summary · Ctrl+C stops a reply · Ctrl+D quits" OFF "\n\n"
                : " /help commands · /info this summary · Ctrl+C stops a reply · Ctrl+D quits\n\n", f);
}

/* "▕██████░░░░▏ 1.2K/8.2K 15%": purple, then orange past 60%, red past 85% (as pf-agent). */
static void bar(FILE *f, size_t used, size_t cap) {
    const int cells = 20;
    double frac = cap ? (double)used / (double)cap : 0;
    int full = (int)(frac * cells + 0.5); if (full > cells) full = cells;
    const char *fill = frac > 0.85 ? "\x1b[38;5;203m" : frac > 0.6 ? "\x1b[38;5;214m" : "\x1b[38;5;177m";
    fputs(LABEL "\xe2\x96\x95", f);
    for (int i = 0; i < cells; i++) {
        fputs(i < full ? fill : "\x1b[38;5;240m", f);
        fputs(i < full ? "\xe2\x96\x88" : "\xe2\x96\x91", f);   /* full block, light shade */
    }
    char u[16], c[16]; count(u, sizeof u, used); count(c, sizeof c, cap);
    fprintf(f, LABEL "\xe2\x96\x8f" VALUE " %s/%s %.0f%%" OFF, u, c, frac * 100);
}

void qw4_ui_turn(FILE *f, const QW4Run *r, const QW4Turn *t, size_t prefilled, size_t used, bool cancelled) {
    double prefill = t->prefill_end - t->prefill_start, decode = t->end - t->decode_start;
    char p[16], g[16]; count(p, sizeof p, prefilled); count(g, sizeof g, t->got);
    fputs("  " LABEL "prefill" VALUE, f);
    fprintf(f, " %.0f t/s " LABEL "(%s tok)" VALUE, prefill > 0 ? (double)prefilled / prefill : 0.0, p);
    fprintf(f, LABEL " · decode" VALUE " %.1f t/s " LABEL "(%s tok, %.1f s)",
            decode > 0 && t->got > 1 ? (double)(t->got - 1) / decode : 0.0, g, decode > 0 ? decode : 0.0);
    if (r->mtp && t->proposed)
        fprintf(f, " · MTP" VALUE " %.0f%%" LABEL " of drafts accepted", 100.0 * (double)t->accepted / (double)t->proposed);
    if (cancelled) fputs(" · \x1b[38;5;214minterrupted" LABEL, f);
    fputs("  ", f);
    bar(f, used, r->capacity);
    fputc('\n', f);
}
