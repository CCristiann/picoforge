/* Choose page-aligned views at tensor boundaries, with at most one page of
 * overlap between adjacent views. Never expose disk-only n-grams to Metal. */
#include "common.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* blk.<L>.ffn_<gate|up|down>_exps.weight: returns L (and the matrix) or -1. */
static long expert_tensor(const char *name, int *matrix) {
    static const char *const kind[3] = {"gate", "up", "down"};
    unsigned layer; char tail[8]; int used = 0;
    if (sscanf(name, "blk.%u.ffn_%7[a-z]_exps.weight%n", &layer, tail, &used) < 2 || name[used]) return -1;
    for (int m = 0; m < 3; m++) if (!strcmp(tail, kind[m])) { *matrix = m; return (long)layer; }
    return -1;
}
/* A view may touch the first and last page of a streamed tensor (shared with its neighbours) but never
 * an interior page: those would be mapped, and Metal would keep them resident. */
static bool hits_table(const GgufPlan *p, size_t start, size_t end) {
    const GgufTensor *t = &p->disk_table;
    if (t->bytes && start < gguf_add(t->file_offset, t->bytes) && t->file_offset < end) return true;
    for (size_t i = 0; i < p->streamed_count; i++) {
        const size_t lo = (p->streamed[i].file_offset + p->page_size - 1) & ~(p->page_size - 1);
        const size_t hi = (p->streamed[i].file_offset + p->streamed[i].bytes) & ~(p->page_size - 1);
        if (lo < hi && start < hi && lo < end) return true;
    }
    return false;
}
const GgufTensor *gguf_plan_streamed(const GgufPlan *p, uint32_t layer, int matrix) {
    for (size_t i = 0; i < p->streamed_count; i++) {
        int m;
        if (expert_tensor(p->streamed[i].name, &m) == (long)layer && m == matrix) return &p->streamed[i];
    }
    return NULL;
}

void gguf_plan(const Gguf *g, size_t page, size_t limit, GgufPlan *p) { gguf_plan_ex(g, page, limit, 0, p); }

void gguf_plan_ex(const Gguf *g, size_t page, size_t limit, uint32_t stream_layers, GgufPlan *p) {
    memset(p, 0, sizeof *p);
    if (!page || (page & (page - 1)) || !(limit &= ~(page - 1)))
        die("gguf: invalid page size or buffer limit");
    p->page_size = page;
    p->source = g;
    p->file_size = g->file_size;
    p->views = calloc(g->tensor_count, sizeof *p->views);
    p->weights = calloc(g->tensor_count, sizeof *p->weights);
    p->streamed = calloc(g->tensor_count, sizeof *p->streamed);
    if (!p->views || !p->weights || !p->streamed) die("gguf: view-plan allocation failed");
    size_t previous_end = 0;
    for (size_t i = 0; i < g->tensor_count; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (!t->bytes || t->file_offset < previous_end || t->file_offset > g->file_size ||
            t->bytes > g->file_size - t->file_offset) die("gguf: invalid plan input span");
        previous_end = t->file_offset + t->bytes;
        if (!strcmp(t->name, "per_layer_token_embd.weight")) {
            if (p->disk_table.bytes || t->type != 30 || t->rank != 2 ||
                t->file_offset % page || previous_end % page)
                die("gguf: disk-only BF16 table must have page-aligned boundaries");
            p->disk_table = *t;
        }
        int m; long layer = stream_layers ? expert_tensor(t->name, &m) : -1;
        if (layer >= 0 && (unsigned long)layer < stream_layers) p->streamed[p->streamed_count++] = *t;
    }
    for (size_t i = 0; i < g->tensor_count; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (!strcmp(t->name, "per_layer_token_embd.weight")) continue;
        int m; long layer = stream_layers ? expert_tensor(t->name, &m) : -1;
        if (layer >= 0 && (unsigned long)layer < stream_layers) continue;   /* streamed: no view */
        size_t start = t->file_offset & ~(page - 1);
        size_t end = gguf_add(t->file_offset + t->bytes, page - 1) & ~(page - 1);
        if (end - start > limit || hits_table(p, start, end))
            die("gguf: tensor cannot fit a resident view: %s", t->name);
        GgufView *v = p->view_count ? &p->views[p->view_count - 1] : NULL;
        if (!v || end - v->file_offset > limit || hits_table(p, v->file_offset, end)) {
            v = &p->views[p->view_count++];
            v->file_offset = start;
        }
        v->bytes = end - v->file_offset;
        p->weights[p->weight_count++] = (GgufBinding){*t, p->view_count - 1, t->file_offset - v->file_offset, p};
    }
    if (!p->weight_count) die("gguf: no resident tensors");
}

void gguf_plan_free(GgufPlan *p) {
    free(p->views);
    free(p->weights);
    free(p->streamed);
    memset(p, 0, sizeof *p);
}
