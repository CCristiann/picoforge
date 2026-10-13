/* Choose page-aligned views at tensor boundaries, with at most one page of
 * overlap between adjacent views. Never expose disk-only n-grams to Metal. */
#include "picoforge.h"
#include "gguf.h"
#include <stdlib.h>

static bool hits_table(const GgufPlan *p, size_t start, size_t end) {
    const GgufTensor *t = &p->disk_table;
    return t->bytes && start < gguf_add(t->file_offset, t->bytes) && t->file_offset < end;
}

void gguf_plan(const Gguf *g, size_t page, size_t limit, GgufPlan *p) {
    memset(p, 0, sizeof *p);
    if (!page || (page & (page - 1)) || !(limit &= ~(page - 1)))
        die("gguf: invalid page size or buffer limit");
    p->page_size = page;
    p->source = g;
    p->file_size = g->file_size;
    p->views = calloc(g->tensor_count, sizeof *p->views);
    p->weights = calloc(g->tensor_count, sizeof *p->weights);
    if (!p->views || !p->weights) die("gguf: view-plan allocation failed");
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
    }
    for (size_t i = 0; i < g->tensor_count; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (!strcmp(t->name, "per_layer_token_embd.weight")) continue;
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
    memset(p, 0, sizeof *p);
}
