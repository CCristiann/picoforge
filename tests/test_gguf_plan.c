#include "gguf.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const size_t page = 16384;
    GgufTensor tensors[] = {
        {.name="a", .file_offset=page+32, .bytes=page},
        {.name="b", .file_offset=3*page, .bytes=2*page},
        {.name="per_layer_token_embd.weight", .type=30, .rank=2, .file_offset=5*page, .bytes=2*page},
        {.name="c", .file_offset=7*page, .bytes=page}
    };
    Gguf g = {.file_size=8*page, .tensors=tensors, .tensor_count=4};
    size_t cap = 2*page;
    if (argc == 2) {
        if (!strcmp(argv[1], "small")) cap = page;
        if (!strcmp(argv[1], "alignment")) tensors[2].file_offset++;
        if (!strcmp(argv[1], "overflow")) tensors[3].bytes = SIZE_MAX;
        if (!strcmp(argv[1], "overlap")) tensors[1].file_offset = page;
    }
    GgufPlan p;
    gguf_plan(&g, page, cap, &p);
    if (argc != 1) return 2;
    assert(p.view_count == 3 && p.weight_count == 3);
    assert(p.views[0].file_offset == page && p.views[0].bytes == 2*page);
    assert(p.weights[0].offset == 32 && p.weights[1].buffer_id == 1);
    assert(p.weights[2].buffer_id == 2 && p.views[2].file_offset == 7*page);
    assert(p.disk_table.bytes == 2*page);
    gguf_plan_free(&p);
    assert(!p.weights && !p.views);
    puts("PASS: tensor boundaries, multiple views and disk-only exclusion");
    return 0;
}
