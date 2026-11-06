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
    assert(p.source == &g);
    Gguf other = g; /* Equal descriptors do not make two openings identical. */
    GgufPlan other_plan;
    gguf_plan(&other, page, cap, &other_plan);
    assert(other_plan.source != p.source);
    gguf_plan_free(&other_plan);
    assert(p.views[0].file_offset == page && p.views[0].bytes == 2*page);
    assert(p.weights[0].offset == 32 && p.weights[1].buffer_id == 1);
    assert(p.weights[2].buffer_id == 2 && p.views[2].file_offset == 7*page);
    assert(p.disk_table.bytes == 2*page);
    gguf_plan_free(&p);
    assert(!p.weights && !p.views && !p.source);
    puts("PASS: tensor boundaries, multiple views and disk-only exclusion");

    /* R5.1b: the expert tensors of the first stream_layers layers get no view; a later layer's do. */
    GgufTensor moe[] = {
        {.name="blk.0.ffn_gate_exps.weight", .rank=3, .shape={4,4,8}, .file_offset=page, .bytes=4*page},
        {.name="blk.0.ffn_norm.weight", .file_offset=5*page+64, .bytes=64},
        {.name="blk.0.ffn_down_exps.weight", .rank=3, .shape={4,4,8}, .file_offset=6*page, .bytes=4*page},
        {.name="blk.1.ffn_up_exps.weight", .rank=3, .shape={4,4,8}, .file_offset=10*page, .bytes=page},
        {.name="blk.0.ffn_gate_exps.weightx", .file_offset=11*page, .bytes=64},
    };
    Gguf mg = {.file_size=12*page, .tensors=moe, .tensor_count=5};
    gguf_plan_ex(&mg, page, 8*page, 1, &p);
    assert(p.streamed_count == 2 && p.weight_count == 3);
    assert(gguf_plan_streamed(&p, 0, 0) == &p.streamed[0] && gguf_plan_streamed(&p, 0, 2) == &p.streamed[1]);
    assert(!gguf_plan_streamed(&p, 0, 1) && !gguf_plan_streamed(&p, 1, 1));
    for (size_t i = 0; i < p.view_count; i++)   /* no view reaches an interior page of a streamed tensor */
        for (size_t j = 0; j < p.streamed_count; j++)
            assert(p.views[i].file_offset + p.views[i].bytes <= p.streamed[j].file_offset + page ||
                   p.views[i].file_offset >= p.streamed[j].file_offset + p.streamed[j].bytes - page);
    gguf_plan_free(&p);
    gguf_plan_ex(&mg, page, 8*page, 0, &p);
    assert(p.streamed_count == 0 && p.weight_count == 5);
    gguf_plan_free(&p);
    puts("PASS: streamed expert tensors are kept out of every view");
    return 0;
}
