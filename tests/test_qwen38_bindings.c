#include "qwen38.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#ifdef Q38_A_GPU
#include "picoforge.h"
#include "qwen38_packed.h"
static void storage_bridge(const Gguf *g, const Qwen38Config *c) {
    /* Run ONLY with the small sparse fixture, under the common GPU lock. */
    assert(g->file_size < 16*1024*1024);
    MetalContext *metal = metal_init("picoforge.metallib");
    GpuWeightStore *store = gpu_weights_open(metal_device(metal),g,1024*1024);
    Qwen38Weights *w; const Qwen38Tensor *t; Qwen38Error e; Qwen38CpuSpan span;
    assert(qwen38_weights_describe(g,c,&w,&e) == Q38_OK);
    assert(qwen38_weights_attach(w,g,gpu_weights_plan(store),&e) == Q38_OK);
    assert(qwen38_weights_get(w,0,Q38_MOE_DOWN,&t,&e) == Q38_OK);
    assert(qwen38_weights_cpu_span(store,t->gpu,&span,&e) == Q38_OK && span.bytes == t->storage.bytes);
    for (size_t i = 0; i < span.bytes; ++i) assert(span.data[i] == 0);
    GgufBinding forged = *t->gpu;
    assert(qwen38_weights_cpu_span(store,&forged,&span,&e) == Q38_OWNER && !span.data && !span.bytes);
    assert(qwen38_weights_get(w,QWEN38_GLOBAL_LAYER,Q38_PLE_TABLE,&t,&e) == Q38_OK && !t->gpu);
    qwen38_weights_free(w); gpu_weights_close(store); metal_shutdown(metal);
    puts("small Metal storage attach/CPU span/forged owner OK");
}
#endif

static void bad_descriptor(Gguf *g, Qwen38Config *c, size_t i, GgufTensor bad, Qwen38Status expected) {
    GgufTensor old = g->tensors[i]; g->tensors[i] = bad;
    Qwen38Weights *w = (void *)1; Qwen38Error e;
    assert(qwen38_weights_describe(g, c, &w, &e) == expected && e.code == expected && !w);
    g->tensors[i] = old;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    Gguf g; gguf_open(argv[1], &g);
    Qwen38Config *c; Qwen38Weights *w; Qwen38Error e;
    assert(qwen38_config_load(&g, &c, &e) == Q38_OK);
    const Qwen38ConfigView *v = qwen38_config_view(c);
    assert(qwen38_weights_describe(&g, c, &w, &e) == Q38_OK);
    size_t count[3] = {0}; const Qwen38Tensor *t, *down;
    for (uint32_t i = 0; i <= v->blocks; ++i) {
        uint32_t layer = i == v->blocks ? QWEN38_GLOBAL_LAYER : i;
        for (Qwen38Role role = 0; role < Q38_ROLE_COUNT; role++) {
            Qwen38Status s = qwen38_weights_get(w, layer, role, &t, &e);
            if (s == Q38_MISSING) { assert(!t); continue; }
            assert(s == Q38_OK && !t->gpu && t->logical_shape[3] == 1);
            count[t->domain]++;
            assert(t->row_bytes == t->storage.shape[0]/t->block_elems*t->block_bytes);
            assert(t->plane_bytes == t->row_bytes*(t->storage.rank > 1 ? t->storage.shape[1] : 1));
            assert(t->logical_shape[0] <= t->storage.shape[0]);
        }
    }
    assert(count[0]+count[1]+count[2] == g.tensor_count);
    /* Predictor-specific roles are now addressable, still in the MTP domain. */
    for (uint32_t i=v->trunk_blocks;i<v->blocks;i++)
        for (Qwen38Role role=Q38_MTP_EH;role<Q38_ROLE_COUNT;role++) {
            assert(qwen38_weights_get(w,i,role,&t,&e)==Q38_OK && t->domain==Q38_MTP);
            assert(qwen38_weights_get(w,0,role,&t,&e)==Q38_MISSING);
        }
    assert(qwen38_weights_get(w, 0, Q38_MOE_DOWN, &down, &e) == Q38_OK);
    assert(down->logical_shape[0] == v->moe.intermediate && down->storage.shape[0] == v->moe.down_physical);
    assert(down->row_bytes == v->moe.down_physical/256*84 && down->block_elems == 256);
    assert(qwen38_weights_get(w, QWEN38_GLOBAL_LAYER, Q38_PLE_TABLE, &t, &e) == Q38_OK);
    assert(t->domain == Q38_DISK_PLE && t->layout == Q38_DISK_ROWS && !t->gpu);
    size_t ix = 0;
    while (strcmp(g.tensors[ix].name, down->storage.name)) ++ix;
    GgufTensor bad = g.tensors[ix]; bad.type = 12;
    bad_descriptor(&g, c, ix, bad, Q38_FORMAT);
    bad = g.tensors[ix]; bad.shape[0] = v->moe.intermediate;
    bad_descriptor(&g, c, ix, bad, Q38_SHAPE);
    bad = g.tensors[ix]; bad.bytes--;
    bad_descriptor(&g, c, ix, bad, Q38_FORMAT);
    bad = g.tensors[ix]; bad.rank = 2;
    bad_descriptor(&g, c, ix, bad, Q38_SHAPE);
    bad = g.tensors[ix]; strcpy(bad.name, "unexpected.weight");
    bad_descriptor(&g, c, ix, bad, Q38_MISSING);
    bad = g.tensors[ix]; bad.file_offset = g.file_size;
    bad_descriptor(&g, c, ix, bad, Q38_FORMAT);
    GgufPlan plan; gguf_plan(&g, 16384, (size_t)8*1024*1024*1024, &plan);
    Gguf foreign = g;
    assert(qwen38_weights_attach(w, &foreign, &plan, &e) == Q38_OWNER);
    const Gguf *old_source = plan.source; plan.source = &foreign;
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_OWNER); plan.source = old_source;
    size_t offset = plan.weights[0].offset; plan.weights[0].offset++;
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_RANGE); plan.weights[0].offset = offset;
    assert(!down->gpu); /* Failed attach never publishes a partial catalog. */
    const GgufPlan *owner = plan.weights[0].owner; plan.weights[0].owner = NULL;
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_OWNER); plan.weights[0].owner = owner;
    GgufBinding saved = plan.weights[1]; plan.weights[1] = plan.weights[0];
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_FORMAT); plan.weights[1] = saved;
    size_t oldbytes = g.tensors[ix].bytes; g.tensors[ix].bytes--;
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_STALE); g.tensors[ix].bytes = oldbytes;
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_OK);
    assert(down->gpu && down->gpu->owner == &plan && !t->gpu);
    assert(qwen38_weights_attach(w, &g, &plan, &e) == Q38_BUSY);
    printf("bindings OK: ordinary=%zu MTP=%zu disk=%zu views=%zu\n",
        count[0], count[1], count[2], plan.view_count);
#ifdef Q38_A_GPU
    storage_bridge(&g,c);
#endif
    qwen38_config_free(c); gguf_close(&g);
    assert(down->gpu->tensor.bytes == down->storage.bytes); /* independent copies */
    qwen38_weights_free(w); gguf_plan_free(&plan);
    puts("binding mutation, owner, bounds, padding and lifetime OK");
    return 0;
}
