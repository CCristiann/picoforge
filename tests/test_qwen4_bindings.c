#include "qwen4.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#ifdef QW4_A_GPU
#include "common.h"
#include "metal_device.h"
#include "qwen4_packed.h"
static void storage_bridge(const Gguf *g, const Qwen4Config *c) {
    /* Run ONLY with the small sparse fixture, under the common GPU lock. */
    assert(g->file_size < 16*1024*1024);
    MetalContext *metal = metal_init("picoforge.metallib");
    GpuWeightStore *store = gpu_weights_open(metal_device(metal),g,1024*1024);
    Qwen4Weights *w; const Qwen4Tensor *t; Qwen4Error e; Qwen4CpuSpan span;
    assert(qwen4_weights_describe(g,c,&w,&e) == QW4_OK);
    assert(qwen4_weights_attach(w,g,gpu_weights_plan(store),&e) == QW4_OK);
    assert(qwen4_weights_get(w,0,QW4_MOE_DOWN,&t,&e) == QW4_OK);
    assert(qwen4_weights_cpu_span(store,t->gpu,&span,&e) == QW4_OK && span.bytes == t->storage.bytes);
    for (size_t i = 0; i < span.bytes; ++i) assert(span.data[i] == 0);
    GgufBinding forged = *t->gpu;
    assert(qwen4_weights_cpu_span(store,&forged,&span,&e) == QW4_OWNER && !span.data && !span.bytes);
    Qwen4Status table = qwen4_weights_get(w,QWEN4_GLOBAL_LAYER,QW4_PLE_TABLE,&t,&e);
    assert(table == QW4_MISSING || (table == QW4_OK && !t->gpu));
    qwen4_weights_free(w); gpu_weights_close(store); metal_shutdown(metal);
    puts("small Metal storage attach/CPU span/forged owner OK");
}
#endif

static void bad_descriptor(Gguf *g, Qwen4Config *c, size_t i, GgufTensor bad, Qwen4Status expected) {
    GgufTensor old = g->tensors[i]; g->tensors[i] = bad;
    Qwen4Weights *w = (void *)1; Qwen4Error e;
    assert(qwen4_weights_describe(g, c, &w, &e) == expected && e.code == expected && !w);
    g->tensors[i] = old;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    Gguf g; gguf_open(argv[1], &g);
    Qwen4Config *c; Qwen4Weights *w; Qwen4Error e;
    assert(qwen4_config_load(&g, &c, &e) == QW4_OK);
    const Qwen4ConfigView *v = qwen4_config_view(c);
    assert(qwen4_weights_describe(&g, c, &w, &e) == QW4_OK);
    size_t count[3] = {0}; const Qwen4Tensor *t, *down;
    for (uint32_t i = 0; i <= v->blocks; ++i) {
        uint32_t layer = i == v->blocks ? QWEN4_GLOBAL_LAYER : i;
        for (Qwen4Role role = 0; role < QW4_ROLE_COUNT; role++) {
            Qwen4Status s = qwen4_weights_get(w, layer, role, &t, &e);
            if (s == QW4_MISSING) { assert(!t); continue; }
            assert(s == QW4_OK && !t->gpu && t->logical_shape[3] == 1);
            count[t->domain]++;
            assert(t->row_bytes == t->storage.shape[0]/t->block_elems*t->block_bytes);
            assert(t->plane_bytes == t->row_bytes*(t->storage.rank > 1 ? t->storage.shape[1] : 1));
            assert(t->logical_shape[0] <= t->storage.shape[0]);
        }
    }
    assert(count[0]+count[1]+count[2] == g.tensor_count);
    /* Predictor-specific roles are now addressable, still in the MTP domain. */
    for (uint32_t i=v->trunk_blocks;i<v->blocks;i++)
        for (Qwen4Role role=QW4_MTP_EH;role<QW4_ROLE_COUNT;role++) {
            assert(qwen4_weights_get(w,i,role,&t,&e)==QW4_OK && t->domain==QW4_MTP);
            assert(qwen4_weights_get(w,0,role,&t,&e)==QW4_MISSING);
        }
    assert(qwen4_weights_get(w, 0, QW4_MOE_DOWN, &down, &e) == QW4_OK);
    assert(down->logical_shape[0] == v->moe.intermediate && down->storage.shape[0] == v->moe.down_physical);
    assert(down->row_bytes == v->moe.down_physical/256*84 && down->block_elems == 256);
    bool has_ple = false; const Qwen4Layer *kind;
    for (uint32_t i = 0; i < v->blocks; ++i) { assert(qwen4_config_layer(c,i,&kind,&e) == QW4_OK); has_ple |= kind->has_ple; }
    const Qwen4Tensor *table = NULL;
    Qwen4Status table_status = qwen4_weights_get(w, QWEN4_GLOBAL_LAYER, QW4_PLE_TABLE, &table, &e);
    if (has_ple) assert(table_status == QW4_OK && table->domain == QW4_DISK_PLE && table->layout == QW4_DISK_ROWS && !table->gpu);
    else assert(table_status == QW4_MISSING && !table && !count[QW4_DISK_PLE]);  /* no PLE: no SSD table */
    size_t ix = 0;
    while (strcmp(g.tensors[ix].name, down->storage.name)) ++ix;
    GgufTensor bad = g.tensors[ix]; bad.type = 12;
    bad_descriptor(&g, c, ix, bad, QW4_FORMAT);
    bad = g.tensors[ix]; bad.shape[0] = v->moe.intermediate;
    bad_descriptor(&g, c, ix, bad, QW4_SHAPE);
    bad = g.tensors[ix]; bad.bytes--;
    bad_descriptor(&g, c, ix, bad, QW4_FORMAT);
    bad = g.tensors[ix]; bad.rank = 2;
    bad_descriptor(&g, c, ix, bad, QW4_SHAPE);
    bad = g.tensors[ix]; strcpy(bad.name, "unexpected.weight");
    bad_descriptor(&g, c, ix, bad, QW4_MISSING);
    bad = g.tensors[ix]; bad.file_offset = g.file_size;
    bad_descriptor(&g, c, ix, bad, QW4_FORMAT);
    GgufPlan plan; gguf_plan(&g, 16384, (size_t)8*1024*1024*1024, &plan);
    Gguf foreign = g;
    assert(qwen4_weights_attach(w, &foreign, &plan, &e) == QW4_OWNER);
    const Gguf *old_source = plan.source; plan.source = &foreign;
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_OWNER); plan.source = old_source;
    size_t offset = plan.weights[0].offset; plan.weights[0].offset++;
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_RANGE); plan.weights[0].offset = offset;
    assert(!down->gpu); /* Failed attach never publishes a partial catalog. */
    const GgufPlan *owner = plan.weights[0].owner; plan.weights[0].owner = NULL;
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_OWNER); plan.weights[0].owner = owner;
    GgufBinding saved = plan.weights[1]; plan.weights[1] = plan.weights[0];
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_FORMAT); plan.weights[1] = saved;
    size_t oldbytes = g.tensors[ix].bytes; g.tensors[ix].bytes--;
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_STALE); g.tensors[ix].bytes = oldbytes;
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_OK);
    assert(down->gpu && down->gpu->owner == &plan && (!table || !table->gpu));
    assert(qwen4_weights_attach(w, &g, &plan, &e) == QW4_BUSY);
    printf("bindings OK: ordinary=%zu MTP=%zu disk=%zu views=%zu\n",
        count[0], count[1], count[2], plan.view_count);
#ifdef QW4_A_GPU
    storage_bridge(&g,c);
#endif
    qwen4_config_free(c); gguf_close(&g);
    assert(down->gpu->tensor.bytes == down->storage.bytes); /* independent copies */
    qwen4_weights_free(w); gguf_plan_free(&plan);
    puts("binding mutation, owner, bounds, padding and lifetime OK");
    return 0;
}
