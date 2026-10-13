/* G2 harness: run the GPU graph on a GGUF and dump every traced boundary.
 *   test_qwen38_forward GGUF OUTDIR LAYERS CHUNK TOKEN...
 * Chunks of CHUNK tokens are appended in order; each (layer, stage) trace
 * is appended to OUTDIR/<layer>.<stage>.f32 so a file holds the whole prefix.
 * All operators run on GPU, with one completion per append.
 * Run under the common lock (tools/qwen38_gpu_lock.py). */
#include "picoforge.h"
#include "qwen38_g_internal.h"
#include <math.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const stage_names[Q38G_STAGE_COUNT] = {"embedding", "ple_delta",
    "attn_mixed", "attn_inject", "attn_contribution", "attn_residual", "ffn_mixed",
    "ffn_inject", "moe", "output", "final"};
static const char *outdir;

static void dump(void *user, uint32_t layer, Qwen38GraphStage stage, Qwen38F32 v) {
    (void)user;
    char path[1024], name[16];
    if (layer == QWEN38_GLOBAL_LAYER) snprintf(name, sizeof name, "global");
    else snprintf(name, sizeof name, "layer%u", layer);
    snprintf(path, sizeof path, "%s/%s.%s.f32", outdir, name, stage_names[stage]);
    FILE *f = fopen(path, "ab");
    assert(f);
    size_t inner = 1;                           /* contiguous elements per token */
    for (size_t i = 1; i < v.rank; ++i) inner *= v.dim[i];
    for (size_t t = 0; t < v.dim[0]; ++t)
        assert(fwrite((const char *)v.data + t*v.stride_bytes[0], 4, inner, f) == inner);
    assert(!fclose(f));
}

static void check(Qwen38Status rc, const Qwen38Error *e) {
    if (rc == Q38_OK) return;
    fprintf(stderr, "FAIL: status %d %s: %s\n", rc, e->subject, e->detail);
    exit(1);
}

/* Canonical histories as well as output must survive chunk/reset/replay.
 * Inject a late layer's NaN AFTER an initial append, so earlier PLE/GDN have
 * already written on device when completion invalidates the whole graph. */
static void graph_state_checks(Qwen38GpuForward *graph,const int32_t *tokens,size_t count,
                               uint32_t chunk,uint32_t layers,uint32_t vocab) {
    Qwen38Error e; NSMutableArray<NSData *> *saved=[NSMutableArray new];
    for (uint32_t l=0;l<layers;l++) for (uint32_t k=0;k<Q38GS_COUNT;k++) {
        Qwen38GpuF32 v=q38g_gpu_state(graph,l,k);
        if (v.span.buffer) [saved addObject:[NSData dataWithBytes:(char *)[(__bridge id<MTLBuffer>)v.span.buffer contents]+v.span.offset_bytes length:v.span.bytes]];
    }
    check(qwen38_gpu_forward_reset(graph,&e),&e);
    uint64_t epoch=qwen38_gpu_forward_info(graph).key.epoch;
    Qwen38F32Mut none={.rank=2,.dim={0,vocab},.stride_bytes={(size_t)vocab*4,4}};
    check(qwen38_gpu_forward(graph,NULL,0,layers,none,NULL,NULL,&e),&e);
    for (size_t i=0;i<count;i++) check(qwen38_gpu_forward(graph,tokens+i,1,layers,none,NULL,NULL,&e),&e);
    size_t n=0;
    for (uint32_t l=0;l<layers;l++) for (uint32_t k=0;k<Q38GS_COUNT;k++) {
        Qwen38GpuF32 v=q38g_gpu_state(graph,l,k);
        if (v.span.buffer) {
            NSData *ref=saved[n++]; assert(ref.length==v.span.bytes);
            assert(!memcmp(ref.bytes,(char *)[(__bridge id<MTLBuffer>)v.span.buffer contents]+v.span.offset_bytes,v.span.bytes));
        }
    }
    check(qwen38_gpu_forward_reset(graph,&e),&e);
    assert(qwen38_gpu_forward_info(graph).key.epoch==epoch+1);
    check(qwen38_gpu_forward(graph,tokens,1,layers,none,NULL,NULL,&e),&e);
    Qwen38GpuF32 poison={0};
    for (uint32_t l=layers;l-->0 && !poison.span.buffer;) {
        poison=q38g_gpu_state(graph,l,Q38GS_VALUE);
        if (!poison.span.buffer) poison=q38g_gpu_state(graph,l,Q38GS_GDN);
    }
    assert(poison.span.buffer);
    float *data=(float *)((char *)[(__bridge id<MTLBuffer>)poison.span.buffer contents]+poison.span.offset_bytes); data[0]=NAN;
    float *output=malloc((size_t)vocab*4); assert(output);
    for (size_t j=0;j<vocab;j++) output[j]=123.0f;
    Qwen38F32Mut y={.data=output,.capacity_bytes=(size_t)vocab*4,.rank=2,.dim={1,vocab},.stride_bytes={(size_t)vocab*4,4}};
    assert(qwen38_gpu_forward(graph,tokens+1,1,layers,y,NULL,NULL,&e)==Q38_GPU_ERROR);
    Qwen38StateInfo info=qwen38_gpu_forward_info(graph);
    assert(!info.valid && info.next_pos==1);
    for (size_t j=0;j<vocab;j++) assert(output[j]==123.0f);
    for (uint32_t l=0;l<layers;l++) { Qwen38StateInfo ei=q38g_gpu_ngram_info(graph,l); if (ei.key.model) assert(ei.next_pos==1); }
    assert(qwen38_gpu_forward(graph,tokens,1,layers,none,NULL,NULL,&e)==Q38_BAD_STATE);
    check(qwen38_gpu_forward_reset(graph,&e),&e);
    for (size_t start=0;start<count;start+=chunk) {
        size_t take=MIN(count-start,chunk);
        check(qwen38_gpu_forward(graph,tokens+start,take,layers,none,NULL,NULL,&e),&e);
    }
    n=0;
    for (uint32_t l=0;l<layers;l++) for (uint32_t k=0;k<Q38GS_COUNT;k++) {
        Qwen38GpuF32 v=q38g_gpu_state(graph,l,k);
        if (v.span.buffer) assert(!memcmp(saved[n++].bytes,(char *)[(__bridge id<MTLBuffer>)v.span.buffer contents]+v.span.offset_bytes,v.span.bytes));
    }
    free(output);
    fprintf(stderr,"G2 histories: chunk/reset/replay bit-exact; late GPU fault withheld logits/hash/position\n");
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s GGUF OUTDIR LAYERS CHUNK TOKEN...\n", argv[0]); return 2; }
    outdir = argv[2];
    uint32_t layers = (uint32_t)strtoul(argv[3], NULL, 10), chunk = (uint32_t)strtoul(argv[4], NULL, 10);
    size_t count = (size_t)(argc - 5);
    int32_t *tokens = malloc(count * sizeof *tokens);
    assert(tokens && chunk);
    for (size_t i = 0; i < count; ++i) tokens[i] = (int32_t)strtol(argv[5 + i], NULL, 10);

    Gguf g; gguf_open(argv[1], &g);
    Qwen38Config *config = NULL; Qwen38Weights *weights = NULL; Qwen38GpuForward *graph = NULL;
    Qwen38Error e;
    check(qwen38_config_load(&g, &config, &e), &e);
    check(qwen38_weights_describe(&g, config, &weights, &e), &e);
    @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@"picoforge.metallib"] error:NULL];
    assert(device && library);
    GpuWeightStore *store = gpu_weights_open((__bridge void *)device, &g, 0);
    check(qwen38_weights_attach(weights, &g, gpu_weights_plan(store), &e), &e);
    check(qwen38_gpu_forward_create(&g, config, weights, store, (__bridge void *)device, (__bridge void *)library, chunk, (uint32_t)count, chunk, &graph, &e), &e);
    uint32_t vocab = qwen38_config_view(config)->vocab_size;
    bool full = layers == qwen38_config_view(config)->trunk_blocks;
    float *logits = malloc(count * vocab * 4), *again = malloc(count * vocab * 4);
    assert(logits && again);
    /* Preflight rejections happen before any mutation: the session stays valid. */
    int32_t bad = (int32_t)vocab;
    Qwen38F32Mut none = {.data=logits, .capacity_bytes=vocab*4, .rank=2, .dim={0, vocab}, .stride_bytes={vocab*4, 4}};
    assert(qwen38_gpu_forward(graph, &bad, 1, layers, none, NULL, NULL, &e) == Q38_RANGE);
    assert(qwen38_gpu_forward(graph, tokens, (size_t)chunk + 1, layers, none, NULL, NULL, &e) == Q38_CAPACITY);
    assert(qwen38_gpu_forward_info(graph).valid && !qwen38_gpu_forward_info(graph).next_pos);
    const char *replay = "none";
    for (int pass = 0; pass < (full && getenv("Q38_G1_REPLAY") ? 2 : 1); ++pass) {
        float *out = pass ? again : logits;
        if (pass) check(qwen38_gpu_forward_reset(graph, &e), &e);      /* reset + replay */
        for (size_t start = 0; start < count; start += chunk) {
            size_t n = count - start < chunk ? count - start : chunk;
            Qwen38F32Mut y = {.data=out + start*vocab, .capacity_bytes=n*vocab*4, .rank=2,
                .dim={n, vocab}, .stride_bytes={vocab*4, 4}};
            check(qwen38_gpu_forward(graph, tokens + start, n, layers, y, pass ? NULL : dump, NULL, &e), &e);
            if (!full) { assert(n == count); break; }               /* prefix slice: one shot */
        }
        if (pass) replay = memcmp(logits, again, count * vocab * 4) ? "different" : "identical";
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/logits.f32", outdir);
    FILE *f = fopen(path, "wb");
    assert(f && fwrite(logits, 4, count*vocab, f) == count*vocab && !fclose(f));
    Qwen38StateInfo info = qwen38_gpu_forward_info(graph);
    printf("{\"tokens\":%zu,\"layers\":%u,\"chunk\":%u,\"next_pos\":%llu,\"valid\":%s,\"replay\":\"%s\"}\n",
           count, layers, chunk, (unsigned long long)info.next_pos, info.valid ? "true" : "false",
           replay);
    if (full && count>1 && getenv("Q38_G2_STATE_CHECKS")) graph_state_checks(graph,tokens,count,chunk,layers,vocab);
    qwen38_gpu_forward_destroy(graph); qwen38_weights_free(weights);
    gpu_weights_close(store); qwen38_config_free(config); gguf_close(&g);
    free(logits); free(again); free(tokens);
    }
    return 0;
}
