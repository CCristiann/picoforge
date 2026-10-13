/* G1 harness: run the CPU graph on a GGUF and dump every traced boundary.
 *   test_qwen38_forward GGUF OUTDIR LAYERS CHUNK TOKEN...
 * Chunks of CHUNK tokens are appended in order; each (layer, stage) trace
 * is appended to OUTDIR/<layer>.<stage>.f32 so a file holds the whole prefix.
 * The Metal device only backs the existing no-copy weight mapping: no GPU
 * work is encoded. Run under the common lock (tools/qwen38_gpu_lock.py). */
#include "picoforge.h"
#include "qwen38_forward.h"
#include <assert.h>
#include <errno.h>
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

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s GGUF OUTDIR LAYERS CHUNK TOKEN...\n", argv[0]); return 2; }
    outdir = argv[2];
    uint32_t layers = (uint32_t)strtoul(argv[3], NULL, 10), chunk = (uint32_t)strtoul(argv[4], NULL, 10);
    size_t count = (size_t)(argc - 5);
    int32_t *tokens = malloc(count * sizeof *tokens);
    assert(tokens && chunk);
    for (size_t i = 0; i < count; ++i) tokens[i] = (int32_t)strtol(argv[5 + i], NULL, 10);

    Gguf g; gguf_open(argv[1], &g);
    Qwen38Config *config = NULL; Qwen38Weights *weights = NULL; Qwen38Forward *graph = NULL;
    Qwen38Error e;
    check(qwen38_config_load(&g, &config, &e), &e);
    check(qwen38_weights_describe(&g, config, &weights, &e), &e);
    MetalContext *metal = metal_init("picoforge.metallib");
    GpuWeightStore *store = gpu_weights_open(metal_device(metal), &g, 0);
    check(qwen38_weights_attach(weights, &g, gpu_weights_plan(store), &e), &e);
    check(qwen38_forward_create(&g, config, weights, store, chunk, (uint32_t)count, &graph, &e), &e);
    uint32_t vocab = qwen38_config_view(config)->vocab_size;
    float *logits = malloc(count * vocab * 4);
    assert(logits);
    for (size_t start = 0; start < count; start += chunk) {
        size_t n = count - start < chunk ? count - start : chunk;
        Qwen38F32Mut y = {.data=logits + start*vocab, .capacity_bytes=n*vocab*4, .rank=2,
            .dim={n, vocab}, .stride_bytes={vocab*4, 4}};
        check(qwen38_forward_cpu(graph, tokens + start, n, layers, y, dump, NULL, &e), &e);
        if (layers < qwen38_config_view(config)->trunk_blocks) {   /* prefix slice: one shot */
            assert(n == count);
            break;
        }
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/logits.f32", outdir);
    FILE *f = fopen(path, "wb");
    assert(f && fwrite(logits, 4, count*vocab, f) == count*vocab && !fclose(f));
    Qwen38StateInfo info = qwen38_forward_info(graph);
    printf("{\"tokens\":%zu,\"layers\":%u,\"chunk\":%u,\"next_pos\":%llu,\"valid\":%s}\n",
           count, layers, chunk, (unsigned long long)info.next_pos, info.valid ? "true" : "false");
    qwen38_forward_destroy(graph); qwen38_weights_free(weights);
    gpu_weights_close(store); metal_shutdown(metal); qwen38_config_free(config); gguf_close(&g);
    free(logits); free(tokens);
    return 0;
}
