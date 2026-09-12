/* gpu_forward.m — the whole forward pass on the GPU.
 *
 * Two design decisions carry this file.
 *
 * The weights are never uploaded. The checkpoint is already mmapped, the
 * memory is unified, and newBufferWithBytesNoCopy over that mapping gives the
 * GPU the same physical pages the CPU sees. One MTLBuffer spans the entire
 * file and every weight is an offset into it. No copy exists, no second
 * gigabyte is allocated, and the cost of "moving the model to the GPU" is one
 * pointer. On a 30B MoE this is the difference between fitting in 64 GB and
 * not.
 *
 * The whole pass goes into ONE command buffer. A layer is 7 matmuls plus 8
 * other dispatches, so 28 layers is about 420 dispatches; at 6 microseconds
 * of command-buffer latency each, submitting them separately would cost
 * 2.5 ms of pure waiting per token before any arithmetic. Inside one buffer
 * they queue back to back. A serial compute encoder also orders them, so the
 * dependencies between the steps need no explicit barriers.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "picoforge.h"

#include <math.h>

enum { K_MATMUL_NAIVE, K_MATMUL_SIMD, K_MATMUL_TENSOR, K_EMBED, K_RMSNORM,
       K_QKROPE, K_ATTN, K_SWIGLU, K_RESIDUAL, K_COUNT };

static const char *gpu_kernel_names[K_COUNT] = {
    "matmul_naive", "matmul_simdgroup", "matmul_tensorops", "embed_lookup",
    "rmsnorm_rows", "qk_norm_rope", "attention", "swiglu", "add_residual",
};

typedef struct { size_t input_ln, q_proj, k_proj, v_proj, q_norm, k_norm,
                        o_proj, post_attn_ln, gate_proj, up_proj, down_proj; } LayerOffsets;

struct GpuModel {
    MetalContext      *ctx;
    Qwen3Config        cfg;
    int                max_seq, max_rows, matmul_kernel;

    void              *pso[K_COUNT];
    void              *weights;          /* one buffer over the whole mapping */
    size_t             embed_off, final_norm_off;
    LayerOffsets      *layer;

    void *x, *xb, *q, *attout, *hb, *hb2, *kcache, *vcache, *logits, *tokens;
};

static size_t offset_of(const SafeTensors *st, const char *name) {
    return (size_t)((const unsigned char *)st_find(st, name)->data
                  - (const unsigned char *)st->map);
}

static size_t layer_offset(const SafeTensors *st, const char *fmt, int l) {
    char name[128];
    snprintf(name, sizeof name, fmt, l);
    return offset_of(st, name);
}

GpuModel *gpu_model_create(MetalContext *ctx, const SafeTensors *st,
                           const Qwen3Config *cfg, int max_seq, int max_rows) {
    GpuModel *g = calloc(1, sizeof *g);
    if (!g) die("out of memory for the GPU model");
    g->ctx = ctx; g->cfg = *cfg;
    g->max_seq = max_seq; g->max_rows = max_rows;
    g->matmul_kernel = K_MATMUL_TENSOR;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device(ctx);
    for (int i = 0; i < K_COUNT; i++)
        g->pso[i] = metal_pipeline(ctx, gpu_kernel_names[i]);

    /* The zero-copy binding. mmap hands back a page-aligned base and rounds
     * the length up to a page, which is exactly what the no-copy path
     * demands; the deallocator is nil because the mapping is not ours to
     * free -- SafeTensors owns it. */
    size_t page = (size_t)getpagesize();
    size_t len = (st->map_len + page - 1) & ~(page - 1);
    id<MTLBuffer> wbuf = [dev newBufferWithBytesNoCopy:st->map
                                                length:len
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (!wbuf) die("cannot map the checkpoint into a Metal buffer (%.2f GB)",
                   (double)len / 1e9);
    g->weights = (__bridge_retained void *)wbuf;

    g->embed_off      = offset_of(st, "model.embed_tokens.weight");
    g->final_norm_off = offset_of(st, "model.norm.weight");
    g->layer = calloc((size_t)cfg->num_hidden_layers, sizeof *g->layer);
    if (!g->layer) die("out of memory for %d layer offsets", cfg->num_hidden_layers);

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        LayerOffsets *o = &g->layer[l];
        o->input_ln     = layer_offset(st, "model.layers.%d.input_layernorm.weight", l);
        o->q_proj       = layer_offset(st, "model.layers.%d.self_attn.q_proj.weight", l);
        o->k_proj       = layer_offset(st, "model.layers.%d.self_attn.k_proj.weight", l);
        o->v_proj       = layer_offset(st, "model.layers.%d.self_attn.v_proj.weight", l);
        o->q_norm       = layer_offset(st, "model.layers.%d.self_attn.q_norm.weight", l);
        o->k_norm       = layer_offset(st, "model.layers.%d.self_attn.k_norm.weight", l);
        o->o_proj       = layer_offset(st, "model.layers.%d.self_attn.o_proj.weight", l);
        o->post_attn_ln = layer_offset(st, "model.layers.%d.post_attention_layernorm.weight", l);
        o->gate_proj    = layer_offset(st, "model.layers.%d.mlp.gate_proj.weight", l);
        o->up_proj      = layer_offset(st, "model.layers.%d.mlp.up_proj.weight", l);
        o->down_proj    = layer_offset(st, "model.layers.%d.mlp.down_proj.weight", l);
    }

    const size_t H = (size_t)cfg->hidden_size;
    const size_t q_dim = (size_t)(cfg->num_attention_heads * cfg->head_dim);
    const size_t kv_dim = (size_t)(cfg->num_key_value_heads * cfg->head_dim);
    const size_t I = (size_t)cfg->intermediate_size;
    const size_t L = (size_t)cfg->num_hidden_layers;
    const size_t S = (size_t)max_seq;

#define ALLOC(field, n) g->field = (__bridge_retained void *) \
    [dev newBufferWithLength:(n) * sizeof(float) options:MTLResourceStorageModeShared]
    ALLOC(x, S * H);        ALLOC(xb, S * H);       ALLOC(q, S * q_dim);
    ALLOC(attout, S * q_dim); ALLOC(hb, S * I);     ALLOC(hb2, S * I);
    ALLOC(kcache, L * S * kv_dim); ALLOC(vcache, L * S * kv_dim);
    ALLOC(logits, (size_t)max_rows * (size_t)cfg->vocab_size);
#undef ALLOC
    g->tokens = (__bridge_retained void *)
        [dev newBufferWithLength:S * sizeof(int) options:MTLResourceStorageModeShared];
    return g;
}

void gpu_model_free(GpuModel *g) {
    if (!g) return;
    void *bufs[] = {g->weights, g->x, g->xb, g->q, g->attout, g->hb, g->hb2,
                    g->kcache, g->vcache, g->logits, g->tokens};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i]) CFRelease(bufs[i]);
    free(g->layer);
    free(g);
}

void gpu_set_matmul_kernel(GpuModel *g, int which) { g->matmul_kernel = which; }

/* ------------------------------------------------------------- encoding */

struct MatmulDimsC { uint32_t M, N, K; };
struct NormDimsC   { uint32_t rows, n; float eps; };
struct RopeDimsC   { uint32_t n, heads, head_dim, dim, pos; float theta, eps; };
struct AttnDimsC   { uint32_t n, heads, kv_heads, head_dim, q_dim, kv_dim, pos, max_seq; float scale; };
struct ElemDimsC   { uint32_t count; };

static void encode_matmul(GpuModel *g, id<MTLComputeCommandEncoder> enc,
                          void *cbuf, size_t coff, void *abuf, size_t aoff,
                          size_t woff, int M, int N, int K) {
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[g->matmul_kernel]];
    [enc setBuffer:(__bridge id<MTLBuffer>)cbuf offset:coff atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)abuf offset:aoff atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:woff atIndex:2];
    struct MatmulDimsC d = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
    [enc setBytes:&d length:sizeof d atIndex:3];

    if (g->matmul_kernel == K_MATMUL_NAIVE) {
        NSUInteger tgh = (NSUInteger)(M < 16 ? M : 16);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1)
       threadsPerThreadgroup:MTLSizeMake(256 / tgh, tgh, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 31) / 32),
                                              (NSUInteger)((M + 31) / 32), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

static void encode_elem(GpuModel *g, id<MTLComputeCommandEncoder> enc, int kernel,
                        void *b0, size_t o0, void *b1, size_t o1, size_t count) {
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[kernel]];
    [enc setBuffer:(__bridge id<MTLBuffer>)b0 offset:o0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b1 offset:o1 atIndex:1];
    struct ElemDimsC d = {(uint32_t)count};
    [enc setBytes:&d length:sizeof d atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_rmsnorm(GpuModel *g, id<MTLComputeCommandEncoder> enc,
                           void *out, size_t ooff, void *in, size_t ioff,
                           size_t woff, int rows, int n, float eps) {
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_RMSNORM]];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:ooff atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)in offset:ioff atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:woff atIndex:2];
    struct NormDimsC d = {(uint32_t)rows, (uint32_t)n, eps};
    [enc setBytes:&d length:sizeof d atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void gpu_forward(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                 float *logits_out) {
    const Qwen3Config *cfg = &g->cfg;
    const int H = cfg->hidden_size, hd = cfg->head_dim;
    const int n_head = cfg->num_attention_heads, n_kv = cfg->num_key_value_heads;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, I = cfg->intermediate_size;
    const float eps = cfg->rms_norm_eps;

    if (pos + n > g->max_seq)
        die("context overflow: %d tokens at position %d exceeds %d", n, pos, g->max_seq);

    id<MTLBuffer> tokbuf = (__bridge id<MTLBuffer>)g->tokens;
    memcpy(tokbuf.contents, tokens, (size_t)n * sizeof *tokens);

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue(g->ctx);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    /* embeddings */
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_EMBED]];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:g->embed_off atIndex:1];
    [enc setBuffer:tokbuf offset:0 atIndex:2];
    struct { uint32_t n, hidden; } ed = {(uint32_t)n, (uint32_t)H};
    [enc setBytes:&ed length:sizeof ed atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(n * H), 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        const LayerOffsets *o = &g->layer[l];
        const size_t layer_kv = (size_t)l * (size_t)g->max_seq * (size_t)kv_dim * sizeof(float);
        const size_t row_kv   = layer_kv + (size_t)pos * (size_t)kv_dim * sizeof(float);

        encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, o->input_ln, n, H, eps);
        encode_matmul(g, enc, g->q, 0, g->xb, 0, o->q_proj, n, q_dim, H);
        encode_matmul(g, enc, g->kcache, row_kv, g->xb, 0, o->k_proj, n, kv_dim, H);
        encode_matmul(g, enc, g->vcache, row_kv, g->xb, 0, o->v_proj, n, kv_dim, H);

        /* QK-norm then RoPE, on Q and on the cache rows just written. */
        for (int side = 0; side < 2; side++) {
            const bool is_q = (side == 0);
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_QKROPE]];
            [enc setBuffer:(__bridge id<MTLBuffer>)(is_q ? g->q : g->kcache)
                    offset:(is_q ? 0 : row_kv) atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)g->weights
                    offset:(is_q ? o->q_norm : o->k_norm) atIndex:1];
            struct RopeDimsC rd = {(uint32_t)n, (uint32_t)(is_q ? n_head : n_kv),
                                   (uint32_t)hd, (uint32_t)(is_q ? q_dim : kv_dim),
                                   (uint32_t)pos, cfg->rope_theta, eps};
            [enc setBytes:&rd length:sizeof rd atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(is_q ? n_head : n_kv),
                                                  (NSUInteger)n, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_ATTN]];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->attout offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->q offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->kcache offset:layer_kv atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->vcache offset:layer_kv atIndex:3];
        struct AttnDimsC ad = {(uint32_t)n, (uint32_t)n_head, (uint32_t)n_kv,
                               (uint32_t)hd, (uint32_t)q_dim, (uint32_t)kv_dim,
                               (uint32_t)pos, (uint32_t)g->max_seq,
                               1.0f / sqrtf((float)hd)};
        [enc setBytes:&ad length:sizeof ad atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)n, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        encode_matmul(g, enc, g->xb, 0, g->attout, 0, o->o_proj, n, H, q_dim);
        encode_elem(g, enc, K_RESIDUAL, g->x, 0, g->xb, 0, (size_t)n * (size_t)H);

        encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, o->post_attn_ln, n, H, eps);
        encode_matmul(g, enc, g->hb, 0, g->xb, 0, o->gate_proj, n, I, H);
        encode_matmul(g, enc, g->hb2, 0, g->xb, 0, o->up_proj, n, I, H);
        encode_elem(g, enc, K_SWIGLU, g->hb, 0, g->hb2, 0, (size_t)n * (size_t)I);
        encode_matmul(g, enc, g->xb, 0, g->hb, 0, o->down_proj, n, H, I);
        encode_elem(g, enc, K_RESIDUAL, g->x, 0, g->xb, 0, (size_t)n * (size_t)H);
    }

    encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, g->final_norm_off, n, H, eps);
    encode_matmul(g, enc, g->logits, 0, g->xb,
                  (size_t)logits_from * (size_t)H * sizeof(float),
                  g->embed_off, n - logits_from, cfg->vocab_size, H);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) die("GPU forward failed: %s", cb.error.localizedDescription.UTF8String);

    if (logits_out) {
        id<MTLBuffer> lb = (__bridge id<MTLBuffer>)g->logits;
        memcpy(logits_out, lb.contents,
               (size_t)(n - logits_from) * (size_t)cfg->vocab_size * sizeof(float));
    }
}
