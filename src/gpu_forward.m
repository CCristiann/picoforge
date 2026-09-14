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
       K_QKROPE, K_ATTN, K_SWIGLU, K_RESIDUAL, K_NARROW, K_EMBED_Q,
       K_Q8_ROW, K_Q8_G32, K_Q4_ROW, K_Q4_G32, K_Q4_G64, K_MATMUL_TENSOR_8X32,
       K_MOE_ROUTE, K_MOE_GROUP, K_MOE_GATHER, K_MOE_MM, K_MOE_SCATTER, K_MOE_Q8, K_COUNT };

static const char *gpu_kernel_names[K_COUNT] = {
    "matmul_naive", "matmul_simdgroup", "matmul_tensorops", "embed_lookup",
    "rmsnorm_rows", "qk_norm_rope", "attention", "swiglu", "add_residual",
    "narrow_bf16", "embed_lookup_q",
    "matmul_q8_row", "matmul_q8_g32", "matmul_q4_row", "matmul_q4_g32", "matmul_q4_g64",
    "matmul_tensorops_8x32",
    "moe_route", "moe_group", "moe_gather", "moe_matmul", "moe_scatter", "moe_matmul_q8row",
};

/* A MoE pass takes at most 4 row tiles of 8 per expert group (kernels.metal),
 * so at most 32 tokens. */
enum { MOE_MAX_ROWS = 32 };

/* A projection as the GPU sees it: offsets into the one weights buffer, and
 * the kernel that multiplies by it. kernel < 0 means bf16 (the selectable
 * Phase 2 matmul); otherwise codes live at q and scales at d. */
typedef struct { size_t w, q, d; int bits, group, kernel; } GLinear;

typedef struct { size_t input_ln, q_norm, k_norm, post_attn_ln;
                 GLinear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
                 bool moe; bool moe_q8; GLinear router; } LayerOffsets;

struct GpuModel {
    MetalContext      *ctx;
    Qwen3Config        cfg;
    int                max_seq, max_rows, matmul_kernel;
    bool               small_tile;       /* 8x32 TensorOps tiles when M <= 8 */

    void              *pso[K_COUNT];
    void              *weights;          /* one buffer over the whole mapping */
    size_t             final_norm_off;
    GLinear            embed, head;      /* head == embed when tied           */
    LayerOffsets      *layer;

    void *x, *xb, *q, *attout, *hb, *hb2, *kcache, *vcache, *logits, *tokens;
    void *act16;                         /* bf16 activations for Q4 matmuls   */

    /* MoE: expert weight offsets, [layer][gate, up, down][expert][weights or
     * codes, scales] -- bfloats for bf16 and scales, bytes for 8-bit codes --
     * and the routing and packed-row buffers of one <= 32-token pass. */
    void *moe_offs, *moe_logits, *moe_idx, *moe_wt, *moe_rop, *moe_tor, *moe_groups,
         *moe_ng, *moe_a, *moe_g, *moe_u, *moe_d;
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

/* The CPU's Linear, translated into offsets and a kernel. weights_bind has
 * already validated every shape, so this only has to pick the kernel -- and
 * refuse loudly if the file holds a block length no kernel was written for,
 * rather than fall back to something slower that nobody measured. */
static GLinear to_glinear(const SafeTensors *st, const Linear *l, int n_in) {
    const unsigned char *base = st->map;
    GLinear g = {0};
    g.kernel = -1;
    if (l->w) { g.w = (size_t)((const unsigned char *)l->w - base); return g; }
    g.q = (size_t)(l->q.q - base);
    g.d = (size_t)((const unsigned char *)l->q.d - base);
    g.bits = l->q.bits;
    g.group = l->q.group;
    const bool row = (g.group == n_in);
    if      (g.bits == 8 && row)             g.kernel = K_Q8_ROW;
    else if (g.bits == 8 && g.group == 32)   g.kernel = K_Q8_G32;
    else if (g.bits == 4 && row)             g.kernel = K_Q4_ROW;
    else if (g.bits == 4 && g.group == 32)   g.kernel = K_Q4_G32;
    else if (g.bits == 4 && g.group == 64)   g.kernel = K_Q4_G64;
    else die("no GPU kernel for %d-bit blocks of %d (there are: q8 row/g32, q4 row/g32/g64)",
             g.bits, g.group);
    return g;
}

GpuModel *gpu_model_create(MetalContext *ctx, const SafeTensors *st,
                           const Qwen3Config *cfg, int max_seq, int max_rows) {
    /* Every weight is an offset into ONE buffer over the mapping, and Metal
     * caps a buffer at maxBufferLength (41.75 GB here). A sharded checkpoint
     * is quantised into one file first (tools/quant/quantize.py). */
    if (st->n_maps > 1)
        die("a sharded checkpoint cannot bind as one GPU buffer; quantise it into one file");
    GpuModel *g = calloc(1, sizeof *g);
    if (!g) die("out of memory for the GPU model");
    g->ctx = ctx; g->cfg = *cfg;
    g->max_seq = max_seq; g->max_rows = max_rows;
    g->matmul_kernel = K_MATMUL_TENSOR;
    g->small_tile = true;

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

    g->final_norm_off = offset_of(st, "model.norm.weight");

    Weights w;
    weights_bind(st, cfg, &w);
    const int Hd = cfg->hidden_size, Id = cfg->intermediate_size;
    const int qd = cfg->num_attention_heads * cfg->head_dim;
    g->embed = to_glinear(st, &w.embed, Hd);
    g->head  = to_glinear(st, &w.head, Hd);
    const size_t E = (size_t)cfg->num_experts;
    uint64_t *offs = NULL;
    if (E) {
        g->moe_offs = (__bridge_retained void *)[dev newBufferWithLength:
            (size_t)cfg->num_hidden_layers * 3 * E * 2 * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        offs = ((__bridge id<MTLBuffer>)g->moe_offs).contents;
    }
    g->layer = calloc((size_t)cfg->num_hidden_layers, sizeof *g->layer);
    if (!g->layer) die("out of memory for %d layer offsets", cfg->num_hidden_layers);

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        LayerOffsets *o = &g->layer[l];
        o->input_ln     = layer_offset(st, "model.layers.%d.input_layernorm.weight", l);
        o->q_norm       = layer_offset(st, "model.layers.%d.self_attn.q_norm.weight", l);
        o->k_norm       = layer_offset(st, "model.layers.%d.self_attn.k_norm.weight", l);
        o->post_attn_ln = layer_offset(st, "model.layers.%d.post_attention_layernorm.weight", l);
        const LayerWeights *L = &w.layers[l];
        o->q_proj    = to_glinear(st, &L->q_proj, Hd);
        o->k_proj    = to_glinear(st, &L->k_proj, Hd);
        o->v_proj    = to_glinear(st, &L->v_proj, Hd);
        o->o_proj    = to_glinear(st, &L->o_proj, qd);
        if (L->experts) {
            /* The grouped kernels read experts by offset: bf16, or q8_row. A
             * layer's experts must share one format, since one dispatch runs
             * them all; anything else is refused rather than misread. */
            o->moe = true;
            o->router = to_glinear(st, &L->router, Hd);
            if (o->router.kernel >= 0) die("layer %d: a quantised router is not on the GPU yet", l);
            for (size_t e = 0; e < E; e++) {
                const Linear *pr[3] = {&L->experts[e].gate_proj, &L->experts[e].up_proj,
                                       &L->experts[e].down_proj};
                for (size_t p = 0; p < 3; p++) {
                    GLinear gl = to_glinear(st, pr[p], p == 2 ? cfg->moe_intermediate_size : Hd);
                    const bool q8 = gl.kernel == K_Q8_ROW;
                    if (gl.kernel >= 0 && !q8)
                        die("layer %d: only bf16 and q8_row experts run on the GPU", l);
                    if (e == 0 && p == 0) o->moe_q8 = q8;
                    if (q8 != o->moe_q8) die("layer %d mixes expert formats", l);
                    const size_t at = (((size_t)l * 3 + p) * E + e) * 2;
                    offs[at]     = q8 ? gl.q : gl.w / 2;
                    offs[at + 1] = q8 ? gl.d / 2 : 0;
                }
            }
            if (o->k_proj.bits != o->q_proj.bits || o->v_proj.bits != o->q_proj.bits)
                die("layer %d mixes bit widths among projections sharing an input", l);
            continue;
        }
        o->gate_proj = to_glinear(st, &L->gate_proj, Hd);
        o->up_proj   = to_glinear(st, &L->up_proj, Hd);
        o->down_proj = to_glinear(st, &L->down_proj, Id);
        /* Projections that read the same input share one narrowing pass, so
         * they must agree on whether they need it. */
        if (o->k_proj.bits != o->q_proj.bits || o->v_proj.bits != o->q_proj.bits ||
            o->up_proj.bits != o->gate_proj.bits)
            die("layer %d mixes bit widths among projections sharing an input", l);
    }
    weights_free(&w);

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
    /* Widest matmul input is I (down_proj); bf16 needs half of a float each,
     * and allocating in floats keeps the macro. */
    ALLOC(act16, S * (I > q_dim ? I : q_dim));
    if (E) {
        const size_t R = MOE_MAX_ROWS, P = R * (size_t)cfg->num_experts_per_tok;
        const size_t Ie = (size_t)cfg->moe_intermediate_size;
        ALLOC(moe_logits, R * E); ALLOC(moe_idx, P); ALLOC(moe_wt, P); ALLOC(moe_rop, P);
        ALLOC(moe_tor, P); ALLOC(moe_groups, 3 * E); ALLOC(moe_ng, 1);
        ALLOC(moe_a, P * H); ALLOC(moe_g, P * Ie); ALLOC(moe_u, P * Ie); ALLOC(moe_d, P * H);
    }
#undef ALLOC
    g->tokens = (__bridge_retained void *)
        [dev newBufferWithLength:S * sizeof(int) options:MTLResourceStorageModeShared];
    return g;
}

void gpu_model_free(GpuModel *g) {
    if (!g) return;
    void *bufs[] = {g->weights, g->x, g->xb, g->q, g->attout, g->hb, g->hb2,
                    g->kcache, g->vcache, g->logits, g->tokens, g->act16, g->moe_offs,
                    g->moe_logits, g->moe_idx, g->moe_wt, g->moe_rop, g->moe_tor, g->moe_groups,
                    g->moe_ng, g->moe_a, g->moe_g, g->moe_u, g->moe_d};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i]) CFRelease(bufs[i]);
    free(g->layer);
    free(g);
}

void gpu_set_matmul_kernel(GpuModel *g, int which) { g->matmul_kernel = which; }
void gpu_set_small_tile(GpuModel *g, bool on) { g->small_tile = on; }

/* ------------------------------------------------------------- encoding */

struct MatmulDimsC { uint32_t M, N, K; };
struct NormDimsC   { uint32_t rows, n; float eps; };
struct RopeDimsC   { uint32_t n, heads, head_dim, dim, pos; float theta, eps; };
struct AttnDimsC   { uint32_t n, heads, kv_heads, head_dim, q_dim, kv_dim, pos, max_seq; float scale; };
struct ElemDimsC   { uint32_t count; };

static void encode_matmul(GpuModel *g, id<MTLComputeCommandEncoder> enc,
                          void *cbuf, size_t coff, void *abuf, size_t aoff,
                          size_t woff, int M, int N, int K) {
    /* Decode and short verifies (M <= 8) take an 8-row tile. The 32x32 op
     * computes its whole tile however few rows exist; step 4.1 measured the
     * 8x32 one bit-identical and 1.6x faster at M=1, 2.2x at M=8. At M=32 the
     * kernel sweep called it level but the whole pass lost 2% (four times the
     * tiles to fan out), so 32 rows and prefill keep the 32-row tile. */
    const bool tile8 = g->matmul_kernel == K_MATMUL_TENSOR && g->small_tile && M <= 8;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
        g->pso[tile8 ? K_MATMUL_TENSOR_8X32 : g->matmul_kernel]];
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
        const int tile_m = tile8 ? 8 : 32;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 31) / 32),
                                              (NSUInteger)((M + tile_m - 1) / tile_m), 1)
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

/* out = W x for one projection, bf16 or quantised. */
static void encode_linear(GpuModel *g, id<MTLComputeCommandEncoder> enc,
                          void *cbuf, size_t coff, void *abuf, size_t aoff,
                          const GLinear *l, int M, int N, int K) {
    if (l->kernel < 0) { encode_matmul(g, enc, cbuf, coff, abuf, aoff, l->w, M, N, K); return; }
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[l->kernel]];
    [enc setBuffer:(__bridge id<MTLBuffer>)cbuf offset:coff atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)abuf offset:aoff atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:l->q atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:l->d atIndex:3];
    struct { uint32_t M, N, K, G; } d = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)l->group};
    [enc setBytes:&d length:sizeof d atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 31) / 32), (NSUInteger)((M + 31) / 32), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void encode_elem(GpuModel *g, id<MTLComputeCommandEncoder> enc, int kernel,
                        void *b0, size_t o0, void *b1, size_t o1, size_t count);

/* Where a projection reads its input from. Q4 kernels take bf16, so the fp32
 * input is narrowed into act16 first; everything else reads it in place. */
static void input_for(GpuModel *g, id<MTLComputeCommandEncoder> enc, const GLinear *l,
                      void *src, size_t off, size_t count, void **buf, size_t *boff) {
    if (l->bits != 4) { *buf = src; *boff = off; return; }
    encode_elem(g, enc, K_NARROW, g->act16, 0, src, off, count);
    *buf = g->act16; *boff = 0;
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

/* ------------------------------------------------------------------ MoE
 * Route, group, gather, grouped SwiGLU, scatter: nine dispatches per layer
 * however many experts the tokens pick. kernels.metal says why. */
struct MoeDimsC { uint32_t n, experts, k, hidden, inter, norm; };

static void pso_set(GpuModel *g, id<MTLComputeCommandEncoder> enc, int kernel, void **bufs, int nb) {
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[kernel]];
    for (int i = 0; i < nb; i++) [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:(NSUInteger)i];
}

/* One projection for every expert group at once: grid column x is a 32-wide
 * tile of the output, row y one of a group's 8-row tiles. Each group gets as
 * many row tiles as n tokens could fill, so a decode step dispatches no empty
 * ones. */
static void encode_moe_matmul(GpuModel *g, id<MTLComputeCommandEncoder> enc, void *C, void *A,
                              int layer, int proj, int n, int N, int K) {
    const int E = g->cfg.num_experts, pairs = n * g->cfg.num_experts_per_tok;
    const bool q8 = g->layer[layer].moe_q8;
    void *bufs[] = {C, A, g->weights, g->moe_offs, g->moe_groups, g->moe_ng};
    pso_set(g, enc, q8 ? K_MOE_Q8 : K_MOE_MM, bufs, 6);
    if (q8) [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:0 atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)g->moe_offs
            offset:((NSUInteger)layer * 3 + (NSUInteger)proj) * (NSUInteger)E * 2 * sizeof(uint64_t) atIndex:3];
    /* M carries the row tiles per group, which the kernel strides the grid by. */
    struct MatmulDimsC d = {(uint32_t)((n + 7) / 8), (uint32_t)N, (uint32_t)K};
    [enc setBytes:&d length:sizeof d atIndex:6];
    const NSUInteger groups = (NSUInteger)(pairs < E ? pairs : E);
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 31) / 32), groups * (NSUInteger)d.M, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void encode_moe(GpuModel *g, id<MTLComputeCommandEncoder> enc, const LayerOffsets *o,
                       int layer, int n, bool run_router) {
    const Qwen3Config *c = &g->cfg;
    const int H = c->hidden_size, E = c->num_experts, Ie = c->moe_intermediate_size;
    const NSUInteger P = (NSUInteger)(n * c->num_experts_per_tok);
    struct MoeDimsC md = {(uint32_t)n, (uint32_t)E, (uint32_t)c->num_experts_per_tok,
                          (uint32_t)H, (uint32_t)Ie, c->norm_topk_prob ? 1u : 0u};

    if (run_router) encode_linear(g, enc, g->moe_logits, 0, g->xb, 0, &o->router, n, E, H);
    void *route[] = {g->moe_logits, g->moe_idx, g->moe_wt};
    pso_set(g, enc, K_MOE_ROUTE, route, 3);
    [enc setBytes:&md length:sizeof md atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    void *group[] = {g->moe_idx, g->moe_rop, g->moe_tor, g->moe_groups, g->moe_ng};
    pso_set(g, enc, K_MOE_GROUP, group, 5);
    [enc setBytes:&md length:sizeof md atIndex:5];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    void *gather[] = {g->moe_a, g->xb, g->moe_tor};
    pso_set(g, enc, K_MOE_GATHER, gather, 3);
    [enc setBytes:&md length:sizeof md atIndex:3];
    [enc dispatchThreads:MTLSizeMake(P * (NSUInteger)H, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    encode_moe_matmul(g, enc, g->moe_g, g->moe_a, layer, 0, n, Ie, H);
    encode_moe_matmul(g, enc, g->moe_u, g->moe_a, layer, 1, n, Ie, H);
    encode_elem(g, enc, K_SWIGLU, g->moe_g, 0, g->moe_u, 0, P * (NSUInteger)Ie);
    encode_moe_matmul(g, enc, g->moe_d, g->moe_g, layer, 2, n, H, Ie);

    void *scatter[] = {g->x, g->moe_d, g->moe_rop, g->moe_wt};
    pso_set(g, enc, K_MOE_SCATTER, scatter, 4);
    [enc setBytes:&md length:sizeof md atIndex:4];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n * (NSUInteger)H, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

/* Phase 4 measurement: ONE MoE block, routing forced. The router matmul is
 * skipped and its logits written here, so that token t picks exactly
 * experts[t*k .. t*k+k-1], most probable first; xb holds fixed pseudo-random
 * activations. Returns GPU seconds. *groups_out is the number of expert groups
 * the GPU actually formed, so the caller can check that the routing it claims
 * to have forced is the routing that ran. */
double gpu_moe_bench(GpuModel *g, int layer, int n, const int *experts, int *groups_out) {
    const Qwen3Config *c = &g->cfg;
    if (layer < 0 || layer >= c->num_hidden_layers || !g->layer[layer].moe)
        die("layer %d is not a MoE layer", layer);
    if (n < 1 || n > MOE_MAX_ROWS) die("a MoE block takes 1..%d tokens, not %d", MOE_MAX_ROWS, n);
    const int E = c->num_experts, k = c->num_experts_per_tok, H = c->hidden_size;

    float *lg = ((__bridge id<MTLBuffer>)g->moe_logits).contents;
    float *xb = ((__bridge id<MTLBuffer>)g->xb).contents;
    for (int t = 0; t < n; t++) {
        for (int e = 0; e < E; e++) lg[t * E + e] = 0.0f;
        for (int j = 0; j < k; j++) lg[t * E + experts[t * k + j]] = 10.0f - 0.5f * (float)j;
        for (int i = 0; i < H; i++) xb[t * H + i] = sinf((float)(t * H + i) * 0.37f);
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue(g->ctx);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode_moe(g, enc, &g->layer[layer], layer, n, false);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) die("MoE bench failed: %s", cb.error.localizedDescription.UTF8String);
    if (groups_out) *groups_out = (int)*(uint32_t *)((__bridge id<MTLBuffer>)g->moe_ng).contents;
    return cb.GPUEndTime - cb.GPUStartTime;
}

/* ------------------------------------------------------------ profiling
 * Optional. With a Prof attached, the pass is cut into one compute encoder
 * per op group and the GPU stamps the start and end of each (Apple GPUs only
 * sample at encoder boundaries, so the boundary is where the cut has to be).
 * Cutting is not free, so the caller reports the profiled pass's total next to
 * an uncut one: a profile that changes what it measures must say by how much. */
static const char *prof_names[GPU_PROF_CLASSES] = {
    "embed", "rmsnorm", "attn proj (q k v o)", "qk-norm + rope", "attention",
    "mlp proj (gate up down)", "elementwise (swiglu, residual, narrow)", "lm head",
};
const char *gpu_prof_name(int cls) { return prof_names[cls]; }

enum { PROF_MAX_ENCODERS = 1024 };
typedef struct {
    id<MTLCounterSampleBuffer> sb;
    int n, cls[PROF_MAX_ENCODERS];
} Prof;

/* End the current encoder and open the next one, stamped, for op group cls. */
static id<MTLComputeCommandEncoder> prof_next(Prof *p, id<MTLCommandBuffer> cb,
                                              id<MTLComputeCommandEncoder> enc, int cls) {
    if (!p) return enc;
    if (enc) [enc endEncoding];
    if (p->n == PROF_MAX_ENCODERS) die("profile needs more than %d encoders", PROF_MAX_ENCODERS);
    MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = p->sb;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = (NSUInteger)(2 * p->n);
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex   = (NSUInteger)(2 * p->n + 1);
    p->cls[p->n++] = cls;
    return [cb computeCommandEncoderWithDescriptor:pd];
}

static double gpu_pass(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                      float *logits_out, Prof *prof);

double gpu_forward(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                   float *logits_out) {
    if (!g->cfg.num_experts || n <= MOE_MAX_ROWS)
        return gpu_pass(g, tokens, n, pos, logits_from, logits_out, NULL);

    /* A MoE model takes longer inputs in chunks of MOE_MAX_ROWS. The cache
     * makes that exact: a chunk at position p reads what earlier chunks wrote,
     * the same arithmetic as one pass. Each chunk returns only the logit rows
     * the caller asked for, landing where one pass would have put them. */
    const size_t V = (size_t)g->cfg.vocab_size;
    double t = 0;
    for (int done = 0; done < n; done += MOE_MAX_ROWS) {
        const int c = n - done < MOE_MAX_ROWS ? n - done : MOE_MAX_ROWS;
        if (done + c <= logits_from) {
            t += gpu_pass(g, tokens + done, c, pos + done, c - 1, NULL, NULL);
            continue;
        }
        const int from = logits_from > done ? logits_from - done : 0;
        t += gpu_pass(g, tokens + done, c, pos + done, from,
                      logits_out ? logits_out + (size_t)(done + from - logits_from) * V : NULL, NULL);
    }
    return t;
}

/* One pass with per-group GPU time added into seconds[GPU_PROF_CLASSES].
 * Returns the command buffer's own GPU time for the (cut) pass. */
double gpu_forward_profile(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                           double *seconds) {
    if (g->cfg.num_experts && n > MOE_MAX_ROWS) die("profile a MoE pass at most %d tokens", MOE_MAX_ROWS);
    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device(g->ctx);
    if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
        die("this GPU cannot timestamp encoder boundaries");
    id<MTLCounterSet> ts_set = nil;
    for (id<MTLCounterSet> cs in dev.counterSets)
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts_set = cs;
    if (!ts_set) die("no timestamp counter set on this GPU");

    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts_set;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = 2 * PROF_MAX_ENCODERS;
    NSError *err = nil;
    Prof p = {.sb = [dev newCounterSampleBufferWithDescriptor:d error:&err]};
    if (!p.sb) die("cannot create a counter sample buffer: %s", err.localizedDescription.UTF8String);

    /* GPU timestamps are in GPU ticks. A (CPU, GPU) pair taken before and
     * after the pass gives the rate. The CPU side is ALREADY nanoseconds, not
     * mach ticks: read as ticks (x 125/3 on this machine) the groups summed to
     * 39x the pass's own GPU time, and bench_profile now checks the sum. */
    MTLTimestamp c0, g0, c1, g1;
    [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
    double t = gpu_pass(g, tokens, n, pos, logits_from, NULL, &p);
    [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
    const double s_per_gpu_tick = (double)(c1 - c0) / (double)(g1 - g0) / 1e9;

    NSData *raw = [p.sb resolveCounterRange:NSMakeRange(0, (NSUInteger)(2 * p.n))];
    const MTLCounterResultTimestamp *r = raw.bytes;
    for (int i = 0; i < p.n; i++) {
        if (r[2 * i].timestamp == MTLCounterErrorValue || r[2 * i + 1].timestamp == MTLCounterErrorValue)
            die("encoder %d has no timestamp", i);
        seconds[p.cls[i]] += (double)(r[2 * i + 1].timestamp - r[2 * i].timestamp) * s_per_gpu_tick;
    }
    return t;
}

static double gpu_pass(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                      float *logits_out, Prof *prof) {
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
    id<MTLComputeCommandEncoder> enc = prof ? nil : [cb computeCommandEncoder];

    /* embeddings */
    enc = prof_next(prof, cb, enc, 0);
    if (g->embed.kernel < 0) {
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_EMBED]];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:g->embed.w atIndex:1];
        [enc setBuffer:tokbuf offset:0 atIndex:2];
        struct { uint32_t n, hidden; } ed = {(uint32_t)n, (uint32_t)H};
        [enc setBytes:&ed length:sizeof ed atIndex:3];
    } else {
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pso[K_EMBED_Q]];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:g->embed.q atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)g->weights offset:g->embed.d atIndex:2];
        [enc setBuffer:tokbuf offset:0 atIndex:3];
        struct { uint32_t n, hidden, bits, group; } ed = {(uint32_t)n, (uint32_t)H,
                                                         (uint32_t)g->embed.bits, (uint32_t)g->embed.group};
        [enc setBytes:&ed length:sizeof ed atIndex:4];
    }
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(n * H), 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        const LayerOffsets *o = &g->layer[l];
        const size_t layer_kv = (size_t)l * (size_t)g->max_seq * (size_t)kv_dim * sizeof(float);
        const size_t row_kv   = layer_kv + (size_t)pos * (size_t)kv_dim * sizeof(float);

        void *in; size_t in_off;
        enc = prof_next(prof, cb, enc, 1);
        encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, o->input_ln, n, H, eps);
        enc = prof_next(prof, cb, enc, 2);
        input_for(g, enc, &o->q_proj, g->xb, 0, (size_t)n * (size_t)H, &in, &in_off);
        encode_linear(g, enc, g->q, 0, in, in_off, &o->q_proj, n, q_dim, H);
        encode_linear(g, enc, g->kcache, row_kv, in, in_off, &o->k_proj, n, kv_dim, H);
        encode_linear(g, enc, g->vcache, row_kv, in, in_off, &o->v_proj, n, kv_dim, H);

        /* QK-norm then RoPE, on Q and on the cache rows just written. */
        enc = prof_next(prof, cb, enc, 3);
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

        enc = prof_next(prof, cb, enc, 4);
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

        enc = prof_next(prof, cb, enc, 2);
        input_for(g, enc, &o->o_proj, g->attout, 0, (size_t)n * (size_t)q_dim, &in, &in_off);
        encode_linear(g, enc, g->xb, 0, in, in_off, &o->o_proj, n, H, q_dim);
        enc = prof_next(prof, cb, enc, 6);
        encode_elem(g, enc, K_RESIDUAL, g->x, 0, g->xb, 0, (size_t)n * (size_t)H);

        enc = prof_next(prof, cb, enc, 1);
        encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, o->post_attn_ln, n, H, eps);
        enc = prof_next(prof, cb, enc, 5);
        if (o->moe) { encode_moe(g, enc, o, l, n, true); continue; }
        input_for(g, enc, &o->gate_proj, g->xb, 0, (size_t)n * (size_t)H, &in, &in_off);
        encode_linear(g, enc, g->hb, 0, in, in_off, &o->gate_proj, n, I, H);
        encode_linear(g, enc, g->hb2, 0, in, in_off, &o->up_proj, n, I, H);
        enc = prof_next(prof, cb, enc, 6);
        encode_elem(g, enc, K_SWIGLU, g->hb, 0, g->hb2, 0, (size_t)n * (size_t)I);
        enc = prof_next(prof, cb, enc, 5);
        input_for(g, enc, &o->down_proj, g->hb, 0, (size_t)n * (size_t)I, &in, &in_off);
        encode_linear(g, enc, g->xb, 0, in, in_off, &o->down_proj, n, H, I);
        enc = prof_next(prof, cb, enc, 6);
        encode_elem(g, enc, K_RESIDUAL, g->x, 0, g->xb, 0, (size_t)n * (size_t)H);
    }

    enc = prof_next(prof, cb, enc, 1);
    encode_rmsnorm(g, enc, g->xb, 0, g->x, 0, g->final_norm_off, n, H, eps);
    enc = prof_next(prof, cb, enc, 7);
    void *hin; size_t hin_off;
    input_for(g, enc, &g->head, g->xb, (size_t)logits_from * (size_t)H * sizeof(float),
              (size_t)(n - logits_from) * (size_t)H, &hin, &hin_off);
    encode_linear(g, enc, g->logits, 0, hin, hin_off, &g->head, n - logits_from,
                  cfg->vocab_size, H);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) die("GPU forward failed: %s", cb.error.localizedDescription.UTF8String);

    if (logits_out) {
        id<MTLBuffer> lb = (__bridge id<MTLBuffer>)g->logits;
        memcpy(logits_out, lb.contents,
               (size_t)(n - logits_from) * (size_t)cfg->vocab_size * sizeof(float));
    }
    /* GPU time for the whole pass, from the command buffer's own clock. */
    return cb.GPUEndTime - cb.GPUStartTime;
}
