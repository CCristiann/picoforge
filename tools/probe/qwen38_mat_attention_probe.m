/* Prefill-gap probe: the materialized attention kernels (q38d_mat_*) against an FP64 CPU
 * reference on random data with the model's shapes (h 24, j 2, d 256, ratio 4, budget 2048).
 * Build: clang -fobjc-arc -Isrc -framework Metal -framework Foundation tools/probe/qwen38_mat_attention_probe.m -o build/pg/matprobe
 * Run:   build/pg/matprobe picoforge.metallib POS0 TOKENS   (prints max scaled error and GPU ms) */
#import <Metal/Metal.h>
#include "qwen38_attention_internal.h"
#include <math.h>
#include <stdlib.h>
typedef struct { uint32_t pos0, step, x_stride, y_stride, ratio, budget, scores_stride, ids_stride, first, count; } Batch;
int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 4) { fprintf(stderr, "usage: matprobe LIB POS0 TOKENS\n"); return 1; }
    const uint32_t pos0 = (uint32_t)atoi(argv[2]), T = (uint32_t)atoi(argv[3]);
    const uint32_t H = 24, J = 2, D = 256, R = 4, BUDGET = 2048, STRIDE = BUDGET + R - 1, SP = (STRIDE + 63) / 64 * 64;
    const uint32_t N = pos0 + T, G = H / J;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
    NSString *names[3] = {@"q38d_mat_scores_gpu", @"q38d_mat_softmax_gpu", @"q38d_mat_pv_gpu"};
    id<MTLComputePipelineState> ps[3];
    for (int i = 0; i < 3; i++) { ps[i] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&e]; if (!ps[i]) { printf("pipeline %d\n", i); return 2; } }
    #define BUF(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    id<MTLBuffer> bq = BUF((size_t)T*H*D*4), bg = BUF((size_t)T*H*D*4), bk = BUF((size_t)N*J*D*4), bv = BUF((size_t)N*J*D*4),
                  bi = BUF((size_t)T*STRIDE*4), bo = BUF((size_t)T*H*D*4), be = BUF(16), bp = BUF(((size_t)T*J*16*SP + (size_t)T*J*16)*4);
    float *q = bq.contents, *g = bg.contents, *k = bk.contents, *v = bv.contents, *o = bo.contents; uint32_t *ids = bi.contents;
    srand48(11);
    for (size_t i = 0; i < (size_t)T*H*D; i++) { q[i] = (float)(drand48()*2-1); g[i] = (float)drand48(); }
    for (size_t i = 0; i < (size_t)N*J*D; i++) { k[i] = (float)(drand48()*2-1); v[i] = (float)(drand48()*2-1); }
    uint32_t *m = calloc(T, 4);
    for (uint32_t z = 0; z < T; z++) {           /* random block selection, sorted, then the causal tail */
        const uint32_t vis = pos0 + z + 1, nb = vis / R, sel = nb < BUDGET / R ? nb : BUDGET / R;
        uint32_t c = 0, need = sel;
        for (uint32_t b = 0; b < nb && need; b++) if (drand48() < (double)need / (nb - b)) { for (uint32_t r = 0; r < R; r++) ids[(size_t)z*STRIDE + c++] = b*R + r; need--; }
        for (uint32_t t = nb*R; t < vis; t++) ids[(size_t)z*STRIDE + c++] = t;
        m[z] = c;
    }
    Q38DGpuArgs a = {.h = H, .j = J, .d = D}; Batch bt = {pos0, 1, H*D, H*D, R, BUDGET, 0, STRIDE, 0, T};
    id<MTLCommandQueue> qu = [dev newCommandQueue]; double best = 1e9;
    for (int rep = 0; rep < 4; rep++) {
        *(uint32_t *)be.contents = 0;
        id<MTLCommandBuffer> cb = [qu commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        id<MTLBuffer> bufs[6] = {bq, bg, bk, bv, bi, bo};
        MTLSize grids[3] = {MTLSizeMake(SP/64, J, T), MTLSizeMake(16, J, T), MTLSizeMake(D/64, J, T)};
        for (int s = 0; s < 3; s++) {
            [en setComputePipelineState:ps[s]];
            for (int i = 0; i < 6; i++) [en setBuffer:bufs[i] offset:0 atIndex:i];
            [en setBytes:&a length:sizeof a atIndex:6]; [en setBuffer:be offset:0 atIndex:7];
            [en setBuffer:bp offset:0 atIndex:8]; [en setBytes:&bt length:sizeof bt atIndex:9];
            [en dispatchThreadgroups:grids[s] threadsPerThreadgroup:MTLSizeMake(s == 1 ? 256 : 128, 1, 1)];
            [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (rep) best = fmin(best, cb.GPUEndTime - cb.GPUStartTime);
    }
    double worst = 0, scale = 0;
    double *sc = malloc(STRIDE * sizeof(double));
    for (uint32_t z = 0; z < T; z++) for (uint32_t h = 0; h < H; h++) {
        const uint32_t kv = h / G; const float *qq = q + ((size_t)z*H + h)*D; double mx = -INFINITY, l = 0;
        for (uint32_t i = 0; i < m[z]; i++) { const float *kk = k + ((size_t)ids[(size_t)z*STRIDE+i]*J + kv)*D; double s = 0;
            for (uint32_t f = 0; f < D; f++) s += (double)qq[f]*kk[f]; sc[i] = s / sqrt((double)D); if (sc[i] > mx) mx = sc[i]; }
        for (uint32_t i = 0; i < m[z]; i++) { sc[i] = exp(sc[i] - mx); l += sc[i]; }
        for (uint32_t f = 0; f < D; f++) { double acc = 0;
            for (uint32_t i = 0; i < m[z]; i++) acc += sc[i] * v[((size_t)ids[(size_t)z*STRIDE+i]*J + kv)*D + f];
            const double y = acc / l * g[((size_t)z*H + h)*D + f], d = fabs(y - o[((size_t)z*H + h)*D + f]);
            if (d > worst) worst = d; if (fabs(y) > scale) scale = fabs(y); }
    }
    for (int s = 0; s < 3; s++) {   /* each pass alone, after the full run filled its inputs */
        double b1 = 1e9;
        for (int rep = 0; rep < 4; rep++) {
            id<MTLCommandBuffer> cb = [qu commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            id<MTLBuffer> bufs[6] = {bq, bg, bk, bv, bi, bo};
            MTLSize grids[3] = {MTLSizeMake(SP/64, J, T), MTLSizeMake(16, J, T), MTLSizeMake(D/64, J, T)};
            [en setComputePipelineState:ps[s]];
            for (int i = 0; i < 6; i++) [en setBuffer:bufs[i] offset:0 atIndex:i];
            [en setBytes:&a length:sizeof a atIndex:6]; [en setBuffer:be offset:0 atIndex:7];
            [en setBuffer:bp offset:0 atIndex:8]; [en setBytes:&bt length:sizeof bt atIndex:9];
            [en dispatchThreadgroups:grids[s] threadsPerThreadgroup:MTLSizeMake(s == 1 ? 256 : 128, 1, 1)];
            [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (rep) b1 = fmin(b1, cb.GPUEndTime - cb.GPUStartTime);
        }
        printf("  pass %d: %.3f ms\n", s + 1, b1 * 1e3);
    }
    printf("pos0 %u tokens %u error word %u max abs %.3e scaled %.3e GPU %.3f ms\n", pos0, T, *(uint32_t *)be.contents, worst, worst/scale, best*1e3);
    return worst/scale < 1e-5 && !*(uint32_t *)be.contents ? 0 : 3;
}}
