/* Compare the Metal attention kernel with an fp64 reference at cache boundaries. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "picoforge.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint32_t n, heads, kv_heads, head_dim, q_dim, kv_dim, pos, max_seq;
    float scale;
} AttnDims;

static uint32_t rng = 20261011;
static float random_value(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 8) / 8388608.0f - 1.0f;
}

static bool check(MetalContext *ctx, int depth, int hd, int rows) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device(ctx);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue(ctx);
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)metal_pipeline(ctx, "attention");
    const int heads = 4, kv_heads = 2, qdim = heads * hd, kvdim = kv_heads * hd;
    const int pos = depth - rows;
    /* Extra future rows must not affect the result. */
    const size_t sizes[] = {(size_t)rows * (size_t)qdim, (size_t)rows * (size_t)qdim,
                           (size_t)(depth + 3) * (size_t)kvdim, (size_t)(depth + 3) * (size_t)kvdim};
    id<MTLBuffer> bufs[4];
    for (int b = 0; b < 4; b++) {
        bufs[b] = [dev newBufferWithLength:sizes[b] * sizeof(float) options:MTLResourceStorageModeShared];
        if (!bufs[b]) die("attention test allocation failed");
        float *p = bufs[b].contents;
        for (size_t i = 0; i < sizes[b]; i++) p[i] = b == 0 ? NAN : random_value();
    }
    float *out = bufs[0].contents, *q = bufs[1].contents;
    float *k = bufs[2].contents, *v = bufs[3].contents;
    const size_t score_bytes = ((size_t)depth * sizeof(float) + 15u) & ~(size_t)15u;
    if (score_bytes + pso.staticThreadgroupMemoryLength > dev.maxThreadgroupMemoryLength)
        die("test context does not fit threadgroup memory");
    AttnDims d = {(uint32_t)rows, (uint32_t)heads, (uint32_t)kv_heads, (uint32_t)hd,
                  (uint32_t)qdim, (uint32_t)kvdim, (uint32_t)pos, (uint32_t)(depth + 3),
                  1.0f / sqrtf((float)hd)};
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (NSUInteger b = 0; b < 4; b++) [enc setBuffer:bufs[b] offset:0 atIndex:b];
    [enc setBytes:&d length:sizeof d atIndex:4];
    [enc setThreadgroupMemoryLength:score_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, (NSUInteger)rows, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) die("attention test: %s", cb.error.localizedDescription.UTF8String);

    double *scores = malloc((size_t)depth * sizeof *scores);
    if (!scores) die("attention reference allocation failed");
    double worst = 0.0;
    for (int t = 0; t < rows; t++) {
        const int len = pos + t + 1;
        for (int h = 0; h < heads; h++) {
            const int kh = h / (heads / kv_heads);
            double mx = -INFINITY, sum = 0.0;
            for (int j = 0; j < len; j++) {
                double dot = 0.0;
                for (int i = 0; i < hd; i++)
                    dot += (double)q[t * qdim + h * hd + i] * k[j * kvdim + kh * hd + i];
                scores[j] = dot * (double)d.scale;
                if (scores[j] > mx) mx = scores[j];
            }
            for (int j = 0; j < len; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
            for (int i = 0; i < hd; i++) {
                double ref = 0.0;
                for (int j = 0; j < len; j++) ref += scores[j] * v[j * kvdim + kh * hd + i];
                ref /= sum;
                double err = fabs((double)out[t * qdim + h * hd + i] - ref);
                if (!isfinite(err)) err = INFINITY;
                if (err > worst) worst = err;
            }
        }
    }
    free(scores);
    /* Absolute error: values lie in [-1,1], and some sums nearly cancel. */
    bool ok = worst < 2e-6;
    printf("attention depth=%d head_dim=%d rows=%d max_abs_error=%.3g %s\n",
           depth, hd, rows, worst, ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    @autoreleasepool {
        MetalContext *ctx = metal_init("picoforge.metallib");
        const int depths[] = {32, 512, 2048, 2049, 4096};
        bool ok = true;
        for (size_t i = 0; i < sizeof depths / sizeof depths[0]; i++)
            for (int hd = 32; hd <= 128; hd *= 4)
                for (int rows = 1; rows <= 3; rows += 2)
                    ok &= check(ctx, depths[i], hd, rows);
        metal_shutdown(ctx);
        return ok ? 0 : 1;
    }
}
