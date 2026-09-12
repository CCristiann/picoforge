/* metal.m — the only Objective-C in the engine, and only because Metal is.
 *
 * Everything here is behind a C interface so the rest of the codebase never
 * sees an object pointer. What lives on this side: the device, the command
 * queue, the compiled kernel library, and the timing.
 *
 * Buffers are created with newBufferWithBytesNoCopy where the caller's memory
 * is page-aligned. On unified memory that is not an optimisation, it is the
 * point: the GPU addresses the same physical pages the CPU just wrote, and
 * the mmapped checkpoint can be handed to a kernel without a second copy of
 * the model existing anywhere.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "picoforge.h"

struct MetalContext {
    void *device;        /* id<MTLDevice>       */
    void *queue;         /* id<MTLCommandQueue> */
    void *pipelines[3];  /* id<MTLComputePipelineState>, one per kernel */
};

static const char *kernel_names[3] = {
    "matmul_naive", "matmul_simdgroup", "matmul_tensorops",
};

MetalContext *metal_init(const char *metallib_path) {
    MetalContext *ctx = calloc(1, sizeof *ctx);
    if (!ctx) die("out of memory for the Metal context");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("no Metal device");
    if (![dev supportsFamily:MTLGPUFamilyMetal4])
        die("this GPU does not support Metal 4");

    NSError *err = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
    id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
    if (!lib) die("cannot load %s: %s", metallib_path, err.localizedDescription.UTF8String);

    ctx->device = (__bridge_retained void *)dev;
    ctx->queue  = (__bridge_retained void *)[dev newCommandQueue];

    /* Kernels that are not written yet simply stay NULL; metal_matmul says so
     * rather than crashing, which keeps the harness usable while versions two
     * and three are still being built. */
    for (int i = 0; i < 3; i++) {
        id<MTLFunction> fn = [lib newFunctionWithName:
                                 [NSString stringWithUTF8String:kernel_names[i]]];
        if (!fn) continue;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn
                                                                            error:&err];
        if (!pso) die("cannot build pipeline %s: %s", kernel_names[i],
                      err.localizedDescription.UTF8String);
        ctx->pipelines[i] = (__bridge_retained void *)pso;
    }
    return ctx;
}

void metal_shutdown(MetalContext *ctx) {
    if (!ctx) return;
    for (int i = 0; i < 3; i++)
        if (ctx->pipelines[i]) CFRelease(ctx->pipelines[i]);
    if (ctx->queue)  CFRelease(ctx->queue);
    if (ctx->device) CFRelease(ctx->device);
    free(ctx);
}

void metal_info(const MetalContext *ctx) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    printf("\n=== Metal ===\n");
    printf("device                : %s\n", dev.name.UTF8String);
    printf("unified memory        : %s\n", dev.hasUnifiedMemory ? "yes" : "no");
    printf("max working set       : %.1f GB\n",
           (double)dev.recommendedMaxWorkingSetSize / 1e9);
    printf("max threads per group : %lu\n",
           (unsigned long)dev.maxThreadsPerThreadgroup.width);
    printf("kernels available     :");
    for (int i = 0; i < 3; i++)
        if (ctx->pipelines[i]) printf(" %s", kernel_names[i]);
    printf("\n");
}

bool metal_has_kernel(const MetalContext *ctx, int which) {
    return which >= 0 && which < 3 && ctx->pipelines[which] != NULL;
}

/* Run one matmul and return the GPU time in seconds.
 *
 * The timing comes from the command buffer's own GPUStartTime/GPUEndTime,
 * not from a clock around the submission. Wall time around a submit measures
 * queueing, driver overhead and scheduling as much as the kernel; these two
 * timestamps are taken by the GPU itself, on either side of this work. */
double metal_matmul(MetalContext *ctx, int which,
                    float *C, const float *A, const uint16_t *B,
                    int M, int N, int K) {
    if (!metal_has_kernel(ctx, which))
        die("kernel %s is not implemented yet", kernel_names[which]);

    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)ctx->pipelines[which];

    const size_t a_bytes = (size_t)M * (size_t)K * sizeof(float);
    const size_t b_bytes = (size_t)N * (size_t)K * sizeof(uint16_t);
    const size_t c_bytes = (size_t)M * (size_t)N * sizeof(float);

    /* MTLResourceStorageModeShared: one allocation both processors address.
     * On a discrete GPU this would be the slow path; on unified memory it is
     * the only sensible one. */
    id<MTLBuffer> bufA = [dev newBufferWithBytes:A length:a_bytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [dev newBufferWithBytes:B length:b_bytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [dev newBufferWithLength:c_bytes
                                          options:MTLResourceStorageModeShared];

    struct { uint32_t M, N, K; } dims = {(uint32_t)M, (uint32_t)N, (uint32_t)K};

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:1];
    [enc setBuffer:bufB offset:0 atIndex:2];
    [enc setBuffer:bufC offset:0 atIndex:0];
    [enc setBytes:&dims length:sizeof dims atIndex:3];

    /* Each kernel wants its grid shaped differently, and that shape is part
     * of the kernel, so it lives next to the dispatch rather than in the
     * caller. The naive one is one thread per output element; the tiled ones
     * are one threadgroup per 32x32 tile, 128 threads = 4 SIMD groups. */
    if (which == MM_NAIVE) {
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1)
       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    } else {
        NSUInteger gx = (NSUInteger)((N + 31) / 32), gy = (NSUInteger)((M + 31) / 32);
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.error) die("kernel %s failed: %s", kernel_names[which],
                      cb.error.localizedDescription.UTF8String);

    memcpy(C, bufC.contents, c_bytes);
    return cb.GPUEndTime - cb.GPUStartTime;
}
