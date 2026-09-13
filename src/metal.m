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
    void *empty;         /* the do-nothing kernel, for the dispatch floor */
    void *library;       /* id<MTLLibrary>, so pipelines can be built by name */

    /* Pipelines built on demand and kept. Building one costs a compile, and
     * the forward pass asks for the same nine every call. */
    struct { char name[48]; void *pso; } cache[32];
    int cached;
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

    ctx->library = (__bridge_retained void *)lib;
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

    id<MTLFunction> efn = [lib newFunctionWithName:@"empty_kernel"];
    if (efn) {
        id<MTLComputePipelineState> epso =
            [dev newComputePipelineStateWithFunction:efn error:&err];
        if (epso) ctx->empty = (__bridge_retained void *)epso;
    }
    return ctx;
}

void metal_shutdown(MetalContext *ctx) {
    if (!ctx) return;
    for (int i = 0; i < 3; i++)
        if (ctx->pipelines[i]) CFRelease(ctx->pipelines[i]);
    if (ctx->empty) CFRelease(ctx->empty);
    for (int i = 0; i < ctx->cached; i++) CFRelease(ctx->cache[i].pso);
    if (ctx->library) CFRelease(ctx->library);
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

/* A prepared matmul: buffers allocated once for one shape, so a benchmark
 * repeats the kernel and nothing else. Allocating and uploading inside the
 * timed region would measure the driver, and the driver is not the subject. */
struct MetalMatmul {
    MetalContext *ctx;
    void *bufA, *bufB, *bufC;    /* id<MTLBuffer> */
    int M, N, K;
};

MetalMatmul *metal_matmul_prepare(MetalContext *ctx, int M, int N, int K) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    MetalMatmul *mm = calloc(1, sizeof *mm);
    if (!mm) die("out of memory for the matmul buffers");
    mm->ctx = ctx; mm->M = M; mm->N = N; mm->K = K;

    /* MTLResourceStorageModeShared: one allocation both processors address.
     * On a discrete GPU this would be the slow path; on unified memory it is
     * the only sensible one, and it is what lets the mmapped checkpoint reach
     * a kernel without a second copy of the model existing. */
    mm->bufA = (__bridge_retained void *)
        [dev newBufferWithLength:(size_t)M * (size_t)K * sizeof(float)
                         options:MTLResourceStorageModeShared];
    mm->bufB = (__bridge_retained void *)
        [dev newBufferWithLength:(size_t)N * (size_t)K * sizeof(uint16_t)
                         options:MTLResourceStorageModeShared];
    mm->bufC = (__bridge_retained void *)
        [dev newBufferWithLength:(size_t)M * (size_t)N * sizeof(float)
                         options:MTLResourceStorageModeShared];
    return mm;
}

void metal_matmul_upload(MetalMatmul *mm, const float *A, const uint16_t *B) {
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)mm->bufA;
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)mm->bufB;
    memcpy(bufA.contents, A, (size_t)mm->M * (size_t)mm->K * sizeof(float));
    memcpy(bufB.contents, B, (size_t)mm->N * (size_t)mm->K * sizeof(uint16_t));
}

void metal_matmul_download(MetalMatmul *mm, float *C) {
    id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)mm->bufC;
    memcpy(C, bufC.contents, (size_t)mm->M * (size_t)mm->N * sizeof(float));
}

void metal_matmul_fill_c(MetalMatmul *mm, float value) {
    float *c = ((__bridge id<MTLBuffer>)mm->bufC).contents;
    for (size_t i = 0; i < (size_t)mm->M * (size_t)mm->N; i++) c[i] = value;
}

void metal_matmul_free(MetalMatmul *mm) {
    if (!mm) return;
    CFRelease(mm->bufA); CFRelease(mm->bufB); CFRelease(mm->bufC);
    free(mm);
}

/* One run. Returns GPU seconds, taken from the command buffer's own
 * timestamps rather than a clock around the submit: wall time around a
 * submission measures queueing, driver work and scheduling as much as the
 * kernel, and those are not what is being compared. */
double metal_matmul_run(MetalMatmul *mm, int which) {
    return metal_matmul_run_n(mm, which, 1, NULL);
}

/* `count` copies of one dispatch inside ONE command buffer and one encoder,
 * the way the forward pass issues ~420 of them. The GPU time answers "what
 * does one more dispatch cost once the buffer is already paid for"; the wall
 * time (encode + commit + wait, if wall_out is given) adds what the CPU spends
 * building the buffer, which a token also has to wait for.
 *
 * per_thread: `grid` counts threads (dispatchThreads); otherwise it counts
 * threadgroups of `group` threads each (dispatchThreadgroups). */
static double run_grid(MetalMatmul *mm, id<MTLComputePipelineState> pso, const char *name,
                       bool per_thread, MTLSize grid, MTLSize group, int count,
                       double *wall_out) {
    const CFAbsoluteTime w0 = CFAbsoluteTimeGetCurrent();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mm->ctx->queue;
    struct { uint32_t M, N, K; } dims = {(uint32_t)mm->M, (uint32_t)mm->N, (uint32_t)mm->K};

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufC offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufA offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufB offset:0 atIndex:2];
    [enc setBytes:&dims length:sizeof dims atIndex:3];
    for (int i = 0; i < count; i++) {
        if (per_thread) [enc dispatchThreads:grid threadsPerThreadgroup:group];
        else            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.error) die("kernel %s failed: %s", name, cb.error.localizedDescription.UTF8String);
    if (wall_out) *wall_out = CFAbsoluteTimeGetCurrent() - w0;
    return cb.GPUEndTime - cb.GPUStartTime;
}

double metal_matmul_run_n(MetalMatmul *mm, int which, int count, double *wall_out) {
    MetalContext *ctx = mm->ctx;
    if (!metal_has_kernel(ctx, which))
        die("kernel %s is not implemented yet", kernel_names[which]);
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)ctx->pipelines[which];

    /* Each kernel wants its grid shaped differently, and that shape is part
     * of the kernel, so it lives beside the dispatch rather than in the
     * caller. The naive one is one thread per output element; the tiled ones
     * are one threadgroup per 32x32 tile, 128 threads = 4 SIMD groups.
     *
     * The naive threadgroup follows M. A fixed 16x16 is a disaster at M=1:
     * Metal clamps the grid, leaving 16 of 256 threads live. Measuring that
     * fix showed it changes almost nothing, which is itself the finding --
     * at M=1 with N=1024 the output has 1024 elements, so the GPU is short of
     * work, not short of occupancy. */
    if (which == MM_NAIVE) {
        NSUInteger tgh = (NSUInteger)(mm->M < 16 ? mm->M : 16);
        return run_grid(mm, pso, kernel_names[which], true,
                        MTLSizeMake((NSUInteger)mm->N, (NSUInteger)mm->M, 1),
                        MTLSizeMake(256 / tgh, tgh, 1), count, wall_out);
    }
    return run_grid(mm, pso, kernel_names[which], false,
                    MTLSizeMake((NSUInteger)((mm->N + 31) / 32), (NSUInteger)((mm->M + 31) / 32), 1),
                    MTLSizeMake(128, 1, 1), count, wall_out);
}

/* A TensorOps variant by name, with its tile. thread_scope kernels run one op
 * per GPU thread, one thread per tile; the others one 4-SIMD-group threadgroup
 * per tile, like matmul_tensorops. */
double metal_matmul_run_tiled(MetalMatmul *mm, const char *kernel, int tile_m, int tile_n,
                              bool thread_scope, int count, double *wall_out) {
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)metal_pipeline(mm->ctx, kernel);
    const NSUInteger gx = (NSUInteger)((mm->N + tile_n - 1) / tile_n);
    const NSUInteger gy = (NSUInteger)((mm->M + tile_m - 1) / tile_m);
    if (thread_scope)
        return run_grid(mm, pso, kernel, true, MTLSizeMake(gx, gy, 1),
                        MTLSizeMake(gx < 64 ? gx : 64, gy < 8 ? gy : 8, 1), count, wall_out);
    return run_grid(mm, pso, kernel, false, MTLSizeMake(gx, gy, 1),
                    MTLSizeMake(128, 1, 1), count, wall_out);
}

/* Convenience wrapper: prepare, upload, run once, download, free. Used by the
 * correctness check, never by the benchmark. */
double metal_matmul(MetalContext *ctx, int which,
                    float *C, const float *A, const uint16_t *B,
                    int M, int N, int K) {
    MetalMatmul *mm = metal_matmul_prepare(ctx, M, N, K);
    metal_matmul_upload(mm, A, B);
    double t = metal_matmul_run(mm, which);
    metal_matmul_download(mm, C);
    metal_matmul_free(mm);
    return t;
}

/* Seconds for a command buffer that does nothing. Everything a GPU
 * measurement reports sits on top of this. */
double metal_dispatch_floor(MetalContext *ctx) {
    if (!ctx->empty) die("the empty kernel is missing from the library");
    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)ctx->empty;
    id<MTLBuffer> sink = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

    double best = 1e9;
    for (int rep = 0; rep < 50; rep++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:sink offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        double t = cb.GPUEndTime - cb.GPUStartTime;
        if (rep >= 5 && t < best) best = t;    /* best of 45, after warmup */
    }
    return best;
}

/* --------------------------------------------------------- accessors
 * gpu_forward.m needs the raw objects. They leave this file as void* so
 * that nothing else in the engine has to include Metal headers. */
void *metal_device(MetalContext *ctx) { return ctx->device; }
void *metal_queue(MetalContext *ctx)  { return ctx->queue; }

void *metal_pipeline(MetalContext *ctx, const char *name) {
    for (int i = 0; i < ctx->cached; i++)
        if (strcmp(ctx->cache[i].name, name) == 0) return ctx->cache[i].pso;

    if (ctx->cached >= (int)(sizeof ctx->cache / sizeof ctx->cache[0]))
        die("more than %zu pipelines requested", sizeof ctx->cache / sizeof ctx->cache[0]);

    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)ctx->library;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) die("kernel %s is not in the library", name);

    NSError *err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) die("cannot build pipeline %s: %s", name,
                  err.localizedDescription.UTF8String);

    snprintf(ctx->cache[ctx->cached].name, sizeof ctx->cache[0].name, "%s", name);
    ctx->cache[ctx->cached].pso = (__bridge_retained void *)pso;
    return ctx->cache[ctx->cached++].pso;
}

/* ------------------------------------------------------ quantised matmul
 * The same prepare/run/free shape as MetalMatmul, for kernels that take a
 * weight in three parts: codes, bf16 scales, and activations whose width
 * depends on the format (bf16 for Q4, which has no float x int4 overload). */
struct MetalQMatmul {
    MetalContext *ctx;
    void *pso, *bufA, *bufQ, *bufD, *bufC;
    int M, N, K, G;
    size_t a_bytes, q_bytes, d_bytes;
};

MetalQMatmul *metal_qmatmul_prepare(MetalContext *ctx, const char *kernel,
                                    int M, int N, int K, int G, int bits, bool bf16_act) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx->device;
    MetalQMatmul *mm = calloc(1, sizeof *mm);
    if (!mm) die("out of memory for a quantised matmul");
    mm->ctx = ctx; mm->M = M; mm->N = N; mm->K = K; mm->G = G;
    mm->pso = metal_pipeline(ctx, kernel);
    mm->a_bytes = (size_t)M * (size_t)K * (bf16_act ? 2u : 4u);
    mm->q_bytes = (size_t)N * (size_t)K * (size_t)bits / 8u;
    mm->d_bytes = (size_t)N * (size_t)(K / G) * 2u;
#define QALLOC(field, n) mm->field = (__bridge_retained void *) \
    [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    QALLOC(bufA, mm->a_bytes); QALLOC(bufQ, mm->q_bytes); QALLOC(bufD, mm->d_bytes);
    QALLOC(bufC, (size_t)M * (size_t)N * sizeof(float));
#undef QALLOC
    return mm;
}

void metal_qmatmul_upload(MetalQMatmul *mm, const void *A, const uint8_t *Q, const uint16_t *D) {
    memcpy(((__bridge id<MTLBuffer>)mm->bufA).contents, A, mm->a_bytes);
    memcpy(((__bridge id<MTLBuffer>)mm->bufQ).contents, Q, mm->q_bytes);
    memcpy(((__bridge id<MTLBuffer>)mm->bufD).contents, D, mm->d_bytes);
}

double metal_qmatmul_run(MetalQMatmul *mm) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mm->ctx->queue;
    struct { uint32_t M, N, K, G; } dims = {(uint32_t)mm->M, (uint32_t)mm->N,
                                            (uint32_t)mm->K, (uint32_t)mm->G};
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)mm->pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufC offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufA offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufQ offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)mm->bufD offset:0 atIndex:3];
    [enc setBytes:&dims length:sizeof dims atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((mm->N + 31) / 32),
                                          (NSUInteger)((mm->M + 31) / 32), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) die("quantised matmul failed: %s", cb.error.localizedDescription.UTF8String);
    return cb.GPUEndTime - cb.GPUStartTime;
}

void metal_qmatmul_download(MetalQMatmul *mm, float *C) {
    memcpy(C, ((__bridge id<MTLBuffer>)mm->bufC).contents,
           (size_t)mm->M * (size_t)mm->N * sizeof(float));
}

void metal_qmatmul_free(MetalQMatmul *mm) {
    if (!mm) return;
    CFRelease(mm->bufA); CFRelease(mm->bufQ); CFRelease(mm->bufD); CFRelease(mm->bufC);
    free(mm);
}
