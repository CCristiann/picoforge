// tensorops_gemm_probe.m — R3.5: does the matmul2d tile of the prefill GEMM
// compute what we think, and how close is it to FP64?
//
// Build:  xcrun -sdk macosx metal -std=metal4.0 -c tools/probe/tensorops_gemm_probe.metal -o build/r35/p.air
//         xcrun -sdk macosx metallib build/r35/p.air -o build/r35/p.metallib
//         clang -fobjc-arc -framework Metal -framework Foundation tools/probe/tensorops_gemm_probe.m -o build/r35/probe
// Run:    build/r35/probe build/r35/p.metallib
//
// Shapes are odd on purpose (M, N, K not multiples of the tile) to exercise the
// zero padding. Two references: FP64 on the FP32 inputs (what the FP32 path
// approximates) and FP64 on the inputs rounded to half (what the hardware
// sees): the second gap is accumulation only, the first adds input rounding.
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: probe LIB.metallib\n"); return 1; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
        id<MTLFunction> fn = [lib newFunctionWithName:@"probe_gemm"];
        id<MTLComputePipelineState> ps = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!ps) { fprintf(stderr, "pipeline: %s\n", e.localizedDescription.UTF8String); return 2; }
        const uint32_t M = 100, N = 130, K = 1000; /* K % 32 = 8: the padded tail */
        id<MTLBuffer> bx = [dev newBufferWithLength:M * K * 4 options:0], bw = [dev newBufferWithLength:N * K * 4 options:0],
                      by = [dev newBufferWithLength:M * N * 4 options:0];
        float *x = bx.contents, *w = bw.contents, *y = by.contents;
        srand48(7);
        for (uint32_t i = 0; i < M * K; i++) x[i] = (float)(drand48() * 2 - 1);
        for (uint32_t i = 0; i < N * K; i++) w[i] = (float)((drand48() * 2 - 1) * 0.05);
        uint32_t mnk[4] = {M, N, K, 0};
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:bx offset:0 atIndex:0]; [enc setBuffer:bw offset:0 atIndex:1]; [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:mnk length:16 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 3; }
        double worst_f = 0, worst_h = 0, scale = 0;
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = 0; n < N; n++) {
                double rf = 0, rh = 0;
                for (uint32_t k = 0; k < K; k++) {
                    rf += (double)x[m * K + k] * w[n * K + k];
                    rh += (double)(_Float16)x[m * K + k] * (double)(_Float16)w[n * K + k];
                }
                const double g = y[m * N + n];
                worst_f = fmax(worst_f, fabs(g - rf)); worst_h = fmax(worst_h, fabs(g - rh)); scale = fmax(scale, fabs(rf));
            }
        printf("M=%u N=%u K=%u  max|ref| %.4f\n", M, N, K, scale);
        printf("vs FP64(fp32 inputs):  max abs %.3e  scaled %.3e\n", worst_f, worst_f / scale);
        printf("vs FP64(half inputs):  max abs %.3e  scaled %.3e\n", worst_h, worst_h / scale);
        /* Accumulation is FP32: against the half-input reference the error must
         * be FP32-sized (K * 2^-24 relative is a loose bound); far larger means
         * the layout or the K tail is wrong. */
        const int ok = worst_h / scale < K * 6e-8;
        printf("verdict: %s\n", ok ? "layout and FP32 accumulation OK" : "MISMATCH");
        return ok ? 0 : 4;
    }
}
