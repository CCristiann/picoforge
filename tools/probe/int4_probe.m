/* int4_probe.m — run int4_probe.metal and decide the nibble layout.
 *
 * Build and run (from the repo root):
 *   xcrun -sdk macosx metal -std=metal4.0 -c tools/probe/int4_probe.metal -o build/int4_probe.air
 *   xcrun -sdk macosx metallib build/int4_probe.air -o build/int4_probe.metallib
 *   clang -fobjc-arc -framework Metal -framework Foundation tools/probe/int4_probe.m -o build/int4_probe
 *   ./build/int4_probe
 *
 * The 16 packed bytes are 0x01 0x23 ... 0xEF twice over. Every plausible
 * reading of them decodes differently, so exactly one hypothesis can match:
 *   low nibble first  vs  high nibble first
 *   two's complement  vs  offset binary (code - 8)
 * If none matches, the op is doing something this probe did not imagine, and
 * that is a finding too.
 */
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static int run(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name,
               void *a, size_t alen, void *b, size_t blen, float *out) {
    NSError *err = nil;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    if (!pso) { printf("pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); return 0; }
    id<MTLBuffer> C = [dev newBufferWithLength:32 * sizeof(float) options:MTLResourceStorageModeShared];
    memset(C.contents, 0, 32 * sizeof(float));  /* the op ACCUMULATES: C = A*B + C */
    id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:C offset:0 atIndex:0];
    [enc setBytes:a length:alen atIndex:1];
    [enc setBytes:b length:blen atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { printf("%s failed: %s\n", name.UTF8String, cb.error.localizedDescription.UTF8String); return 0; }
    memcpy(out, C.contents, 32 * sizeof(float));
    return 1;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"build/int4_probe.metallib"] error:&err];
        if (!lib) { printf("library: %s\n", err.localizedDescription.UTF8String); return 1; }

        uint16_t a16[32 * 32] = {0};        /* identity in bf16: 1.0 is 0x3F80 */
        float    a32[32 * 32] = {0};
        for (int i = 0; i < 32; i++) { a16[i * 32 + i] = 0x3F80; a32[i * 32 + i] = 1.0f; }

        uint8_t packed[16];
        for (int i = 0; i < 16; i++) packed[i] = (uint8_t)(0x01 + 0x22 * (i % 8));

        float got[32];
        if (!run(dev, lib, @"probe_i4", a16, sizeof a16, packed, sizeof packed, got)) return 1;
        printf("int4 codes as read    :");
        for (int i = 0; i < 32; i++) printf(" %g", got[i]);
        printf("\n");

        const char *names[4] = {"low-first, two's complement", "high-first, two's complement",
                                "low-first, offset binary", "high-first, offset binary"};
        int matches = 0;
        for (int h = 0; h < 4; h++) {
            int ok = 1;
            for (int i = 0; i < 32 && ok; i++) {
                int byte = packed[i / 2];
                int first = (h % 2 == 0) ? 0 : 4;
                int nib = (byte >> ((i % 2 == 0) ? first : 4 - first)) & 0xF;
                int want = (h < 2) ? (nib >= 8 ? nib - 16 : nib) : nib - 8;
                ok = (got[i] == (float)want);
            }
            if (ok) { printf("layout                : %s\n", names[h]); matches++; }
        }
        if (matches != 1) printf("layout                : %d hypotheses match -- UNEXPLAINED\n", matches);

        int8_t codes8[32];
        for (int i = 0; i < 32; i++) codes8[i] = (int8_t)(i * 9 - 128);
        if (!run(dev, lib, @"probe_i8", a32, sizeof a32, codes8, sizeof codes8, got)) return 1;
        int ok8 = 1;
        for (int i = 0; i < 32; i++) ok8 &= (got[i] == (float)codes8[i]);
        printf("int8 codes            : %s\n", ok8 ? "read back exactly, as signed bytes" : "MISMATCH");
        return matches == 1 && ok8 ? 0 : 1;
    }
}
