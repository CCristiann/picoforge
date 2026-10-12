/* GPU bytes must survive closing both C owners before command submission. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gguf.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const bool real = argc == 2 && argv[1][0] != '-';
    const char *mode = argc == 2 && !real ? argv[1] : "";
    if (argc > 2 || (*mode && strcmp(mode, "--foreign") && strcmp(mode, "--oversize-cap") &&
                     strcmp(mode, "--truncated"))) return 2;
    Gguf g = {0};
    GgufTensor fixture[] = {
        {.name="first", .type=0, .rank=1, .shape={page/4}, .file_offset=page+32, .bytes=page},
        {.name="second", .type=0, .rank=1, .shape={page/2}, .file_offset=3*page, .bytes=2*page},
        {.name="per_layer_token_embd.weight", .type=30, .rank=2, .shape={8,page/8}, .file_offset=5*page, .bytes=2*page},
        {.name="third", .type=0, .rank=1, .shape={page/4}, .file_offset=7*page, .bytes=page}
    };
    if (real) gguf_open(argv[1], &g);
    else {
        char path[] = "/tmp/picoforge-weights-XXXXXX";
        g.fd = mkstemp(path);
        assert(g.fd >= 0 && unlink(path) == 0);
        g.file_size = 8*page; g.tensors = fixture; g.tensor_count = 4;
        unsigned char *bytes = malloc(g.file_size);
        assert(bytes);
        for (size_t i = 0; i < g.file_size; i++) bytes[i] = (unsigned char)((i*17 + i/251) & 255);
        assert(write(g.fd, bytes, g.file_size) == (ssize_t)g.file_size);
        free(bytes);
    }
    const size_t initial_maps = gpu_weights_live_mapped_bytes();
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        assert(dev);
        size_t cap = real ? 0 : 2*page;
        if (!strcmp(mode, "--truncated")) assert(ftruncate(g.fd, 1) == 0);
        if (!strcmp(mode, "--oversize-cap")) cap = dev.maxBufferLength + page;
        GpuWeightStore *store = gpu_weights_open((__bridge void *)dev, &g, cap);
        const GgufPlan *plan = gpu_weights_plan(store);
        const size_t count = plan->weight_count, views = plan->view_count;
        if (!real) assert(count == 3 && views == 3);
        assert(!gpu_weights_find(store, "per_layer_token_embd.weight"));
        if (!strcmp(mode, "--foreign")) {
            GgufBinding foreign = plan->weights[0]; foreign.owner = NULL;
            (void)gpu_weights_buffer(store, &foreign);
            return 2;
        }
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void probe(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],"
            "constant ulong *at [[buffer(2)]], constant uint &base [[buffer(3)]],"
            "uint i [[thread_position_in_grid]]) { if(i<3) dst[base+i]=src[at[i]]; }";
        id<MTLLibrary> library = [dev newLibraryWithSource:source options:nil error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pipeline = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
        assert(pipeline);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        assert(command.retainedReferences);
        id<MTLBuffer> output = [dev newBufferWithLength:count*3 options:MTLResourceStorageModeShared];
        assert(output);
        unsigned char *expected = malloc(count*3);
        assert(expected);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:output offset:0 atIndex:1];
        for (size_t i = 0; i < count; i++) {
            const GgufBinding *b = &plan->weights[i];
            assert(b->owner == plan);
            assert(gpu_weights_find(store, b->tensor.name) == b);
            uint64_t at[] = {b->offset, b->offset + b->tensor.bytes/2, b->offset + b->tensor.bytes-1};
            uint32_t base = (uint32_t)(i*3);
            for (size_t j = 0; j < 3; j++) {
                size_t offset = b->tensor.file_offset + (size_t)(at[j] - b->offset);
                assert(pread(g.fd, expected+i*3+j, 1, (off_t)offset) == 1);
            }
            [encoder setBuffer:(__bridge id<MTLBuffer>)gpu_weights_buffer(store, b) offset:0 atIndex:0];
            [encoder setBytes:at length:sizeof at atIndex:2];
            [encoder setBytes:&base length:sizeof base atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(3,1,1)];
        }
        [encoder endEncoding];
        size_t mapped = gpu_weights_live_mapped_bytes() - initial_maps;
        size_t allocated = dev.currentAllocatedSize;
        if (real) gguf_close(&g); else assert(close(g.fd) == 0);
        gpu_weights_close(store);
        assert(gpu_weights_live_mapped_bytes() == initial_maps + mapped);
        [command commit];
        [command waitUntilCompleted];
        assert(command.status == MTLCommandBufferStatusCompleted);
        assert(memcmp(output.contents, expected, count*3) == 0);
        free(expected);
        printf("{\"weights\":%zu,\"views\":%zu,\"byte_probes\":%zu,\"mapped_virtual_bytes\":%zu,"
               "\"metal_allocated_bytes\":%zu,\"byte_exact\":true,\"survived_owner_close\":true}\n",
               count, views, count*3, mapped, allocated);
    }
    assert(gpu_weights_live_mapped_bytes() == initial_maps);
    puts("PASS: no-copy GPU bytes, disk-only exclusion, deferred unmap and complete release");
    return 0;
}
