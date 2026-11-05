/* metal_device.m — the system Metal device, for tests written in plain C.
 *
 * The engine itself opens its device inside the qwen38_*.m files; a C test
 * that only needs "a device to hand to the weight store" uses this instead of
 * pulling in Objective-C. The library path is accepted and ignored: these
 * tests never launch a kernel through this context. */
#import <Metal/Metal.h>

#include "metal_device.h"
#include "common.h"

#include <stdlib.h>

struct MetalContext { void *device; };

MetalContext *metal_init(const char *metallib_path) {
    (void)metallib_path;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("no Metal device");
    MetalContext *ctx = calloc(1, sizeof *ctx);
    if (!ctx) die("out of memory for the Metal context");
    ctx->device = (__bridge_retained void *)dev;
    return ctx;
}

void *metal_device(MetalContext *ctx) { return ctx->device; }

void metal_shutdown(MetalContext *ctx) {
    if (!ctx) return;
    CFRelease(ctx->device);
    free(ctx);
}
