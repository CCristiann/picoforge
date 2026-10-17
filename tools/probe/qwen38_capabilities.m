/* R0.1 capability probe: what this Mac and OS actually support.
 *
 * Metal 4 and residency sets are young APIs, so each fact is asked of the live
 * device and every newer selector is guarded by respondsToSelector:/@available.
 * Counter sampling has distinct points: "stage" samples once per encoder pass,
 * "dispatch" around each dispatch inside an encoder (what per-op timing needs).
 *
 * Output: one JSON object on stdout, keys sorted. Exit 1 without a device.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSNumber *flag(BOOL value) { return value ? @YES : @NO; }

static NSDictionary *counter_sampling(id<MTLDevice> device) {
    static const struct { const char *name; MTLCounterSamplingPoint point; } points[] = {
        {"stage", MTLCounterSamplingPointAtStageBoundary},
        {"dispatch", MTLCounterSamplingPointAtDispatchBoundary},
        {"draw", MTLCounterSamplingPointAtDrawBoundary},
        {"blit", MTLCounterSamplingPointAtBlitBoundary},
        {"tile", MTLCounterSamplingPointAtTileDispatchBoundary},
    };
    BOOL known = [device respondsToSelector:@selector(supportsCounterSampling:)];
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < sizeof points / sizeof points[0]; i++)
        result[@(points[i].name)] = flag(known && [device supportsCounterSampling:points[i].point]);
    return result;
}

/* Run a one-byte blit fill and read the driver's GPU timestamps back. */
static BOOL command_buffer_gpu_times(id<MTLDevice> device) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buffer = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    if (!queue || !buffer) return NO;
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, 16) value:0];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    return commands.status == MTLCommandBufferStatusCompleted && commands.GPUStartTime > 0 &&
           commands.GPUEndTime >= commands.GPUStartTime;
}

static BOOL metal4_family(id<MTLDevice> device) {
    if (@available(macOS 26.0, *))
        return [device supportsFamily:MTLGPUFamilyMetal4];
    return NO;
}

static BOOL mtl4_command_queue(id<MTLDevice> device) {
    if (@available(macOS 26.0, *))
        return [device respondsToSelector:@selector(newMTL4CommandQueue)] &&
               [device newMTL4CommandQueue] != nil;
    return NO;
}

static BOOL residency_set(id<MTLDevice> device) {
    if (@available(macOS 15.0, *)) {
        if (![device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]) return NO;
        MTLResidencySetDescriptor *descriptor = [MTLResidencySetDescriptor new];
        return [device newResidencySetWithDescriptor:descriptor error:NULL] != nil;
    }
    return NO;
}

static NSString *thermal_state(void) {
    switch (NSProcessInfo.processInfo.thermalState) {
    case NSProcessInfoThermalStateNominal: return @"nominal";
    case NSProcessInfoThermalStateFair: return @"fair";
    case NSProcessInfoThermalStateSerious: return @"serious";
    case NSProcessInfoThermalStateCritical: return @"critical";
    }
    return @"unknown";
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        NSMutableArray<NSString *> *sets = [NSMutableArray array];
        for (id<MTLCounterSet> set in device.counterSets) [sets addObject:set.name];
        NSDictionary *caps = @{
            @"device": device.name,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"counter_sampling": counter_sampling(device),
            @"counter_sets": sets,
            @"timestamp_counter": flag([sets containsObject:MTLCommonCounterSetTimestamp]),
            @"command_buffer_gpu_times": flag(command_buffer_gpu_times(device)),
            @"metal4_family": flag(metal4_family(device)),
            @"mtl4_command_queue": flag(mtl4_command_queue(device)),
            @"residency_set": flag(residency_set(device)),
            @"recommended_max_working_set_bytes": @(device.recommendedMaxWorkingSetSize),
            @"max_buffer_length": @(device.maxBufferLength),
            @"unified_memory": flag(device.hasUnifiedMemory),
            @"thermal_state": thermal_state(),
        };
        NSError *error = nil;
        NSJSONWritingOptions options = NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys;
        NSData *data = [NSJSONSerialization dataWithJSONObject:caps options:options error:&error];
        if (!data) {
            fprintf(stderr, "JSON error: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fwrite(data.bytes, 1, data.length, stdout);
        putchar('\n');
    }
    return 0;
}
