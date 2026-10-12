/* Probe the selected runtime, not a published GPU specification.
 * clang -fobjc-arc -framework Foundation -framework Metal
 *       tools/probe/memory_budget.m -o build/memory_budget
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;
        NSDictionary *values = @{
            @"device": device.name,
            @"physical_memory_bytes": @(NSProcessInfo.processInfo.physicalMemory),
            @"recommended_working_set_bytes": @(device.recommendedMaxWorkingSetSize),
            @"max_buffer_bytes": @(device.maxBufferLength),
            @"max_threadgroup_bytes": @(device.maxThreadgroupMemoryLength),
            @"unified_memory": @(device.hasUnifiedMemory),
            @"probe_allocated_bytes": @(device.currentAllocatedSize),
            @"thermal_state": @(NSProcessInfo.processInfo.thermalState),
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString
        };
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:values
                                     options:NSJSONWritingPrettyPrinted error:&error];
        if (!data) return 1;
        fwrite(data.bytes, 1, data.length, stdout);
        putchar('\n');
    }
    return 0;
}
