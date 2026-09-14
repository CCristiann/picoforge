// metal4_probe.m — runtime probe for Metal 4 / tensor API availability.
//
// Why a compiled probe instead of trusting version strings: the SDK headers
// tell us what we can *compile against*; only the runtime can tell us what
// this specific GPU + driver actually *supports*. docs/DESIGN.md rule: never
// assume, always probe.
//
// Build:  clang -fobjc-arc -framework Metal -framework Foundation \
//               -o metal4_probe metal4_probe.m
// Run:    ./metal4_probe

#import <Metal/Metal.h>

static void check(id<MTLDevice> dev, MTLGPUFamily fam, const char *name) {
    printf("  %-22s %s\n", name, [dev supportsFamily:fam] ? "YES" : "no");
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            printf("FATAL: no Metal device found\n");
            return 1;
        }

        printf("device: %s\n", dev.name.UTF8String);
        printf("recommended max working set: %.1f GB\n",
               (double)dev.recommendedMaxWorkingSetSize / 1e9);
        printf("unified memory: %s\n\n", dev.hasUnifiedMemory ? "YES" : "no");

        printf("GPU family support:\n");
        check(dev, MTLGPUFamilyApple7,  "Apple7  (M1)");
        check(dev, MTLGPUFamilyApple8,  "Apple8  (M2)");
        check(dev, MTLGPUFamilyApple9,  "Apple9  (M3/M4)");
        check(dev, MTLGPUFamilyApple10, "Apple10 (M5?)");
        check(dev, MTLGPUFamilyMetal3,  "Metal3");
        check(dev, MTLGPUFamilyMetal4,  "Metal4");

        // Friction test: actually create a tiny fp16 tensor. If this
        // returns a live object, the tensor APIs are real on this machine,
        // not just declared in the headers.
        printf("\nMTLTensor creation test: ");
        if (@available(macOS 26.0, *)) {
            MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
            td.dataType = MTLTensorDataTypeFloat16;
            td.dimensions = [[MTLTensorExtents alloc] initWithRank:2
                                                            values:(NSInteger[]){4, 4}];
            NSError *err = nil;
            id<MTLTensor> t = [dev newTensorWithDescriptor:td error:&err];
            if (t) {
                printf("OK (rank-2 fp16 4x4 tensor created)\n");
            } else {
                printf("FAILED: %s\n", err.localizedDescription.UTF8String);
                return 2;
            }
        } else {
            printf("SKIPPED (macOS < 26.0)\n");
            return 3;
        }

        printf("\nverdict: Metal 4 tensor APIs are usable on this machine.\n");
        return 0;
    }
}
