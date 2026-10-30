/* R0.2 profile recorder: one start/end timestamp pair per group, resolved
 * after the command completes and written as one csv row per group. */
#import <Metal/Metal.h>
#include "qwen38_profile.h"
#include "qwen38.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>
/* 4096 timestamps x 8 B = 32 KiB, the largest sample buffer Metal accepts on
 * this M5 Pro (measured: 4097 is refused, and so is a 33rd buffer). */
enum { SAMPLES = 4096 };
typedef struct { uint32_t layer, type, dispatches, barriers; uint64_t bytes; char op[32]; } Group;
struct Q38Profile {
    void *device, *set, *buffers, *pass, *proxy;  /* retained Objective-C objects */
    FILE *csv; char phase[16]; uint64_t token;
    Group *groups; size_t count, capacity;
    MTLTimestamp cpu0, gpu0;              /* calibration origin */
    uint64_t floor;                       /* last end timestamp of the previous append */
};
/* R0.3: counts what the graph encodes, without touching any component. The
 * counted selectors are implemented here; every other message takes the
 * runtime's fast forwarding path to the real encoder. Profile mode only, so
 * the extra message cost lands in encode_s, never in normal runs. */
@interface Q38CountingEncoder : NSProxy { @public id<MTLComputeCommandEncoder> target; uint32_t dispatches, barriers; }
@end
@implementation Q38CountingEncoder
- (id)forwardingTargetForSelector:(SEL)s { (void)s; return target; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)s { return [(NSObject *)target methodSignatureForSelector:s]; }
- (void)forwardInvocation:(NSInvocation *)i { [i invokeWithTarget:target]; }
- (void)dispatchThreads:(MTLSize)n threadsPerThreadgroup:(MTLSize)g { dispatches++; [target dispatchThreads:n threadsPerThreadgroup:g]; }
- (void)dispatchThreadgroups:(MTLSize)n threadsPerThreadgroup:(MTLSize)g { dispatches++; [target dispatchThreadgroups:n threadsPerThreadgroup:g]; }
- (void)memoryBarrierWithScope:(MTLBarrierScope)s { barriers++; [target memoryBarrierWithScope:s]; }
- (void)memoryBarrierWithResources:(const id<MTLResource> __unsafe_unretained [])r count:(NSUInteger)n {
    barriers++; [target memoryBarrierWithResources:r count:n];
}
@end
/* The proxy's counts so far belong to the group opened last; restart them. */
static void take_counts(Q38Profile *p) {
    Q38CountingEncoder *c = (__bridge Q38CountingEncoder *)p->proxy;
    if (p->count) { p->groups[p->count - 1].dispatches = c->dispatches; p->groups[p->count - 1].barriers = c->barriers; }
    c->dispatches = c->barriers = 0;
}
static Qwen38Status fail(Qwen38Error *e, Qwen38Status code, const char *detail) {
    if (e) { memset(e, 0, sizeof *e); e->code = code;
        if (code) { snprintf(e->subject, sizeof e->subject, "qwen38/profile"); snprintf(e->detail, sizeof e->detail, "%s", detail); } }
    return code;
}
Qwen38Status q38_profile_create(void *device, FILE *csv, Q38Profile **out, Qwen38Error *e) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device; id<MTLCounterSet> set = nil; Q38Profile *p = NULL;
    for (id<MTLCounterSet> s in dev.counterSets) if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) set = s;
    Qwen38Status rc = !csv || !out || !dev ? Q38_INVALID_ARGUMENT
        : !set || ![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] ? Q38_UNSUPPORTED
        : (p = calloc(1, sizeof *p)) ? Q38_OK : Q38_OOM;
    if (rc) { if (csv) fclose(csv); return fail(e, rc, "stage-boundary timestamp sampling"); }
    p->device = (__bridge_retained void *)dev; p->set = (__bridge_retained void *)set;
    p->buffers = (__bridge_retained void *)[NSMutableArray new];
    p->pass = (__bridge_retained void *)[MTLComputePassDescriptor computePassDescriptor];
    p->proxy = (__bridge_retained void *)[Q38CountingEncoder alloc];
    p->csv = csv; fputs("phase,token,layer,op,format,weight_bytes,dispatches,barriers,gpu_us,gbps\n", csv);
    [dev sampleTimestamps:&p->cpu0 gpuTimestamp:&p->gpu0];
    *out = p; return fail(e, Q38_OK, NULL);
}
void q38_profile_begin_append(Q38Profile *p, const char *phase, uint64_t token) {
    if (p) { snprintf(p->phase, sizeof p->phase, "%s", phase); p->token = token; }
}
void *q38_profile_group(Q38Profile *p, uint32_t layer, const char *op, uint32_t type, uint64_t bytes) {
    NSMutableArray *buffers = (__bridge NSMutableArray *)p->buffers; size_t slot = 2 * p->count;
    take_counts(p);
    if (p->count == p->capacity) {
        size_t n = p->capacity ? 2 * p->capacity : 256; Group *grown = realloc(p->groups, n * sizeof *grown);
        if (!grown) return NULL;
        p->groups = grown; p->capacity = n;
    }
    if (slot / SAMPLES == buffers.count) {
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = (__bridge id<MTLCounterSet>)p->set; d.storageMode = MTLStorageModeShared; d.sampleCount = SAMPLES;
        id<MTLCounterSampleBuffer> b = [(__bridge id<MTLDevice>)p->device newCounterSampleBufferWithDescriptor:d error:NULL];
        if (!b) return NULL;
        [buffers addObject:b];
    }
    Group *g = &p->groups[p->count++]; *g = (Group){.layer = layer, .type = type, .bytes = bytes};
    snprintf(g->op, sizeof g->op, "%s", op);
    /* The encoder copies the descriptor when it is created, so one is reused. */
    MTLComputePassSampleBufferAttachmentDescriptor *a = ((__bridge MTLComputePassDescriptor *)p->pass).sampleBufferAttachments[0];
    a.sampleBuffer = buffers[slot / SAMPLES];
    a.startOfEncoderSampleIndex = slot % SAMPLES; a.endOfEncoderSampleIndex = slot % SAMPLES + 1;
    return p->pass;
}
void *q38_profile_wrap(Q38Profile *p, void *encoder) {
    ((__bridge Q38CountingEncoder *)p->proxy)->target = (__bridge id<MTLComputeCommandEncoder>)encoder; return p->proxy;
}
void *q38_profile_unwrap(Q38Profile *p, void *encoder) {
    return encoder == p->proxy ? (__bridge void *)((__bridge Q38CountingEncoder *)p->proxy)->target : encoder;
}
bool q38_profile_end_append(Q38Profile *p, bool ok) {
    take_counts(p); ((__bridge Q38CountingEncoder *)p->proxy)->target = nil;   /* the command has ended */
    size_t n = p->count; p->count = 0;
    if (!ok || !n) return true;
    /* Timestamp units are calibrated, not assumed: two (CPU ns, GPU tick) pairs
     * give ns per tick (measured 1.0 here; CPU side equals CLOCK_UPTIME_RAW). */
    MTLTimestamp cpu, gpu; [(__bridge id<MTLDevice>)p->device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    double ns = cpu > p->cpu0 && gpu > p->gpu0 ? (double)(cpu - p->cpu0) / (double)(gpu - p->gpu0) : 0;
    NSArray *buffers = (__bridge NSArray *)p->buffers; NSData *data = nil; bool good = ns > 0; uint64_t last = p->floor;
    for (size_t i = 0, slot = 0; i < n && good; i++, slot += 2) {
        if (slot % SAMPLES == 0) data = [buffers[slot / SAMPLES] resolveCounterRange:NSMakeRange(0, MIN((size_t)SAMPLES, 2 * n - slot))];
        const uint64_t *t = data.length >= (slot % SAMPLES + 2) * 8 ? (const uint64_t *)data.bytes + slot % SAMPLES : NULL;
        /* A missing sample (error value, 0 for an encoder the GPU never ran, or a
         * stale pair left by an earlier append in these reused slots: it starts
         * before that append's last end) is a failure, never a number. */
        good = t && t[0] > p->floor && t[1] != MTLCounterErrorValue && t[1] > t[0];
        if (!good) break;
        last = MAX(last, t[1]);
        const Group *g = &p->groups[i]; double us = (double)(t[1] - t[0]) * ns / 1e3;
        char layer[16] = "global"; if (g->layer != QWEN38_GLOBAL_LAYER) snprintf(layer, sizeof layer, "%u", g->layer);
        fprintf(p->csv, "%s,%llu,%s,%s,%s,%llu,%u,%u,%.3f,%.3f\n", p->phase, (unsigned long long)p->token, layer, g->op,
            g->type == Q38_PROFILE_NONE ? "NONE" : g->type == Q38_PROFILE_MIXED ? "MIXED" : gguf_format(g->type)->name,
            (unsigned long long)g->bytes, g->dispatches, g->barriers, us, (double)g->bytes / (us * 1e3));
    }
    if (good) p->floor = last;
    return good && !ferror(p->csv);
}
bool q38_profile_close(Q38Profile *p) {
    if (!p) return true;
    bool ok = !ferror(p->csv); if (fclose(p->csv)) ok = false;
    CFRelease(p->device); CFRelease(p->set); CFRelease(p->buffers); CFRelease(p->pass); CFRelease(p->proxy);
    free(p->groups); free(p); return ok;
}
