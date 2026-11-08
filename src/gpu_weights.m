/* Each MTLBuffer owns its own mmap. Retained command-buffer references keep
 * that mapping alive even after the store and GGUF header have been closed. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "common.h"
#include "gguf.h"
#include "qwen38_packed.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* R5.1b: pool[m] are the expert-pool regions (gate, up, down) as no-copy buffers, listed after
 * the file views as buffer ids view_count+m; streamed[] are the virtual bindings of the streamed
 * expert tensors into them (slot s of a tensor at s * expert bytes). */
struct GpuWeightStore { GgufPlan plan; void **buffers; void *device; void *pool[3]; GgufBinding *streamed; uint32_t slots; };
static _Atomic size_t live_mapped_bytes;

size_t gpu_weights_live_mapped_bytes(void) { return atomic_load(&live_mapped_bytes); }
const GgufPlan *gpu_weights_plan(const GpuWeightStore *s) { return &s->plan; }

GpuWeightStore *gpu_weights_open(void *device, const Gguf *g, size_t cap) { return gpu_weights_open_ex(device, g, cap, 0); }
GpuWeightStore *gpu_weights_open_ex(void *device, const Gguf *g, size_t cap, uint32_t stream_layers) {
    struct stat file;
    if (fstat(g->fd, &file) || !S_ISREG(file.st_mode) || file.st_size < 0 ||
        (size_t)file.st_size != g->file_size) die("gguf: file size changed before Metal mapping");
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    if (!dev || !dev.hasUnifiedMemory) die("gguf: weight views require unified Metal memory");
    size_t max_buffer = dev.maxBufferLength;
    if (cap > max_buffer) die("gguf: requested view exceeds maxBufferLength");
    if (!cap) cap = max_buffer;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) die("gguf: cannot query host page size");
    GpuWeightStore *s = calloc(1, sizeof *s);
    if (!s) die("gguf: weight-store allocation failed");
    s->device = device;
    gguf_plan_ex(g, (size_t)page, cap, stream_layers, &s->plan);
    size_t planned = 0;
    for (size_t i = 0; i < s->plan.view_count; i++) planned = gguf_add(planned, s->plan.views[i].bytes);
    if (planned > dev.recommendedMaxWorkingSetSize)
        die("gguf: resident views exceed recommended GPU working set before runtime buffers");
    s->buffers = calloc(s->plan.view_count, sizeof *s->buffers);
    if (!s->buffers) die("gguf: buffer-list allocation failed");
    for (size_t i = 0; i < s->plan.view_count; i++) {
        const GgufView *v = &s->plan.views[i];
        void *map = mmap(NULL, v->bytes, PROT_READ, MAP_SHARED, g->fd, (off_t)v->file_offset);
        if (map == MAP_FAILED) die("gguf: resident-view mapping failed");
        id<MTLBuffer> buffer = [dev newBufferWithBytesNoCopy:map length:v->bytes
            options:MTLResourceStorageModeShared deallocator:^(void *pointer, NSUInteger length) {
                munmap(pointer, length);
                atomic_fetch_sub(&live_mapped_bytes, (size_t)length);
            }];
        if (!buffer) { munmap(map, v->bytes); die("gguf: Metal rejected a resident view"); }
        atomic_fetch_add(&live_mapped_bytes, v->bytes);
        buffer.label = [NSString stringWithFormat:@"picoforge_weights_%zu", i];
        s->buffers[i] = (__bridge_retained void *)buffer;
    }
    return s;
}

const GgufBinding *gpu_weights_find(const GpuWeightStore *s, const char *name) {
    for (size_t i = 0; i < s->plan.weight_count; i++)
        if (!strcmp(s->plan.weights[i].tensor.name, name)) return &s->plan.weights[i];
    return NULL; /* includes the intentionally disk-only n-gram table */
}

void gpu_weights_attach_pool(GpuWeightStore *s, void *const region[3], const size_t region_bytes[3], uint32_t slots) {
    if (!s->plan.streamed_count || s->streamed || !slots) die("gguf: expert pool attach without streamed tensors");
    id<MTLDevice> dev = (__bridge id<MTLDevice>)s->device;
    for (int m = 0; m < 3; m++) {
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:region[m] length:region_bytes[m]
            options:MTLResourceStorageModeShared deallocator:nil];   /* the pool owns the memory */
        if (!b) die("gguf: Metal rejected an expert-pool region");
        b.label = [NSString stringWithFormat:@"picoforge_expert_pool_%d", m];
        s->pool[m] = (__bridge_retained void *)b;
    }
    s->streamed = calloc(s->plan.streamed_count, sizeof *s->streamed);
    if (!s->streamed) die("gguf: streamed-binding allocation failed");
    for (size_t i = 0; i < s->plan.streamed_count; i++) {
        const GgufTensor *t = &s->plan.streamed[i];
        static const char *const kind[3] = {"_gate_exps.", "_up_exps.", "_down_exps."};
        int m = -1;
        for (int k = 0; k < 3; k++) if (strstr(t->name, kind[k])) m = k;
        if (m < 0 || t->rank != 3 || !t->shape[2] || t->bytes % t->shape[2]) die("gguf: streamed tensor shape: %s", t->name);
        const size_t expert = t->bytes / t->shape[2];
        if ((size_t)slots * expert > region_bytes[m]) die("gguf: expert pool smaller than its slots: %s", t->name);
        GgufBinding *b = &s->streamed[i];
        *b = (GgufBinding){*t, s->plan.view_count + (size_t)m, 0, &s->plan};
        b->tensor.shape[2] = slots; b->tensor.bytes = (size_t)slots * expert;
    }
    s->slots = slots;
}
const GgufBinding *gpu_weights_streamed(const GpuWeightStore *s, const char *name) {
    for (size_t i = 0; s->streamed && i < s->plan.streamed_count; i++)
        if (!strcmp(s->streamed[i].tensor.name, name)) return &s->streamed[i];
    return NULL;
}
uint32_t gpu_weights_stream_slots(const GpuWeightStore *s) { return s->slots; }
size_t gpu_weights_pool_buffers(const GpuWeightStore *s, void *out[3]) {
    for (int m = 0; m < 3; m++) out[m] = s->pool[m];
    return s->pool[0] ? 3 : 0;
}

void *gpu_weights_buffer(const GpuWeightStore *s, const GgufBinding *b) {
    if (b && b->owner == &s->plan && s->pool[0] && b->buffer_id >= s->plan.view_count && b->buffer_id < s->plan.view_count + 3) {
        id<MTLBuffer> pool = (__bridge id<MTLBuffer>)s->pool[b->buffer_id - s->plan.view_count];
        if (b->offset || b->tensor.bytes > pool.length) die("gguf: streamed binding exceeds its pool");
        return s->pool[b->buffer_id - s->plan.view_count];
    }
    if (!b || b->owner != &s->plan || b->buffer_id >= s->plan.view_count)
        die("gguf: binding belongs to a different store");
    const GgufView *v = &s->plan.views[b->buffer_id];
    if (b->offset > v->bytes || b->tensor.bytes > v->bytes - b->offset)
        die("gguf: binding exceeds its view");
    return s->buffers[b->buffer_id]; /* borrowed; the command encoder retains it */
}

Qwen38Status qwen38_weights_cpu_span(const GpuWeightStore *s,
        const GgufBinding *b, Qwen38CpuSpan *out, Qwen38Error *err) {
    Qwen38Status status = Q38_OK;
    if (err) memset(err, 0, sizeof *err);
    if (out) *out = (Qwen38CpuSpan){0};
    if (!s || !b || !out) status = Q38_INVALID_ARGUMENT;
    else if (b->owner != &s->plan) status = Q38_OWNER;
    else {
        /* Accept a live catalog record, not a forged/copy-by-value binding. */
        bool found = false;
        for (size_t i = 0; i < s->plan.weight_count; i++)
            if (b == &s->plan.weights[i]) { found = true; break; }
        for (size_t i = 0; s->streamed && !found && i < s->plan.streamed_count; i++)
            if (b == &s->streamed[i]) found = true;   /* R5.1b: a virtual binding into the expert pool */
        if (!found) status = Q38_OWNER;
        else if (b->buffer_id >= s->plan.view_count && b->buffer_id < s->plan.view_count + 3 && s->pool[0]) {
            id<MTLBuffer> pool = (__bridge id<MTLBuffer>)s->pool[b->buffer_id - s->plan.view_count];
            if (b->offset || b->tensor.bytes > pool.length) status = Q38_RANGE;
            else *out = (Qwen38CpuSpan){(const unsigned char *)pool.contents, b->tensor.bytes};
        }
        else if (b->buffer_id >= s->plan.view_count) status = Q38_RANGE;
        else {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)s->buffers[b->buffer_id];
            size_t bytes = (size_t)buffer.length;
            if (b->offset > bytes || b->tensor.bytes > bytes - b->offset)
                status = Q38_RANGE;
            else *out = (Qwen38CpuSpan){(const unsigned char *)buffer.contents + b->offset,
                                       b->tensor.bytes};
        }
    }
    if (err) {
        err->code = status;
        if (status != Q38_OK) {
            snprintf(err->subject, sizeof err->subject, "weight CPU span");
            snprintf(err->detail, sizeof err->detail, "invalid argument, owner or bounds");
        }
    }
    return status;
}

void gpu_weights_close(GpuWeightStore *s) {
    if (!s) return;
    for (size_t i = 0; i < s->plan.view_count; i++) CFRelease(s->buffers[i]);
    for (int m = 0; m < 3; m++) if (s->pool[m]) CFRelease(s->pool[m]);
    free(s->buffers); free(s->streamed);
    gguf_plan_free(&s->plan);
    free(s);
}
