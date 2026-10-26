/* G0: the common GPU session. The ledger is plain bookkeeping; the session
 * adds the Metal queue, command buffer and error word it protects. */
#import <Metal/Metal.h>
#include "qwen38_gpu.h"
#include "qwen38_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Qwen38Status report(Qwen38Error *e, Qwen38Status code) {
    if (e) { memset(e, 0, sizeof(*e)); e->code = code; }
    return code;
}
static bool busy(const Qwen38GpuLedger *l) {
    return l->state == Q38_SESSION_ENCODING || l->state == Q38_SESSION_IN_FLIGHT;
}
/* A ticket is honoured only if it is exactly the command begin() admitted, in
 * the expected phase. Everything else is a stale, foreign or replayed ticket. */
static Qwen38Status check(const Qwen38GpuLedger *l, const Qwen38GpuTicket *t,
    Qwen38SessionState want) {
    if (!l || !t) return Q38_INVALID_ARGUMENT;
    if (t->model != l->model) return Q38_OWNER;
    if (t->epoch != l->epoch) return Q38_STALE;
    if (l->state != want || t->start_pos != l->open.start_pos
        || t->count != l->open.count) return Q38_BAD_STATE;
    return Q38_OK;
}

Qwen38Status qwen38_gpu_ledger_check(const Qwen38GpuLedger *l, const Qwen38GpuTicket *t,
    Qwen38SessionState want, Qwen38Error *e) {
    return report(e, check(l, t, want));
}
Qwen38Status qwen38_gpu_ledger_init(Qwen38GpuLedger *l, const void *model,
    uint64_t capacity, Qwen38Error *e) {
    if (l) memset(l, 0, sizeof(*l));
    if (!l || !model || !capacity) return report(e, Q38_INVALID_ARGUMENT);
    l->model = model; l->capacity = capacity;
    l->epoch = 1;                       /* epoch 0 is never valid */
    return report(e, Q38_OK);           /* IDLE is the zero state */
}
Qwen38Status qwen38_gpu_ledger_begin(Qwen38GpuLedger *l, const void *model,
    uint64_t start_pos, uint64_t count, Qwen38GpuTicket *t, Qwen38Error *e) {
    if (t) *t = (Qwen38GpuTicket){0};
    if (!l || !t) return report(e, Q38_INVALID_ARGUMENT);
    if (model != l->model) return report(e, Q38_OWNER);
    if (busy(l)) return report(e, Q38_BUSY);
    if (l->state == Q38_SESSION_INVALID || start_pos != l->next_pos)
        return report(e, Q38_BAD_STATE);
    /* next_pos <= capacity always holds, so this subtraction cannot wrap. */
    if (count > l->capacity - start_pos) return report(e, Q38_CAPACITY);
    l->resume = l->state; l->state = Q38_SESSION_ENCODING;
    l->open = *t = (Qwen38GpuTicket){model, l->epoch, start_pos, count};
    return report(e, Q38_OK);
}
Qwen38Status qwen38_gpu_ledger_submit(Qwen38GpuLedger *l, const Qwen38GpuTicket *t,
    Qwen38Error *e) {
    Qwen38Status s = check(l, t, Q38_SESSION_ENCODING);
    if (s == Q38_OK) l->state = Q38_SESSION_IN_FLIGHT;
    return report(e, s);
}
Qwen38Status qwen38_gpu_ledger_abort(Qwen38GpuLedger *l, const Qwen38GpuTicket *t,
    bool state_mutated, Qwen38Error *e) {
    Qwen38Status s = check(l, t, Q38_SESSION_ENCODING);
    if (s == Q38_OK) l->state = state_mutated ? Q38_SESSION_INVALID : l->resume;
    return report(e, s);
}
Qwen38Status qwen38_gpu_ledger_complete(Qwen38GpuLedger *l, const Qwen38GpuTicket *t,
    bool command_ok, uint32_t error_bits, Qwen38Error *e) {
    Qwen38Status s = check(l, t, Q38_SESSION_IN_FLIGHT);
    if (s != Q38_OK) return report(e, s);
    if (command_ok && !error_bits) {
        l->next_pos += t->count;
        l->state = t->count ? Q38_SESSION_VALID : l->resume;  /* T=0 publishes nothing */
        return report(e, Q38_OK);
    }
    /* Later layers may already have written: no partial history is trusted. */
    l->state = Q38_SESSION_INVALID;
    report(e, Q38_GPU_ERROR);
    if (e) snprintf(e->detail, sizeof(e->detail), "command_ok=%d error_bits=0x%x",
        command_ok, error_bits);
    return Q38_GPU_ERROR;
}
Qwen38Status qwen38_gpu_ledger_reset(Qwen38GpuLedger *l, const void *model,
    Qwen38Error *e) {
    if (!l) return report(e, Q38_INVALID_ARGUMENT);
    if (model != l->model) return report(e, Q38_OWNER);
    if (busy(l)) return report(e, Q38_BUSY);
    if (l->epoch == UINT64_MAX) return report(e, Q38_OVERFLOW);
    l->epoch++; l->next_pos = 0;
    l->state = l->resume = Q38_SESSION_IDLE;
    l->open = (Qwen38GpuTicket){0};
    return report(e, Q38_OK);
}
Qwen38StateInfo qwen38_gpu_ledger_info(const Qwen38GpuLedger *l) {
    if (!l) return (Qwen38StateInfo){0};
    return (Qwen38StateInfo){.key = {l->model, 0, l->epoch}, .next_pos = l->next_pos,
        .valid = l->state != Q38_SESSION_INVALID, .busy = busy(l)};
}

/* Retained Metal objects are held as void * (bridge-retained), like the other
 * Qwen38 GPU glue, so this struct stays a plain C struct under ARC. */
typedef struct {
    void *buffer;                      /* retained until the command ends */
    const void *ident;                 /* NULL for CPU-visible memory, else the buffer */
    uintptr_t lo, hi;                  /* address range, or offsets when ident is set */
    bool write;
} Claim;
struct Qwen38GpuSession {
    Qwen38GpuLedger ledger;
    void *device, *queue, *error_word;
    void *command, *encoder;           /* alive only from begin to finish/abort */
    Claim *claims;
    size_t claim_count, claim_capacity;
    double gpu_seconds;
    /* R1.6b: concurrent encoders unless Q38_REF_KERNELS selects `barrier`.
     * A concurrent encoder orders nothing by itself: every dependency must be
     * an explicit memory barrier, so this needs the barrier audit of R1.6b. */
    bool concurrent;
};
static void drop(void **p) {
    if (*p) { (void)(__bridge_transfer id)*p; *p = NULL; }
}
static void release_claims(Qwen38GpuSession *g) {
    for (size_t i = 0; i < g->claim_count; i++) drop(&g->claims[i].buffer);
    g->claim_count = 0;
}
static Qwen38Status add_claim(Qwen38GpuSession *g, Qwen38GpuSpan span, Qwen38GpuUse use,
    Qwen38Error *e) {
    if (!span.buffer || !span.bytes) return report(e, Q38_INVALID_ARGUMENT);
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)span.buffer;
    if (b.device != (__bridge id<MTLDevice>)g->device) return report(e, Q38_OWNER);
    if (span.offset_bytes % 4) return report(e, Q38_STRIDE);
    if (span.offset_bytes > b.length || span.bytes > b.length - span.offset_bytes)
        return report(e, Q38_CAPACITY);
    if (b.hazardTrackingMode == MTLHazardTrackingModeUntracked)
        return report(e, Q38_UNSUPPORTED);
    /* Separate no-copy buffers may wrap the same memory, so CPU-visible buffers
     * are compared by address; private ones can only overlap themselves. */
    uintptr_t base = b.storageMode == MTLStorageModeShared ? (uintptr_t)b.contents : 0;
    Claim c = {.ident = base ? NULL : span.buffer, .write = use == Q38_GPU_USE_WRITE};
    c.lo = base + span.offset_bytes; c.hi = c.lo + span.bytes;
    for (size_t i = 0; i < g->claim_count; i++) {
        const Claim *o = &g->claims[i];
        if (o->ident == c.ident && c.lo < o->hi && o->lo < c.hi && (c.write || o->write))
            return report(e, Q38_INVALID_ARGUMENT);
    }
    if (g->claim_count == g->claim_capacity) {
        Claim *grown = realloc(g->claims, 2 * g->claim_capacity * sizeof(*grown));
        if (!grown) return report(e, Q38_OOM);
        g->claims = grown; g->claim_capacity *= 2;
    }
    c.buffer = (__bridge_retained void *)b;
    g->claims[g->claim_count++] = c;
    return report(e, Q38_OK);
}
static id<MTLBuffer> word_of(const Qwen38GpuSession *g) {
    return (__bridge id<MTLBuffer>)g->error_word;
}

Qwen38Status qwen38_gpu_session_create(void *device, const void *model,
    uint64_t capacity, Qwen38GpuSession **out, Qwen38Error *e) {
    if (out) *out = NULL;
    if (!out || !device) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38GpuSession *g = calloc(1, sizeof(*g));
    if (!g) return report(e, Q38_OOM);
    g->claim_capacity = 64;
    g->claims = calloc(g->claim_capacity, sizeof(*g->claims));
    if (!g->claims) { free(g); return report(e, Q38_OOM); }
    g->concurrent = !q38_ref_kernels("barrier");   /* once per session, never per token */
    Qwen38Status s = qwen38_gpu_ledger_init(&g->ledger, model, capacity, e);
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = s ? nil : [dev newCommandQueue];
    id<MTLBuffer> word = s ? nil : [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    if (s || !queue || !word) { free(g->claims); free(g); return s ? s : report(e, Q38_OOM); }
    g->device = (__bridge_retained void *)dev;
    g->queue = (__bridge_retained void *)queue;
    g->error_word = (__bridge_retained void *)word;
    *out = g;
    return report(e, Q38_OK);
}
static MTLDispatchType dispatch_type(const Qwen38GpuSession *g) {
    return g->concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
}
Qwen38Status qwen38_gpu_session_begin(Qwen38GpuSession *g, const void *model,
    uint64_t start_pos, uint64_t count, Qwen38GpuTicket *t, Qwen38GpuCommand *cmd,
    Qwen38Error *e) {
    if (cmd) *cmd = (Qwen38GpuCommand){0};
    if (!g || !cmd) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38Status s = qwen38_gpu_ledger_begin(&g->ledger, model, start_pos, count, t, e);
    if (s) return s;
    g->gpu_seconds=0;
    id<MTLCommandBuffer> cb = [(__bridge id<MTLCommandQueue>)g->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:dispatch_type(g)];
    if (!enc) {                         /* nothing was touched: the session stays valid */
        qwen38_gpu_ledger_abort(&g->ledger, t, false, NULL);
        *t = (Qwen38GpuTicket){0};
        return report(e, Q38_GPU_ERROR);
    }
    memset(word_of(g).contents, 0, 8);    /* the only reset of both error words */
    /* The empty, preallocated list cannot refuse this: nothing may alias the words. */
    (void)add_claim(g, (Qwen38GpuSpan){g->error_word, 0, 8}, Q38_GPU_USE_WRITE, NULL);
    g->command = (__bridge_retained void *)cb;
    g->encoder = (__bridge_retained void *)enc;
    *cmd = (Qwen38GpuCommand){g->device, g->encoder, {g->error_word, 0, 4}, g->concurrent};
    return report(e, Q38_OK);
}
Qwen38Status qwen38_gpu_session_claim(Qwen38GpuSession *g, const Qwen38GpuTicket *t,
    Qwen38GpuSpan span, Qwen38GpuUse use, Qwen38Error *e) {
    if (!g) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38Status s = qwen38_gpu_ledger_check(&g->ledger, t, Q38_SESSION_ENCODING, e);
    return s ? s : add_claim(g, span, use, e);
}
Qwen38Status qwen38_gpu_session_submit(Qwen38GpuSession *g, const Qwen38GpuTicket *t,
    Qwen38Error *e) {
    if (!g) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38Status s = qwen38_gpu_ledger_submit(&g->ledger, t, e);
    if (s) return s;                    /* a double submit never reaches commit */
    [(__bridge id<MTLComputeCommandEncoder>)g->encoder endEncoding];
    drop(&g->encoder);
    [(__bridge id<MTLCommandBuffer>)g->command commit];
    return s;
}
Qwen38Status qwen38_gpu_session_abort(Qwen38GpuSession *g, const Qwen38GpuTicket *t,
    bool state_mutated, Qwen38Error *e) {
    if (!g) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38Status s = qwen38_gpu_ledger_abort(&g->ledger, t, state_mutated, e);
    if (s) return s;
    [(__bridge id<MTLComputeCommandEncoder>)g->encoder endEncoding];
    drop(&g->encoder); drop(&g->command);   /* never committed, so nothing runs */
    release_claims(g);
    return s;
}
Qwen38Status qwen38_gpu_session_finish(Qwen38GpuSession *g, const Qwen38GpuTicket *t,
    Qwen38Error *e) {
    if (!g) return report(e, Q38_INVALID_ARGUMENT);
    bool ok = true; uint32_t bits = 0; NSString *why = nil;
    /* Only a committed command may be waited on; anything else is the ledger's
     * BAD_STATE to report (finish before submit, double finish). */
    if (g->ledger.state == Q38_SESSION_IN_FLIGHT) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)g->command;
        [cb waitUntilCompleted];
        ok = cb.status == MTLCommandBufferStatusCompleted && !cb.error;
        if (ok && cb.GPUStartTime>0 && cb.GPUEndTime>=cb.GPUStartTime)
            g->gpu_seconds=cb.GPUEndTime-cb.GPUStartTime;
        why = cb.error.localizedDescription;
        const volatile uint32_t *w = word_of(g).contents;
        bits = w[0] | w[1];                 /* w[1]: checks of a concurrent command */
    }
    Qwen38Status s = qwen38_gpu_ledger_complete(&g->ledger, t, ok, bits, e);
    if (s == Q38_OK || s == Q38_GPU_ERROR) {
        drop(&g->command); drop(&g->encoder); release_claims(g);   /* only now may scratch be reused */
    }
    if (s == Q38_GPU_ERROR && e && why) {
        size_t n = strlen(e->detail);
        snprintf(e->detail + n, sizeof(e->detail) - n, " metal=%s", why.UTF8String);
    }
    return s;
}
Qwen38Status qwen38_gpu_session_reset(Qwen38GpuSession *g, const void *model,
    Qwen38Error *e) {
    return g ? qwen38_gpu_ledger_reset(&g->ledger, model, e) : report(e, Q38_INVALID_ARGUMENT);
}
Qwen38StateInfo qwen38_gpu_session_info(const Qwen38GpuSession *g) {
    return g ? qwen38_gpu_ledger_info(&g->ledger) : (Qwen38StateInfo){0};
}
double qwen38_gpu_session_seconds(const Qwen38GpuSession *g) { return g ? g->gpu_seconds : 0; }
Qwen38Status q38g_session_rewind(Qwen38GpuSession *g,const void *model,uint64_t epoch,
    uint64_t position,Qwen38Error *e) {
    if (!g) return report(e,Q38_INVALID_ARGUMENT);
    Qwen38GpuLedger *l=&g->ledger;
    if (l->model!=model) return report(e,Q38_OWNER);
    if (busy(l)) return report(e,Q38_BUSY);
    if (l->epoch!=epoch) return report(e,Q38_STALE);
    if (l->state==Q38_SESSION_INVALID || position>l->next_pos || l->next_pos-position>2)
        return report(e,Q38_BAD_STATE);
    Qwen38Status rc=qwen38_gpu_ledger_reset(l,model,e);
    if (!rc) { l->next_pos=position; l->state=position ? Q38_SESSION_VALID : Q38_SESSION_IDLE; }
    return rc;
}
/* R0.2 profile mode only: end the current encoder and continue the SAME
 * command in a new one (from an MTLComputePassDescriptor, or plain if NULL).
 * Encoders of one command buffer keep Metal's order and hazard tracking. */
Qwen38Status q38g_session_split(Qwen38GpuSession *g, const Qwen38GpuTicket *t, void *pass,
    Qwen38GpuCommand *cmd, Qwen38Error *e) {
    if (!g || !cmd || cmd->compute_encoder != g->encoder) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38Status s = qwen38_gpu_ledger_check(&g->ledger, t, Q38_SESSION_ENCODING, e);
    if (s) return s;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)g->command;
    [(__bridge id<MTLComputeCommandEncoder>)g->encoder endEncoding]; drop(&g->encoder);
    if (pass) ((__bridge MTLComputePassDescriptor *)pass).dispatchType = dispatch_type(g);
    id<MTLComputeCommandEncoder> enc = pass ? [cb computeCommandEncoderWithDescriptor:
        (__bridge MTLComputePassDescriptor *)pass] : [cb computeCommandEncoderWithDispatchType:dispatch_type(g)];
    cmd->compute_encoder = g->encoder = enc ? (__bridge_retained void *)enc : NULL;
    return report(e, enc ? Q38_OK : Q38_GPU_ERROR);   /* nil: the caller must abort */
}
Qwen38Status qwen38_gpu_session_destroy(Qwen38GpuSession *g, Qwen38Error *e) {
    if (!g) return report(e, Q38_INVALID_ARGUMENT);
    if (qwen38_gpu_ledger_info(&g->ledger).busy) return report(e, Q38_BUSY);
    drop(&g->command); drop(&g->encoder); drop(&g->error_word);
    drop(&g->queue); drop(&g->device);
    release_claims(g); free(g->claims); free(g);
    return report(e, Q38_OK);
}
