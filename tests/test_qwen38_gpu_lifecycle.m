/* G0 lifecycle: ledger rules, then the Metal session. Run only through
 * qwen38_gpu_lock.py (make test-qwen38-g0); a missing lock is a failure, not a skip. */
#import <Metal/Metal.h>
#include "qwen38_gpu.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPECT(call, want) do { Qwen38Status got_ = (call); \
    if (got_ != (want)) { fprintf(stderr, "line %d: %s -> %d, want %d\n", \
        __LINE__, #call, got_, (int)(want)); abort(); } } while (0)
#define IS(cond) do { if (!(cond)) { fprintf(stderr, "line %d: %s\n", __LINE__, #cond); abort(); } } while (0)

static int model_a, model_b;   /* only their addresses matter */
static Qwen38GpuLedger fresh(uint64_t capacity) {
    Qwen38GpuLedger l; EXPECT(qwen38_gpu_ledger_init(&l, &model_a, capacity, NULL), Q38_OK);
    return l;
}
/* Runs one whole clean command and returns its status. */
static Qwen38Status run(Qwen38GpuLedger *l, uint64_t start, uint64_t count) {
    Qwen38GpuTicket t; Qwen38Status s = qwen38_gpu_ledger_begin(l, &model_a, start, count, &t, NULL);
    if (s) return s;
    EXPECT(qwen38_gpu_ledger_submit(l, &t, NULL), Q38_OK);
    return qwen38_gpu_ledger_complete(l, &t, true, 0, NULL);
}

static void positions_and_capacity(void) {
    Qwen38GpuLedger l = fresh(8); Qwen38GpuTicket t;
    IS(l.state == Q38_SESSION_IDLE && l.epoch == 1);
    EXPECT(run(&l, 0, 4), Q38_OK); IS(l.state == Q38_SESSION_VALID && l.next_pos == 4);
    EXPECT(run(&l, 3, 1), Q38_BAD_STATE);                  /* must append at next_pos */
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 4, 5, &t, NULL), Q38_CAPACITY);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 4, UINT64_MAX, &t, NULL), Q38_CAPACITY);
    IS(l.state == Q38_SESSION_VALID && l.next_pos == 4 && !t.epoch);   /* refusal changes nothing */
    EXPECT(run(&l, 4, 4), Q38_OK); IS(l.next_pos == 8);
}
static void zero_tokens_publish_nothing(void) {
    Qwen38GpuLedger l = fresh(8);
    EXPECT(run(&l, 0, 0), Q38_OK); IS(l.state == Q38_SESSION_IDLE && l.next_pos == 0);
    EXPECT(run(&l, 0, 3), Q38_OK); EXPECT(run(&l, 3, 0), Q38_OK);
    IS(l.state == Q38_SESSION_VALID && l.next_pos == 3);
}
static void failure_before_submit(void) {
    Qwen38GpuLedger l = fresh(8); Qwen38GpuTicket t;
    EXPECT(run(&l, 0, 2), Q38_OK);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 2, 2, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_abort(&l, &t, false, NULL), Q38_OK);   /* nothing was touched */
    IS(l.state == Q38_SESSION_VALID && l.next_pos == 2); EXPECT(run(&l, 2, 2), Q38_OK);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 4, 1, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_abort(&l, &t, true, NULL), Q38_OK);    /* something was touched */
    IS(l.state == Q38_SESSION_INVALID && !qwen38_gpu_ledger_info(&l).valid);
    EXPECT(run(&l, 4, 1), Q38_BAD_STATE);
    EXPECT(qwen38_gpu_ledger_reset(&l, &model_a, NULL), Q38_OK);    /* replay from zero */
    IS(l.state == Q38_SESSION_IDLE && l.epoch == 2 && l.next_pos == 0);
    EXPECT(run(&l, 0, 4), Q38_OK);
}
static void kernel_error_after_writes(void) {
    const struct { bool ok; uint32_t bits; } bad[] = {
        {true, Q38_GPU_NONFINITE}, {true, Q38_GPU_BOUNDS | Q38_GPU_STATE}, {false, 0}};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        Qwen38GpuLedger l = fresh(8); Qwen38GpuTicket t; Qwen38Error e;
        EXPECT(run(&l, 0, 2), Q38_OK);
        EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 2, 3, &t, NULL), Q38_OK);
        EXPECT(qwen38_gpu_ledger_submit(&l, &t, NULL), Q38_OK);
        EXPECT(qwen38_gpu_ledger_complete(&l, &t, bad[i].ok, bad[i].bits, &e), Q38_GPU_ERROR);
        IS(e.code == Q38_GPU_ERROR && strstr(e.detail, "error_bits"));
        IS(l.state == Q38_SESSION_INVALID && l.next_pos == 2);      /* nothing published */
    }
}
static void tickets_and_owners(void) {
    Qwen38GpuLedger l = fresh(8); Qwen38GpuTicket t, old, forged;
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_b, 0, 1, &t, NULL), Q38_OWNER);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 0, 2, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 0, 2, &forged, NULL), Q38_BUSY);
    EXPECT(qwen38_gpu_ledger_reset(&l, &model_a, NULL), Q38_BUSY);
    EXPECT(qwen38_gpu_ledger_complete(&l, &t, true, 0, NULL), Q38_BAD_STATE);   /* not submitted */
    forged = t; forged.count = 3; EXPECT(qwen38_gpu_ledger_submit(&l, &forged, NULL), Q38_BAD_STATE);
    forged = t; forged.model = &model_b; EXPECT(qwen38_gpu_ledger_submit(&l, &forged, NULL), Q38_OWNER);
    EXPECT(qwen38_gpu_ledger_submit(&l, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_submit(&l, &t, NULL), Q38_BAD_STATE);              /* double submit */
    EXPECT(qwen38_gpu_ledger_abort(&l, &t, false, NULL), Q38_BAD_STATE);        /* too late */
    EXPECT(qwen38_gpu_ledger_reset(&l, &model_a, NULL), Q38_BUSY);
    EXPECT(qwen38_gpu_ledger_complete(&l, &t, true, 0, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_complete(&l, &t, true, 0, NULL), Q38_BAD_STATE);   /* double finish */
    IS(l.next_pos == 2);                                                         /* published once */
    old = t; EXPECT(qwen38_gpu_ledger_reset(&l, &model_b, NULL), Q38_OWNER);
    EXPECT(qwen38_gpu_ledger_reset(&l, &model_a, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_begin(&l, &model_a, 0, 2, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_ledger_submit(&l, &old, NULL), Q38_STALE);                /* previous epoch */
}
static void epoch_and_arguments(void) {
    Qwen38GpuLedger l = fresh(8); Qwen38GpuTicket t;
    l.epoch = UINT64_MAX; EXPECT(qwen38_gpu_ledger_reset(&l, &model_a, NULL), Q38_OVERFLOW);
    IS(l.epoch == UINT64_MAX);
    EXPECT(qwen38_gpu_ledger_init(&l, NULL, 8, NULL), Q38_INVALID_ARGUMENT);
    EXPECT(qwen38_gpu_ledger_init(&l, &model_a, 0, NULL), Q38_INVALID_ARGUMENT);
    EXPECT(qwen38_gpu_ledger_begin(NULL, &model_a, 0, 1, &t, NULL), Q38_INVALID_ARGUMENT);
    IS(qwen38_gpu_ledger_info(NULL).key.epoch == 0);
}

/* ---- Metal session -------------------------------------------------- */
/* Writes 42+i to every element, THEN (if asked) ORs flag into the error word. */
static NSString *const kSource = @"#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void write_then_flag(device uint *state [[buffer(0)]],"
    " device atomic_uint *err [[buffer(1)]], constant uint &flag [[buffer(2)]],"
    " uint i [[thread_position_in_grid]]) {\n"
    "  state[i] = 42u + i;\n"
    "  if (flag && i == 0) atomic_fetch_or_explicit(err, flag, memory_order_relaxed);\n}\n";
static id<MTLDevice> dev;
static id<MTLComputePipelineState> pso;
static id<MTLBuffer> state_buf;   /* 4 x uint, Shared, so the test can read what the GPU wrote */
static uint32_t *state(void) { return (uint32_t *)state_buf.contents; }
static uint32_t word_value(Qwen38GpuCommand c) {
    return *(const uint32_t *)((id<MTLBuffer>)(__bridge id)c.error_word.buffer).contents;
}
static void encode(Qwen38GpuCommand c, uint32_t flag) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c.compute_encoder;
    [enc setComputePipelineState:pso];
    [enc setBuffer:state_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)c.error_word.buffer offset:c.error_word.offset_bytes atIndex:1];
    [enc setBytes:&flag length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
}
static Qwen38GpuSession *session(void) {
    Qwen38GpuSession *s; EXPECT(qwen38_gpu_session_create((__bridge void *)dev, &model_a, 16, &s, NULL), Q38_OK);
    return s;
}
/* One full command; returns the status of finish(). */
static Qwen38Status gpu_run(Qwen38GpuSession *s, uint64_t start, uint64_t count, uint32_t flag, Qwen38Error *e) {
    Qwen38GpuTicket t; Qwen38GpuCommand c;
    EXPECT(qwen38_gpu_session_begin(s, &model_a, start, count, &t, &c, NULL), Q38_OK);
    IS(word_value(c) == 0);                                    /* clean even right after a failed command */
    encode(c, flag);
    EXPECT(qwen38_gpu_session_submit(s, &t, NULL), Q38_OK);
    return qwen38_gpu_session_finish(s, &t, e);
}

static void session_publishes_only_clean_commands(void) {
    Qwen38GpuSession *s = session(); Qwen38Error e;
    memset(state(), 0, 16);
    EXPECT(gpu_run(s, 0, 4, 0, NULL), Q38_OK);
    IS(qwen38_gpu_session_info(s).next_pos == 4 && state()[3] == 45);
    memset(state(), 0, 16);                                    /* the next command fails after writing */
    EXPECT(gpu_run(s, 4, 4, Q38_GPU_NONFINITE, &e), Q38_GPU_ERROR);
    IS(state()[0] == 42 && state()[3] == 45);                  /* the GPU did write ... */
    IS(!qwen38_gpu_session_info(s).valid && qwen38_gpu_session_info(s).next_pos == 4);   /* ... nothing published */
    IS(strstr(e.detail, "error_bits=0x1"));
    Qwen38GpuTicket t; Qwen38GpuCommand c;
    EXPECT(qwen38_gpu_session_begin(s, &model_a, 4, 1, &t, &c, NULL), Q38_BAD_STATE);
    EXPECT(qwen38_gpu_session_reset(s, &model_a, NULL), Q38_OK);
    EXPECT(gpu_run(s, 0, 4, 0, NULL), Q38_OK);                 /* replay: the old bit must not survive begin() */
    IS(qwen38_gpu_session_info(s).key.epoch == 2 && qwen38_gpu_session_info(s).next_pos == 4);
    EXPECT(gpu_run(s, 4, 0, 0, NULL), Q38_OK);                 /* T=0: admitted, nothing published */
    IS(qwen38_gpu_session_info(s).next_pos == 4);
    EXPECT(qwen38_gpu_session_destroy(s, NULL), Q38_OK);
}
static void session_failure_before_submit(void) {
    Qwen38GpuSession *s = session(); Qwen38GpuTicket t; Qwen38GpuCommand c;
    EXPECT(gpu_run(s, 0, 2, 0, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_begin(s, &model_a, 2, 2, &t, &c, NULL), Q38_OK);
    encode(c, Q38_GPU_BOUNDS);                                 /* recorded but never committed */
    EXPECT(qwen38_gpu_session_abort(s, &t, false, NULL), Q38_OK);
    IS(qwen38_gpu_session_info(s).valid && !qwen38_gpu_session_info(s).busy);
    EXPECT(qwen38_gpu_session_submit(s, &t, NULL), Q38_BAD_STATE);   /* aborted ticket is dead */
    EXPECT(gpu_run(s, 2, 2, 0, NULL), Q38_OK);                 /* no leaked bit, no leaked encoder */
    EXPECT(qwen38_gpu_session_begin(s, &model_a, 4, 1, &t, &c, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_abort(s, &t, true, NULL), Q38_OK);
    IS(!qwen38_gpu_session_info(s).valid);
    EXPECT(qwen38_gpu_session_destroy(s, NULL), Q38_OK);
}
static void session_tickets_busy_and_owner(void) {
    Qwen38GpuSession *s = session(); Qwen38GpuTicket t, old; Qwen38GpuCommand c;
    EXPECT(qwen38_gpu_session_begin(s, &model_b, 0, 1, &t, &c, NULL), Q38_OWNER);
    IS(!c.compute_encoder && !t.epoch);
    EXPECT(qwen38_gpu_session_begin(s, &model_a, 0, 4, &t, &c, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_finish(s, &t, NULL), Q38_BAD_STATE);        /* before submit: must not hang */
    EXPECT(qwen38_gpu_session_destroy(s, NULL), Q38_BUSY);
    EXPECT(qwen38_gpu_session_reset(s, &model_a, NULL), Q38_BUSY);
    encode(c, 0);
    EXPECT(qwen38_gpu_session_submit(s, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_submit(s, &t, NULL), Q38_BAD_STATE);        /* never committed twice */
    EXPECT(qwen38_gpu_session_destroy(s, NULL), Q38_BUSY);                /* in flight: still alive */
    EXPECT(qwen38_gpu_session_finish(s, &t, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_finish(s, &t, NULL), Q38_BAD_STATE);
    IS(qwen38_gpu_session_info(s).next_pos == 4);                         /* published once */
    old = t; EXPECT(qwen38_gpu_session_reset(s, &model_a, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_submit(s, &old, NULL), Q38_STALE);
    EXPECT(qwen38_gpu_session_abort(s, &t, false, NULL), Q38_OK);
    EXPECT(qwen38_gpu_session_destroy(s, NULL), Q38_OK);
}

int main(void) {
    const char *lock = getenv("PICOFORGE_GPU_LOCK_FD");
    if (!lock || fcntl(atoi(lock), F_GETFD) < 0) {
        fprintf(stderr, "run through tools/qwen38_gpu_lock.py; inherited lock fd required\n"); return 2;
    }
    positions_and_capacity(); zero_tokens_publish_nothing(); failure_before_submit();
    kernel_error_after_writes(); tickets_and_owners(); epoch_and_arguments();
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice(); IS(dev);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"write_then_flag"] error:&err];
        state_buf = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared]; IS(pso && state_buf);
        session_publishes_only_clean_commands(); session_failure_before_submit();
        session_tickets_busy_and_owner();
    }
    puts("qwen38 gpu lifecycle: 6 ledger + 3 session groups passed");
    return 0;
}
