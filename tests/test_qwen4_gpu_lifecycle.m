/* G0 lifecycle: ledger rules, then the Metal session. Run only through
 * qwen4_gpu_lock.py (make test-qwen4-g0); a missing lock is a failure, not a skip. */
#import <Metal/Metal.h>
#include "qwen4_gpu.h"
#include "qwen4_ref.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#define EXPECT(call, want) do { Qwen4Status got_ = (call); \
    if (got_ != (want)) { fprintf(stderr, "line %d: %s -> %d, want %d\n", \
        __LINE__, #call, got_, (int)(want)); abort(); } } while (0)
#define IS(cond) do { if (!(cond)) { fprintf(stderr, "line %d: %s\n", __LINE__, #cond); abort(); } } while (0)

static int model_a, model_b;   /* only their addresses matter */
static Qwen4GpuLedger fresh(uint64_t capacity) {
    Qwen4GpuLedger l; EXPECT(qwen4_gpu_ledger_init(&l, &model_a, capacity, NULL), QW4_OK);
    return l;
}
/* Runs one whole clean command and returns its status. */
static Qwen4Status run(Qwen4GpuLedger *l, uint64_t start, uint64_t count) {
    Qwen4GpuTicket t; Qwen4Status s = qwen4_gpu_ledger_begin(l, &model_a, start, count, &t, NULL);
    if (s) return s;
    EXPECT(qwen4_gpu_ledger_submit(l, &t, NULL), QW4_OK);
    return qwen4_gpu_ledger_complete(l, &t, true, 0, NULL);
}

static void positions_and_capacity(void) {
    Qwen4GpuLedger l = fresh(8); Qwen4GpuTicket t;
    IS(l.state == QW4_SESSION_IDLE && l.epoch == 1);
    EXPECT(run(&l, 0, 4), QW4_OK); IS(l.state == QW4_SESSION_VALID && l.next_pos == 4);
    EXPECT(run(&l, 3, 1), QW4_BAD_STATE);                  /* must append at next_pos */
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 4, 5, &t, NULL), QW4_CAPACITY);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 4, UINT64_MAX, &t, NULL), QW4_CAPACITY);
    IS(l.state == QW4_SESSION_VALID && l.next_pos == 4 && !t.epoch);   /* refusal changes nothing */
    EXPECT(run(&l, 4, 4), QW4_OK); IS(l.next_pos == 8);
}
static void zero_tokens_publish_nothing(void) {
    Qwen4GpuLedger l = fresh(8);
    EXPECT(run(&l, 0, 0), QW4_OK); IS(l.state == QW4_SESSION_IDLE && l.next_pos == 0);
    EXPECT(run(&l, 0, 3), QW4_OK); EXPECT(run(&l, 3, 0), QW4_OK);
    IS(l.state == QW4_SESSION_VALID && l.next_pos == 3);
}
static void failure_before_submit(void) {
    Qwen4GpuLedger l = fresh(8); Qwen4GpuTicket t;
    EXPECT(run(&l, 0, 2), QW4_OK);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 2, 2, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_abort(&l, &t, false, NULL), QW4_OK);   /* nothing was touched */
    IS(l.state == QW4_SESSION_VALID && l.next_pos == 2); EXPECT(run(&l, 2, 2), QW4_OK);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 4, 1, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_abort(&l, &t, true, NULL), QW4_OK);    /* something was touched */
    IS(l.state == QW4_SESSION_INVALID && !qwen4_gpu_ledger_info(&l).valid);
    EXPECT(run(&l, 4, 1), QW4_BAD_STATE);
    EXPECT(qwen4_gpu_ledger_reset(&l, &model_a, NULL), QW4_OK);    /* replay from zero */
    IS(l.state == QW4_SESSION_IDLE && l.epoch == 2 && l.next_pos == 0);
    EXPECT(run(&l, 0, 4), QW4_OK);
}
static void kernel_error_after_writes(void) {
    const struct { bool ok; uint32_t bits; } bad[] = {
        {true, QW4_GPU_NONFINITE}, {true, QW4_GPU_BOUNDS | QW4_GPU_STATE}, {false, 0}};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        Qwen4GpuLedger l = fresh(8); Qwen4GpuTicket t; Qwen4Error e;
        EXPECT(run(&l, 0, 2), QW4_OK);
        EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 2, 3, &t, NULL), QW4_OK);
        EXPECT(qwen4_gpu_ledger_submit(&l, &t, NULL), QW4_OK);
        EXPECT(qwen4_gpu_ledger_complete(&l, &t, bad[i].ok, bad[i].bits, &e), QW4_GPU_ERROR);
        IS(e.code == QW4_GPU_ERROR && strstr(e.detail, "error_bits"));
        IS(l.state == QW4_SESSION_INVALID && l.next_pos == 2);      /* nothing published */
    }
}
static void tickets_and_owners(void) {
    Qwen4GpuLedger l = fresh(8); Qwen4GpuTicket t, old, forged;
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_b, 0, 1, &t, NULL), QW4_OWNER);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 0, 2, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 0, 2, &forged, NULL), QW4_BUSY);
    EXPECT(qwen4_gpu_ledger_reset(&l, &model_a, NULL), QW4_BUSY);
    EXPECT(qwen4_gpu_ledger_complete(&l, &t, true, 0, NULL), QW4_BAD_STATE);   /* not submitted */
    forged = t; forged.count = 3; EXPECT(qwen4_gpu_ledger_submit(&l, &forged, NULL), QW4_BAD_STATE);
    forged = t; forged.model = &model_b; EXPECT(qwen4_gpu_ledger_submit(&l, &forged, NULL), QW4_OWNER);
    EXPECT(qwen4_gpu_ledger_submit(&l, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_submit(&l, &t, NULL), QW4_BAD_STATE);              /* double submit */
    EXPECT(qwen4_gpu_ledger_abort(&l, &t, false, NULL), QW4_BAD_STATE);        /* too late */
    EXPECT(qwen4_gpu_ledger_reset(&l, &model_a, NULL), QW4_BUSY);
    EXPECT(qwen4_gpu_ledger_complete(&l, &t, true, 0, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_complete(&l, &t, true, 0, NULL), QW4_BAD_STATE);   /* double finish */
    IS(l.next_pos == 2);                                                         /* published once */
    old = t; EXPECT(qwen4_gpu_ledger_reset(&l, &model_b, NULL), QW4_OWNER);
    EXPECT(qwen4_gpu_ledger_reset(&l, &model_a, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_begin(&l, &model_a, 0, 2, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_ledger_submit(&l, &old, NULL), QW4_STALE);                /* previous epoch */
}
static void epoch_and_arguments(void) {
    Qwen4GpuLedger l = fresh(8); Qwen4GpuTicket t;
    l.epoch = UINT64_MAX; EXPECT(qwen4_gpu_ledger_reset(&l, &model_a, NULL), QW4_OVERFLOW);
    IS(l.epoch == UINT64_MAX);
    EXPECT(qwen4_gpu_ledger_init(&l, NULL, 8, NULL), QW4_INVALID_ARGUMENT);
    EXPECT(qwen4_gpu_ledger_init(&l, &model_a, 0, NULL), QW4_INVALID_ARGUMENT);
    EXPECT(qwen4_gpu_ledger_begin(NULL, &model_a, 0, 1, &t, NULL), QW4_INVALID_ARGUMENT);
    IS(qwen4_gpu_ledger_info(NULL).key.epoch == 0);
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
static uint32_t word_value(Qwen4GpuCommand c) {
    return *(const uint32_t *)((id<MTLBuffer>)(__bridge id)c.error_word.buffer).contents;
}
static void encode(Qwen4GpuCommand c, uint32_t flag) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c.compute_encoder;
    [enc setComputePipelineState:pso];
    [enc setBuffer:state_buf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)c.error_word.buffer offset:c.error_word.offset_bytes atIndex:1];
    [enc setBytes:&flag length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
}
static Qwen4GpuSession *session(void) {
    Qwen4GpuSession *s; EXPECT(qwen4_gpu_session_create((__bridge void *)dev, &model_a, 16, &s, NULL), QW4_OK);
    return s;
}
/* One full command; returns the status of finish(). */
static Qwen4Status gpu_run(Qwen4GpuSession *s, uint64_t start, uint64_t count, uint32_t flag, Qwen4Error *e) {
    Qwen4GpuTicket t; Qwen4GpuCommand c;
    EXPECT(qwen4_gpu_session_begin(s, &model_a, start, count, &t, &c, NULL), QW4_OK);
    IS(word_value(c) == 0);                                    /* clean even right after a failed command */
    encode(c, flag);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    return qwen4_gpu_session_finish(s, &t, e);
}

static void session_publishes_only_clean_commands(void) {
    Qwen4GpuSession *s = session(); Qwen4Error e;
    memset(state(), 0, 16);
    EXPECT(gpu_run(s, 0, 4, 0, NULL), QW4_OK);
    IS(qwen4_gpu_session_info(s).next_pos == 4 && state()[3] == 45);
    memset(state(), 0, 16);                                    /* the next command fails after writing */
    EXPECT(gpu_run(s, 4, 4, QW4_GPU_NONFINITE, &e), QW4_GPU_ERROR);
    IS(state()[0] == 42 && state()[3] == 45);                  /* the GPU did write ... */
    IS(!qwen4_gpu_session_info(s).valid && qwen4_gpu_session_info(s).next_pos == 4);   /* ... nothing published */
    IS(strstr(e.detail, "error_bits=0x1"));
    Qwen4GpuTicket t; Qwen4GpuCommand c;
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 4, 1, &t, &c, NULL), QW4_BAD_STATE);
    EXPECT(qwen4_gpu_session_reset(s, &model_a, NULL), QW4_OK);
    EXPECT(gpu_run(s, 0, 4, 0, NULL), QW4_OK);                 /* replay: the old bit must not survive begin() */
    IS(qwen4_gpu_session_info(s).key.epoch == 2 && qwen4_gpu_session_info(s).next_pos == 4);
    EXPECT(gpu_run(s, 4, 0, 0, NULL), QW4_OK);                 /* T=0: admitted, nothing published */
    IS(qwen4_gpu_session_info(s).next_pos == 4);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}
/* R1.6b: a concurrent session hands out both words and reports a bit set only in word 1. */
static void session_reports_second_word(void) {
    Qwen4GpuSession *s = session(); Qwen4Error e; Qwen4GpuTicket t; Qwen4GpuCommand c;
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 4, &t, &c, NULL), QW4_OK);
    IS(c.concurrent == !qw4_ref_kernels("barrier") && c.error_word.bytes == (c.concurrent ? 8u : 4u));
    if (c.concurrent) c.error_word.offset_bytes += 4;            /* write only word 1, as a check does */
    encode(c, QW4_GPU_NONFINITE);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_finish(s, &t, &e), QW4_GPU_ERROR);
    IS(strstr(e.detail, "error_bits=0x1"));
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}
static void session_failure_before_submit(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t; Qwen4GpuCommand c;
    EXPECT(gpu_run(s, 0, 2, 0, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 2, 2, &t, &c, NULL), QW4_OK);
    encode(c, QW4_GPU_BOUNDS);                                 /* recorded but never committed */
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    IS(qwen4_gpu_session_info(s).valid && !qwen4_gpu_session_info(s).busy);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_BAD_STATE);   /* aborted ticket is dead */
    EXPECT(gpu_run(s, 2, 2, 0, NULL), QW4_OK);                 /* no leaked bit, no leaked encoder */
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 4, 1, &t, &c, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_abort(s, &t, true, NULL), QW4_OK);
    IS(!qwen4_gpu_session_info(s).valid);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}
static void session_tickets_busy_and_owner(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t, old; Qwen4GpuCommand c;
    EXPECT(qwen4_gpu_session_begin(s, &model_b, 0, 1, &t, &c, NULL), QW4_OWNER);
    IS(!c.compute_encoder && !t.epoch);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 4, &t, &c, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_finish(s, &t, NULL), QW4_BAD_STATE);        /* before submit: must not hang */
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_BUSY);
    EXPECT(qwen4_gpu_session_reset(s, &model_a, NULL), QW4_BUSY);
    encode(c, 0);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_BAD_STATE);        /* never committed twice */
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_BUSY);                /* in flight: still alive */
    EXPECT(qwen4_gpu_session_finish(s, &t, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_finish(s, &t, NULL), QW4_BAD_STATE);
    IS(qwen4_gpu_session_info(s).next_pos == 4);                         /* published once */
    old = t; EXPECT(qwen4_gpu_session_reset(s, &model_a, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_submit(s, &old, NULL), QW4_STALE);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}

/* ---- claims: alias, reuse and lifetime ------------------------------ */
static Qwen4GpuSpan span(id<MTLBuffer> b, size_t offset, size_t bytes) {
    return (Qwen4GpuSpan){(__bridge void *)b, offset, bytes};
}
static id<MTLBuffer> no_copy(void *p, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:p length:bytes
        options:MTLResourceStorageModeShared deallocator:nil];
    IS(b); return b;
}
#define CLAIM(s, t, sp, use, want) EXPECT(qwen4_gpu_session_claim(s, &t, sp, use, NULL), want)
static void session_claims_refuse_aliases(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t, old; Qwen4GpuCommand c;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *mem; IS(posix_memalign(&mem, page, 4 * page) == 0);
    /* Three separate Metal objects over one allocation: weights, an overlapping
     * view of it, and a disjoint tail. Only the address can reveal the overlap. */
    id<MTLBuffer> weights = no_copy(mem, 2 * page), overlap = no_copy((char *)mem + page, 2 * page);
    id<MTLBuffer> tail = no_copy((char *)mem + 3 * page, page);
    id<MTLBuffer> priv = [dev newBufferWithLength:1024 options:MTLResourceStorageModePrivate];
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, t, span(weights, 0, 2 * page), QW4_GPU_USE_READ, QW4_OK);
    CLAIM(s, t, span(overlap, 0, page), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);   /* scratch over weights */
    CLAIM(s, t, span(overlap, 0, 2 * page), QW4_GPU_USE_READ, QW4_OK);              /* shared read-only */
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_OK);
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);      /* scratch reused in-command */
    CLAIM(s, t, span(tail, 0, 64), QW4_GPU_USE_READ, QW4_INVALID_ARGUMENT);         /* reading what is written */
    Qwen4GpuSpan word = c.error_word;                                              /* the error word itself */
    CLAIM(s, t, word, QW4_GPU_USE_READ, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, word, QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, span(priv, 0, 512), QW4_GPU_USE_WRITE, QW4_OK);                     /* private: by offset */
    CLAIM(s, t, span(priv, 256, 512), QW4_GPU_USE_READ, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, span(priv, 512, 512), QW4_GPU_USE_WRITE, QW4_OK);
    CLAIM(s, t, span(tail, 2, 4), QW4_GPU_USE_READ, QW4_STRIDE);
    CLAIM(s, t, span(tail, page, 4), QW4_GPU_USE_READ, QW4_CAPACITY);
    CLAIM(s, t, span(tail, 0, page + 4), QW4_GPU_USE_READ, QW4_CAPACITY);
    CLAIM(s, t, span(tail, 0, 0), QW4_GPU_USE_READ, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, ((Qwen4GpuSpan){0}), QW4_GPU_USE_READ, QW4_INVALID_ARGUMENT);
    old = t;
    encode(c, 0);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    CLAIM(s, t, span(tail, 0, 64), QW4_GPU_USE_READ, QW4_BAD_STATE);                /* in flight: no new claims */
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 1, 1, &t, &c, NULL), QW4_BUSY);
    EXPECT(qwen4_gpu_session_finish(s, &old, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 1, 1, &t, &c, NULL), QW4_OK);      /* claims were released */
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_OK);                    /* same scratch, new command */
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 1, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_OK);                    /* abort released them too */
    CLAIM(s, old, span(priv, 0, 64), QW4_GPU_USE_READ, QW4_BAD_STATE);              /* finished ticket is dead */
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_reset(s, &model_a, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, old, span(tail, 0, 64), QW4_GPU_USE_READ, QW4_STALE);                  /* previous epoch */
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
    free(mem);
}
static void session_retains_buffers_until_completion(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t; Qwen4GpuCommand c;
    __weak id<MTLBuffer> weak;
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    @autoreleasepool {      /* the caller's only reference dies at the end of this block */
        id<MTLBuffer> scratch = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        weak = scratch;
        CLAIM(s, t, span(scratch, 0, 64), QW4_GPU_USE_WRITE, QW4_OK);
    }
    IS(weak);                                                   /* kept alive by the session */
    encode(c, 0);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    IS(weak);                                                   /* ... and while the GPU may use it */
    EXPECT(qwen4_gpu_session_finish(s, &t, NULL), QW4_OK);
    IS(!weak);                                                  /* released at completion */
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}

/* ---- R1.7a: read claims registered once per session ------------------ */
#define PIN(s, t, sp, want) EXPECT(qwen4_gpu_session_pin_read(s, &t, sp, NULL), want)
static void session_pinned_reads(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t, old; Qwen4GpuCommand c;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *mem; IS(posix_memalign(&mem, page, 8 * page) == 0);
    id<MTLBuffer> big = no_copy(mem, 8 * page);                        /* [0,8) pages */
    id<MTLBuffer> view = no_copy((char *)mem + 4 * page, 2 * page);    /* alias of pages [4,6) */
    id<MTLBuffer> tail = no_copy((char *)mem + 7 * page, page);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    if (!qwen4_gpu_session_pins_enabled(s)) {     /* QW4_REF_KERNELS=claim: today's path */
        PIN(s, t, span(big, 0, page), QW4_UNSUPPORTED);
        IS(qwen4_gpu_session_pin_count(s) == 0);
        EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
        EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK); free(mem); return;
    }
    /* A wide pin hides narrow ones behind it: only a running max of hi finds it. */
    PIN(s, t, span(big, 0, 6 * page), QW4_OK);
    PIN(s, t, span(big, page, page), QW4_OK);
    PIN(s, t, span(big, 2 * page, page), QW4_OK);                      /* READ/READ overlap allowed */
    PIN(s, t, span(big, 0, 0), QW4_INVALID_ARGUMENT);                   /* same checks as claim */
    PIN(s, t, span(big, 2, 4), QW4_STRIDE);
    PIN(s, t, span(tail, 0, 2 * page), QW4_CAPACITY);
    IS(qwen4_gpu_session_pin_count(s) == 3);
    CLAIM(s, t, span(big, 4 * page, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);   /* inside the wide pin */
    CLAIM(s, t, span(view, 0, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);         /* other object, same memory */
    CLAIM(s, t, span(big, 6 * page - 4, 8), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT); /* straddles the end */
    CLAIM(s, t, span(big, 6 * page, 64), QW4_GPU_USE_WRITE, QW4_OK);                  /* first byte after */
    CLAIM(s, t, span(view, 0, 2 * page), QW4_GPU_USE_READ, QW4_OK);                  /* READ over a pin */
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_OK);
    CLAIM(s, t, span(tail, 0, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);          /* per-command still works */
    PIN(s, t, span(tail, 0, page), QW4_INVALID_ARGUMENT);              /* pinning over a claimed WRITE */
    old = t; encode(c, 0);
    EXPECT(qwen4_gpu_session_submit(s, &t, NULL), QW4_OK);
    PIN(s, t, span(big, 0, page), QW4_BAD_STATE);                      /* in flight: no new pins */
    EXPECT(qwen4_gpu_session_finish(s, &old, NULL), QW4_OK);
    /* The pins outlive the command, an abort and a reset; the claims do not. */
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 1, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, t, span(big, 0, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, span(tail, 0, page), QW4_GPU_USE_WRITE, QW4_OK);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_reset(s, &model_a, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, t, span(big, 5 * page, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    PIN(s, old, span(big, 0, page), QW4_STALE);
    IS(qwen4_gpu_session_pin_count(s) == 3);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    /* Private buffers pin by offset within the buffer. */
    id<MTLBuffer> priv = [dev newBufferWithLength:1024 options:MTLResourceStorageModePrivate];
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    PIN(s, t, span(priv, 0, 512), QW4_OK);
    CLAIM(s, t, span(priv, 512, 512), QW4_GPU_USE_WRITE, QW4_OK);
    CLAIM(s, t, span(priv, 256, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    __weak id<MTLBuffer> weak;                      /* the session keeps a pinned buffer alive */
    @autoreleasepool {
        id<MTLBuffer> w = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared]; weak = w;
        EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
        PIN(s, t, span(w, 0, 64), QW4_OK);
        EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    }
    IS(weak);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
    IS(!weak);                                      /* ... and releases it at destroy */
    free(mem);
}
/* R1.7a: a failed pin pass is cleared, so a later command is fully covered. */
static void session_partial_pins_recover(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t; Qwen4GpuCommand c;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *mem; IS(posix_memalign(&mem, page, 4 * page) == 0);
    id<MTLBuffer> w = no_copy(mem, 4 * page);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    if (!qwen4_gpu_session_pins_enabled(s)) {
        EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
        EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK); free(mem); return;
    }
    PIN(s, t, span(w, 0, page), QW4_OK);                                /* first half of a pass ... */
    PIN(s, t, span(w, 2, 4), QW4_STRIDE);                               /* ... fails */
    qwen4_gpu_session_pins_clear(s);
    IS(qwen4_gpu_session_pin_count(s) == 0 && !qwen4_gpu_session_pins_sealed(s));
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    CLAIM(s, t, span(w, 0, 64), QW4_GPU_USE_WRITE, QW4_OK);             /* nothing stale is pinned */
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    PIN(s, t, span(w, 0, page), QW4_OK); PIN(s, t, span(w, page, page), QW4_OK);   /* the retry pass */
    qwen4_gpu_session_pins_seal(s);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    IS(qwen4_gpu_session_pins_sealed(s) && qwen4_gpu_session_pin_count(s) == 2);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);   /* sealed: fully covered */
    CLAIM(s, t, span(w, 0, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    CLAIM(s, t, span(w, page + 8, 64), QW4_GPU_USE_WRITE, QW4_INVALID_ARGUMENT);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK); free(mem);
}
/* R1.7b: a residency set on the queue changes nothing about a command's result. */
static void session_residency_set(void) {
    Qwen4GpuSession *s = session(); Qwen4GpuTicket t; Qwen4GpuCommand c;
    id<MTLBuffer> w = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    void *list[1] = {(__bridge void *)w};
    EXPECT(qwen4_gpu_session_use_residency(s, NULL, 1, 0, NULL), QW4_INVALID_ARGUMENT);
    EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
    EXPECT(qwen4_gpu_session_use_residency(s, list, 1, 0, NULL),
        qwen4_gpu_session_pins_enabled(s) ? QW4_BUSY : QW4_UNSUPPORTED);
    EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    Qwen4Status want = qwen4_gpu_session_pins_enabled(s) ? QW4_OK : QW4_UNSUPPORTED;
    EXPECT(qwen4_gpu_session_use_residency(s, list, 1, 0, NULL), want);
    if (want == QW4_OK) EXPECT(qwen4_gpu_session_use_residency(s, list, 1, 0, NULL), QW4_UNSUPPORTED);  /* once */
    memset(state(), 0, 16); EXPECT(gpu_run(s, 0, 4, 0, NULL), QW4_OK);   /* commands run with the set attached */
    IS(state()[3] == 45);
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK);
}
/* Encode cost of the claims of one token: N weight spans + 60 scratch buffers. */
static double claims_seconds(bool oracle, size_t n_weights) {
    setenv("QW4_REF_KERNELS", oracle ? "claim" : "0", 1);
    Qwen4GpuSession *s = session(); unsetenv("QW4_REF_KERNELS");
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *mem; IS(posix_memalign(&mem, page, n_weights * 64 + page) == 0);
    id<MTLBuffer> w = no_copy(mem, (n_weights * 64 + page) / page * page);
    NSMutableArray *scratch = [NSMutableArray array];
    for (int i = 0; i < 60; i++) [scratch addObject:[dev newBufferWithLength:4096 options:MTLResourceStorageModeShared]];
    Qwen4GpuTicket t; Qwen4GpuCommand c; double total = 0; const int tokens = 50;
    for (int k = 0; k < tokens; k++) {
        EXPECT(qwen4_gpu_session_begin(s, &model_a, 0, 1, &t, &c, NULL), QW4_OK);
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        for (id<MTLBuffer> x in scratch) CLAIM(s, t, span(x, 0, 4096), QW4_GPU_USE_WRITE, QW4_OK);
        bool pinned = qwen4_gpu_session_pins_enabled(s) && qwen4_gpu_session_pin_count(s) >= n_weights;
        for (size_t i = 0; i < n_weights && !pinned; i++) {
            if (qwen4_gpu_session_pins_enabled(s)) PIN(s, t, span(w, i * 64, 64), QW4_OK);
            else CLAIM(s, t, span(w, i * 64, 64), QW4_GPU_USE_READ, QW4_OK);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        if (k) total += (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec);
        EXPECT(qwen4_gpu_session_abort(s, &t, false, NULL), QW4_OK);
    }
    EXPECT(qwen4_gpu_session_destroy(s, NULL), QW4_OK); free(mem);
    return total / (tokens - 1);
}

int main(void) {
    const char *lock = getenv("PICOFORGE_GPU_LOCK_FD");
    if (!lock || fcntl(atoi(lock), F_GETFD) < 0) {
        fprintf(stderr, "run through tools/qwen4_gpu_lock.py; inherited lock fd required\n"); return 2;
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
        session_publishes_only_clean_commands(); session_reports_second_word(); session_failure_before_submit();
        session_tickets_busy_and_owner(); session_claims_refuse_aliases();
        session_retains_buffers_until_completion(); session_pinned_reads(); session_residency_set(); session_partial_pins_recover();
        if (getenv("QW4_CLAIM_BENCH"))
            printf("claims per token, 1500 weights + 60 scratch: per-command %.3f ms, pinned %.3f ms\n",
                1e3 * claims_seconds(true, 1500), 1e3 * claims_seconds(false, 1500));
    }
    puts("qwen4 gpu lifecycle: 6 ledger + 9 session groups passed");
    return 0;
}
