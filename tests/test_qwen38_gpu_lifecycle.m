/* G0 ledger rules, CPU only: no Metal device is touched, so no GPU lock. */
#include "qwen38_gpu.h"
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

int main(void) {
    positions_and_capacity(); zero_tokens_publish_nothing(); failure_before_submit();
    kernel_error_after_writes(); tickets_and_owners(); epoch_and_arguments();
    puts("qwen38 gpu ledger: 6 groups passed");
    return 0;
}
