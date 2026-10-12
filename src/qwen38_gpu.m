/* G0: the common GPU session. This part is the ledger, plain bookkeeping with
 * no Metal calls; the queue, command buffer and error word join it next. */
#include "qwen38_gpu.h"
#include <stdio.h>
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
