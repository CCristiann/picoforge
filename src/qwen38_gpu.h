/* C-facing Metal contract; no Objective-C types or implicit CPU pointers. */
#ifndef PICOFORGE_QWEN38_GPU_H
#define PICOFORGE_QWEN38_GPU_H
#include "qwen38_types.h"
typedef struct {
    void *buffer;                  /* borrowed MTLBuffer */
    size_t offset_bytes, bytes;     /* bytes is accessible length FROM offset */
} Qwen38GpuSpan;
typedef struct {
    Qwen38GpuSpan span;
    size_t rank, dim[QWEN38_MAX_RANK], stride_bytes[QWEN38_MAX_RANK];
} Qwen38GpuF32;
typedef struct {
    void *device, *compute_encoder; /* borrowed, same device as every resource */
    Qwen38GpuSpan error_word;       /* aligned uint32, zero before command */
} Qwen38GpuCommand;
enum {
    Q38_GPU_NONFINITE = 1u,
    Q38_GPU_BOUNDS = 2u,
    Q38_GPU_STATE = 4u
};
/* Host status means encoded, NOT completed. Kernel atomically ORs error bits.
 * Caller checks command-buffer error AND this word after completion. No
 * component commits/waits on the command buffer or resets this word itself. */

/* G0 ledger: the one place that decides whether a command may run and whether
 * its result may be published. Pure bookkeeping, no Metal types, no I/O. */
typedef enum {
    Q38_SESSION_IDLE,      /* fresh or just reset, next_pos == 0 */
    Q38_SESSION_ENCODING,  /* begin done, components are recording work */
    Q38_SESSION_IN_FLIGHT, /* submitted: the GPU owns every resource */
    Q38_SESSION_VALID,     /* last command completed cleanly, history published */
    Q38_SESSION_INVALID    /* a failure touched state; only reset recovers */
} Qwen38SessionState;
typedef struct {
    const void *model;     /* identity token, never dereferenced */
    uint64_t epoch, start_pos, count;
} Qwen38GpuTicket;
typedef struct {
    const void *model;
    uint64_t epoch, next_pos, capacity;
    Qwen38SessionState state, resume;  /* resume: where abort and T=0 return */
    Qwen38GpuTicket open;              /* the one command begin() admitted */
} Qwen38GpuLedger;

Qwen38Status qwen38_gpu_ledger_init(Qwen38GpuLedger *, const void *model,
    uint64_t capacity, Qwen38Error *);
/* start_pos must equal next_pos. count==0 is admitted and publishes nothing. */
Qwen38Status qwen38_gpu_ledger_begin(Qwen38GpuLedger *, const void *model,
    uint64_t start_pos, uint64_t count, Qwen38GpuTicket *, Qwen38Error *);
Qwen38Status qwen38_gpu_ledger_submit(Qwen38GpuLedger *, const Qwen38GpuTicket *,
    Qwen38Error *);
/* Before submit only. state_mutated=true poisons the session, else it is kept. */
Qwen38Status qwen38_gpu_ledger_abort(Qwen38GpuLedger *, const Qwen38GpuTicket *,
    bool state_mutated, Qwen38Error *);
/* Publishes next_pos only if the command buffer AND the error word are clean;
 * otherwise the whole session becomes INVALID and Q38_GPU_ERROR is returned. */
Qwen38Status qwen38_gpu_ledger_complete(Qwen38GpuLedger *, const Qwen38GpuTicket *,
    bool command_ok, uint32_t error_bits, Qwen38Error *);
/* Not while busy. New epoch, next_pos 0; every older ticket becomes STALE. */
Qwen38Status qwen38_gpu_ledger_reset(Qwen38GpuLedger *, const void *model,
    Qwen38Error *);
Qwen38StateInfo qwen38_gpu_ledger_info(const Qwen38GpuLedger *);
#endif
