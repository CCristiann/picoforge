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
    /* R1.6b, set only by the session: the encoder is concurrent (only explicit
     * barriers order dispatches) and error_word spans 8 bytes: word 0, read by
     * the kernels' early exits, then a second zeroed word that finish() ORs in.
     * Input checks may write word 1 instead: no kernel reads it, so a check
     * needs no barrier before an op that does not write what the check reads.
     * Then a failed check stops neither its op nor the later kernels: the
     * failure bits differ from the serial check+oracle pattern (deterministic,
     * bit 1 still set for non-finite inputs, but not a superset). The contract
     * is the status: Q38_GPU_ERROR and an INVALID session. */
    bool concurrent;
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
/* OK only if the ticket is exactly the open command and the ledger is in `want`:
 * OWNER for another model, STALE for an old epoch, BAD_STATE otherwise. */
Qwen38Status qwen38_gpu_ledger_check(const Qwen38GpuLedger *, const Qwen38GpuTicket *,
    Qwen38SessionState want, Qwen38Error *);

/* The session is the ledger plus what it protects: a command queue, one command
 * buffer per command and the shared error word. Components receive only the
 * Qwen38GpuCommand from begin() and encode into it; they never commit or wait. */
typedef struct Qwen38GpuSession Qwen38GpuSession;
/* device is a borrowed id<MTLDevice>; the session keeps its own reference. */
Qwen38Status qwen38_gpu_session_create(void *device, const void *model,
    uint64_t capacity, Qwen38GpuSession **, Qwen38Error *);
/* Admits the command, zeroes the error word and opens the encoder. */
Qwen38Status qwen38_gpu_session_begin(Qwen38GpuSession *, const void *model,
    uint64_t start_pos, uint64_t count, Qwen38GpuTicket *, Qwen38GpuCommand *,
    Qwen38Error *);
/* Every range a command touches is declared while it is ENCODING. The session
 * retains the buffer until finish/abort, and refuses any overlap with a WRITE
 * claim, including the error word and separate no-copy views of one memory.
 * READ/READ overlap is allowed (shared weights); in-place state claims once. */
typedef enum { Q38_GPU_USE_READ, Q38_GPU_USE_WRITE } Qwen38GpuUse;
Qwen38Status qwen38_gpu_session_claim(Qwen38GpuSession *, const Qwen38GpuTicket *,
    Qwen38GpuSpan, Qwen38GpuUse, Qwen38Error *);
/* R1.7a: a READ span registered once for the session's life (retained until
 * destroy, kept across commands and reset). Later WRITE claims overlapping it are
 * refused as before; the per-command cost is a binary search, not a scan. Like claim,
 * only while ENCODING. Q38_UNSUPPORTED when Q38_REF_KERNELS selects `claim`
 * (today's per-command claims). */
Qwen38Status qwen38_gpu_session_pin_read(Qwen38GpuSession *, const Qwen38GpuTicket *,
    Qwen38GpuSpan, Qwen38Error *);
/* R1.7b: attach a residency set of these MTLBuffers to the session's queue (not while
 * busy, once). Skipped with Q38_UNSUPPORTED (and a stderr line) unless buffers plus
 * headroom_bytes <= 85% of the recommended working set; also UNSUPPORTED on an old OS or
 * with the `claim` oracle. Callers ignore UNSUPPORTED: the session works without it.
 * share_residency puts an existing set (residency() of another session) on this queue. */
Qwen38Status qwen38_gpu_session_use_residency(Qwen38GpuSession *, void *const *buffers,
    size_t count, uint64_t headroom_bytes, Qwen38Error *);
void *qwen38_gpu_session_residency(const Qwen38GpuSession *);   /* borrowed, or NULL */
Qwen38Status qwen38_gpu_session_share_residency(Qwen38GpuSession *, void *set, Qwen38Error *);
/* Pins are all-or-nothing: seal a complete set; clear (and unseal) after a failed pass. */
void qwen38_gpu_session_pins_seal(Qwen38GpuSession *);
bool qwen38_gpu_session_pins_sealed(const Qwen38GpuSession *);
void qwen38_gpu_session_pins_clear(Qwen38GpuSession *);
bool qwen38_gpu_session_pins_enabled(const Qwen38GpuSession *);
size_t qwen38_gpu_session_pin_count(const Qwen38GpuSession *);
Qwen38Status qwen38_gpu_session_submit(Qwen38GpuSession *, const Qwen38GpuTicket *,
    Qwen38Error *);
Qwen38Status qwen38_gpu_session_abort(Qwen38GpuSession *, const Qwen38GpuTicket *,
    bool state_mutated, Qwen38Error *);
/* Waits, reads the Metal error AND the error word, then publishes or poisons. */
Qwen38Status qwen38_gpu_session_finish(Qwen38GpuSession *, const Qwen38GpuTicket *,
    Qwen38Error *);
Qwen38Status qwen38_gpu_session_reset(Qwen38GpuSession *, const void *model,
    Qwen38Error *);
Qwen38StateInfo qwen38_gpu_session_info(const Qwen38GpuSession *);
/* Last successful command's GPU timestamps; zero until available. This is
 * device execution time, not submit/wait wall time. */
double qwen38_gpu_session_seconds(const Qwen38GpuSession *);
/* BUSY leaves the session and every resource alive. */
Qwen38Status qwen38_gpu_session_destroy(Qwen38GpuSession *, Qwen38Error *);
#endif
