/* C-facing Metal contract; no Objective-C types or implicit CPU pointers. */
#ifndef PICOFORGE_QWEN4_GPU_H
#define PICOFORGE_QWEN4_GPU_H
#include "qwen4_types.h"
typedef struct {
    void *buffer;                  /* borrowed MTLBuffer */
    size_t offset_bytes, bytes;     /* bytes is accessible length FROM offset */
} Qwen4GpuSpan;
typedef struct {
    Qwen4GpuSpan span;
    size_t rank, dim[QWEN4_MAX_RANK], stride_bytes[QWEN4_MAX_RANK];
} Qwen4GpuF32;
typedef struct {
    void *device, *compute_encoder; /* borrowed, same device as every resource */
    Qwen4GpuSpan error_word;       /* aligned uint32, zero before command */
    /* R1.6b, set only by the session: the encoder is concurrent (only explicit
     * barriers order dispatches) and error_word spans 8 bytes: word 0, read by
     * the kernels' early exits, then a second zeroed word that finish() ORs in.
     * Input checks may write word 1 instead: no kernel reads it, so a check
     * needs no barrier before an op that does not write what the check reads.
     * Then a failed check stops neither its op nor the later kernels: the
     * failure bits differ from the serial check+oracle pattern (deterministic,
     * bit 1 still set for non-finite inputs, but not a superset). The contract
     * is the status: QW4_GPU_ERROR and an INVALID session. */
    bool concurrent;
} Qwen4GpuCommand;
enum {
    QW4_GPU_NONFINITE = 1u,
    QW4_GPU_BOUNDS = 2u,
    QW4_GPU_STATE = 4u
};
/* Host status means encoded, NOT completed. Kernel atomically ORs error bits.
 * Caller checks command-buffer error AND this word after completion. No
 * component commits/waits on the command buffer or resets this word itself. */

/* G0 ledger: the one place that decides whether a command may run and whether
 * its result may be published. Pure bookkeeping, no Metal types, no I/O. */
typedef enum {
    QW4_SESSION_IDLE,      /* fresh or just reset, next_pos == 0 */
    QW4_SESSION_ENCODING,  /* begin done, components are recording work */
    QW4_SESSION_IN_FLIGHT, /* submitted: the GPU owns every resource */
    QW4_SESSION_VALID,     /* last command completed cleanly, history published */
    QW4_SESSION_INVALID    /* a failure touched state; only reset recovers */
} Qwen4SessionState;
typedef struct {
    const void *model;     /* identity token, never dereferenced */
    uint64_t epoch, start_pos, count;
} Qwen4GpuTicket;
typedef struct {
    const void *model;
    uint64_t epoch, next_pos, capacity;
    Qwen4SessionState state, resume;  /* resume: where abort and T=0 return */
    Qwen4GpuTicket open;              /* the one command begin() admitted */
} Qwen4GpuLedger;

Qwen4Status qwen4_gpu_ledger_init(Qwen4GpuLedger *, const void *model,
    uint64_t capacity, Qwen4Error *);
/* start_pos must equal next_pos. count==0 is admitted and publishes nothing. */
Qwen4Status qwen4_gpu_ledger_begin(Qwen4GpuLedger *, const void *model,
    uint64_t start_pos, uint64_t count, Qwen4GpuTicket *, Qwen4Error *);
Qwen4Status qwen4_gpu_ledger_submit(Qwen4GpuLedger *, const Qwen4GpuTicket *,
    Qwen4Error *);
/* Before submit only. state_mutated=true poisons the session, else it is kept. */
Qwen4Status qwen4_gpu_ledger_abort(Qwen4GpuLedger *, const Qwen4GpuTicket *,
    bool state_mutated, Qwen4Error *);
/* Publishes next_pos only if the command buffer AND the error word are clean;
 * otherwise the whole session becomes INVALID and QW4_GPU_ERROR is returned. */
Qwen4Status qwen4_gpu_ledger_complete(Qwen4GpuLedger *, const Qwen4GpuTicket *,
    bool command_ok, uint32_t error_bits, Qwen4Error *);
/* Not while busy. New epoch, next_pos 0; every older ticket becomes STALE. */
Qwen4Status qwen4_gpu_ledger_reset(Qwen4GpuLedger *, const void *model,
    Qwen4Error *);
Qwen4StateInfo qwen4_gpu_ledger_info(const Qwen4GpuLedger *);
/* OK only if the ticket is exactly the open command and the ledger is in `want`:
 * OWNER for another model, STALE for an old epoch, BAD_STATE otherwise. */
Qwen4Status qwen4_gpu_ledger_check(const Qwen4GpuLedger *, const Qwen4GpuTicket *,
    Qwen4SessionState want, Qwen4Error *);

/* The session is the ledger plus what it protects: a command queue, one command
 * buffer per command and the shared error word. Components receive only the
 * Qwen4GpuCommand from begin() and encode into it; they never commit or wait. */
typedef struct Qwen4GpuSession Qwen4GpuSession;
/* device is a borrowed id<MTLDevice>; the session keeps its own reference. */
Qwen4Status qwen4_gpu_session_create(void *device, const void *model,
    uint64_t capacity, Qwen4GpuSession **, Qwen4Error *);
/* Admits the command, zeroes the error word and opens the encoder. */
Qwen4Status qwen4_gpu_session_begin(Qwen4GpuSession *, const void *model,
    uint64_t start_pos, uint64_t count, Qwen4GpuTicket *, Qwen4GpuCommand *,
    Qwen4Error *);
/* Every range a command touches is declared while it is ENCODING. The session
 * retains the buffer until finish/abort, and refuses any overlap with a WRITE
 * claim, including the error word and separate no-copy views of one memory.
 * READ/READ overlap is allowed (shared weights); in-place state claims once. */
typedef enum { QW4_GPU_USE_READ, QW4_GPU_USE_WRITE } Qwen4GpuUse;
Qwen4Status qwen4_gpu_session_claim(Qwen4GpuSession *, const Qwen4GpuTicket *,
    Qwen4GpuSpan, Qwen4GpuUse, Qwen4Error *);
/* R1.7a: a READ span registered once for the session's life (retained until
 * destroy, kept across commands and reset). Later WRITE claims overlapping it are
 * refused as before; the per-command cost is a binary search, not a scan. Like claim,
 * only while ENCODING. QW4_UNSUPPORTED when QW4_REF_KERNELS selects `claim`
 * (today's per-command claims). */
Qwen4Status qwen4_gpu_session_pin_read(Qwen4GpuSession *, const Qwen4GpuTicket *,
    Qwen4GpuSpan, Qwen4Error *);
/* R1.7b: attach a residency set of these MTLBuffers to the session's queue (not while
 * busy, once). Skipped with QW4_UNSUPPORTED (and a stderr line) unless buffers plus
 * headroom_bytes <= 85% of the recommended working set; also UNSUPPORTED on an old OS or
 * with the `claim` oracle. Callers ignore UNSUPPORTED: the session works without it.
 * share_residency puts an existing set (residency() of another session) on this queue. */
Qwen4Status qwen4_gpu_session_use_residency(Qwen4GpuSession *, void *const *buffers,
    size_t count, uint64_t headroom_bytes, Qwen4Error *);
void *qwen4_gpu_session_residency(const Qwen4GpuSession *);   /* borrowed, or NULL */
Qwen4Status qwen4_gpu_session_share_residency(Qwen4GpuSession *, void *set, Qwen4Error *);
/* Pins are all-or-nothing: seal a complete set; clear (and unseal) after a failed pass. */
void qwen4_gpu_session_pins_seal(Qwen4GpuSession *);
bool qwen4_gpu_session_pins_sealed(const Qwen4GpuSession *);
void qwen4_gpu_session_pins_clear(Qwen4GpuSession *);
bool qwen4_gpu_session_pins_enabled(const Qwen4GpuSession *);
size_t qwen4_gpu_session_pin_count(const Qwen4GpuSession *);
Qwen4Status qwen4_gpu_session_submit(Qwen4GpuSession *, const Qwen4GpuTicket *,
    Qwen4Error *);
Qwen4Status qwen4_gpu_session_abort(Qwen4GpuSession *, const Qwen4GpuTicket *,
    bool state_mutated, Qwen4Error *);
/* Waits, reads the Metal error AND the error word, then publishes or poisons. */
Qwen4Status qwen4_gpu_session_finish(Qwen4GpuSession *, const Qwen4GpuTicket *,
    Qwen4Error *);
Qwen4Status qwen4_gpu_session_reset(Qwen4GpuSession *, const void *model,
    Qwen4Error *);
Qwen4StateInfo qwen4_gpu_session_info(const Qwen4GpuSession *);
/* Last successful command's GPU timestamps; zero until available. This is
 * device execution time, not submit/wait wall time. */
double qwen4_gpu_session_seconds(const Qwen4GpuSession *);
/* BUSY leaves the session and every resource alive. */
Qwen4Status qwen4_gpu_session_destroy(Qwen4GpuSession *, Qwen4Error *);
#endif
