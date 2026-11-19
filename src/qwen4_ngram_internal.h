/* Private n-gram (PLE) reader and test seams; public contracts are in qwen4_ngram.h. */
#ifndef QWEN4_NGRAM_INTERNAL_H
#define QWEN4_NGRAM_INTERNAL_H
#include "qwen4_ngram.h"
#include "qwen4_gpu.h"
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

typedef struct QW4eReader QW4eReader;
typedef struct { const QW4eReader *owner; uint64_t generation; } QW4eLease;
typedef ssize_t (*QW4ePread)(void *, int, void *, size_t, off_t);
typedef struct {
    void *context;
    void *(*allocate)(void *, size_t);
    void (*release)(void *, void *);
} QW4eAllocator;
typedef struct {
    size_t staging_bytes, calls, bytes_read, unique_rows, intervals;
    size_t requested_bytes, completed_bytes;
    uint64_t error_offset;
    int os_errno;
} QW4eReadStats;

/* Owned duplicated read-only fd; no file mapping. One synchronous job per reader.
 * budget includes object, output, request plan and raw staging; no per-read malloc.
 * max_read <= 64 KiB, rows/width describe ONLY the disk table, base is absolute.
 * Optional allocator and pread callbacks are private fault-injection seams. */
Qwen4Status qw4e_reader_open(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const QW4eAllocator *allocator,
    QW4eReader **out, Qwen4Error *);
Qwen4Status qw4e_reader_read(QW4eReader *reader, const uint64_t *rows, size_t count,
    QW4ePread read_fn, void *context, QW4eLease *lease, Qwen4Error *);
/* Copy only a complete, current batch. No borrowed pointer to recyclable rows.
 * On failure destination is unchanged. Empty batches allow dst=NULL, size=0. */
Qwen4Status qw4e_reader_copy(const QW4eReader *reader, QW4eLease lease,
    float *dst, size_t float_capacity, Qwen4Error *);
QW4eReadStats qw4e_reader_stats(const QW4eReader *reader);
void qw4e_reader_close(QW4eReader *reader);
/* Test-visible exact FP32 reduction used by the normalized PLE core. */
float qw4e_ple_scaled_dot(const float *key, const float *query, size_t width);
/* One synchronous job per store. References stay pinned until job_release,
 * including after commit/abort. Handles are invalid after destroy/release. */
typedef enum { QW4E_PREPARED, QW4E_READY, QW4E_COMPUTED, QW4E_COMMITTED, QW4E_ABORTED } QW4ePhase;
typedef struct { void *allocation; size_t bytes; Qwen4ConvState *handle; } QW4eConv;
struct Qwen4NgramStore {
    Qwen4NgramConfig config;
    Qwen4ConvShape conv;
    Qwen4StateSize conv_size;
    QW4eAllocator allocator;
    QW4eReader *reader;
    uint64_t *arrays;
    size_t max_tokens, width, channels, used, peak, states, workspaces;
    bool changed;
    Qwen4NgramJob *job;
    QW4ePread read_fn;
    void *read_context;
};
struct Qwen4NgramState {
    Qwen4NgramStore *store;
    Qwen4StateInfo info;
    int32_t *previous;
    QW4eConv conv;
    Qwen4GpuSession *gpu_session; /* borrowed; explicit external conv history */
};
struct Qwen4NgramJob {
    Qwen4NgramStore *store;
    Qwen4NgramState *state;
    Qwen4StateKey key;
    size_t count;
    uint64_t position, *ids;
    int32_t *next;
    QW4eConv candidate;
    QW4eLease lease;
    QW4ePhase phase;
    Qwen4PleWorkspace *workspace;
};
struct Qwen4PleWorkspace {
    Qwen4NgramStore *store;
    Qwen4NgramJob *job;
    float *data, *key, *query, *value, *gate, *u, *z, *conv;
    size_t bytes;
};
Qwen4Status qw4e_error(Qwen4Error *, Qwen4Status, const char *);
Qwen4Status qw4e_allocate(Qwen4NgramStore *, size_t, void **, Qwen4Error *);
void qw4e_free(Qwen4NgramStore *, void *, size_t);
Qwen4Status qw4e_job_check(const Qwen4NgramJob *, Qwen4Error *);
/* Bind a fresh state to the graph's G0 session. CPU PLE is then forbidden.
 * Caller owns/zeros GPU conv history on reset and keeps session alive. */
Qwen4Status qw4e_ngram_gpu_bind(Qwen4NgramState *, Qwen4GpuSession *, Qwen4Error *);
/* Only after G0 finish succeeds: publishes hash history, never reads back conv. */
Qwen4Status qw4e_ngram_gpu_commit(Qwen4NgramJob *, Qwen4Error *);
/* Fault seams use the same production path. Allocator context must outlive store. */
Qwen4Status qw4e_ngram_open_test(const Gguf *, const Qwen4NgramConfig *, size_t,
    size_t, const QW4eAllocator *, Qwen4NgramStore **, Qwen4Error *);
#endif
