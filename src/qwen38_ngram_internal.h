/* Private E reader and test seams; public contracts are in qwen38_ngram.h. */
#ifndef QWEN38_NGRAM_INTERNAL_H
#define QWEN38_NGRAM_INTERNAL_H
#include "qwen38_ngram.h"
#include "qwen38_gpu.h"
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

typedef struct Q38eReader Q38eReader;
typedef struct { const Q38eReader *owner; uint64_t generation; } Q38eLease;
typedef ssize_t (*Q38ePread)(void *, int, void *, size_t, off_t);
typedef struct {
    void *context;
    void *(*allocate)(void *, size_t);
    void (*release)(void *, void *);
} Q38eAllocator;
typedef struct {
    size_t staging_bytes, calls, bytes_read, unique_rows, intervals;
    size_t requested_bytes, completed_bytes;
    uint64_t error_offset;
    int os_errno;
} Q38eReadStats;

/* Owned duplicated read-only fd; no file mapping. One synchronous job per reader.
 * budget includes object, output, request plan and raw staging; no per-read malloc.
 * max_read <= 64 KiB, rows/width describe ONLY the disk table, base is absolute.
 * Optional allocator and pread callbacks are private fault-injection seams. */
Qwen38Status q38e_reader_open(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const Q38eAllocator *allocator,
    Q38eReader **out, Qwen38Error *);
Qwen38Status q38e_reader_read(Q38eReader *reader, const uint64_t *rows, size_t count,
    Q38ePread read_fn, void *context, Q38eLease *lease, Qwen38Error *);
/* Copy only a complete, current batch. No borrowed pointer to recyclable rows.
 * On failure destination is unchanged. Empty batches allow dst=NULL, size=0. */
Qwen38Status q38e_reader_copy(const Q38eReader *reader, Q38eLease lease,
    float *dst, size_t float_capacity, Qwen38Error *);
Q38eReadStats q38e_reader_stats(const Q38eReader *reader);
void q38e_reader_close(Q38eReader *reader);
/* Test-visible exact FP32 reduction used by the normalized PLE core. */
float q38e_ple_scaled_dot(const float *key, const float *query, size_t width);
/* One synchronous job per store. References stay pinned until job_release,
 * including after commit/abort. Handles are invalid after destroy/release. */
typedef enum { Q38E_PREPARED, Q38E_READY, Q38E_COMPUTED, Q38E_COMMITTED, Q38E_ABORTED } Q38ePhase;
typedef struct { void *allocation; size_t bytes; Qwen38ConvState *handle; } Q38eConv;
struct Qwen38NgramStore {
    Qwen38NgramConfig config;
    Qwen38ConvShape conv;
    Qwen38StateSize conv_size;
    Q38eAllocator allocator;
    Q38eReader *reader;
    uint64_t *arrays;
    size_t max_tokens, width, channels, used, peak, states, workspaces;
    bool changed;
    Qwen38NgramJob *job;
    Q38ePread read_fn;
    void *read_context;
};
struct Qwen38NgramState {
    Qwen38NgramStore *store;
    Qwen38StateInfo info;
    int32_t *previous;
    Q38eConv conv;
    Qwen38GpuSession *gpu_session; /* borrowed; explicit external conv history */
};
struct Qwen38NgramJob {
    Qwen38NgramStore *store;
    Qwen38NgramState *state;
    Qwen38StateKey key;
    size_t count;
    uint64_t position, *ids;
    int32_t *next;
    Q38eConv candidate;
    Q38eLease lease;
    Q38ePhase phase;
    Qwen38PleWorkspace *workspace;
};
struct Qwen38PleWorkspace {
    Qwen38NgramStore *store;
    Qwen38NgramJob *job;
    float *data, *key, *query, *value, *gate, *u, *z, *conv;
    size_t bytes;
};
Qwen38Status q38e_error(Qwen38Error *, Qwen38Status, const char *);
Qwen38Status q38e_allocate(Qwen38NgramStore *, size_t, void **, Qwen38Error *);
void q38e_free(Qwen38NgramStore *, void *, size_t);
Qwen38Status q38e_job_check(const Qwen38NgramJob *, Qwen38Error *);
/* Bind a fresh state to the graph's G0 session. CPU PLE is then forbidden.
 * Caller owns/zeros GPU conv history on reset and keeps session alive. */
Qwen38Status q38e_ngram_gpu_bind(Qwen38NgramState *, Qwen38GpuSession *, Qwen38Error *);
/* Only after G0 finish succeeds: publishes hash history, never reads back conv. */
Qwen38Status q38e_ngram_gpu_commit(Qwen38NgramJob *, Qwen38Error *);
/* Fault seams use the same production path. Allocator context must outlive store. */
Qwen38Status q38e_ngram_open_test(const Gguf *, const Qwen38NgramConfig *, size_t,
    size_t, const Q38eAllocator *, Qwen38NgramStore **, Qwen38Error *);
#endif
