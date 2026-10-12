/* Private E reader and test seams; public contracts are in qwen38_ngram.h. */
#ifndef QWEN38_NGRAM_INTERNAL_H
#define QWEN38_NGRAM_INTERNAL_H
#include "qwen38_ngram.h"
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
#endif
