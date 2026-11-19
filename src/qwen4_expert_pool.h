/* qwen4_expert_pool.h — a bounded cache of routed experts read from the GGUF (R5.1).
 *
 * An expert is the gate, up and down slice of one (layer, expert). The pool holds `slots` of
 * them in three page-aligned regions, one per matrix, so a GPU kernel that indexes experts as
 * `slot * bytes_per_expert` reads the pool exactly as it reads the full expert tensor. The
 * least recently used unpinned slot is evicted on a miss. Experts acquired for the current
 * token stay pinned until released, so an eviction can never pull one out from under the GPU.
 * Plain C, no Metal: the regions are anonymous mmaps, page-aligned for no-copy MTLBuffers.
 */
#ifndef PICOFORGE_QWEN4_EXPERT_POOL_H
#define PICOFORGE_QWEN4_EXPERT_POOL_H
#include <stddef.h>
#include <stdint.h>

typedef struct QW4ExpertPool QW4ExpertPool;
typedef struct { uint64_t offset[3]; } QW4ExpertLayer;   /* file offset of the layer's gate, up, down tensors */
/* acquire_ns: wall time inside qw4_pool_acquire, i.e. the time the token waits for the SSD (R5.2). */
typedef struct { uint64_t hits, misses, evictions, bytes_read, acquire_ns; } QW4PoolStats;

/* fd: the GGUF, opened for reading (the caller sets F_NOCACHE if wanted). bytes[m]: size of one
 * expert's slice of matrix m (gate, up, down). Expert e of layer l, matrix m, lives at
 * layers[l].offset[m] + e * bytes[m]. NULL on failure. */
QW4ExpertPool *qw4_pool_create(int fd, const QW4ExpertLayer *layers, uint32_t n_layers, uint32_t experts,
                               const size_t bytes[3], uint32_t slots);
void qw4_pool_destroy(QW4ExpertPool *);
/* Makes the n experts of `layer` resident and pins them; slots[i] is where experts[i] now lives. An
 * expert listed twice shares its slot and is pinned twice. Returns 0, or -1 when every slot is pinned
 * or a read fails (nothing stays pinned by this call). */
int qw4_pool_acquire(QW4ExpertPool *, uint32_t layer, const uint32_t *experts, size_t n, uint32_t *slots);
/* R5.2a: how many reads of one acquire run at once (1 = serial, the R5.1 behaviour; default 8). */
void qw4_pool_set_readers(QW4ExpertPool *, uint32_t readers);
void qw4_pool_release(QW4ExpertPool *, const uint32_t *slots, size_t n);
void *qw4_pool_region(const QW4ExpertPool *, int matrix, size_t *bytes);   /* the whole matrix pool */
QW4PoolStats qw4_pool_stats(const QW4ExpertPool *);
#endif
