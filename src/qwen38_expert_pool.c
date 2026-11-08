#include "qwen38_expert_pool.h"

#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#define NONE UINT32_MAX
#define PAGE 16384u

struct Q38ExpertPool {
    int fd;
    uint32_t n_layers, experts, slots;
    size_t bytes[3], region_bytes[3];
    Q38ExpertLayer *layers;
    void *region[3];
    uint32_t *slot_of;          /* [layer * experts + expert] -> slot, or NONE */
    uint32_t *owner;            /* [slot] -> layer * experts + expert, or NONE */
    uint32_t *pins;             /* [slot] */
    uint32_t *prev, *next;      /* LRU list over all slots: head = most recent, tail = next victim */
    uint32_t head, tail;
    uint32_t readers;
    uint32_t *miss_slot, *miss_expert;   /* [slots]: the misses of the current acquire */
    Q38PoolStats stats;
};

static void unlink_slot(Q38ExpertPool *p, uint32_t s) {
    if (p->prev[s] != NONE) p->next[p->prev[s]] = p->next[s]; else p->head = p->next[s];
    if (p->next[s] != NONE) p->prev[p->next[s]] = p->prev[s]; else p->tail = p->prev[s];
}
static void push_front(Q38ExpertPool *p, uint32_t s) {
    p->prev[s] = NONE; p->next[s] = p->head;
    if (p->head != NONE) p->prev[p->head] = s; else p->tail = s;
    p->head = s;
}
static void touch(Q38ExpertPool *p, uint32_t s) { unlink_slot(p, s); push_front(p, s); }

Q38ExpertPool *q38_pool_create(int fd, const Q38ExpertLayer *layers, uint32_t n_layers, uint32_t experts,
                               const size_t bytes[3], uint32_t slots) {
    if (fd < 0 || !layers || !n_layers || !experts || !slots || slots == NONE) return NULL;
    Q38ExpertPool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fd = fd; p->n_layers = n_layers; p->experts = experts; p->slots = slots; p->head = p->tail = NONE;
    p->layers = malloc(n_layers * sizeof *layers);
    p->slot_of = malloc((size_t)n_layers * experts * sizeof *p->slot_of);
    p->owner = malloc(slots * sizeof *p->owner); p->pins = calloc(slots, sizeof *p->pins);
    p->prev = malloc(slots * sizeof *p->prev); p->next = malloc(slots * sizeof *p->next);
    p->miss_slot = malloc(slots * sizeof *p->miss_slot); p->miss_expert = malloc(slots * sizeof *p->miss_expert);
    p->readers = 8;
    if (!p->layers || !p->slot_of || !p->owner || !p->pins || !p->prev || !p->next || !p->miss_slot || !p->miss_expert) { q38_pool_destroy(p); return NULL; }
    for (uint32_t l = 0; l < n_layers; l++) p->layers[l] = layers[l];
    for (size_t i = 0; i < (size_t)n_layers * experts; i++) p->slot_of[i] = NONE;
    for (int m = 0; m < 3; m++) {
        p->bytes[m] = bytes[m];
        p->region_bytes[m] = ((size_t)slots * bytes[m] + PAGE - 1) / PAGE * PAGE;
        p->region[m] = mmap(NULL, p->region_bytes[m], PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (p->region[m] == MAP_FAILED) { p->region[m] = NULL; q38_pool_destroy(p); return NULL; }
    }
    for (uint32_t s = slots; s-- > 0;) { p->owner[s] = NONE; push_front(p, s); }   /* slot 0 is the first victim */
    return p;
}

void q38_pool_destroy(Q38ExpertPool *p) {
    if (!p) return;
    for (int m = 0; m < 3; m++) if (p->region[m]) munmap(p->region[m], p->region_bytes[m]);
    free(p->layers); free(p->slot_of); free(p->owner); free(p->pins); free(p->prev); free(p->next);
    free(p->miss_slot); free(p->miss_expert); free(p);
}

static void *slot_data(const Q38ExpertPool *p, int m, uint32_t s) { return (char *)p->region[m] + (size_t)s * p->bytes[m]; }

static int read_all(int fd, void *dst, size_t n, uint64_t off) {
    for (size_t got = 0; got < n;) {
        const ssize_t r = pread(fd, (char *)dst + got, n - got, (off_t)(off + got));
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

void q38_pool_release(Q38ExpertPool *p, const uint32_t *slots, size_t n) {
    for (size_t i = 0; i < n; i++) if (slots[i] < p->slots && p->pins[slots[i]]) p->pins[slots[i]]--;
}

void q38_pool_set_readers(Q38ExpertPool *p, uint32_t readers) { if (p) p->readers = readers ? readers : 1; }

/* One read job = one matrix of one missed expert, so a token's misses become 3 x misses independent
 * preads that the SSD serves in parallel (R5.0: 1 thread 384 us per 2.6 MB, >= 2 threads 13 GB/s). */
typedef struct { Q38ExpertPool *p; uint32_t layer, n, width; int failed; } ReadJobs;
static void read_job(void *ctx, size_t j) {
    ReadJobs *r = ctx; Q38ExpertPool *p = r->p;
    const size_t i = j / 3; const int m = (int)(j % 3);
    if (read_all(p->fd, slot_data(p, m, p->miss_slot[i]), p->bytes[m],
                 p->layers[r->layer].offset[m] + (uint64_t)p->miss_expert[i] * p->bytes[m]))
        __atomic_store_n(&r->failed, 1, __ATOMIC_RELAXED);
}
static void read_group(void *ctx, size_t g) {   /* reader g takes jobs g, g + width, ... */
    ReadJobs *r = ctx;
    for (size_t j = g; j < (size_t)r->n * 3; j += r->width) read_job(ctx, j);
}
static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

/* Two passes: the bookkeeping (which slot, which victim) runs first and in order, exactly as when
 * every miss was read on the spot, so hits, evictions and slots do not depend on the reader count;
 * then the misses are read together. A failed read undoes this call's new slots and pins. */
int q38_pool_acquire(Q38ExpertPool *p, uint32_t layer, const uint32_t *experts, size_t n, uint32_t *slots) {
    if (!p || layer >= p->n_layers) return -1;
    const uint64_t t0 = now_ns();
    uint32_t misses = 0; int rc = 0;
    for (size_t i = 0; i < n; i++) {
        if (experts[i] >= p->experts) { q38_pool_release(p, slots, i); rc = -1; goto undo; }
        const uint32_t key = layer * p->experts + experts[i];
        uint32_t s = p->slot_of[key];
        if (s != NONE) {
            p->stats.hits++;
        } else {
            for (s = p->tail; s != NONE && p->pins[s]; s = p->prev[s]) {}
            if (s == NONE) { q38_pool_release(p, slots, i); rc = -1; goto undo; }
            if (p->owner[s] != NONE) { p->slot_of[p->owner[s]] = NONE; p->owner[s] = NONE; p->stats.evictions++; }
            p->owner[s] = key; p->slot_of[key] = s; p->stats.misses++;
            p->stats.bytes_read += p->bytes[0] + p->bytes[1] + p->bytes[2];
            p->miss_slot[misses] = s; p->miss_expert[misses] = experts[i]; misses++;
        }
        p->pins[s]++; touch(p, s); slots[i] = s;
    }
    ReadJobs jobs = {p, layer, misses, p->readers < misses * 3 ? p->readers : misses * 3, 0};
    if (jobs.width <= 1) { jobs.width = 1; read_group(&jobs, 0); }
    else dispatch_apply_f(jobs.width, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &jobs, read_group);
    if (jobs.failed) { q38_pool_release(p, slots, n); rc = -1; }
undo:
    /* On failure the new slots hold garbage or nothing: forget them (they stay at the LRU position). */
    if (rc) for (uint32_t i = 0; i < misses; i++) { p->slot_of[p->owner[p->miss_slot[i]]] = NONE; p->owner[p->miss_slot[i]] = NONE; }
    p->stats.acquire_ns += now_ns() - t0;
    return rc;
}

void *q38_pool_region(const Q38ExpertPool *p, int matrix, size_t *bytes) {
    if (!p || matrix < 0 || matrix > 2) return NULL;
    if (bytes) *bytes = p->region_bytes[matrix];
    return p->region[matrix];
}
Q38PoolStats q38_pool_stats(const Q38ExpertPool *p) { return p->stats; }
