/* The expert pool against a naive model: contents are the file's bytes, hits/misses/evictions are those
 * of an LRU kept as a plain list, pins protect the current batch, and failures leave nothing pinned. */
#include "qwen38_expert_pool.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

enum { L = 3, E = 12, SLOTS = 8, MAXB = 4 };
static const size_t B[3] = {3000, 3100, 4500};

static uint8_t byte_of(uint32_t l, int m, uint32_t e, size_t i) {
    uint64_t x = ((uint64_t)l << 40) ^ ((uint64_t)m << 32) ^ ((uint64_t)e << 16) ^ i;
    x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 32;
    return (uint8_t)x;
}
static uint64_t offset_of(uint32_t l, int m) {
    uint64_t o = 0;
    for (uint32_t i = 0; i < l; i++) for (int k = 0; k < 3; k++) o += E * B[k];
    for (int k = 0; k < m; k++) o += E * B[k];
    return o + 12345;                   /* a header, so offsets do not start at zero */
}
static int make_file(char *path) {
    int fd = mkstemp(path); assert(fd >= 0);
    uint8_t *buf = calloc(1, offset_of(L - 1, 2) + E * B[2]); assert(buf);
    for (uint32_t l = 0; l < L; l++) for (int m = 0; m < 3; m++) for (uint32_t e = 0; e < E; e++)
        for (size_t i = 0; i < B[m]; i++) buf[offset_of(l, m) + e * B[m] + i] = byte_of(l, m, e, i);
    size_t total = offset_of(L - 1, 2) + E * B[2];
    assert(write(fd, buf, total) == (ssize_t)total);
    free(buf);
    return fd;
}

/* Reference: keys oldest -> newest; a batch pins the keys it has already touched. */
static uint32_t ref_keys[SLOTS], ref_n;
static uint64_t ref_hits, ref_misses, ref_evict;
static int ref_acquire(uint32_t l, const uint32_t *ex, size_t n) {
    uint32_t pinned[MAXB], np = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t key = l * E + ex[i]; size_t at = ref_n;
        for (size_t j = 0; j < ref_n; j++) if (ref_keys[j] == key) at = j;
        if (at < ref_n) { ref_hits++; for (size_t j = at; j + 1 < ref_n; j++) ref_keys[j] = ref_keys[j + 1]; ref_keys[ref_n - 1] = key; }
        else {
            if (ref_n == SLOTS) {
                size_t v = SLOTS;
                for (size_t j = 0; j < ref_n && v == SLOTS; j++) { int p = 0; for (uint32_t q = 0; q < np; q++) p |= pinned[q] == ref_keys[j]; if (!p) v = j; }
                if (v == SLOTS) return -1;
                for (size_t j = v; j + 1 < ref_n; j++) ref_keys[j] = ref_keys[j + 1];
                ref_n--; ref_evict++;
            }
            ref_keys[ref_n++] = key; ref_misses++;
        }
        pinned[np++] = key;
    }
    return 0;
}

int main(void) {
    char path[] = "/tmp/q38pool-XXXXXX";
    int fd = make_file(path);
    Q38ExpertLayer layers[L];
    for (uint32_t l = 0; l < L; l++) for (int m = 0; m < 3; m++) layers[l].offset[m] = offset_of(l, m);
    Q38ExpertPool *p = q38_pool_create(fd, layers, L, E, B, SLOTS); assert(p);

    uint64_t rng = 88172645463325252ull;
    for (int round = 0; round < 4000; round++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint32_t l = (uint32_t)(rng % L), n = 1 + (uint32_t)((rng >> 8) % MAXB), ex[MAXB], sl[MAXB];
        for (uint32_t i = 0; i < n; i++) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ex[i] = (uint32_t)(rng % (round % 5 ? 5 : E)); }
        int r = q38_pool_acquire(p, l, ex, n, sl), rr = ref_acquire(l, ex, n);
        assert(r == rr && r == 0);
        for (uint32_t i = 0; i < n; i++) for (int m = 0; m < 3; m++) {
            const uint8_t *d = (const uint8_t *)q38_pool_region(p, m, NULL) + (size_t)sl[i] * B[m];
            for (size_t k = 0; k < B[m]; k += 61) assert(d[k] == byte_of(l, m, ex[i], k));
            assert(d[B[m] - 1] == byte_of(l, m, ex[i], B[m] - 1));
        }
        q38_pool_release(p, sl, n);
    }
    Q38PoolStats s = q38_pool_stats(p);
    assert(s.hits == ref_hits && s.misses == ref_misses && s.evictions == ref_evict);
    assert(s.bytes_read == s.misses * (B[0] + B[1] + B[2]) && s.misses > 100 && s.hits > 100);

    /* A batch bigger than the pool fails and leaves nothing pinned; a duplicate shares its slot. */
    uint32_t big[SLOTS + 1], sl[SLOTS + 1];
    for (uint32_t i = 0; i < SLOTS + 1; i++) big[i] = i;
    assert(q38_pool_acquire(p, 0, big, SLOTS + 1, sl) == -1);
    assert(q38_pool_acquire(p, 1, big, SLOTS, sl) == 0);
    q38_pool_release(p, sl, SLOTS);
    uint32_t dup[3] = {7, 7, 9}, ds[3];
    assert(q38_pool_acquire(p, 2, dup, 3, ds) == 0 && ds[0] == ds[1] && ds[0] != ds[2]);
    q38_pool_release(p, ds, 3);
    uint32_t bad = E;
    assert(q38_pool_acquire(p, 0, &bad, 1, ds) == -1);
    size_t bytes; assert(q38_pool_region(p, 2, &bytes) && bytes % 16384 == 0 && bytes >= SLOTS * B[2]);
    q38_pool_destroy(p);
    close(fd); unlink(path);
    printf("expert pool OK: %llu hits, %llu misses, %llu evictions match the reference\n",
           (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.evictions);
    return 0;
}
