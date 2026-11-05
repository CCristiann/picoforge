/* ssd_bench.c — what the internal SSD gives for the reads an expert cache would issue.
 *
 *   ssd_bench FILE REGION_OFFSET REGION_BYTES OUT.csv
 *
 * Random reads of a given size at random 16 KiB-aligned offsets inside one region of FILE,
 * issued by T threads with pread and F_NOCACHE (the unified buffer cache neither serves nor
 * keeps them, so this is the device and not memory). The region must be cold: for the Qwen3.8
 * Q2 file use the per-layer n-gram table, which the model reads only by rows. Sizes are the
 * 256 KB of a projection slice, 1.42 MB (a Q2 expert) and 2.59 MB (a Q4K expert).
 * Protocol (docs/BENCHMARKS.md): 3 warm-up and 20 measured repetitions of ~512 MiB per
 * configuration, median and p10/p90 of GB/s, latency percentiles over every measured read.
 * Plus a sequential read for the peak. Nothing is mapped, nothing is written.
 */
#define _DARWIN_C_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define WARMUP 3
#define REPS 20
#define PAGE 16384u
#define REP_BYTES (512ull << 20)
#define MAX_THREADS 16

typedef struct {
    int fd; uint64_t base, span; size_t size; unsigned reads; int sequential;
    uint64_t seed, start; double *lat; unsigned done; void *buf;
} Job;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }
static uint64_t next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static void *worker(void *arg) {
    Job *j = arg; uint64_t rng = j->seed, off = j->start;
    const uint64_t slots = (j->span - j->size) / PAGE;
    for (unsigned i = 0; i < j->reads; i++) {
        if (!j->sequential) off = j->base + (next(&rng) % slots) * PAGE;
        const double t0 = now();
        size_t got = 0;
        while (got < j->size) {
            const ssize_t n = pread(j->fd, (char *)j->buf + got, j->size - got, (off_t)(off + got));
            if (n <= 0) { perror("pread"); exit(1); }
            got += (size_t)n;
        }
        j->lat[i] = (now() - t0) * 1e6;
        if (j->sequential) { off += j->size; if (off + j->size > j->base + j->span) off = j->base; }
    }
    return NULL;
}
static int cmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double pct(double *v, size_t n, double p) { qsort(v, n, sizeof *v, cmp); return v[(size_t)(p * (double)(n - 1) + 0.5)]; }

static void run(int fd, uint64_t base, uint64_t span, size_t size, int threads, int sequential, FILE *csv, const char *name) {
    const unsigned per = (unsigned)(REP_BYTES / size / (uint64_t)threads);
    pthread_t th[MAX_THREADS]; Job job[MAX_THREADS];
    double rate[REPS]; double *lat = malloc((size_t)REPS * per * (size_t)threads * sizeof *lat); size_t nlat = 0;
    for (int t = 0; t < threads; t++) {
        if (posix_memalign(&job[t].buf, PAGE, size)) exit(1);
        job[t].lat = malloc(per * sizeof(double));
    }
    for (int rep = 0; rep < WARMUP + REPS; rep++) {
        for (int t = 0; t < threads; t++) {
            Job *j = &job[t];
            j->fd = fd; j->base = base; j->span = span; j->size = size; j->reads = per; j->sequential = sequential;
            j->seed = 0x9e3779b97f4a7c15ull * (uint64_t)(rep * 64 + t + 1);
            j->start = base + ((span / (uint64_t)threads) * (uint64_t)t) / PAGE * PAGE;
        }
        const double t0 = now();
        for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, worker, &job[t]);
        for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
        const double dt = now() - t0;
        if (rep >= WARMUP) {
            rate[rep - WARMUP] = (double)per * (double)threads * (double)size / dt / 1e9;
            fprintf(csv, "%s,%zu,%d,%d,%.6f,%.3f\n", name, size, threads, rep - WARMUP, dt, rate[rep - WARMUP]);
            for (int t = 0; t < threads; t++) for (unsigned i = 0; i < per; i++) lat[nlat++] = job[t].lat[i];
        }
    }
    const double med = pct(rate, REPS, 0.5), p10 = pct(rate, REPS, 0.1), p90 = pct(rate, REPS, 0.9);
    printf("%-10s size %7zu B  threads %2d  GB/s median %6.2f  p10 %6.2f  p90 %6.2f  latency us p50 %8.0f p99 %8.0f\n",
           name, size, threads, med, p10, p90, pct(lat, nlat, 0.5), pct(lat, nlat, 0.99));
    for (int t = 0; t < threads; t++) { free(job[t].buf); free(job[t].lat); }
    free(lat);
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: ssd_bench FILE REGION_OFFSET REGION_BYTES OUT.csv\n"); return 2; }
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    if (fcntl(fd, F_NOCACHE, 1) < 0) { perror("F_NOCACHE"); return 1; }
    const uint64_t base = strtoull(argv[2], NULL, 10) / PAGE * PAGE, span = strtoull(argv[3], NULL, 10);
    FILE *csv = fopen(argv[4], "wx");
    if (!csv) { perror(argv[4]); return 1; }
    fprintf(csv, "pattern,size_bytes,threads,rep,seconds,gb_per_s\n");
    static const size_t sizes[] = {262144, 1486848, 2711552};
    static const int threads[] = {1, 2, 4, 8, 16};
    for (size_t s = 0; s < 3; s++) for (size_t t = 0; t < 5; t++) run(fd, base, span, sizes[s], threads[t], 0, csv, "random");
    run(fd, base, span, 8u << 20, 1, 1, csv, "sequential");
    run(fd, base, span, 8u << 20, 4, 1, csv, "sequential");
    fclose(csv); close(fd);
    return 0;
}
