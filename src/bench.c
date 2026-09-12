/* bench.c — the measurement harness, following docs/BENCHMARKS.md.
 *
 * The protocol, and why each part of it is there:
 *
 *   >= 3 warmup runs      the first dispatch of a pipeline pays for shader
 *                         caches, page faults on freshly allocated buffers,
 *                         and a GPU that may still be at idle clocks.
 *   >= 20 measured runs   one run is an anecdote. Silicon on a laptop is a
 *                         noisy instrument: other processes, thermal state
 *                         and the scheduler all leak into a single sample.
 *   median, p10, p90      not the mean. A mean lets one descheduled run
 *                         dominate; the median ignores it and the spread
 *                         between p10 and p90 says whether to trust the run
 *                         at all. A wide spread is a result, not noise to
 *                         be hidden.
 *   buffers reused        allocation and upload happen once, outside the
 *                         timed region. Otherwise the driver is the subject.
 *   raw CSV committed     under bench/, so the plots can be redrawn and the
 *                         claims rechecked by someone who was not here.
 */
#include "picoforge.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 5
#define REPS   25

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Percentile of a sorted array, nearest-rank. With 25 samples there is no
 * call for interpolation, and nearest-rank always reports a number that was
 * actually observed. */
static double pct(const double *sorted, int n, double p) {
    int i = (int)(p * (n - 1) + 0.5);
    return sorted[i];
}

static void fill_inputs(float *A, uint16_t *B, int M, int N, int K) {
    uint32_t rng = 987654321u;
    for (long i = 0; i < (long)M * K; i++) {
        rng = rng * 1664525u + 1013904223u;
        A[i] = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
    }
    for (long i = 0; i < (long)N * K; i++) {
        rng = rng * 1664525u + 1013904223u;
        float v = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
        uint32_t bits;
        memcpy(&bits, &v, sizeof bits);
        B[i] = (uint16_t)(bits >> 16);
    }
}

/* One (kernel, shape) cell: WARMUP + REPS runs, one CSV row. */
static void measure(MetalContext *ctx, FILE *csv, const char *kernel_name,
                    int which, int M, int N, int K, const char *tag) {
    if (!metal_has_kernel(ctx, which)) return;

    float *A = malloc((size_t)M * (size_t)K * sizeof *A);
    uint16_t *B = malloc((size_t)N * (size_t)K * sizeof *B);
    if (!A || !B) die("out of memory for a %dx%dx%d benchmark", M, N, K);
    fill_inputs(A, B, M, N, K);

    MetalMatmul *mm = metal_matmul_prepare(ctx, M, N, K);
    metal_matmul_upload(mm, A, B);

    for (int i = 0; i < WARMUP; i++) (void)metal_matmul_run(mm, which);

    double t[REPS];
    for (int i = 0; i < REPS; i++) t[i] = metal_matmul_run(mm, which);
    qsort(t, REPS, sizeof *t, cmp_double);

    const double med = pct(t, REPS, 0.50);
    const double p10 = pct(t, REPS, 0.10);
    const double p90 = pct(t, REPS, 0.90);
    const double flops = 2.0 * M * N * K;
    /* Bytes that MUST cross the memory system at minimum: read A, read B,
     * write C. Real traffic can exceed this when a kernel re-reads; that is
     * precisely what the tiled versions exist to avoid. */
    const double bytes = (double)M * K * 4 + (double)N * K * 2 + (double)M * N * 4;

    fprintf(csv, "%s,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.3f,%.3f,%.2f\n",
            tag, kernel_name, M, N, K, med, p10, p90,
            flops / med / 1e9, bytes / med / 1e9, 100.0 * (bytes / med) / 307e9);

    printf("  %-18s %-6s M=%-5d N=%-6d K=%-5d  %8.3f ms  [%.3f, %.3f]  "
           "%8.1f GFLOP/s  %7.1f GB/s\n",
           tag, kernel_name, M, N, K, med * 1e3, p10 * 1e3, p90 * 1e3,
           flops / med / 1e9, bytes / med / 1e9);

    metal_matmul_free(mm);
    free(A); free(B);
}

void bench_matmul(MetalContext *ctx, const char *csv_path) {
    FILE *csv = fopen(csv_path, "wb");
    if (!csv) die("cannot write %s", csv_path);
    fprintf(csv, "sweep,kernel,M,N,K,median_s,p10_s,p90_s,gflops,gbps,pct_of_307\n");

    static const char *names[3] = {"naive", "simd", "tensor"};
    const int H = 1024, I = 3072;          /* Qwen3-0.6B's real widths */

    printf("\n=== matmul benchmark (%d warmup + %d reps, median [p10, p90]) ===\n",
           WARMUP, REPS);
    printf("dispatch floor: %.1f us — anything near it is measuring the queue\n\n",
           metal_dispatch_floor(ctx) * 1e6);

    /* Sweep 1: batch size, at the engine's real widths. This is the axis that
     * separates decode (M=1) from prefill (M large), and it is where the
     * tiled kernels are expected to change from a loss to a win. */
    for (int which = 0; which < 3; which++)
        for (int M = 1; M <= 512; M *= 2)
            measure(ctx, csv, names[which], which, M, I, H, "batch-sweep");

    /* Sweep 2: output width at M=1. If decode is short of parallelism rather
     * than short of bandwidth, throughput should climb with N — there is
     * nothing else in this sweep that could make it do so. */
    for (int which = 0; which < 3; which++)
        for (int N = 256; N <= 16384; N *= 2)
            measure(ctx, csv, names[which], which, 1, N, H, "width-at-M1");

    /* Sweep 3: square, the shape everyone quotes. Included so these numbers
     * can be compared with other people's, not because the engine issues it. */
    for (int which = 0; which < 3; which++)
        for (int S = 128; S <= 2048; S *= 2)
            measure(ctx, csv, names[which], which, S, S, S, "square");

    fclose(csv);
    printf("\nraw results -> %s\n", csv_path);
}

/* ------------------------------------------------------ quantised kernels
 * The same protocol, over the Phase 3 kernels, with the bf16 TensorOps kernel
 * as the baseline in the same file. The question is not only "how fast" but
 * "what did the blocks cost": a per-row kernel runs the op once per tile, a
 * G=32 kernel runs it K/32 = 32 times, and only a measurement says whether the
 * bytes saved pay for the calls added.
 *
 * Bytes per run count what must cross the memory system: activations (fp32,
 * or bf16 for Q4), codes, scales, and the fp32 output. */
typedef struct { const char *name; int bits, group; bool bf16_act; } QBench;

/* Warm-up by TIME, not by count. The first sweep of this harness measured
 * M=1 at 154 us for every kernel, bf16 included, then watched the same cells
 * get faster in the order they were measured -- 155, 153, 108, 89, 69 us --
 * straight across kernels that do very different work. That is the GPU
 * leaving a low-power state, not the kernels: five warm-up runs of a 0.1 ms
 * kernel are half a millisecond of work, and the idle gap while the next
 * cell's buffers are allocated is enough to let it drop back. */
#define WARM_SECONDS 0.5

static double wall_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void stats_row(FILE *csv, const char *tag, const char *kernel, int M, int N, int K,
                      double *t, double bytes) {
    qsort(t, REPS, sizeof *t, cmp_double);
    const double med = pct(t, REPS, 0.50), p10 = pct(t, REPS, 0.10), p90 = pct(t, REPS, 0.90);
    const double flops = 2.0 * M * N * K;
    fprintf(csv, "%s,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.3f,%.3f,%.2f\n", tag, kernel, M, N, K,
            med, p10, p90, flops / med / 1e9, bytes / med / 1e9, 100.0 * (bytes / med) / 307e9);
    printf("  %-14s M=%-4d %8.3f ms  [%.3f, %.3f]  %8.1f GFLOP/s  %7.1f GB/s\n",
           kernel, M, med * 1e3, p10 * 1e3, p90 * 1e3, flops / med / 1e9, bytes / med / 1e9);
}

void bench_qmatmul(MetalContext *ctx, const char *csv_path) {
    static const QBench kernels[] = {
        {"matmul_q8_row", 8, 0, false}, {"matmul_q8_g32", 8, 32, false},
        {"matmul_q4_row", 4, 0, true},  {"matmul_q4_g64", 4, 64, true},
        {"matmul_q4_g32", 4, 32, true},
    };
    const int N = 3072, K = 1024;          /* gate_proj: the widest matmul per layer */
    FILE *csv = fopen(csv_path, "wb");
    if (!csv) die("cannot write %s", csv_path);
    fprintf(csv, "sweep,kernel,M,N,K,median_s,p10_s,p90_s,gflops,gbps,pct_of_307\n");
    printf("\n=== quantised matmul benchmark (%.1f s warmup + %d reps) ===\n", WARM_SECONDS, REPS);
    printf("dispatch floor: %.1f us\n", metal_dispatch_floor(ctx) * 1e6);

    for (int M = 1; M <= 512; M *= 2) {
        printf("M=%d\n", M);
        double t[REPS];

        float *A = malloc((size_t)M * K * sizeof *A);
        uint16_t *B = malloc((size_t)N * K * sizeof *B);
        uint8_t *Q = malloc((size_t)N * K);
        uint16_t *D = malloc((size_t)N * K * sizeof *D);
        if (!A || !B || !Q || !D) die("out of memory for the quantised benchmark");
        fill_inputs(A, B, M, N, K);
        for (size_t i = 0; i < (size_t)N * K; i++) { Q[i] = (uint8_t)(B[i] >> 3); D[i] = B[i]; }

        MetalMatmul *base = metal_matmul_prepare(ctx, M, N, K);
        metal_matmul_upload(base, A, B);
        for (double t0 = wall_s(); wall_s() - t0 < WARM_SECONDS;) (void)metal_matmul_run(base, MM_TENSOROPS);
        for (int i = 0; i < REPS; i++) t[i] = metal_matmul_run(base, MM_TENSOROPS);
        stats_row(csv, "quant-batch", "bf16", M, N, K, t,
                  (double)M * K * 4 + (double)N * K * 2 + (double)M * N * 4);
        metal_matmul_free(base);

        for (size_t k = 0; k < sizeof kernels / sizeof kernels[0]; k++) {
            const QBench *qb = &kernels[k];
            const int G = qb->group ? qb->group : K;
            MetalQMatmul *mm = metal_qmatmul_prepare(ctx, qb->name, M, N, K, G, qb->bits, qb->bf16_act);
            /* Values do not matter to the timing; the activation buffer is
             * uploaded as raw bytes of the right width either way. */
            metal_qmatmul_upload(mm, A, Q, D);
            for (double t0 = wall_s(); wall_s() - t0 < WARM_SECONDS;) (void)metal_qmatmul_run(mm);
            for (int i = 0; i < REPS; i++) t[i] = metal_qmatmul_run(mm);
            double bytes = (double)M * K * (qb->bf16_act ? 2 : 4) + (double)N * K * qb->bits / 8.0
                         + (double)N * (K / G) * 2 + (double)M * N * 4;
            stats_row(csv, "quant-batch", qb->name + 7, M, N, K, t, bytes);
            metal_qmatmul_free(mm);
        }
        free(A); free(B); free(Q); free(D);
    }
    fclose(csv);
    printf("\nraw results -> %s\n", csv_path);
}
