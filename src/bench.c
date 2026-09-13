/* bench.c — the measurement harness, following docs/BENCHMARKS.md.
 *
 * The protocol, and why each part of it is there:
 *
 *   0.5 s of warm-up      the first dispatch of a pipeline pays for shader
 *                         caches, page faults on freshly allocated buffers,
 *                         and a GPU that may still be at idle clocks. By
 *                         time, not count: see WARM_SECONDS below.
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

/* One (kernel, shape) cell: a timed warm-up + REPS runs, one CSV row. */
static void measure(MetalContext *ctx, FILE *csv, const char *kernel_name,
                    int which, int M, int N, int K, const char *tag) {
    if (!metal_has_kernel(ctx, which)) return;

    float *A = malloc((size_t)M * (size_t)K * sizeof *A);
    uint16_t *B = malloc((size_t)N * (size_t)K * sizeof *B);
    if (!A || !B) die("out of memory for a %dx%dx%d benchmark", M, N, K);
    fill_inputs(A, B, M, N, K);

    MetalMatmul *mm = metal_matmul_prepare(ctx, M, N, K);
    metal_matmul_upload(mm, A, B);

    for (double t0 = wall_s(); wall_s() - t0 < WARM_SECONDS;) (void)metal_matmul_run(mm, which);

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

    printf("\n=== matmul benchmark (%.1f s warm-up + %d reps, median [p10, p90]) ===\n",
           WARM_SECONDS, REPS);
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

/* ------------------------------------------------------- dispatch cost
 * Phase 4 step 4.1. Phase 3 measured ~33 us for one TensorOps matmul at M=1
 * whether it moved 6 MB or 1.5 MB -- but always ONE dispatch per command
 * buffer, so that number may be the buffer, not the matmul. A MoE verify step
 * issues thousands of matmuls, and whether each costs bytes or a fixed toll
 * decides how speculation over experts should be priced. So: the same matmul
 * N times in one buffer, N = 1..1024, at three shapes:
 *
 *   dense    1 x 3072 x 1024   Qwen3-0.6B's gate_proj at decode
 *   expert   1 x  768 x 2048   one Qwen3-30B-A3B expert projection at decode
 *   tiny     1 x   32 x   32   almost no work: whatever is left is the toll
 *
 * If time grows with N at the tiny shape as fast as at the dense one, the
 * cost is per call. If tiny stays flat and dense grows, it is per byte. */
/* One cell: `count` copies of an M x N x K matmul inside one command buffer.
 * `tiles` is the number of 32x32 threadgroups the tiled kernels fan out to. */
typedef struct { const char *name; int tile_m, tile_n; bool thread_scope; } TileVariant;

static double run_cell(MetalMatmul *mm, int which, const TileVariant *v, int count, double *wall) {
    return v ? metal_matmul_run_tiled(mm, v->name, v->tile_m, v->tile_n, v->thread_scope, count, wall)
             : metal_matmul_run_n(mm, which, count, wall);
}

static void dispatch_cell(MetalContext *ctx, FILE *csv, const char *sweep, int which,
                          const TileVariant *v, int M, int N, int K, int count) {
    static const char *names[3] = {"naive", "simd", "tensor"};
    if (!v && !metal_has_kernel(ctx, which)) return;
    const char *kname = v ? v->name + strlen("matmul_") : names[which];
    float *A = malloc((size_t)M * (size_t)K * sizeof *A);
    uint16_t *B = malloc((size_t)N * (size_t)K * sizeof *B);
    if (!A || !B) die("out of memory for the dispatch benchmark");
    fill_inputs(A, B, M, N, K);
    MetalMatmul *mm = metal_matmul_prepare(ctx, M, N, K);
    metal_matmul_upload(mm, A, B);

    double gpu[REPS], wall[REPS];
    /* A variant is judged before it is timed: its output against the 32x32
     * TensorOps kernel's, bit-exact with the CPU oracle since Phase 2. C is
     * poisoned first, or a kernel that wrote nothing would pass. */
    double max_rel = 0.0;
    if (v) {
        float *ref = malloc((size_t)M * (size_t)N * sizeof *ref);
        float *got = malloc((size_t)M * (size_t)N * sizeof *got);
        if (!ref || !got) die("out of memory for the variant check");
        (void)metal_matmul_run(mm, MM_TENSOROPS);
        metal_matmul_download(mm, ref);
        metal_matmul_fill_c(mm, NAN);
        (void)run_cell(mm, which, v, 1, NULL);
        metal_matmul_download(mm, got);
        for (size_t i = 0; i < (size_t)M * (size_t)N; i++) {
            const double rel = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-6);
            if (!(rel <= max_rel)) max_rel = isnan(rel) ? INFINITY : rel;
        }
        free(ref); free(got);
        if (!(max_rel < 1e-5)) die("%s disagrees with matmul_tensorops at %dx%dx%d: rel %.3g",
                                   v->name, M, N, K, max_rel);
    }

    for (double t0 = wall_s(); wall_s() - t0 < WARM_SECONDS;) (void)run_cell(mm, which, v, count, NULL);
    for (int i = 0; i < REPS; i++) gpu[i] = run_cell(mm, which, v, count, &wall[i]);
    qsort(gpu, REPS, sizeof *gpu, cmp_double);
    qsort(wall, REPS, sizeof *wall, cmp_double);
    const double med = pct(gpu, REPS, 0.5), wmed = pct(wall, REPS, 0.5);
    const int tm = v ? v->tile_m : 32, tn = v ? v->tile_n : 32;
    const int tiles = ((N + tn - 1) / tn) * ((M + tm - 1) / tm);

    fprintf(csv, "%s,%s,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3g\n", sweep, kname,
            M, N, K, tiles, count, med, pct(gpu, REPS, 0.1), pct(gpu, REPS, 0.9), wmed,
            med / count, wmed / count, max_rel);
    printf("  %-9s %-18s %2dx%-5dx%-5d tiles %-4d x%-5d gpu %8.3f ms [%.3f, %.3f] %6.1f us/disp"
           "   wall %6.1f us/disp\n", sweep, kname, M, N, K, tiles, count, med * 1e3,
           pct(gpu, REPS, 0.1) * 1e3, pct(gpu, REPS, 0.9) * 1e3, med / count * 1e6,
           wmed / count * 1e6);
    metal_matmul_free(mm);
    free(A); free(B);
}

void bench_dispatch(MetalContext *ctx, const char *csv_path) {
    static const struct { const char *tag; int M, N, K; } shapes[] = {
        {"dense", 1, 3072, 1024}, {"expert", 1, 768, 2048}, {"tiny", 1, 32, 32},
    };
    FILE *csv = fopen(csv_path, "wb");
    if (!csv) die("cannot write %s", csv_path);
    fprintf(csv, "sweep,kernel,M,N,K,tiles,count,gpu_median_s,gpu_p10_s,gpu_p90_s,"
                 "wall_median_s,gpu_per_dispatch_s,wall_per_dispatch_s,max_rel_vs_32x32\n");
    printf("\n=== dispatches per command buffer (%.1f s warm-up + %d reps) ===\n",
           WARM_SECONDS, REPS);
    printf("empty-buffer floor: %.1f us\n", metal_dispatch_floor(ctx) * 1e6);

    /* 1. Does a dispatch cost the same alone as among 1024 others? */
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        for (int which = 0; which < 3; which++)
            for (int count = 1; count <= 1024; count *= 4)
                dispatch_cell(ctx, csv, shapes[s].tag, which, NULL,
                              shapes[s].M, shapes[s].N, shapes[s].K, count);

    /* The TensorOps kernel costs ~31 us at both real shapes but 1.4 us on a
     * single tile. Three sweeps, 64 dispatches each, to find which axis the
     * toll lives on: */
    /* 2. fan-out: more tiles, each the same size (K fixed, N grows). */
    for (int N = 32; N <= 4096; N *= 2)
        dispatch_cell(ctx, csv, "fanout", MM_TENSOROPS, NULL, 1, N, 2048, 64);
    /* 3. work inside ONE tile (N = 32, K grows). */
    for (int K = 32; K <= 8192; K *= 4)
        dispatch_cell(ctx, csv, "tilework", MM_TENSOROPS, NULL, 1, 32, K, 64);
    /* 4. rows: tokens routed to one expert, as in a verify step. */
    for (int M = 1; M <= 32; M *= 2)
        dispatch_cell(ctx, csv, "rows", MM_TENSOROPS, NULL, M, 768, 2048, 64);

    /* 5. tile shape: the kernels.metal variants, at decode and verify shapes. */
    static const TileVariant variants[] = {
        {"matmul_tensorops_8x16", 8, 16, false},   {"matmul_tensorops_8x32", 8, 32, false},
        {"matmul_tensorops_8x64", 8, 64, false},   {"matmul_tensorops_16x32", 16, 32, false},
        {"matmul_tensorops_8x256", 8, 256, false}, {"matmul_tensorops_t1x32", 1, 32, true},
    };
    static const struct { int M, N, K; } tshapes[] = {
        {1, 3072, 1024}, {1, 768, 2048}, {8, 768, 2048}, {32, 3072, 1024},
    };
    for (size_t s = 0; s < sizeof tshapes / sizeof tshapes[0]; s++) {
        dispatch_cell(ctx, csv, "tileshape", MM_TENSOROPS, NULL,
                      tshapes[s].M, tshapes[s].N, tshapes[s].K, 64);
        for (size_t v = 0; v < sizeof variants / sizeof variants[0]; v++)
            dispatch_cell(ctx, csv, "tileshape", MM_TENSOROPS, &variants[v],
                          tshapes[s].M, tshapes[s].N, tshapes[s].K, 64);
    }

    fclose(csv);
    printf("\nraw results -> %s\n", csv_path);
}

/* ------------------------------------------------------------ end to end
 * Whole forward passes of one model, three regimes at the same cache depth:
 *
 *   prefill  512 tokens from position 0            (compute-bound)
 *   decode     1 token at position 512             (bandwidth-bound, M=1)
 *   verify    32 tokens at position 512            (M=32: what speculative
 *                                                   decoding pays per step)
 *
 * The same position matters: a decode at position 10 reads almost no cache
 * and would flatter every format equally. Repeating a pass at one position
 * is idempotent -- it rewrites the same cache rows with the same values.
 *
 * Decode bytes: every weight is read once per token (the embedding in full,
 * as the LM head), plus the K/V cache up to the position. That is the traffic
 * the 307 GB/s ceiling applies to. */
static double weight_bytes(const SafeTensors *st) {
    double b = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const Tensor *t = &st->tensors[i];
        if (strcmp(t->name, "lm_head.weight") == 0) continue;     /* tied copy, never read */
        b += (double)t->nelem * (t->dtype == DT_F32 ? 4 : (t->dtype == DT_I8 || t->dtype == DT_U8) ? 1 : 2);
    }
    return b;
}

void bench_e2e(const char *model_dir, const char *csv_path) {
    Qwen3Config cfg;
    SafeTensors st;
    config_load(model_dir, &cfg);
    st_open(model_dir, &st);
    MetalContext *mtl = metal_init("picoforge.metallib");
    GpuModel *g = gpu_model_create(mtl, &st, &cfg, 1024, 32);

    enum { POS = 512, VERIFY = 32 };
    int tokens[1024];
    for (int i = 0; i < 1024; i++) tokens[i] = (i * 7919 + 13) % cfg.vocab_size;
    float *logits = malloc(32u * (size_t)cfg.vocab_size * sizeof *logits);
    if (!logits) die("out of memory for e2e logits");

    const double wbytes = weight_bytes(&st);
    const double kv_bytes = (double)cfg.num_hidden_layers * (POS + 1)
                          * cfg.num_key_value_heads * cfg.head_dim * 2 * 4;

    FILE *csv = fopen(csv_path, "ab");
    if (!csv) die("cannot append to %s", csv_path);
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0)
        fprintf(csv, "model,regime,n,pos,median_s,p10_s,p90_s,tok_s,weight_gb,kv_gb,gbps,pct_of_307\n");
    printf("\n=== end to end: %s (%.0f MB of weights, %.1f s warm-up + %d reps) ===\n",
           model_dir, wbytes / 1e6, WARM_SECONDS, REPS);

    (void)gpu_forward(g, tokens, POS, 0, POS - 1, logits);         /* fill the cache */
    struct { const char *name; int n, pos, from; } regimes[] = {
        {"prefill", POS, 0, POS - 1}, {"decode", 1, POS, 0}, {"verify32", VERIFY, POS, 0},
    };
    for (size_t r = 0; r < sizeof regimes / sizeof regimes[0]; r++) {
        const int *t = tokens + regimes[r].pos;
        double ts[REPS];
        for (double t0 = wall_s(); wall_s() - t0 < WARM_SECONDS;)
            (void)gpu_forward(g, t, regimes[r].n, regimes[r].pos, regimes[r].from, logits);
        for (int i = 0; i < REPS; i++)
            ts[i] = gpu_forward(g, t, regimes[r].n, regimes[r].pos, regimes[r].from, logits);
        qsort(ts, REPS, sizeof *ts, cmp_double);
        const double med = pct(ts, REPS, 0.5), p10 = pct(ts, REPS, 0.1), p90 = pct(ts, REPS, 0.9);
        const bool dec = strcmp(regimes[r].name, "decode") == 0;
        const double gbps = dec ? (wbytes + kv_bytes) / med / 1e9 : 0.0;
        fprintf(csv, "%s,%s,%d,%d,%.6f,%.6f,%.6f,%.2f,%.4f,%.4f,%.2f,%.2f\n", model_dir,
                regimes[r].name, regimes[r].n, regimes[r].pos, med, p10, p90, regimes[r].n / med,
                wbytes / 1e9, dec ? kv_bytes / 1e9 : 0.0, gbps, 100.0 * gbps / 307.0);
        printf("  %-9s n=%-4d pos=%-4d %8.2f ms  [%.2f, %.2f]  %8.1f tok/s%s",
               regimes[r].name, regimes[r].n, regimes[r].pos, med * 1e3, p10 * 1e3, p90 * 1e3,
               regimes[r].n / med, dec ? "" : "\n");
        if (dec) printf("  %.1f GB/s (%.0f%% of 307)\n", gbps, 100.0 * gbps / 307.0);
    }
    fclose(csv);
    free(logits);
    gpu_model_free(g);
    metal_shutdown(mtl);
    st_close(&st);
}
