/* main.c — the matmul microbenchmark: one kernel, three implementations.
 *
 *   microbench --check             every GPU matmul against a CPU reference
 *   microbench --matmul OUT.csv    naive / simdgroup / TensorOps sweep
 *   microbench --quant OUT.csv     the quantised kernels, same protocol
 *   microbench --dispatch OUT.csv  cost per dispatch inside one command buffer
 *
 * Model-agnostic on purpose: it measures the machine, not an LLM. The CSVs
 * from it (bench/matmul_*.csv, bench/dispatch_m5pro.csv) are the numbers the
 * roadmap's performance reasoning starts from.
 */
#include "microbench.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define METALLIB "build/microbench.metallib"

/* The oracle for the check. bf16 is the top half of an fp32, so widening is
 * a shift; the sum runs in fp64, so the reference is more accurate than any
 * kernel being judged. `mag` receives sum |a*w| per output: the scale an fp32
 * accumulation error is proportional to. Dividing by |ref| instead would blow
 * up on outputs that cancel to nearly zero, which says nothing about the kernel. */
static void cpu_matmul(float *C, float *mag, const float *A, const uint16_t *B, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0, abs_acc = 0.0;
            for (int k = 0; k < K; k++) {
                uint32_t bits = (uint32_t)B[(size_t)n * (size_t)K + (size_t)k] << 16;
                float w;
                memcpy(&w, &bits, sizeof w);
                const double term = (double)A[(size_t)m * (size_t)K + (size_t)k] * (double)w;
                acc += term;
                abs_acc += fabs(term);
            }
            C[(size_t)m * (size_t)N + (size_t)n] = (float)acc;
            mag[(size_t)m * (size_t)N + (size_t)n] = (float)abs_acc;
        }
}

static bool check(MetalContext *mtl) {
    struct { int M, N, K; const char *what; } cases[] = {
        {  1, 1024, 1024, "M=1: the bandwidth-bound case" },
        {  1, 3072, 1024, "M=1, wide" },
        { 40, 2048, 1024, "40 rows" },
        {128, 3072, 1024, "128 rows" },
        {  7,  333,  517, "awkward sizes: nothing divides anything" },
    };
    printf("dispatch floor: %.1f us (an empty kernel)\n\n", metal_dispatch_floor(mtl) * 1e6);
    printf("%-46s %10s %10s %9s %9s\n", "case", "max err/mag", "GPU ms", "GFLOP/s", "GB/s");

    bool all_ok = true;
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const int M = cases[c].M, N = cases[c].N, K = cases[c].K;
        float *A = malloc((size_t)M * (size_t)K * sizeof *A);
        uint16_t *B = malloc((size_t)N * (size_t)K * sizeof *B);
        float *gpu = malloc((size_t)M * (size_t)N * sizeof *gpu);
        float *ref = malloc((size_t)M * (size_t)N * sizeof *ref);
        float *mag = malloc((size_t)M * (size_t)N * sizeof *mag);
        if (!A || !B || !gpu || !ref || !mag) die("out of memory for the %dx%dx%d case", M, N, K);

        uint32_t rng = 12345u;
        for (int i = 0; i < M * K; i++) {
            rng = rng * 1664525u + 1013904223u;
            A[i] = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
        }
        for (int i = 0; i < N * K; i++) {
            rng = rng * 1664525u + 1013904223u;
            float v = (float)((rng >> 16) % 2048u) / 1024.0f - 1.0f;
            uint32_t bits;
            memcpy(&bits, &v, sizeof bits);
            B[i] = (uint16_t)(bits >> 16);
        }
        cpu_matmul(ref, mag, A, B, M, N, K);

        for (int which = 0; which < 3; which++) {
            if (!metal_has_kernel(mtl, which)) continue;
            const double secs = metal_matmul(mtl, which, gpu, A, B, M, N, K);
            double worst = 0.0;
            for (int i = 0; i < M * N; i++) {
                const double d = fabs((double)gpu[i] - (double)ref[i]);
                const double rel = d / ((double)mag[i] + 1e-12);
                if (rel > worst) worst = rel;
            }
            const double flops = 2.0 * M * N * K;
            const double bytes = (double)M * K * 4 + (double)N * K * 2 + (double)M * N * 4;
            char label[96];
            snprintf(label, sizeof label, "[%d] %s", which, cases[c].what);
            printf("%-46.46s %10.2e %10.3f %9.1f %9.1f\n", label, worst,
                   secs * 1e3, flops / secs / 1e9, bytes / secs / 1e9);
            /* fp32 accumulation in a different ORDER is the only difference
             * available: its error is a few ulps of the sum of magnitudes. */
            if (worst > 1e-5) { printf("    ^^ FAIL\n"); all_ok = false; }
        }
        free(A); free(B); free(gpu); free(ref); free(mag);
    }
    printf("\n%s\n", all_ok ? "every GPU kernel agrees with the CPU reference"
                             : "A GPU KERNEL DIVERGES");
    return all_ok;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: microbench --check | --matmul OUT.csv | --quant OUT.csv | --dispatch OUT.csv");
    MetalContext *mtl = metal_init(METALLIB);
    metal_info(mtl);
    bool ok = true;
    if (!strcmp(argv[1], "--check")) ok = check(mtl);
    else if (argc > 2 && !strcmp(argv[1], "--matmul")) bench_matmul(mtl, argv[2]);
    else if (argc > 2 && !strcmp(argv[1], "--quant")) bench_qmatmul(mtl, argv[2]);
    else if (argc > 2 && !strcmp(argv[1], "--dispatch")) bench_dispatch(mtl, argv[2]);
    else die("unknown option %s", argv[1]);
    metal_shutdown(mtl);
    return ok ? 0 : 1;
}
