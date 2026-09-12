/* quant.c — the quantised kernels, judged by the CPU reference.
 *
 * Same contract as --metal-check: real shapes, the CPU path as the oracle.
 * Weights here are random codes and random bf16 scales, not a quantised
 * model. That is deliberate. This checks ARITHMETIC — does the GPU compute
 * sum_k a[k] * d * q[k] — and random codes exercise every code value and every
 * nibble position, where a real model would mostly exercise the small ones.
 * What quantisation costs in quality is measured elsewhere, on text.
 *
 * Q4 activations are narrowed to bf16 before EITHER side sees them, so the two
 * compute on identical inputs. Narrowing is a Q4 error of its own; folding it
 * into this comparison would hide a kernel bug inside an expected loss.
 */
#include "picoforge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_next(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }

static uint16_t f32_to_bf16(float v) {         /* truncating narrow: exact widen-back */
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return (uint16_t)(bits >> 16);
}

typedef struct { const char *kernel; int bits; int group; } QKernel;   /* group 0 = K */

/* One (kernel, shape): run both sides, return the worst error relative to the
 * summands, and the worst absolute difference through *worst_abs_out. */
static double check_one(MetalContext *ctx, const QKernel *qk, int M, int N, int K,
                        double *worst_abs_out) {
    const int G = qk->group ? qk->group : K;
    const bool bf16_act = (qk->bits == 4);
    const size_t q_bytes = (size_t)N * (size_t)K * (size_t)qk->bits / 8u;
    const size_t nd = (size_t)N * (size_t)(K / G);

    float    *x   = malloc((size_t)M * (size_t)K * sizeof *x);
    uint16_t *x16 = malloc((size_t)M * (size_t)K * sizeof *x16);
    uint8_t  *q   = malloc(q_bytes);
    uint16_t *d   = malloc(nd * sizeof *d);
    float    *gpu = malloc((size_t)M * (size_t)N * sizeof *gpu);
    float    *ref = malloc((size_t)M * (size_t)N * sizeof *ref);
    if (!x || !x16 || !q || !d || !gpu || !ref) die("out of memory for a quant check");

    uint32_t s = 20260912u;
    for (size_t i = 0; i < (size_t)M * (size_t)K; i++) {
        x16[i] = f32_to_bf16((float)(rng_next(&s) % 4096u) / 1024.0f - 2.0f);
        /* The CPU gets exactly what the GPU will read: bf16 for Q4. */
        x[i] = bf16_act ? bf16_to_f32(x16[i]) : (float)(rng_next(&s) % 4096u) / 1024.0f - 2.0f;
    }
    for (size_t i = 0; i < q_bytes; i++) q[i] = (uint8_t)rng_next(&s);
    for (size_t i = 0; i < nd; i++)          /* scales across two decades, both signs */
        d[i] = f32_to_bf16(((rng_next(&s) & 1) ? -1.0f : 1.0f)
                           * (0.002f + (float)(rng_next(&s) % 1000u) * 2e-5f));

    QWeight w = {qk->bits, G, q, d, NULL};
    for (int m = 0; m < M; m++)
        matmul_q(ref + (size_t)m * (size_t)N, x + (size_t)m * (size_t)K, &w, K, N);

    MetalQMatmul *mm = metal_qmatmul_prepare(ctx, qk->kernel, M, N, K, G, qk->bits, bf16_act);
    metal_qmatmul_upload(mm, bf16_act ? (void *)x16 : (void *)x, q, d);
    (void)metal_qmatmul_run(mm);
    metal_qmatmul_download(mm, gpu);
    metal_qmatmul_free(mm);

    /* Error relative to the SUMMANDS, not to the result. A sum of 1024 random
     * signed terms routinely lands near zero, and |gpu - ref| / |ref| then
     * reports cancellation as error: the first version of this check did, and
     * its worst case grew with M only because more outputs meant more near-
     * zero ones. What fp32 accumulation can actually be charged with is a
     * small multiple of n * FLT_EPSILON * sum|a_k w_k|. */
    double worst = 0.0, worst_abs = 0.0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double mag = 0.0;
            for (int k = 0; k < K; k++) {
                float wk = bf16_to_f32(d[(size_t)n * (size_t)(K / G) + (size_t)(k / G)]);
                int code = (qk->bits == 8) ? (int8_t)q[(size_t)n * (size_t)K + (size_t)k]
                         : (int8_t)(uint8_t)(((q[(size_t)n * (size_t)(K / 2) + (size_t)(k >> 1)]
                                              >> ((k & 1) << 2)) & 0xF) << 4) >> 4;
                mag += fabs((double)x[(size_t)m * (size_t)K + (size_t)k] * (double)wk * code);
            }
            size_t i = (size_t)m * (size_t)N + (size_t)n;
            double diff = fabs((double)gpu[i] - (double)ref[i]);
            if (diff > worst_abs) worst_abs = diff;
            if (mag > 0 && diff / mag > worst) worst = diff / mag;
        }
    *worst_abs_out = worst_abs;
    free(x); free(x16); free(q); free(d); free(gpu); free(ref);
    return worst;
}

/* The budget, measured before it was set: the worst case over every kernel and
 * shape below is 2.98e-7 = 2.5 FLT_EPSILON of the summands. That is two fp32
 * accumulations in different orders and nothing else. A wrong index or a
 * misapplied scale moves an output by a whole term, >= 1e-2 of the summands
 * for these widths. 16 epsilon sits 6x above the noise and four orders of
 * magnitude below the smallest real bug. */
#define QUANT_BUDGET (16.0 * 1.1920929e-7)

bool quant_check(MetalContext *ctx) {
    static const QKernel kernels[] = {
        {"matmul_q8_row", 8, 0},
        {"matmul_q4_row", 4, 0},
        {"matmul_q8_g32", 8, 32},
        {"matmul_q4_g32", 4, 32},
        {"matmul_q4_g64", 4, 64},
    };
    struct { int M, N, K; const char *what; } cases[] = {
        {  1, 3072, 1024, "decode  gate_proj" },
        { 32, 3072, 1024, "M=32: speculative verify" },
        {128, 2048, 1024, "prefill q_proj, 128 tokens" },
        {  7,  333,  512, "awkward M and N" },
    };

    printf("\n=== quantised kernels vs the CPU reference ===\n");
    printf("%-16s %-28s %16s %12s\n", "kernel", "case", "err/sum|terms|", "max |diff|");
    bool ok = true;
    for (size_t k = 0; k < sizeof kernels / sizeof kernels[0]; k++)
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            double abs_diff;
            double worst = check_one(ctx, &kernels[k], cases[c].M, cases[c].N, cases[c].K,
                                     &abs_diff);
            printf("%-16s %-28s %16.2e %12.2e\n", kernels[k].kernel, cases[c].what, worst, abs_diff);
            if (!(worst <= QUANT_BUDGET)) { printf("    ^^ FAIL\n"); ok = false; }
        }
    printf("budget: %.2e of the summands (16 fp32 epsilons)\n", QUANT_BUDGET);
    printf("\n%s\n", ok ? "every quantised kernel agrees with the CPU reference"
                        : "A QUANTISED KERNEL DIVERGES");
    return ok;
}
