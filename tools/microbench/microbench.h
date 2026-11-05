/* microbench.h — the matmul microbenchmark's C interface to Metal.
 *
 * The GPU side lives behind a C interface: no object pointer ever escapes
 * metal.m. Kernel indices are positional and match kernels.metal. */
#ifndef PICOFORGE_MICROBENCH_H
#define PICOFORGE_MICROBENCH_H

#include "common.h"

typedef struct MetalContext MetalContext;
enum { MM_NAIVE = 0, MM_SIMDGROUP = 1, MM_TENSOROPS = 2 };

MetalContext *metal_init(const char *metallib_path);
void   metal_shutdown(MetalContext *ctx);
void   metal_info(const MetalContext *ctx);
bool   metal_has_kernel(const MetalContext *ctx, int which);

/* The cost of dispatching a kernel that does nothing, in seconds. */
double metal_dispatch_floor(MetalContext *ctx);

/* One-shot C[M,N] = A[M,K] * B[N,K]^T, with A fp32, B bf16 and C fp32.
 * Returns the GPU time in seconds, measured by the GPU rather than around
 * the submit. */
double metal_matmul(MetalContext *ctx, int which,
                    float *C, const float *A, const uint16_t *B,
                    int M, int N, int K);

/* A matmul with its buffers allocated once, so a benchmark can repeat the
 * kernel without re-measuring the driver. */
typedef struct MetalMatmul MetalMatmul;
MetalMatmul *metal_matmul_prepare(MetalContext *ctx, int M, int N, int K);
void   metal_matmul_upload(MetalMatmul *mm, const float *A, const uint16_t *B);
double metal_matmul_run(MetalMatmul *mm, int which);
double metal_matmul_run_n(MetalMatmul *mm, int which, int count, double *wall_out);
double metal_matmul_run_tiled(MetalMatmul *mm, const char *kernel, int tile_m, int tile_n,
                              bool thread_scope, int count, double *wall_out);
void   metal_matmul_fill_c(MetalMatmul *mm, float value);   /* poison C before a check */
void   metal_matmul_download(MetalMatmul *mm, float *C);
void   metal_matmul_free(MetalMatmul *mm);

/* A quantised matmul, prepared once. `kernel` names the Metal function; A is
 * bf16 when bf16_act (Q4), fp32 otherwise. Pipelines come from the cache. */
typedef struct MetalQMatmul MetalQMatmul;
MetalQMatmul *metal_qmatmul_prepare(MetalContext *ctx, const char *kernel,
                                    int M, int N, int K, int G, int bits, bool bf16_act);
void   metal_qmatmul_upload(MetalQMatmul *mm, const void *A, const uint8_t *Q, const uint16_t *D);
double metal_qmatmul_run(MetalQMatmul *mm);
void   metal_qmatmul_download(MetalQMatmul *mm, float *C);
void   metal_qmatmul_free(MetalQMatmul *mm);

/* Raw Metal objects, as void* so nothing else has to include Metal headers. */
void  *metal_device(MetalContext *ctx);
void  *metal_queue(MetalContext *ctx);
void  *metal_pipeline(MetalContext *ctx, const char *name);

/* The measurement harness (docs/BENCHMARKS.md). One CSV row per (kernel, shape). */
void   bench_matmul(MetalContext *ctx, const char *csv_path);
void   bench_qmatmul(MetalContext *ctx, const char *csv_path);
void   bench_dispatch(MetalContext *ctx, const char *csv_path);

#endif
