/* Private PLE composition on the GPU graph. The packed layer owns projections, the CPU ops own norms and conv. */
#ifndef PICOFORGE_QWEN4_PLE_INTERNAL_H
#define PICOFORGE_QWEN4_PLE_INTERNAL_H
#include "qwen4_ngram_internal.h"
#include "qwen4_cpu_internal.h"
typedef struct QW4EGpu QW4EGpu;
typedef struct {
    Qwen4GpuF32 rows, residual;             /* [T,heads*head_dim], [T,H,D] */
    Qwen4GpuF32 gamma_key, gamma_query, gamma_conv, taps; /* [H,D] x3, [H*D,K] */
    Qwen4GpuF32 key, query, value, gate, u, z, conv, delta;
    Qwen4GpuF32 history;                    /* [(K-1)*order,H*D], mutable */
} QW4EGpuWork;
Qwen4Status qw4e_gpu_create(void *, void *, QW4EGpu **, Qwen4Error *);
void qw4e_gpu_destroy(QW4EGpu *);
/* Dense disjoint views (read/read aliases allowed), all preflighted before
 * encode. rows are a copied bounded SSD batch. No submit, no state publication. */
Qwen4Status qw4e_ple_encode(QW4EGpu *, QW4PackedGpu *, QW4CGpu *, Qwen4GpuCommand,
    const Qwen4NgramConfig *, const GpuWeightStore *, const GgufBinding *const[2],
    const QW4PackedLayout[2], const QW4EGpuWork *, Qwen4Error *);
/* Elementwise sum for Delta and graph residual, dense equal shapes. Output
 * can exactly alias either input; partial overlaps are forbidden. */
Qwen4Status qw4e_add_encode(QW4EGpu *, Qwen4GpuCommand, Qwen4GpuF32,
    Qwen4GpuF32, Qwen4GpuF32, Qwen4Error *);
#endif
