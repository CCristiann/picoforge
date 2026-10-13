/* Private G2 PLE composition. B owns projections, C owns norms and conv. */
#ifndef PICOFORGE_QWEN38_E_INTERNAL_H
#define PICOFORGE_QWEN38_E_INTERNAL_H
#include "qwen38_ngram_internal.h"
#include "qwen38_c_internal.h"
typedef struct Q38EGpu Q38EGpu;
typedef struct {
    Qwen38GpuF32 rows, residual;             /* [T,heads*head_dim], [T,H,D] */
    Qwen38GpuF32 gamma_key, gamma_query, gamma_conv, taps; /* [H,D] x3, [H*D,K] */
    Qwen38GpuF32 key, query, value, gate, u, z, conv, delta;
    Qwen38GpuF32 history;                    /* [(K-1)*order,H*D], mutable */
} Q38EGpuWork;
Qwen38Status q38e_gpu_create(void *, void *, Q38EGpu **, Qwen38Error *);
void q38e_gpu_destroy(Q38EGpu *);
/* Dense disjoint views (read/read aliases allowed), all preflighted before
 * encode. rows are a copied bounded SSD batch. No submit, no state publication. */
Qwen38Status q38e_ple_encode(Q38EGpu *, Q38PackedGpu *, Q38CGpu *, Qwen38GpuCommand,
    const Qwen38NgramConfig *, const GpuWeightStore *, const GgufBinding *const[2],
    const Q38PackedLayout[2], const Q38EGpuWork *, Qwen38Error *);
/* Elementwise sum for Delta and graph residual, dense equal shapes. Output
 * can exactly alias either input; partial overlaps are forbidden. */
Qwen38Status q38e_add_encode(Q38EGpu *, Qwen38GpuCommand, Qwen38GpuF32,
    Qwen38GpuF32, Qwen38GpuF32, Qwen38Error *);
#endif
