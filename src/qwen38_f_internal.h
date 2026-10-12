/* Private F CPU helpers; not a second public ABI. */
#ifndef PICOFORGE_QWEN38_F_INTERNAL_H
#define PICOFORGE_QWEN38_F_INTERNAL_H
#include "qwen38_moe.h"

/* Optional synchronous diagnostics, private to F tests. Never substitutes an
 * operator. Views are borrowed only during the call; observers must not mutate
 * data or reenter/destroy the workspace. Projection stages are weight indices
 * 0..7; stage 8=SiLU, 9=product, 10=sigmoid, 11=routed, 12=Y. */
typedef void (*Q38MoeObserver)(void *, size_t stage, size_t expert, size_t token,
                             Qwen38F32 input, Qwen38F32 output);
void q38_moe_observe(Qwen38MoeWork *, Q38MoeObserver, void *);

/* Private post-routing GPU experiment. CPU route IDs/weights are frozen before
 * encode; B's expert index is host-side. No GPU-dependent dispatch, lifecycle
 * ABI, commit or wait here. Caller serializes reuse and keeps store/bindings,
 * context and all resources alive and unchanged until command completion. */
typedef struct Q38FMoeGpu Q38FMoeGpu;
typedef void (*Q38FMoeGpuObserver)(void *,Qwen38GpuCommand,size_t stage,size_t expert,
                                  Qwen38GpuF32 input,Qwen38GpuF32 output);
void q38f_gpu_observe(Q38FMoeGpu *,Q38FMoeGpuObserver,void *);
Qwen38Status q38f_gpu_create(void *device,void *library,const Qwen38MoeConfig *,
    const GpuWeightStore *,const GgufBinding *const weights[8],size_t token_cap,
    size_t byte_budget,Q38FMoeGpu **,Qwen38Error *);
void q38f_gpu_destroy(Q38FMoeGpu *);
size_t q38f_gpu_bytes(const Q38FMoeGpu *);
Qwen38Status q38f_gpu_encode(Q38FMoeGpu *,Qwen38GpuCommand,Qwen38GpuF32 x,
    Qwen38GpuF32 y,Qwen38MoeRoutes frozen_routes,Qwen38Error *);

typedef struct {
    size_t *offsets;                 /* [experts+1], group [offsets[e],offsets[e+1]) */
    size_t offsets_capacity_bytes;
    size_t *cursors;                 /* [experts], caller-owned scratch */
    size_t cursors_capacity_bytes;
    size_t *pair_to_row, *row_to_pair; /* [T*K], inverse permutations */
    size_t pair_to_row_capacity_bytes, row_to_pair_capacity_bytes;
} Q38MoeGrouping;

/* IDs logically const despite the shared view type's mutable data pointer.
 * Stable order: expert ascending, then pair=t*K+slot ascending. No weights
 * changed, no dropping or per-expert capacity. All spans must be disjoint,
 * size_t-aligned, and caller-owned for the duration of this synchronous call.
 * Error preflight leaves all arrays unchanged. T=0 writes zero offsets/cursors
 * and permits NULL permutation arrays. IDs must be distinct within each token.
 * Scratch cursors end at offsets[e+1] and have no subsequent semantic role. */
Qwen38Status q38_moe_group_pairs(const Qwen38MoeConfig *, Qwen38U32Mut ids,
    Q38MoeGrouping, Qwen38Error *);
#endif
