/* Private F CPU helpers; not a second public ABI. */
#ifndef PICOFORGE_QWEN38_F_INTERNAL_H
#define PICOFORGE_QWEN38_F_INTERNAL_H
#include "qwen38_moe.h"
#ifndef __METAL_VERSION__
typedef struct Q38CGpu Q38CGpu;
#endif

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

/* Device-routed MoE for the G2 graph: router product, softmax/top-k, indexed
 * expert products and reduction all stay on the GPU; IDs are never read back.
 * Every view below is caller-owned scratch sized for T tokens / P=T*K pairs,
 * disjoint from x/y/weights; the caller declares it to the session. */
typedef struct {
    Qwen38GpuF32 logits, probs, weights;          /* [T,E], [T,E], [T,K] */
    Qwen38GpuF32 gate, up, act, down;             /* [P,I], [P,I], [P,I], [P,H] */
    Qwen38GpuF32 sgate, sup, sact, shared;        /* [T,Is] x3, [T,H] */
    Qwen38GpuF32 slogit, ssig, routed;            /* [T,1], [T,1], [T,H] */
    Qwen38GpuSpan ids, identity;                  /* uint32 [P] each */
} Q38FRoutedWork;
typedef struct Q38FRouted Q38FRouted;
/* R1.5d parallel router (q38_moe_route_fast): one threadgroup of this many
 * threads per token, for at most this many experts (the kernel mirrors the
 * capacity); a larger E, k > E, or `moe` selected keeps the one-thread
 * q38f_route. */
#define Q38F_ROUTE_THREADS 256
#define Q38F_ROUTE_MAX_EXPERTS 1024
Qwen38Status q38f_routed_create(void *device, void *library, Q38FRouted **, Qwen38Error *);
void q38f_routed_destroy(Q38FRouted *);
/* True iff q38f_routed_encode routes [E experts, top K] with the parallel kernel. */
bool q38f_routed_fast_route(const Q38FRouted *, size_t experts, size_t slots);
/* weights/layouts in Qwen38MoeWeights order: router, gate, up, down,
 * shared gate, shared up, shared down, shared router. */
Qwen38Status q38f_routed_encode(Q38FRouted *, Q38PackedGpu *, Q38CGpu *, Qwen38GpuCommand,
    const Qwen38MoeConfig *, const GpuWeightStore *, const GgufBinding *const weights[8],
    const Q38PackedLayout layouts[8], Qwen38GpuF32 x, Qwen38GpuF32 y,
    const Q38FRoutedWork *, Qwen38Error *);

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
