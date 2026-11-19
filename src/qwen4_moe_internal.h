/* Private MoE CPU helpers; not a second public ABI. */
#ifndef PICOFORGE_QWEN4_MOE_INTERNAL_H
#define PICOFORGE_QWEN4_MOE_INTERNAL_H
#include "qwen4_moe.h"
#ifndef __METAL_VERSION__
typedef struct QW4CGpu QW4CGpu;
#endif

/* Optional synchronous diagnostics, private to the MoE tests. Never substitutes an
 * operator. Views are borrowed only during the call; observers must not mutate
 * data or reenter/destroy the workspace. Projection stages are weight indices
 * 0..7; stage 8=SiLU, 9=product, 10=sigmoid, 11=routed, 12=Y. */
typedef void (*QW4MoeObserver)(void *, size_t stage, size_t expert, size_t token,
                             Qwen4F32 input, Qwen4F32 output);
void qw4_moe_observe(Qwen4MoeWork *, QW4MoeObserver, void *);

/* Private post-routing GPU experiment. CPU route IDs/weights are frozen before
 * encode; the packed expert index is host-side. No GPU-dependent dispatch, lifecycle
 * ABI, commit or wait here. Caller serializes reuse and keeps store/bindings,
 * context and all resources alive and unchanged until command completion. */
typedef struct QW4FMoeGpu QW4FMoeGpu;
typedef void (*QW4FMoeGpuObserver)(void *,Qwen4GpuCommand,size_t stage,size_t expert,
                                  Qwen4GpuF32 input,Qwen4GpuF32 output);
void qw4f_gpu_observe(QW4FMoeGpu *,QW4FMoeGpuObserver,void *);
Qwen4Status qw4f_gpu_create(void *device,void *library,const Qwen4MoeConfig *,
    const GpuWeightStore *,const GgufBinding *const weights[8],size_t token_cap,
    size_t byte_budget,QW4FMoeGpu **,Qwen4Error *);
void qw4f_gpu_destroy(QW4FMoeGpu *);
size_t qw4f_gpu_bytes(const QW4FMoeGpu *);
Qwen4Status qw4f_gpu_encode(QW4FMoeGpu *,Qwen4GpuCommand,Qwen4GpuF32 x,
    Qwen4GpuF32 y,Qwen4MoeRoutes frozen_routes,Qwen4Error *);

/* Device-routed MoE for the G2 graph: router product, softmax/top-k, indexed
 * expert products and reduction all stay on the GPU; IDs are never read back.
 * Every view below is caller-owned scratch sized for T tokens / P=T*K pairs,
 * disjoint from x/y/weights; the caller declares it to the session. */
typedef struct {
    Qwen4GpuF32 logits, probs, weights;          /* [T,E], [T,E], [T,K] */
    Qwen4GpuF32 gate, up, act, down;             /* [P,I], [P,I], [P,I], [P,H] */
    Qwen4GpuF32 sgate, sup, sact, shared;        /* [T,Is] x3, [T,H] */
    Qwen4GpuF32 slogit, ssig, routed;            /* [T,1], [T,1], [T,H] */
    Qwen4GpuSpan ids, identity;                  /* uint32 [P] each */
    /* R1.9a profile only (NULL otherwise: no extra dispatch, barrier or split):
     * called before each sub-group with the command it must continue on (the
     * hook may replace cmd->compute_encoder). */
    void (*mark)(void *ctx, Qwen4GpuCommand *cmd, const char *op, uint32_t type, uint64_t bytes);
    void *mark_ctx;
    Qwen4GpuSpan group;                          /* R3.2: qw4_packed_group_bytes(E, P); empty: never grouped */
    /* R5.1b streaming (NULL otherwise): called right after the router with the command it must
     * continue on; it makes the routed experts resident and writes their pool slots to `slots`
     * (uint32 [P]), which the expert kernels then read instead of `ids`. */
    Qwen4Status (*remap)(void *ctx, Qwen4GpuCommand *cmd, Qwen4GpuSpan ids, Qwen4GpuSpan slots, size_t pairs, Qwen4Error *);
    void *remap_ctx;
    Qwen4GpuSpan slots;
    /* Decode III (buffer NULL otherwise): the HC_ADD after the MoE, residual [T,h,H] += inject [T,h] * y,
     * done by the fused down when it can; *hc_added (non-NULL) reports it, else the caller adds. */
    Qwen4GpuF32 hc_residual, hc_inject;
    bool *hc_added;
} QW4FRoutedWork;
typedef struct QW4FRouted QW4FRouted;
/* R1.5d parallel router (qw4_moe_route_fast): one threadgroup of this many
 * threads per token, for at most this many experts (the kernel mirrors the
 * capacity); a larger E, k > E, or `moe` selected keeps the one-thread
 * qw4f_route. */
#define QW4F_ROUTE_THREADS 256
#define QW4F_ROUTE_MAX_EXPERTS 1024
Qwen4Status qw4f_routed_create(void *device, void *library, QW4FRouted **, Qwen4Error *);
void qw4f_routed_destroy(QW4FRouted *);
/* True iff qw4f_routed_encode routes [E experts, top K] with the parallel kernel. */
bool qw4f_routed_fast_route(const QW4FRouted *, size_t experts, size_t slots);
/* weights/layouts in Qwen4MoeWeights order: router, gate, up, down,
 * shared gate, shared up, shared down, shared router. */
Qwen4Status qw4f_routed_encode(QW4FRouted *, QW4PackedGpu *, QW4CGpu *, Qwen4GpuCommand,
    const Qwen4MoeConfig *, const GpuWeightStore *, const GgufBinding *const weights[8],
    const QW4PackedLayout layouts[8], Qwen4GpuF32 x, Qwen4GpuF32 y,
    const QW4FRoutedWork *, Qwen4Error *);

typedef struct {
    size_t *offsets;                 /* [experts+1], group [offsets[e],offsets[e+1]) */
    size_t offsets_capacity_bytes;
    size_t *cursors;                 /* [experts], caller-owned scratch */
    size_t cursors_capacity_bytes;
    size_t *pair_to_row, *row_to_pair; /* [T*K], inverse permutations */
    size_t pair_to_row_capacity_bytes, row_to_pair_capacity_bytes;
} QW4MoeGrouping;

/* IDs logically const despite the shared view type's mutable data pointer.
 * Stable order: expert ascending, then pair=t*K+slot ascending. No weights
 * changed, no dropping or per-expert capacity. All spans must be disjoint,
 * size_t-aligned, and caller-owned for the duration of this synchronous call.
 * Error preflight leaves all arrays unchanged. T=0 writes zero offsets/cursors
 * and permits NULL permutation arrays. IDs must be distinct within each token.
 * Scratch cursors end at offsets[e+1] and have no subsequent semantic role. */
Qwen4Status qw4_moe_group_pairs(const Qwen4MoeConfig *, Qwen4U32Mut ids,
    QW4MoeGrouping, Qwen4Error *);
#endif
