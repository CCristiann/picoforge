/* F: routed+shared contribution, never adds the residual. */
#ifndef PICOFORGE_QWEN4_MOE_H
#define PICOFORGE_QWEN4_MOE_H
#include "qwen4_packed.h"
typedef struct Qwen4Moe Qwen4Moe;
typedef struct Qwen4MoeWork Qwen4MoeWork;
typedef struct {
    QW4PackedCpuView router, gate, up, down;
    QW4PackedCpuView shared_gate, shared_up, shared_down, shared_router;
} Qwen4MoeWeights;
typedef struct { Qwen4U32Mut ids; Qwen4F32Mut weights; } Qwen4MoeRoutes;
/* CPU weights borrowed until destroy. Direct B calls; no callback plugin ABI. */
Qwen4Status qwen4_moe_create(const Qwen4MoeConfig *,
    const Qwen4MoeWeights *, Qwen4Moe **, Qwen4Error *);
void qwen4_moe_destroy(Qwen4Moe *);
Qwen4Status qwen4_moe_work_size(const Qwen4Moe *, size_t token_cap,
    size_t *bytes, Qwen4Error *);
Qwen4Status qwen4_moe_work_create(const Qwen4Moe *, size_t token_cap,
    size_t byte_budget, Qwen4MoeWork **, Qwen4Error *);
Qwen4Status qwen4_moe_work_reset(Qwen4MoeWork *, Qwen4Error *);
void qwen4_moe_work_destroy(Qwen4MoeWork *);
/* Pure routing is independent of weights/B: owns temporary softmax scratch
 * inside caller-provided [T,E] probability view, disjoint from logits/routes. */
Qwen4Status qwen4_moe_route(const Qwen4MoeConfig *, Qwen4F32 logits,
    Qwen4MoeRoutes, Qwen4F32Mut probabilities, Qwen4Error *);
Qwen4Status qwen4_moe_forward(const Qwen4Moe *, Qwen4F32 x,
    Qwen4F32Mut y, Qwen4MoeRoutes *optional_routes, Qwen4MoeWork *, Qwen4Error *);
#endif
