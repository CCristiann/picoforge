/* F: routed+shared contribution, never adds the residual. */
#ifndef PICOFORGE_QWEN38_MOE_H
#define PICOFORGE_QWEN38_MOE_H
#include "qwen38_packed.h"
typedef struct Qwen38Moe Qwen38Moe;
typedef struct Qwen38MoeWork Qwen38MoeWork;
typedef struct {
    Q38PackedCpuView router, gate, up, down;
    Q38PackedCpuView shared_gate, shared_up, shared_down, shared_router;
} Qwen38MoeWeights;
typedef struct { Qwen38U32Mut ids; Qwen38F32Mut weights; } Qwen38MoeRoutes;
/* CPU weights borrowed until destroy. Direct B calls; no callback plugin ABI. */
Qwen38Status qwen38_moe_create(const Qwen38MoeConfig *,
    const Qwen38MoeWeights *, Qwen38Moe **, Qwen38Error *);
void qwen38_moe_destroy(Qwen38Moe *);
Qwen38Status qwen38_moe_work_size(const Qwen38Moe *, size_t token_cap,
    size_t *bytes, Qwen38Error *);
Qwen38Status qwen38_moe_work_create(const Qwen38Moe *, size_t token_cap,
    size_t byte_budget, Qwen38MoeWork **, Qwen38Error *);
Qwen38Status qwen38_moe_work_reset(Qwen38MoeWork *, Qwen38Error *);
void qwen38_moe_work_destroy(Qwen38MoeWork *);
/* Pure routing is independent of weights/B: owns temporary softmax scratch
 * inside caller-provided [T,E] probability view, disjoint from logits/routes. */
Qwen38Status qwen38_moe_route(const Qwen38MoeConfig *, Qwen38F32 logits,
    Qwen38MoeRoutes, Qwen38F32Mut probabilities, Qwen38Error *);
Qwen38Status qwen38_moe_forward(const Qwen38Moe *, Qwen38F32 x,
    Qwen38F32Mut y, Qwen38MoeRoutes *optional_routes, Qwen38MoeWork *, Qwen38Error *);
#endif
