/* D: projected-input attention; caller applies output projection and HC. */
#ifndef PICOFORGE_QWEN38_ATTENTION_H
#define PICOFORGE_QWEN38_ATTENTION_H
#include "qwen38_types.h"
typedef struct Qwen38AttentionState Qwen38AttentionState;
typedef struct Qwen38AttentionWorkspace Qwen38AttentionWorkspace;
typedef struct Qwen38AttentionTicket Qwen38AttentionTicket;
typedef struct {
    size_t cache_bytes, workspace_bytes, alignment;
    uint32_t capacity, max_tokens;
} Qwen38AttentionPlan;
typedef struct {
    Qwen38F32 qg, k, v, iq, ik;
    Qwen38F32 gamma_q, gamma_k, gamma_iq, gamma_ik;
} Qwen38AttentionInput;
Qwen38Status qwen38_attention_plan(const Qwen38AttentionConfig *,
    uint32_t capacity, uint32_t max_tokens, Qwen38AttentionPlan *, Qwen38Error *);
/* create owns CPU allocations, copies config/plan and enforces total budget. */
Qwen38Status qwen38_attention_create(const Qwen38AttentionConfig *,
    const Qwen38AttentionPlan *, Qwen38StateKey, size_t byte_budget,
    Qwen38AttentionState **, Qwen38AttentionWorkspace **, Qwen38Error *);
Qwen38StateInfo qwen38_attention_state_info(const Qwen38AttentionState *);
Qwen38Status qwen38_attention_begin(Qwen38AttentionState *, uint64_t pos0,
    uint32_t tokens, uint64_t epoch, Qwen38AttentionTicket **, Qwen38Error *);
Qwen38Status qwen38_attention_cpu(Qwen38AttentionTicket *,
    const Qwen38AttentionInput *, Qwen38F32Mut out,
    Qwen38AttentionWorkspace *, Qwen38Error *);
/* Ticket owned by state, expires at finish; no per-append allocation. */
Qwen38Status qwen38_attention_finish(Qwen38AttentionTicket *, bool success,
    Qwen38Error *);
Qwen38Status qwen38_attention_reset(Qwen38AttentionState *, uint64_t epoch,
    Qwen38Error *);
Qwen38Status qwen38_attention_destroy(Qwen38AttentionState *,
    Qwen38AttentionWorkspace *, Qwen38Error *); /* BUSY leaves both alive */
#endif
