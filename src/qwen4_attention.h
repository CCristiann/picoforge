/* D: projected-input attention; caller applies output projection and HC. */
#ifndef PICOFORGE_QWEN4_ATTENTION_H
#define PICOFORGE_QWEN4_ATTENTION_H
#include "qwen4_types.h"
typedef struct Qwen4AttentionState Qwen4AttentionState;
typedef struct Qwen4AttentionWorkspace Qwen4AttentionWorkspace;
typedef struct Qwen4AttentionTicket Qwen4AttentionTicket;
typedef struct {
    size_t cache_bytes, workspace_bytes, alignment;
    uint32_t capacity, max_tokens;
} Qwen4AttentionPlan;
typedef struct {
    Qwen4F32 qg, k, v, iq, ik;
    Qwen4F32 gamma_q, gamma_k, gamma_iq, gamma_ik;
} Qwen4AttentionInput;
Qwen4Status qwen4_attention_plan(const Qwen4AttentionConfig *,
    uint32_t capacity, uint32_t max_tokens, Qwen4AttentionPlan *, Qwen4Error *);
/* create owns CPU allocations, copies config/plan and enforces total budget. */
Qwen4Status qwen4_attention_create(const Qwen4AttentionConfig *,
    const Qwen4AttentionPlan *, Qwen4StateKey, size_t byte_budget,
    Qwen4AttentionState **, Qwen4AttentionWorkspace **, Qwen4Error *);
Qwen4StateInfo qwen4_attention_state_info(const Qwen4AttentionState *);
Qwen4Status qwen4_attention_begin(Qwen4AttentionState *, uint64_t pos0,
    uint32_t tokens, uint64_t epoch, Qwen4AttentionTicket **, Qwen4Error *);
Qwen4Status qwen4_attention_cpu(Qwen4AttentionTicket *,
    const Qwen4AttentionInput *, Qwen4F32Mut out,
    Qwen4AttentionWorkspace *, Qwen4Error *);
/* Ticket owned by state, expires at finish; no per-append allocation. */
Qwen4Status qwen4_attention_finish(Qwen4AttentionTicket *, bool success,
    Qwen4Error *);
Qwen4Status qwen4_attention_reset(Qwen4AttentionState *, uint64_t epoch,
    Qwen4Error *);
Qwen4Status qwen4_attention_destroy(Qwen4AttentionState *,
    Qwen4AttentionWorkspace *, Qwen4Error *); /* BUSY leaves both alive */
#endif
