/* G1: the ordinary Qwen3.8 graph on the CPU. Coordinator-owned contract.
 * Composition only: every operator is A/B/C/D/E/F's public (or C's private
 * composition) code. Order mirrors Qwen4ExpTextModel/DecoderLayer.forward:
 *   embed -> repeat H -> per layer [PLE R+=Delta] -> HC-attn -> GDN|QSA
 *   -> inject -> HC-ffn -> MoE -> inject  ...  -> output HC mixer -> LM head.
 * MTP is described by A but never executed here. */
#ifndef PICOFORGE_QWEN38_FORWARD_H
#define PICOFORGE_QWEN38_FORWARD_H
#include "qwen38.h"
#include "qwen38_packed.h"

typedef struct Qwen38Forward Qwen38Forward;
typedef enum {
    Q38G_EMBEDDING, Q38G_PLE_DELTA, Q38G_ATTN_MIXED, Q38G_ATTN_INJECT,
    Q38G_ATTN_CONTRIBUTION, Q38G_ATTN_RESIDUAL, Q38G_FFN_MIXED, Q38G_FFN_INJECT,
    Q38G_MOE, Q38G_OUTPUT, Q38G_FINAL, Q38G_STAGE_COUNT
} Qwen38GraphStage;
/* Optional observer: borrowed FP32 views of graph scratch, valid only during
 * the call. layer is QWEN38_GLOBAL_LAYER for EMBEDDING/FINAL. */
typedef void (*Qwen38GraphTrace)(void *user, uint32_t layer, Qwen38GraphStage, Qwen38F32);

/* g, config, weights (attached) and store stay borrowed until destroy.
 * max_tokens bounds one append (a prefill chunk); capacity the context. */
Qwen38Status qwen38_forward_create(const Gguf *g, const Qwen38Config *,
    const Qwen38Weights *, const GpuWeightStore *, uint32_t max_tokens,
    uint32_t capacity, Qwen38Forward **, Qwen38Error *);
Qwen38StateInfo qwen38_forward_info(const Qwen38Forward *);
/* Appends tokens at next_pos. layers < trunk_blocks runs a PREFIX of the
 * stack (debugging slices) from a fresh session and leaves it invalid: only
 * a full stack advances next_pos. logits=[rows,V] receives the LAST rows
 * tokens (rows may be 0).
 * Any failure after a mutation invalidates the session until reset. */
Qwen38Status qwen38_forward_cpu(Qwen38Forward *, const int32_t *tokens,
    size_t count, uint32_t layers, Qwen38F32Mut logits,
    Qwen38GraphTrace, void *user, Qwen38Error *);
Qwen38Status qwen38_forward_reset(Qwen38Forward *, Qwen38Error *);
void qwen38_forward_destroy(Qwen38Forward *);
#endif
