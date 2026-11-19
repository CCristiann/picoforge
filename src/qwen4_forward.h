/* G1: the ordinary Qwen3.8 graph on the CPU.
 * Composition only: every operator is a layer's public code (or the CPU ops' private
 * composition) code. Order mirrors Qwen4ExpTextModel/DecoderLayer.forward:
 *   embed -> repeat H -> per layer [PLE R+=Delta] -> HC-attn -> GDN|QSA
 *   -> inject -> HC-ffn -> MoE -> inject  ...  -> output HC mixer -> LM head.
 * MTP is described by A but never executed here. */
#ifndef PICOFORGE_QWEN4_FORWARD_H
#define PICOFORGE_QWEN4_FORWARD_H
#include "qwen4.h"
#include "qwen4_packed.h"

typedef struct Qwen4Forward Qwen4Forward;
typedef enum {
    QW4G_EMBEDDING, QW4G_PLE_DELTA, QW4G_ATTN_MIXED, QW4G_ATTN_INJECT,
    QW4G_ATTN_CONTRIBUTION, QW4G_ATTN_RESIDUAL, QW4G_FFN_MIXED, QW4G_FFN_INJECT,
    QW4G_MOE, QW4G_OUTPUT, QW4G_FINAL, QW4G_STAGE_COUNT
} Qwen4GraphStage;
/* Optional observer: borrowed FP32 views of graph scratch, valid only during
 * the call. layer is QWEN4_GLOBAL_LAYER for EMBEDDING/FINAL. */
typedef void (*Qwen4GraphTrace)(void *user, uint32_t layer, Qwen4GraphStage, Qwen4F32);

/* g, config, weights (attached) and store stay borrowed until destroy.
 * max_tokens bounds one append (a prefill chunk); capacity the context. */
Qwen4Status qwen4_forward_create(const Gguf *g, const Qwen4Config *,
    const Qwen4Weights *, const GpuWeightStore *, uint32_t max_tokens,
    uint32_t capacity, Qwen4Forward **, Qwen4Error *);
Qwen4StateInfo qwen4_forward_info(const Qwen4Forward *);
/* Appends tokens at next_pos. layers < trunk_blocks runs a PREFIX of the
 * stack (debugging slices) from a fresh session and leaves it invalid: only
 * a full stack advances next_pos. logits=[rows,V] receives the LAST rows
 * tokens (rows may be 0).
 * Any failure after a mutation invalidates the session until reset. */
Qwen4Status qwen4_forward_cpu(Qwen4Forward *, const int32_t *tokens,
    size_t count, uint32_t layers, Qwen4F32Mut logits,
    Qwen4GraphTrace, void *user, Qwen4Error *);
Qwen4Status qwen4_forward_reset(Qwen4Forward *, Qwen4Error *);
void qwen4_forward_destroy(Qwen4Forward *);
#endif
