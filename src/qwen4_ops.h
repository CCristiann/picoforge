/* C: CPU primitives on projected inputs; no packed decoder here. */
#ifndef PICOFORGE_QWEN4_OPS_H
#define PICOFORGE_QWEN4_OPS_H
#include "qwen4_types.h"
typedef struct Qwen4ConvState Qwen4ConvState;
typedef struct Qwen4GdnState Qwen4GdnState;
Qwen4Status qwen4_conv_state_bytes(const Qwen4ConvShape *,
    Qwen4StateSize *, Qwen4Error *);
Qwen4Status qwen4_gdn_state_bytes(const Qwen4GdnShape *,
    Qwen4StateSize *, Qwen4Error *);
/* Handles live inside caller-owned metadata; payload stays borrowed.
 * No destroy/free: caller releases BOTH backings once no operation uses them. */
Qwen4Status qwen4_conv_state_init(const Qwen4ConvShape *, Qwen4StateKey,
    Qwen4Memory metadata, Qwen4Memory payload, Qwen4ConvState **, Qwen4Error *);
Qwen4Status qwen4_gdn_state_init(const Qwen4GdnShape *, Qwen4StateKey,
    Qwen4Memory metadata, Qwen4Memory payload, Qwen4GdnState **, Qwen4Error *);
Qwen4StateInfo qwen4_conv_state_info(const Qwen4ConvState *);
Qwen4StateInfo qwen4_gdn_state_info(const Qwen4GdnState *);
Qwen4Status qwen4_conv_state_reset(Qwen4ConvState *, uint64_t epoch, Qwen4Error *);
Qwen4Status qwen4_gdn_state_reset(Qwen4GdnState *, uint64_t epoch, Qwen4Error *);
/* Canonical payload export: history [L,C] oldest-first; S [Hv,Dv,Dk]. */
Qwen4Status qwen4_conv_state_export(const Qwen4ConvState *, Qwen4F32Mut, Qwen4Error *);
Qwen4Status qwen4_gdn_state_export(const Qwen4GdnState *, Qwen4F32Mut, Qwen4Error *);
/* Copy into separately initialized backing with identical key/shape. Used by
 * E's candidate history; never copy an opaque handle with memcpy. */
Qwen4Status qwen4_conv_state_copy(Qwen4ConvState *dst,
    const Qwen4ConvState *src, Qwen4Error *);
Qwen4Status qwen4_conv_causal(const Qwen4ConvShape *, Qwen4F32 x,
    Qwen4F32 taps, Qwen4F32Mut out, Qwen4ConvState *, uint64_t start_pos,
    Qwen4Error *);
Qwen4Status qwen4_gdn_prepare(const Qwen4GdnShape *, Qwen4F32 qkv,
    Qwen4F32 a, Qwen4F32 b, Qwen4F32 ssm_a, Qwen4F32 dt_bias,
    Qwen4F32Mut prepared, Qwen4F32Mut log_decay, Qwen4F32Mut beta, Qwen4Error *);
Qwen4Status qwen4_gdn_scan(const Qwen4GdnShape *, Qwen4F32 prepared,
    Qwen4F32 log_decay, Qwen4F32 beta, Qwen4F32Mut out,
    Qwen4GdnState *, uint64_t start_pos, Qwen4Error *);
Qwen4Status qwen4_gdn_norm_gate(const Qwen4GdnShape *, Qwen4F32 o,
    Qwen4F32 z, Qwen4F32 gain, Qwen4F32Mut out, Qwen4Error *);
Qwen4Status qwen4_rmsnorm_groups(Qwen4F32 x, Qwen4F32 gain,
    Qwen4F32Mut out, float eps, Qwen4Error *);
Qwen4Status qwen4_hc_lowrank_activate(const Qwen4HcShape *, Qwen4F32 lo,
    Qwen4F32Mut activated, Qwen4Error *);
Qwen4Status qwen4_hc_read(const Qwen4HcShape *, Qwen4F32 normed,
    Qwen4F32 gate_logits, Qwen4F32Mut mixed, Qwen4Error *);
Qwen4Status qwen4_hc_inject_weights(const Qwen4HcShape *, Qwen4F32 logits,
    Qwen4F32Mut inject, Qwen4Error *);
Qwen4Status qwen4_hc_add(const Qwen4HcShape *, Qwen4F32Mut residual,
    Qwen4F32 block_out, Qwen4F32 inject, Qwen4Error *);
typedef enum { QW4_SIGMOID, QW4_SILU, QW4_SOFTPLUS } Qwen4Activation;
Qwen4Status qwen4_activate(Qwen4Activation, Qwen4F32 x,
    Qwen4F32Mut out, Qwen4Error *);
#endif
