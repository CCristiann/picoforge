/* C: CPU primitives on projected inputs; no packed decoder here. */
#ifndef PICOFORGE_QWEN38_OPS_H
#define PICOFORGE_QWEN38_OPS_H
#include "qwen38_types.h"
typedef struct Qwen38ConvState Qwen38ConvState;
typedef struct Qwen38GdnState Qwen38GdnState;
Qwen38Status qwen38_conv_state_bytes(const Qwen38ConvShape *,
    Qwen38StateSize *, Qwen38Error *);
Qwen38Status qwen38_gdn_state_bytes(const Qwen38GdnShape *,
    Qwen38StateSize *, Qwen38Error *);
/* Handles live inside caller-owned metadata; payload stays borrowed.
 * No destroy/free: caller releases BOTH backings once no operation uses them. */
Qwen38Status qwen38_conv_state_init(const Qwen38ConvShape *, Qwen38StateKey,
    Qwen38Memory metadata, Qwen38Memory payload, Qwen38ConvState **, Qwen38Error *);
Qwen38Status qwen38_gdn_state_init(const Qwen38GdnShape *, Qwen38StateKey,
    Qwen38Memory metadata, Qwen38Memory payload, Qwen38GdnState **, Qwen38Error *);
Qwen38StateInfo qwen38_conv_state_info(const Qwen38ConvState *);
Qwen38StateInfo qwen38_gdn_state_info(const Qwen38GdnState *);
Qwen38Status qwen38_conv_state_reset(Qwen38ConvState *, uint64_t epoch, Qwen38Error *);
Qwen38Status qwen38_gdn_state_reset(Qwen38GdnState *, uint64_t epoch, Qwen38Error *);
/* Canonical payload export: history [L,C] oldest-first; S [Hv,Dv,Dk]. */
Qwen38Status qwen38_conv_state_export(const Qwen38ConvState *, Qwen38F32Mut, Qwen38Error *);
Qwen38Status qwen38_gdn_state_export(const Qwen38GdnState *, Qwen38F32Mut, Qwen38Error *);
/* Copy into separately initialized backing with identical key/shape. Used by
 * E's candidate history; never copy an opaque handle with memcpy. */
Qwen38Status qwen38_conv_state_copy(Qwen38ConvState *dst,
    const Qwen38ConvState *src, Qwen38Error *);
Qwen38Status qwen38_conv_causal(const Qwen38ConvShape *, Qwen38F32 x,
    Qwen38F32 taps, Qwen38F32Mut out, Qwen38ConvState *, uint64_t start_pos,
    Qwen38Error *);
Qwen38Status qwen38_gdn_prepare(const Qwen38GdnShape *, Qwen38F32 qkv,
    Qwen38F32 a, Qwen38F32 b, Qwen38F32 ssm_a, Qwen38F32 dt_bias,
    Qwen38F32Mut prepared, Qwen38F32Mut log_decay, Qwen38F32Mut beta, Qwen38Error *);
Qwen38Status qwen38_gdn_scan(const Qwen38GdnShape *, Qwen38F32 prepared,
    Qwen38F32 log_decay, Qwen38F32 beta, Qwen38F32Mut out,
    Qwen38GdnState *, uint64_t start_pos, Qwen38Error *);
Qwen38Status qwen38_gdn_norm_gate(const Qwen38GdnShape *, Qwen38F32 o,
    Qwen38F32 z, Qwen38F32 gain, Qwen38F32Mut out, Qwen38Error *);
Qwen38Status qwen38_rmsnorm_groups(Qwen38F32 x, Qwen38F32 gain,
    Qwen38F32Mut out, float eps, Qwen38Error *);
Qwen38Status qwen38_hc_lowrank_activate(const Qwen38HcShape *, Qwen38F32 lo,
    Qwen38F32Mut activated, Qwen38Error *);
Qwen38Status qwen38_hc_read(const Qwen38HcShape *, Qwen38F32 normed,
    Qwen38F32 gate_logits, Qwen38F32Mut mixed, Qwen38Error *);
Qwen38Status qwen38_hc_inject_weights(const Qwen38HcShape *, Qwen38F32 logits,
    Qwen38F32Mut inject, Qwen38Error *);
Qwen38Status qwen38_hc_add(const Qwen38HcShape *, Qwen38F32Mut residual,
    Qwen38F32 block_out, Qwen38F32 inject, Qwen38Error *);
typedef enum { Q38_SIGMOID, Q38_SILU, Q38_SOFTPLUS } Qwen38Activation;
Qwen38Status qwen38_activate(Qwen38Activation, Qwen38F32 x,
    Qwen38F32Mut out, Qwen38Error *);
#endif
