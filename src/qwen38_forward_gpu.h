/* G2: ordinary GPU graph, one G0 command and one completion per append.
 * Inputs/weights remain borrowed until destroy. MTP is explicitly enabled. */
#ifndef PICOFORGE_QWEN38_FORWARD_GPU_H
#define PICOFORGE_QWEN38_FORWARD_GPU_H
#include "qwen38_forward.h"
typedef struct Qwen38GpuForward Qwen38GpuForward;
/* Last admitted nonempty append. Read preparation includes bounded pread,
 * dedup/decode and staging. GPU timestamps overlap wait_s; never add them. */
typedef struct {
    double prepare_s, encode_s, wait_s, gpu_s;
    size_t ngram_calls, ngram_bytes, ngram_rows;
} Qwen38GpuForwardStats;
Qwen38GpuForwardStats qwen38_gpu_forward_stats(const Qwen38GpuForward *);
/* max_tokens <=1024, capacity <=32768; max_logits_rows<=max_tokens bounds
 * readback/workspace independently from prefill. Optional traces have a
 * fixed 256 MiB per-append ceiling, checked before mutation. */
Qwen38Status qwen38_gpu_forward_create(const Gguf *,const Qwen38Config *,const Qwen38Weights *,
    const GpuWeightStore *,void *device,void *library,uint32_t max_tokens,uint32_t capacity,
    uint32_t max_logits_rows,Qwen38GpuForward **,Qwen38Error *);
/* Same append/prefix/logits semantics as G1. Synchronous, single caller.
 * Traces and logits are published only after successful completion. Failed
 * GPU commands poison all state; reset zeros histories and advances epoch. */
Qwen38Status qwen38_gpu_forward(Qwen38GpuForward *,const int32_t *,size_t count,uint32_t layers,
    Qwen38F32Mut logits,Qwen38GraphTrace,void *,Qwen38Error *);
/* Same append, but only the greedy id of the LAST token's logits comes back: the argmax
 * runs on the GPU (ties -> first index, non-finite -> Q38_GPU_ERROR). Q38_REF_KERNELS
 * naming `argmax` reads the row back and scans it on the CPU instead. */
Qwen38Status qwen38_gpu_forward_argmax(Qwen38GpuForward *,const int32_t *,size_t count,uint32_t layers,
    int32_t *id,Qwen38Error *);
Qwen38StateInfo qwen38_gpu_forward_info(const Qwen38GpuForward *);
Qwen38Status qwen38_gpu_forward_reset(Qwen38GpuForward *,Qwen38Error *);
void qwen38_gpu_forward_destroy(Qwen38GpuForward *);
/* R5.1b: the expert pool whose regions the store attached (gpu_weights_attach_pool); borrowed, it
 * must outlive the graph. Required before the first append when any expert tensor is streamed. */
struct Q38ExpertPool;
void qwen38_gpu_forward_set_pool(Qwen38GpuForward *, struct Q38ExpertPool *);
/* Allocated Metal workspace/state/constants (excludes borrowed weights and
 * optional transient trace storage). SSD staging is owned separately by E. */
size_t qwen38_gpu_forward_bytes(const Qwen38GpuForward *);
/* One predictor block, own QSA cache. Enable only on a fresh graph. No PLE
 * and no additional weight load. A failure requires destroying the graph. */
Qwen38Status qwen38_gpu_mtp_enable(Qwen38GpuForward *,Qwen38Error *);
/* Pair residual rows from the LAST successful full trunk append with next
 * tokens. Predictor position must equal that append's start + offset. Rows
 * of logits (0..count) are the final rows. Does not overwrite trunk residuals.
 * Errors after admission poison both graphs; reset recovers. */
Qwen38Status qwen38_gpu_mtp(Qwen38GpuForward *,size_t offset,const int32_t *next_tokens,
    size_t count,Qwen38F32Mut logits,Qwen38Error *);
Qwen38StateInfo qwen38_gpu_mtp_info(const Qwen38GpuForward *);
/* One reusable host snapshot, excluding weights and disk tables. Save only
 * idle/valid state. Restore only within two successful trunk tokens, with
 * predictor unchanged. Saving/reset/restore invalidates previous handles.
 * Restore advances the trunk epoch and restores PLE hash history as well as
 * all device histories and the last trunk residual. */
/* R4 speculative verify: 1 confirmed token + up to 3 drafts in one append.
 * Needs MTP enabled (it allocates the per-row snapshots). The append is an
 * ordinary one (logits rows as qwen38_gpu_forward) that also records, for rows
 * 0..count-2, the recurrent state and convolution/n-gram histories. keep(n)
 * then keeps the first n rows (1..count) and drops the rest without replay:
 * the state equals n ordinary appends of those tokens, bit for bit. A pending
 * verify ends at keep, reset, or the next trunk append. */
#define QWEN38_VERIFY_ROWS 4
Qwen38Status qwen38_gpu_forward_verify(Qwen38GpuForward *,const int32_t *,size_t count,
    Qwen38F32Mut logits,Qwen38Error *);
Qwen38Status qwen38_gpu_forward_keep(Qwen38GpuForward *,size_t keep,Qwen38Error *);
/* R4 chained draft: the predictor on its own last output row and `token`, one
 * position past its last row. Speculative: mtp_rewind(position) must return the
 * predictor to its confirmed position before the next qwen38_gpu_mtp. */
Qwen38Status qwen38_gpu_mtp_chain(Qwen38GpuForward *,int32_t token,Qwen38F32Mut logits,Qwen38Error *);
Qwen38Status qwen38_gpu_mtp_rewind(Qwen38GpuForward *,uint64_t position,Qwen38Error *);
typedef struct { const void *owner; uint64_t epoch,serial,position,mtp_position; } Qwen38GpuCheckpoint;
Qwen38Status qwen38_gpu_checkpoint_save(Qwen38GpuForward *,Qwen38GpuCheckpoint *,Qwen38Error *);
Qwen38Status qwen38_gpu_checkpoint_restore(Qwen38GpuForward *,Qwen38GpuCheckpoint,Qwen38Error *);
size_t qwen38_gpu_checkpoint_bytes(const Qwen38GpuForward *);
#endif
