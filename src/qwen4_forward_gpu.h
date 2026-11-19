/* G2: ordinary GPU graph, one G0 command and one completion per append.
 * Inputs/weights remain borrowed until destroy. MTP is explicitly enabled. */
#ifndef PICOFORGE_QWEN4_FORWARD_GPU_H
#define PICOFORGE_QWEN4_FORWARD_GPU_H
#include "qwen4_forward.h"
typedef struct Qwen4GpuForward Qwen4GpuForward;
/* Last admitted nonempty append. Read preparation includes bounded pread,
 * dedup/decode and staging. GPU timestamps overlap wait_s; never add them. */
typedef struct {
    double prepare_s, encode_s, wait_s, gpu_s;
    size_t ngram_calls, ngram_bytes, ngram_rows;
} Qwen4GpuForwardStats;
Qwen4GpuForwardStats qwen4_gpu_forward_stats(const Qwen4GpuForward *);
/* max_tokens <=2048, capacity <=QWEN4_MAX_CONTEXT; max_logits_rows<=max_tokens bounds
 * readback/workspace independently from prefill. Optional traces have a
 * fixed 256 MiB per-append ceiling, checked before mutation. */
Qwen4Status qwen4_gpu_forward_create(const Gguf *,const Qwen4Config *,const Qwen4Weights *,
    const GpuWeightStore *,void *device,void *library,uint32_t max_tokens,uint32_t capacity,
    uint32_t max_logits_rows,Qwen4GpuForward **,Qwen4Error *);
/* Same append/prefix/logits semantics as G1. Synchronous, single caller.
 * Traces and logits are published only after successful completion. Failed
 * GPU commands poison all state; reset zeros histories and advances epoch. */
Qwen4Status qwen4_gpu_forward(Qwen4GpuForward *,const int32_t *,size_t count,uint32_t layers,
    Qwen4F32Mut logits,Qwen4GraphTrace,void *,Qwen4Error *);
/* Same append, but only the greedy id of the LAST token's logits comes back: the argmax
 * runs on the GPU (ties -> first index, non-finite -> QW4_GPU_ERROR). QW4_REF_KERNELS
 * naming `argmax` reads the row back and scans it on the CPU instead. */
Qwen4Status qwen4_gpu_forward_argmax(Qwen4GpuForward *,const int32_t *,size_t count,uint32_t layers,
    int32_t *id,Qwen4Error *);
Qwen4StateInfo qwen4_gpu_forward_info(const Qwen4GpuForward *);
Qwen4Status qwen4_gpu_forward_reset(Qwen4GpuForward *,Qwen4Error *);
void qwen4_gpu_forward_destroy(Qwen4GpuForward *);
/* R5.1b: the expert pool whose regions the store attached (gpu_weights_attach_pool); borrowed, it
 * must outlive the graph. Required before the first append when any expert tensor is streamed. */
struct QW4ExpertPool;
void qwen4_gpu_forward_set_pool(Qwen4GpuForward *, struct QW4ExpertPool *);
/* Allocated Metal workspace/state/constants (excludes borrowed weights and
 * optional transient trace storage). SSD staging is owned separately by E. */
size_t qwen4_gpu_forward_bytes(const Qwen4GpuForward *);
/* One predictor block, own QSA cache. Enable only on a fresh graph. No PLE
 * and no additional weight load. A failure requires destroying the graph. */
Qwen4Status qwen4_gpu_mtp_enable(Qwen4GpuForward *,Qwen4Error *);
/* Pair residual rows from the LAST successful full trunk append with next
 * tokens. Predictor position must equal that append's start + offset. Rows
 * of logits (0..count) are the final rows. Does not overwrite trunk residuals.
 * Errors after admission poison both graphs; reset recovers. */
Qwen4Status qwen4_gpu_mtp(Qwen4GpuForward *,size_t offset,const int32_t *next_tokens,
    size_t count,Qwen4F32Mut logits,Qwen4Error *);
Qwen4StateInfo qwen4_gpu_mtp_info(const Qwen4GpuForward *);
/* One reusable host snapshot, excluding weights and disk tables. Save only
 * idle/valid state. Restore only within two successful trunk tokens, with
 * predictor unchanged. Saving/reset/restore invalidates previous handles.
 * Restore advances the trunk epoch and restores PLE hash history as well as
 * all device histories and the last trunk residual. */
/* R4 speculative verify: 1 confirmed token + up to 3 drafts in one append.
 * Needs MTP enabled (it allocates the per-row snapshots). The append is an
 * ordinary one (logits rows as qwen4_gpu_forward) that also records, for rows
 * 0..count-2, the recurrent state and convolution/n-gram histories. keep(n)
 * then keeps the first n rows (1..count) and drops the rest without replay:
 * the state equals n ordinary appends of those tokens, bit for bit. A pending
 * verify ends at keep, reset, or the next trunk append. */
#define QWEN4_VERIFY_ROWS 4
Qwen4Status qwen4_gpu_forward_verify(Qwen4GpuForward *,const int32_t *,size_t count,
    Qwen4F32Mut logits,Qwen4Error *);
Qwen4Status qwen4_gpu_forward_keep(Qwen4GpuForward *,size_t keep,Qwen4Error *);
/* R4 chained draft: the predictor on its own last output row and `token`, one
 * position past its last row. Speculative: mtp_rewind(position) must return the
 * predictor to its confirmed position before the next qwen4_gpu_mtp. */
Qwen4Status qwen4_gpu_mtp_chain(Qwen4GpuForward *,int32_t token,Qwen4F32Mut logits,Qwen4Error *);
Qwen4Status qwen4_gpu_mtp_rewind(Qwen4GpuForward *,uint64_t position,Qwen4Error *);
typedef struct { const void *owner; uint64_t epoch,serial,position,mtp_position; } Qwen4GpuCheckpoint;
Qwen4Status qwen4_gpu_checkpoint_save(Qwen4GpuForward *,Qwen4GpuCheckpoint *,Qwen4Error *);
Qwen4Status qwen4_gpu_checkpoint_restore(Qwen4GpuForward *,Qwen4GpuCheckpoint,Qwen4Error *);
size_t qwen4_gpu_checkpoint_bytes(const Qwen4GpuForward *);
/* Saved sessions (pf-agent): the whole state to a file and back. Every state
 * buffer (KV and indexer caches, recurrent and convolution states, PLE n-gram
 * histories, the MTP predictor's), the residual rows of the last trunk append
 * and the positions. Write needs an idle, valid graph with no pending verify.
 * Read needs a graph created with the same model, capacity, chunk and MTP
 * setting (checked against the buffer sizes, not the weights); it replaces
 * the current state, and on failure leaves the graph reset. Continuing after a
 * read equals continuing the graph that wrote the file, bit for bit. */
#include <stdio.h>
Qwen4Status qwen4_gpu_state_write(Qwen4GpuForward *,FILE *,Qwen4Error *);
Qwen4Status qwen4_gpu_state_read(Qwen4GpuForward *,FILE *,Qwen4Error *);
#endif
