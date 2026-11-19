/* Bounded diagnostic/fault seam, not a public inference ABI. Borrowed views
 * exist only while the graph is idle; tests must never change live weights. */
#ifndef PICOFORGE_QWEN4_GRAPH_INTERNAL_H
#define PICOFORGE_QWEN4_GRAPH_INTERNAL_H
#include "qwen4_forward_gpu.h"
#include "qwen4_profile.h"
enum { QW4GS_CONV,QW4GS_GDN,QW4GS_KEY,QW4GS_VALUE,QW4GS_INDEX,QW4GS_BLOCK,QW4GS_PLE,QW4GS_COUNT };
Qwen4GpuF32 qw4g_gpu_state(Qwen4GpuForward *,uint32_t layer,uint32_t slot);
Qwen4StateInfo qw4g_gpu_ngram_info(const Qwen4GpuForward *,uint32_t layer);
/* Graph-only rewind after complete validation, with no active command. */
Qwen4Status qw4g_session_rewind(Qwen4GpuSession *,const void *model,uint64_t epoch,
    uint64_t position,Qwen4Error *);
/* Saved sessions: a fresh (reset, idle) session jumps to position. */
Qwen4Status qw4g_session_seek(Qwen4GpuSession *,const void *model,uint64_t position,Qwen4Error *);
/* R0.2 opt-in profile (qwen4_profile.h). ENCODING only: ends the current
 * encoder, opens the next (pass: MTLComputePassDescriptor, NULL = plain) and
 * updates cmd. On failure the caller must abort the command. */
Qwen4Status qw4g_session_split(Qwen4GpuSession *,const Qwen4GpuTicket *,void *pass,
    Qwen4GpuCommand *,Qwen4Error *);
/* D1: ENCODING only. Commits the command buffer encoded so far and continues the same command
 * in a new one (updates cmd). The caller aborts the command on failure. */
Qwen4Status qw4g_session_flush(Qwen4GpuSession *,const Qwen4GpuTicket *,Qwen4GpuCommand *,Qwen4Error *);
Qwen4Status qw4g_session_sync(Qwen4GpuSession *,const Qwen4GpuTicket *,Qwen4GpuCommand *,Qwen4Error *);
/* R5.2b: ENCODING only. Commits what is encoded; the GPU then waits until fn(ctx) has run on the
 * listener queue (after all earlier work completed). Updates cmd; the caller aborts on failure. */
Qwen4Status qw4g_session_gate(Qwen4GpuSession *,const Qwen4GpuTicket *,Qwen4GpuCommand *,void (*fn)(void *),void *ctx,Qwen4Error *);
/* Attach a borrowed profile (NULL detaches) while idle. */
Qwen4Status qw4g_forward_set_profile(Qwen4GpuForward *,QW4Profile *,Qwen4Error *);
/* Closes the current group and opens the next. A no-op, with zero extra
 * encoders, without a profile or outside an ordinary trunk append (MTP). */
void qw4g_profile_mark(Qwen4GpuForward *,Qwen4GpuCommand *,uint32_t layer,const char *op,
    uint32_t gguf_type,uint64_t weight_bytes);
/* T15c: the argmax kernel alone on one host row (n >= 1). */
Qwen4Status qw4g_argmax_probe(void *device,void *library,const float *row,uint32_t n,int32_t *id,
    uint32_t *error_bits,Qwen4Error *);
#endif
