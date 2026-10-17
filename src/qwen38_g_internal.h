/* Bounded diagnostic/fault seam, not a public inference ABI. Borrowed views
 * exist only while the graph is idle; tests must never change live weights. */
#ifndef PICOFORGE_QWEN38_G_INTERNAL_H
#define PICOFORGE_QWEN38_G_INTERNAL_H
#include "qwen38_forward_gpu.h"
#include "qwen38_profile.h"
enum { Q38GS_CONV,Q38GS_GDN,Q38GS_KEY,Q38GS_VALUE,Q38GS_INDEX,Q38GS_BLOCK,Q38GS_PLE,Q38GS_COUNT };
Qwen38GpuF32 q38g_gpu_state(Qwen38GpuForward *,uint32_t layer,uint32_t slot);
Qwen38StateInfo q38g_gpu_ngram_info(const Qwen38GpuForward *,uint32_t layer);
/* Graph-only rewind after complete validation, with no active command. */
Qwen38Status q38g_session_rewind(Qwen38GpuSession *,const void *model,uint64_t epoch,
    uint64_t position,Qwen38Error *);
/* R0.2 opt-in profile (qwen38_profile.h). ENCODING only: ends the current
 * encoder, opens the next (pass: MTLComputePassDescriptor, NULL = plain) and
 * updates cmd. On failure the caller must abort the command. */
Qwen38Status q38g_session_split(Qwen38GpuSession *,const Qwen38GpuTicket *,void *pass,
    Qwen38GpuCommand *,Qwen38Error *);
/* Attach a borrowed profile (NULL detaches) while idle. */
Qwen38Status q38g_forward_set_profile(Qwen38GpuForward *,Q38Profile *,Qwen38Error *);
/* Closes the current group and opens the next. A no-op, with zero extra
 * encoders, without a profile or outside an ordinary trunk append (MTP). */
void q38g_profile_mark(Qwen38GpuForward *,Qwen38GpuCommand *,uint32_t layer,const char *op,
    uint32_t gguf_type,uint64_t weight_bytes);
#endif
