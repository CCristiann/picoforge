/* Bounded diagnostic/fault seam, not a public inference ABI. Borrowed views
 * exist only while the graph is idle; tests must never change live weights. */
#ifndef PICOFORGE_QWEN38_G_INTERNAL_H
#define PICOFORGE_QWEN38_G_INTERNAL_H
#include "qwen38_forward_gpu.h"
enum { Q38GS_CONV,Q38GS_GDN,Q38GS_KEY,Q38GS_VALUE,Q38GS_INDEX,Q38GS_BLOCK,Q38GS_PLE,Q38GS_COUNT };
Qwen38GpuF32 q38g_gpu_state(Qwen38GpuForward *,uint32_t layer,uint32_t slot);
Qwen38StateInfo q38g_gpu_ngram_info(const Qwen38GpuForward *,uint32_t layer);
/* Graph-only rewind after complete validation, with no active command. */
Qwen38Status q38g_session_rewind(Qwen38GpuSession *,const void *model,uint64_t epoch,
    uint64_t position,Qwen38Error *);
#endif
