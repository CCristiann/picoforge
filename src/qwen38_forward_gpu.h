/* G2: ordinary GPU graph, one G0 command and one completion per append.
 * Inputs/weights remain borrowed until destroy. No MTP execution. */
#ifndef PICOFORGE_QWEN38_FORWARD_GPU_H
#define PICOFORGE_QWEN38_FORWARD_GPU_H
#include "qwen38_forward.h"
typedef struct Qwen38GpuForward Qwen38GpuForward;
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
Qwen38StateInfo qwen38_gpu_forward_info(const Qwen38GpuForward *);
Qwen38Status qwen38_gpu_forward_reset(Qwen38GpuForward *,Qwen38Error *);
void qwen38_gpu_forward_destroy(Qwen38GpuForward *);
/* Allocated Metal workspace/state/constants (excludes borrowed weights and
 * optional transient trace storage). SSD staging is owned separately by E. */
size_t qwen38_gpu_forward_bytes(const Qwen38GpuForward *);
#endif
