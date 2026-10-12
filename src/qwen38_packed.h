/* B: one linear contract for packed AND scalar GGUF formats. */
#ifndef PICOFORGE_QWEN38_PACKED_H
#define PICOFORGE_QWEN38_PACKED_H
#include "gguf.h"
#include "qwen38_gpu.h"
typedef struct {
    uint32_t gguf_type;
    size_t k_storage, k_logical, n_out, n_experts;
    size_t row_bytes, expert_bytes, tensor_bytes;
} Q38PackedLayout;
typedef struct { Qwen38CpuSpan span; Q38PackedLayout layout; } Q38PackedCpuView;
typedef struct Q38PackedGpu Q38PackedGpu;
Qwen38Status q38_packed_layout_init(const GgufTensor *, size_t k_logical,
    Q38PackedLayout *, Qwen38Error *);
/* Decode emits k_storage floats. Matmul reads ONLY k_logical columns of X.
 * x=[M,k_logical], y=[M,n_out]; rank 2 byte-strided common views. */
Qwen38Status q38_packed_decode_row(const Q38PackedCpuView *, size_t expert,
    size_t row, float *out, size_t out_elements, Qwen38Error *);
Qwen38Status q38_packed_matmul(const Q38PackedCpuView *, size_t expert,
    Qwen38F32 x, Qwen38F32Mut y, Qwen38Error *);
Qwen38Status q38_packed_gpu_create(void *device, void *library,
    Q38PackedGpu **, Qwen38Error *);
void q38_packed_gpu_free(Q38PackedGpu *); /* only after all commands complete */
Qwen38Status q38_packed_encode(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
/* Weight storage bridge: coordinator implementation in gpu_weights.m.
 * Borrowed read-only mapped bytes, no new mmap/copy and no disk PLE span. */
Qwen38Status qwen38_weights_cpu_span(const GpuWeightStore *,
    const GgufBinding *, Qwen38CpuSpan *, Qwen38Error *);
#endif
