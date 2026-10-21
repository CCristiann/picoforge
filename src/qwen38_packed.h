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
/* Q8_0, F16 and F32 run the fast matvecs (not bit-identical to the oracle
 * kernel) unless Q38_REF_KERNELS selects `linear` at creation; other formats
 * run the oracle. */
Qwen38Status q38_packed_encode(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
/* Indexed rows, for MoE without reading route IDs back: row p of y uses
 * expert ids[p] (uint32, GPU span of rows(y) entries) and row p/x_divisor of
 * x. An ID outside the tensor sets Q38_GPU_BOUNDS and leaves that row unset.
 * IQ2_XXS, Q2_K, Q4_K and MXFP4 run the indexed fast matvecs (not
 * bit-identical to the oracle kernel) unless Q38_REF_KERNELS selects
 * `indexed` at creation; other formats run the oracle. */
Qwen38Status q38_packed_encode_indexed(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    Qwen38GpuSpan expert_ids, size_t x_divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *);
/* Gather embedding rows using device uint32 token IDs; y=[T,k_logical].
 * One matrix only; invalid IDs set the command bounds error. */
Qwen38Status q38_packed_lookup(Q38PackedGpu *, const Qwen38GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const Q38PackedLayout *,
    Qwen38GpuSpan ids, Qwen38GpuF32 y, Qwen38Error *);
/* Weight storage bridge: coordinator implementation in gpu_weights.m.
 * Borrowed read-only mapped bytes, no new mmap/copy and no disk PLE span. */
Qwen38Status qwen38_weights_cpu_span(const GpuWeightStore *,
    const GgufBinding *, Qwen38CpuSpan *, Qwen38Error *);
#endif
