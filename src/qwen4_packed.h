/* B: one linear contract for packed AND scalar GGUF formats. */
#ifndef PICOFORGE_QWEN4_PACKED_H
#define PICOFORGE_QWEN4_PACKED_H
#include "gguf.h"
#include "qwen4_gpu.h"
typedef struct {
    uint32_t gguf_type;
    size_t k_storage, k_logical, n_out, n_experts;
    size_t row_bytes, expert_bytes, tensor_bytes;
} QW4PackedLayout;
typedef struct { Qwen4CpuSpan span; QW4PackedLayout layout; } QW4PackedCpuView;
typedef struct QW4PackedGpu QW4PackedGpu;
Qwen4Status qw4_packed_layout_init(const GgufTensor *, size_t k_logical,
    QW4PackedLayout *, Qwen4Error *);
/* Decode emits k_storage floats. Matmul reads ONLY k_logical columns of X.
 * x=[M,k_logical], y=[M,n_out]; rank 2 byte-strided common views. */
Qwen4Status qw4_packed_decode_row(const QW4PackedCpuView *, size_t expert,
    size_t row, float *out, size_t out_elements, Qwen4Error *);
Qwen4Status qw4_packed_matmul(const QW4PackedCpuView *, size_t expert,
    Qwen4F32 x, Qwen4F32Mut y, Qwen4Error *);
Qwen4Status qw4_packed_gpu_create(void *device, void *library,
    QW4PackedGpu **, Qwen4Error *);
void qw4_packed_gpu_free(QW4PackedGpu *); /* only after all commands complete */
/* Q8_0, F16 and F32 run the fast matvecs (not bit-identical to the oracle
 * kernel) unless QW4_REF_KERNELS selects `linear` at creation; other formats
 * run the oracle. */
Qwen4Status qw4_packed_encode(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const QW4PackedLayout *,
    size_t expert, Qwen4GpuF32 x, Qwen4GpuF32 y, Qwen4Error *);
/* Indexed rows, for MoE without reading route IDs back: row p of y uses
 * expert ids[p] (uint32, GPU span of rows(y) entries) and row p/x_divisor of
 * x. An ID outside the tensor sets QW4_GPU_BOUNDS and leaves that row unset.
 * IQ2_XXS, Q2_K, Q4_K and MXFP4 run the indexed fast matvecs (not
 * bit-identical to the oracle kernel) unless QW4_REF_KERNELS selects
 * `indexed` at creation; other formats run the oracle. */
Qwen4Status qw4_packed_encode_indexed(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const QW4PackedLayout *,
    Qwen4GpuSpan expert_ids, size_t x_divisor, Qwen4GpuF32 x, Qwen4GpuF32 y, Qwen4Error *);
/* Gather embedding rows using device uint32 token IDs; y=[T,k_logical].
 * One matrix only; invalid IDs set the command bounds error. */
Qwen4Status qw4_packed_lookup(QW4PackedGpu *, const Qwen4GpuCommand *,
    const GpuWeightStore *, const GgufBinding *, const QW4PackedLayout *,
    Qwen4GpuSpan ids, Qwen4GpuF32 y, Qwen4Error *);
/* Weight storage bridge, implemented in gpu_weights.m.
 * Borrowed read-only mapped bytes, no new mmap/copy and no disk PLE span. */
Qwen4Status qwen4_weights_cpu_span(const GpuWeightStore *,
    const GgufBinding *, Qwen4CpuSpan *, Qwen4Error *);
#endif
