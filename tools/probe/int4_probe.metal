/* int4_probe.metal — what does TensorOps actually READ from an int4 tensor?
 *
 * The header says int4b_format is accepted and nothing else: not the nibble
 * order, not the sign encoding. The compile probe cannot answer either, so
 * this asks the silicon. A is the 32x32 identity, so
 *
 *     C[m, 0] = sum_k A[m, k] * q[0, k] = q[0, m]
 *
 * and the kernel hands back each code exactly as it decoded it.
 *
 * Two findings from the compile probe are encoded here: the storage pointer
 * for a sub-byte tensor is `device uchar *` (NOT `device int4b_format *`,
 * which fails with "no matching constructor"), and element types carry no
 * const.
 */
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constexpr constant auto probe_desc = matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent),
                                                false, /*transpose_right=*/true);

kernel void probe_i4(device float  *C [[buffer(0)]],
                     device bfloat *A [[buffer(1)]],
                     device uchar  *B [[buffer(2)]]) {
    matmul2d<probe_desc, execution_simdgroups<4>> op;
    auto tA = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(32, 32));
    auto tB = tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(32, 1));
    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(1, 32));
    op.run(tA, tB, tC);
}

kernel void probe_i8(device float  *C [[buffer(0)]],
                     device float  *A [[buffer(1)]],
                     device int8_t *B [[buffer(2)]]) {
    matmul2d<probe_desc, execution_simdgroups<4>> op;
    auto tA = tensor<device float, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(32, 32));
    auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(32, 1));
    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(1, 32));
    op.run(tA, tB, tC);
}
