// tensorops_gemm_probe.metal — R3.5 probe: the tile shape the prefill GEMM uses.
// Y[m,n] = sum_k X[m,k] W[n,k] with half operands staged in threadgroup memory,
// FP32 accumulation in a cooperative tensor, K walked in 32-wide steps (the
// matmul2d descriptor's k; a K that is not a multiple of it is truncated in
// silence, so the tail is zero-padded). Layout idea from DwarfStar fc80bd6
// (MIT), metal/qwen4.metal kernel_qwen4_moe_mm_mid_nax_t: left operand = the
// activations (k contiguous, one row per token), right operand = the weights
// stored n-major and read transposed, destination extents (N, M).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define PM 64
#define PN 64
#define PK 32

kernel void probe_gemm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
        device float *y [[buffer(2)]], constant uint3 &mnk [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half xs[PM * PK], ws[PN * PK];
    threadgroup float cs[PM * PN];
    const uint M = mnk.x, N = mnk.y, K = mnk.z, n0 = group.x * PN, m0 = group.y * PM;
    auto tx = tensor(xs, dextents<int32_t, 2>(PK, PM));
    auto tw = tensor(ws, dextents<int32_t, 2>(PK, PN));
    matmul2d<matmul2d_descriptor(PM, PN, PK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto c = mm.get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();
    for (uint16_t i = 0; i < c.get_capacity(); i++) if (c.is_valid_element(i)) c[i] = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += PK) {
        for (uint i = tid; i < PM * PK; i += 128) {
            const uint r = i / PK, k = k0 + i % PK;
            xs[i] = m0 + r < M && k < K ? half(x[(m0 + r) * K + k]) : 0.0h;
        }
        for (uint i = tid; i < PN * PK; i += 128) {
            const uint r = i / PK, k = k0 + i % PK;
            ws[i] = n0 + r < N && k < K ? half(w[(n0 + r) * K + k]) : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sx = tx.slice(0, 0);
        auto sw = tw.slice(0, 0);
        mm.run(sx, sw, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tc = tensor(cs, dextents<int32_t, 2>(PN, PM));
    auto sc = tc.slice(0, 0);
    c.store(sc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < PM * PN; i += 128) {
        const uint r = i / PN, n = i % PN;
        if (m0 + r < M && n0 + n < N) y[(m0 + r) * N + n0 + n] = cs[i];
    }
}
