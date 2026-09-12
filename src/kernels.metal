/* kernels.metal — the matmul, version one of three.
 *
 * The problem, stated once for all three versions:
 *
 *     C[M, N] = A[M, K] * B[N, K]^T
 *
 * A is activations in fp32, B is weights in bf16 exactly as they sit in the
 * mmapped checkpoint, C is fp32. B is indexed [n][k] rather than [k][n]
 * because that is PyTorch's storage order and transposing 1.2 GB to please a
 * kernel would be a strange way to save time.
 *
 * Weights stay bf16 and are widened in the inner loop, as on the CPU. At
 * 2 flops per 2 bytes there is no arithmetic intensity to speak of, so the
 * bytes moved are the whole story and halving them is the single biggest
 * lever available.
 */
#include <metal_stdlib>
using namespace metal;

struct MatmulDims { uint M, N, K; };

/* bf16 is the top half of an fp32: shift left 16 and reinterpret. Identical
 * to the CPU's bf16_to_f32, deliberately — the two paths must agree bit for
 * bit, and the cheapest way to guarantee that is to do the same thing. */
inline float bf16_to_float(ushort h) {
    return as_type<float>(uint(h) << 16);
}

/* Version 1: naive. One thread per output element, one loop over K.
 *
 * This is the baseline the other two are measured against, and its job is to
 * be obviously correct rather than fast. It is also instructive about what
 * goes wrong: every thread in a SIMD group reads the SAME row of A (they
 * share m) but a DIFFERENT row of B, so the B reads are strided by K floats
 * and touch a fresh cache line each. Nothing is reused, nothing is
 * coalesced, and the machine spends its time waiting. */
kernel void matmul_naive(device float         *C [[buffer(0)]],
                         device const float   *A [[buffer(1)]],
                         device const ushort  *B [[buffer(2)]],
                         constant MatmulDims  &d [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint n = gid.x;
    const uint m = gid.y;
    if (m >= d.M || n >= d.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < d.K; k++)
        acc += A[m * d.K + k] * bf16_to_float(B[n * d.K + k]);

    C[m * d.N + n] = acc;
}

/* ------------------------------------------------------------ version 2
 * Tiled, using the SIMD-group matrix units.
 *
 * What the naive kernel wastes: it reads A[m][k] once per output column and
 * B[n][k] once per output row. Every value crosses the memory system N or M
 * times when it needs to cross once. A tile fixes that — stage a 32x8 block
 * of A and a 32x8 block of B in threadgroup memory, and every thread in the
 * group reads them from there instead of from DRAM.
 *
 * On top of the tiling sit the matrix units. simdgroup_multiply_accumulate
 * does an 8x8 by 8x8 multiply-accumulate across the 32 lanes of a SIMD group
 * in one instruction. A threadgroup here is 128 threads = 4 SIMD groups, each
 * owning 8 rows of a 32x32 output tile and holding four 8x8 accumulators in
 * registers for the whole K loop. The accumulators never touch memory until
 * the end.
 *
 * Two API details that were probed rather than assumed:
 *
 *   - float x bfloat -> float multiply-accumulate is a real overload, so the
 *     activations stay fp32 and the weights stay bf16. No conversion pass and
 *     no precision given away.
 *   - simdgroup_load takes a transpose flag, so B[n][k] can be read as the
 *     B^T[k][n] the product wants. Transposing 1.2 GB of weights in memory to
 *     please a kernel would have been a strange way to save time.
 *
 * Everything is staged through threadgroup memory rather than loaded from
 * device memory directly, which costs a little and buys the edge cases: M, N
 * and K need not be multiples of anything, because out-of-range elements are
 * staged as zeros and a zero contributes nothing to a sum.
 */
#define TM 32
#define TN 32
#define TK 8
#define TG_THREADS 128

kernel void matmul_simdgroup(device float         *C [[buffer(0)]],
                             device const float   *A [[buffer(1)]],
                             device const bfloat  *B [[buffer(2)]],
                             constant MatmulDims  &d [[buffer(3)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint  tid  [[thread_index_in_threadgroup]],
                             uint  sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float  As[TM * TK];
    threadgroup bfloat Bs[TN * TK];
    threadgroup float  Cs[TM * TN];

    const uint m0 = tgid.y * TM;
    const uint n0 = tgid.x * TN;

    simdgroup_float8x8 acc[4];
    for (uint j = 0; j < 4; j++)
        acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < d.K; k0 += TK) {
        for (uint i = tid; i < TM * TK; i += TG_THREADS) {
            const uint m = m0 + i / TK, k = k0 + i % TK;
            As[i] = (m < d.M && k < d.K) ? A[m * d.K + k] : 0.0f;
        }
        for (uint i = tid; i < TN * TK; i += TG_THREADS) {
            const uint n = n0 + i / TK, k = k0 + i % TK;
            Bs[i] = (n < d.N && k < d.K) ? B[n * d.K + k] : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma;
        simdgroup_load(ma, As + sgid * 8 * TK, TK);
        for (uint j = 0; j < 4; j++) {
            simdgroup_bfloat8x8 mb;
            /* Bs holds B[n][k]; the transpose flag turns that 8x8 block into
             * the B^T[k][n] block the accumulation wants. */
            simdgroup_load(mb, Bs + j * 8 * TK, TK, ulong2(0), true);
            simdgroup_multiply_accumulate(acc[j], ma, mb, acc[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Through threadgroup memory again, so the store can be bounds-checked:
     * simdgroup_store writes a full 8x8 whether or not the matrix has room. */
    for (uint j = 0; j < 4; j++)
        simdgroup_store(acc[j], Cs + sgid * 8 * TN + j * 8, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < TM * TN; i += TG_THREADS) {
        const uint m = m0 + i / TN, n = n0 + i % TN;
        if (m < d.M && n < d.N) C[m * d.N + n] = Cs[i];
    }
}
