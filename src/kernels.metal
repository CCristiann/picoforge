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

/* An empty kernel, dispatched over a single thread. It exists to measure the
 * cost of asking the GPU to do anything at all: encode a command buffer,
 * submit it, have the GPU pick it up, run nothing, and signal completion.
 *
 * That floor decides whether a measurement means anything. A 1024x1024 decode
 * matmul moves 2 MB, which at 307 GB/s is under 7 microseconds of real work.
 * If the floor is larger than that, every decode number in the table is a
 * measurement of dispatch latency wearing a matmul's name. */
kernel void empty_kernel(device float *sink [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) sink[0] = 0.0f;
}

/* ------------------------------------------------------------ version 3
 * Metal 4 TensorOps, on the M5's Neural Accelerators.
 *
 * Versions 1 and 2 moved data by hand. This one hands the whole tile to a
 * hardware unit: describe the shapes, wrap the buffers in tensors, call run.
 * There is no inner loop to write, which is the point — and also the reason
 * it is the version most likely to be wrong for reasons nobody can see.
 *
 * Four things were probed rather than assumed, per the fresh-API rule, and
 * all four cost real time:
 *
 * 1. tensor_inline, not the default tensor_handle. The default expects an
 *    MTLTensor bound from the host; tensor_inline wraps a plain device
 *    pointer, which is what an engine that already has buffers wants.
 *
 * 2. NO const on the element types. `tensor<device const float, ...>` makes
 *    leftValueType const-qualified, the dispatch chain's is_same<T, float>
 *    tests all fail, and control falls through to a terminal static_assert
 *    that reports "Unsupported type" naming the DESTINATION type. The error
 *    blames the wrong parameter entirely; only reading the header finds it.
 *
 * 3. float x bfloat -> float is supported here, unlike the older reports of
 *    strict type matching. Activations stay fp32 and weights stay bf16.
 *
 * 4. The dispatch chain also accepts bfloat x int4b_format and
 *    bfloat x int8_t with fp32 accumulation — 4-bit and 8-bit weights
 *    straight into the matrix units, no dequantisation pass. That is Phase 3
 *    arriving early, and it is worth knowing before designing the format.
 *
 * The descriptor's transpose_right does what the simdgroup version's
 * transpose flag did: read the weights as stored, [n][k], and let the unit
 * treat them as the B^T the product needs.
 */
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

kernel void matmul_tensorops(device float        *C [[buffer(0)]],
                             device float        *A [[buffer(1)]],
                             device bfloat       *B [[buffer(2)]],
                             constant MatmulDims &d [[buffer(3)]],
                             uint2 tgid [[threadgroup_position_in_grid]]) {
    /* Tile shape is compile-time; K is dynamic and comes from the tensors. */
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                              /*transpose_left=*/false,
                                              /*transpose_right=*/true);
    matmul2d<desc, execution_simdgroups<4>> op;

    /* Extents are (columns, rows): A is M x K, B is N x K, C is M x N.
     * The tensors carry their own bounds, so slices at the edge of a matrix
     * whose size is not a multiple of the tile are handled by the operation
     * rather than by staging zeros as version 2 has to. */
    auto tA = tensor<device float,  dextents<int32_t, 2>, tensor_inline>(
                  A, dextents<int32_t, 2>(int(d.K), int(d.M)));
    auto tB = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
                  B, dextents<int32_t, 2>(int(d.K), int(d.N)));
    auto tC = tensor<device float,  dextents<int32_t, 2>, tensor_inline>(
                  C, dextents<int32_t, 2>(int(d.N), int(d.M)));

    auto mA = tA.slice(0, int(tgid.y) * TM);
    auto mB = tB.slice(0, int(tgid.x) * TN);
    auto mC = tC.slice(int(tgid.x) * TN, int(tgid.y) * TM);

    op.run(mA, mB, mC);
}

/* =======================================================================
 * The rest of the forward pass.
 *
 * These are the obvious versions, written to be judged by the CPU oracle
 * before anything clever happens to them. Together with the matmul they let
 * a whole layer run without the result ever leaving the GPU, which is the
 * point: 28 layers x 7 matmuls is 196 dispatches per token, and at 6 us of
 * command-buffer latency each that would be 1.2 ms of pure waiting. They all
 * go into ONE command buffer instead.
 * ======================================================================= */

struct NormDims  { uint rows, n; float eps; };
struct RopeDims  { uint n, heads, head_dim, dim, pos; float theta, eps; };
struct AttnDims  { uint n, heads, kv_heads, head_dim, q_dim, kv_dim, pos, max_seq; float scale; };
struct ElemDims  { uint count; };
struct EmbedDims { uint n, hidden; };

/* Embedding: a row lookup, widened to fp32. One thread per element. */
kernel void embed_lookup(device float        *x      [[buffer(0)]],
                         device const bfloat *table  [[buffer(1)]],
                         device const int    *tokens [[buffer(2)]],
                         constant EmbedDims  &d      [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint t = gid / d.hidden, i = gid % d.hidden;
    if (t >= d.n) return;
    x[gid] = float(table[uint(tokens[t]) * d.hidden + i]);
}

/* RMSNorm, one threadgroup per row.
 *
 * The sum of squares is a reduction over 1024 elements, so it is done as a
 * tree in threadgroup memory rather than by one thread looping. eps stays
 * INSIDE the sqrt, as on the CPU and in the reference: outside it would
 * still prevent the division by zero, still look correct, and still shift
 * the low digits of every activation in the model. */
#define NORM_THREADS 256

kernel void rmsnorm_rows(device float        *out [[buffer(0)]],
                         device const float  *in  [[buffer(1)]],
                         device const bfloat *w   [[buffer(2)]],
                         constant NormDims   &d   [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[NORM_THREADS];

    const uint base = row * d.n;
    float acc = 0.0f;
    for (uint i = tid; i < d.n; i += NORM_THREADS) {
        float v = in[base + i];
        acc += v * v;
    }
    part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = NORM_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms = sqrt(part[0] / float(d.n) + d.eps);
    for (uint i = tid; i < d.n; i += NORM_THREADS)
        out[base + i] = (in[base + i] / rms) * float(w[i]);
}

/* Qwen3's per-head QK-norm followed by RoPE, in place.
 *
 * One threadgroup per (token, head), head_dim threads. The two must happen in
 * this order and never touch V.
 *
 * The rotation uses the ABSOLUTE position pos + t. This is where a K/V cache
 * goes wrong most easily: rotating by t alone is invisible during prefill,
 * because the whole prompt starts at zero, and only breaks once generation
 * begins and every new token believes it is at the start of the sequence. */
kernel void qk_norm_rope(device float        *x [[buffer(0)]],
                         device const bfloat *w [[buffer(1)]],
                         constant RopeDims   &d [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint  tid  [[thread_index_in_threadgroup]]) {
    threadgroup float part[128];
    threadgroup float rms_shared;

    const uint t = tgid.y, h = tgid.x;
    if (t >= d.n || h >= d.heads) return;

    device float *v = x + t * d.dim + h * d.head_dim;

    float s = (tid < d.head_dim) ? v[tid] * v[tid] : 0.0f;
    part[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 64; k > 0; k >>= 1) {
        if (tid < k) part[tid] += part[tid + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) rms_shared = sqrt(part[0] / float(d.head_dim) + d.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float normed = 0.0f;
    if (tid < d.head_dim) normed = (v[tid] / rms_shared) * float(w[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < d.head_dim) v[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* rotate_half: dimension i pairs with i + head_dim/2, NOT with i + 1.
     * The wrong pairing produces no error and wrong logits. */
    const uint half_d = d.head_dim / 2;
    if (tid < half_d) {
        const float inv_freq = pow(d.theta, -float(2 * tid) / float(d.head_dim));
        const float angle = float(d.pos + t) * inv_freq;
        const float c = cos(angle), sn = sin(angle);
        const float x1 = normed, x2 = v[tid + half_d];
        v[tid]          = x1 * c - x2 * sn;
        v[tid + half_d] = x2 * c + x1 * sn;
    }
}

/* Causal attention over the K/V cache. One threadgroup per (token, head).
 *
 * Causality is structural: the key loop stops at the query's own absolute
 * position. There is no mask to build and therefore none to build wrong.
 *
 * GQA is an index, not a copy: query head h reads KV head h / group. */
#define ATTN_THREADS 128
#define ATTN_MAX_CTX 2048

kernel void attention(device float        *out [[buffer(0)]],
                      device const float  *q   [[buffer(1)]],
                      device const float  *kc  [[buffer(2)]],
                      device const float  *vc  [[buffer(3)]],
                      constant AttnDims   &d   [[buffer(4)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint  tid  [[thread_index_in_threadgroup]]) {
    threadgroup float score[ATTN_MAX_CTX];
    threadgroup float part[ATTN_THREADS];
    threadgroup float shared_max, shared_sum;

    const uint t = tgid.y, h = tgid.x;
    if (t >= d.n || h >= d.heads) return;

    const uint abs_t = d.pos + t;
    const uint len   = abs_t + 1;
    const uint kvh   = h / (d.heads / d.kv_heads);
    device const float *qh = q + t * d.q_dim + h * d.head_dim;

    for (uint j = tid; j < len; j += ATTN_THREADS) {
        device const float *kh = kc + j * d.kv_dim + kvh * d.head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < d.head_dim; i++) dot += qh[i] * kh[i];
        score[j] = dot * d.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Stable softmax: subtract the row max first. Free mathematically, and
     * it keeps every exponent <= 0 so exp can never overflow. */
    float m = -INFINITY;
    for (uint j = tid; j < len; j += ATTN_THREADS) m = max(m, score[j]);
    part[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] = max(part[tid], part[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_max = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint j = tid; j < len; j += ATTN_THREADS) {
        score[j] = exp(score[j] - shared_max);
        sum += score[j];
    }
    part[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_sum = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Weighted sum of values: one thread per output dimension, each walking
     * the whole context. */
    for (uint i = tid; i < d.head_dim; i += ATTN_THREADS) {
        float acc = 0.0f;
        for (uint j = 0; j < len; j++)
            acc += score[j] * vc[j * d.kv_dim + kvh * d.head_dim + i];
        out[t * d.q_dim + h * d.head_dim + i] = acc / shared_sum;
    }
}

/* SwiGLU's elementwise half: gate <- silu(gate) * up. */
kernel void swiglu(device float       *gate [[buffer(0)]],
                   device const float *up   [[buffer(1)]],
                   constant ElemDims  &d    [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    const float z = gate[i];
    gate[i] = (z / (1.0f + exp(-z))) * up[i];
}

/* The residual add: x <- x + delta. */
kernel void add_residual(device float       *x     [[buffer(0)]],
                         device const float *delta [[buffer(1)]],
                         constant ElemDims  &d     [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    x[i] += delta[i];
}
