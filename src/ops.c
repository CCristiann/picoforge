/* ops.c — the numeric primitives, written the obvious way.
 *
 * Every function here is the slow, readable version. That is deliberate:
 * these are the reference the Metal kernels will be judged against in
 * Phase 2 (CLAUDE.md: chain of oracles), and a reference that is clever is
 * a reference you cannot trust.
 *
 * One design decision runs through all of them: WEIGHTS STAY bf16. They are
 * read straight out of the mmapped checkpoint and widened inside the loop.
 * Materialising fp32 copies would cost 2.4 GB and throw away the mapping —
 * and since widening is a shift (see bf16_to_f32), reading 2 bytes per weight
 * instead of 4 halves the traffic of an operation that is bound by nothing
 * else. Correct and fast agree here, which is rare enough to point out.
 */
#include "picoforge.h"

#include <math.h>

/* out[j] = dot(x, W[j]) for W stored [n_out, n_in] row-major.
 *
 * That layout is PyTorch's (nn.Linear keeps [out_features, in_features] and
 * computes x @ W.T), and it happens to be the one we want: row j is
 * contiguous, so each output is one linear sweep through memory.
 *
 * Arithmetic intensity is 2 flops per 2 bytes read. There is no reuse to
 * exploit, which is why token generation is bound by memory bandwidth alone
 * and why no amount of cleverness in Phase 2 will beat 307 GB/s. */
void matmul(float *out, const float *x, const uint16_t *w, int n_in, int n_out) {
    for (int j = 0; j < n_out; j++) {
        const uint16_t *row = w + (size_t)j * (size_t)n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += x[i] * bf16_to_f32(row[i]);
        out[j] = acc;
    }
}

/* One integer code out of a quantised row. For 4 bits: pick the nibble (low
 * first), then sign-extend it — shifting left so bit 3 lands on bit 7 and
 * shifting back arithmetically turns 0x8..0xF into -8..-1. */
static inline int q_code(const uint8_t *row, int i, int bits) {
    if (bits == 8) return (int8_t)row[i];
    int nib = (row[i >> 1] >> ((i & 1) << 2)) & 0xF;
    return (int8_t)(uint8_t)(nib << 4) >> 4;
}

void matmul_q(float *out, const float *x, const QWeight *w, int n_in, int n_out) {
    const int per_row = (w->bits == 8) ? n_in : n_in / 2;
    const int blocks  = n_in / w->group;
    for (int j = 0; j < n_out; j++) {
        const uint8_t  *row = w->q + (size_t)j * (size_t)per_row;
        const uint16_t *d   = w->d + (size_t)j * (size_t)blocks;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            float wi = bf16_to_f32(d[i / w->group]) * (float)q_code(row, i, w->bits);
            if (w->c) wi *= bf16_to_f32(w->c[i]);
            acc += x[i] * wi;
        }
        out[j] = acc;
    }
}

/* RMSNorm: rescale the row to unit RMS, then apply the learned gain.
 * LayerNorm without mean-centering and without bias — it turned out that
 * rescaling is what matters and recentering was never earning its cost.
 *
 * eps goes INSIDE the sqrt, as in the reference. Outside it would still
 * prevent the division by zero, and would still look correct, and would
 * shift the low digits of every activation in the model.
 *
 * The division is also deliberate. Multiplying by a precomputed reciprocal
 * is what a production engine does and what Phase 2 will measure, but it
 * differs from the oracle in the last ulp. Phase 1 buys parity, not speed. */
void rmsnorm(float *out, const float *x, const uint16_t *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] / rms) * bf16_to_f32(w[i]);
}

/* In-place row softmax. Subtracting the max is free mathematically — the
 * factor cancels between numerator and denominator — and it guarantees
 * every exponent is <= 0, so expf() can never overflow. Without it, a score
 * of 90 alone is enough to produce inf/inf = NaN in fp32. */
void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* RoPE, in place, on one head's vector at absolute position `pos`.
 *
 * Position is encoded by ROTATING pairs of dimensions by an angle
 * proportional to pos. Pair i turns at frequency theta^(-2i/head_dim): the
 * first pairs spin about a radian per token, the last are nearly frozen, so
 * the head reads the sequence at many zoom levels at once. The payoff is
 * that a dot product between Q at position m and K at position n depends
 * only on m - n: absolute positions go in, relative distance comes out.
 *
 * The pairing is the classic trap. HF and Qwen pair dimension i with
 * i + head_dim/2 ("rotate_half"), NOT with its neighbour i+1. Pairing
 * adjacently produces no error, no warning, and wrong logits. */
void rope_apply(float *x, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float inv_freq = powf(theta, -(float)(2 * i) / (float)head_dim);
        float angle = (float)pos * inv_freq;
        float c = cosf(angle), s = sinf(angle);

        float x1 = x[i], x2 = x[i + half];
        x[i]        = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

/* silu(z) = z * sigmoid(z), the config's hidden_act: a ReLU with the corner
 * smoothed off, and slightly negative just below zero. */
float silu(float z) { return z / (1.0f + expf(-z)); }
