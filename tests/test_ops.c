/* test_ops.c — print what src/ops.c computes, for tests/test_ops.py to judge.
 *
 * This binary makes no claims. It generates deterministic inputs, runs each
 * primitive and prints the result; the oracle's own functions decide whether
 * those numbers are right. Keeping the judgement on the Python side means we
 * compare against the reference, not against a second guess of our own.
 *
 * Inputs come from an integer LCG, and every value is a multiple of 1/1024 in
 * [-1, 1). Integers are exact in both languages and so are binary fractions,
 * so C and Python start from bit-identical inputs and every difference in the
 * output is attributable to the operation itself.
 *
 * Sizes and seeds below are mirrored verbatim in test_ops.py. They must stay
 * in step: the PRNG is consumed in order.
 */
#include "picoforge.h"

#include <stdio.h>

static uint32_t rng_state;

static void rng_seed(uint32_t s) { rng_state = s; }

static float rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 16) % 2048u) / 1024.0f - 1.0f;
}

/* A weight as the checkpoint stores it: an fp32 value truncated to its top
 * 16 bits. Round-trips through bf16_to_f32 exactly, by construction. */
static uint16_t rnd_bf16(void) {
    float f = rnd();
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static void emit(const char *name, const float *v, int n) {
    printf("%s", name);
    for (int i = 0; i < n; i++) printf(" %.9g", (double)v[i]);
    printf("\n");
}

int main(void) {
    float x[64], w_f[64], out[64];
    uint16_t w[64];

    /* rmsnorm: n=16, eps=1e-6 */
    rng_seed(1);
    for (int i = 0; i < 16; i++) x[i] = rnd();
    for (int i = 0; i < 16; i++) w[i] = rnd_bf16();
    rmsnorm(out, x, w, 16, 1e-6f);
    emit("rmsnorm", out, 16);

    /* matmul: n_in=8, n_out=5, W row-major [n_out][n_in] */
    rng_seed(2);
    for (int i = 0; i < 8; i++) x[i] = rnd();
    for (int i = 0; i < 40; i++) w[i] = rnd_bf16();
    matmul(out, x, w, 8, 5);
    emit("matmul", out, 5);

    /* softmax: n=12, scaled up so the max-subtraction actually matters */
    rng_seed(3);
    for (int i = 0; i < 12; i++) out[i] = rnd() * 8.0f;
    softmax(out, 12);
    emit("softmax", out, 12);

    /* rope: head_dim=16 at position 7, theta as in Qwen3's config */
    rng_seed(4);
    for (int i = 0; i < 16; i++) out[i] = rnd();
    rope_apply(out, 16, 7, 1000000.0f);
    emit("rope", out, 16);

    /* silu: n=8 */
    rng_seed(5);
    for (int i = 0; i < 8; i++) w_f[i] = rnd() * 4.0f;
    for (int i = 0; i < 8; i++) out[i] = silu(w_f[i]);
    emit("silu", out, 8);

    return 0;
}
