/* qwen38_sample.c — see qwen38_sample.h. */
#include "qwen38_sample.h"
#include <math.h>

void q38_sampler_init(Q38Sampler *s, float temperature, uint32_t top_k, float top_p, float min_p, uint64_t seed) {
    s->temperature = temperature;
    s->top_k = top_k == 0 || top_k > Q38_SAMPLE_MAX_K ? Q38_SAMPLE_MAX_K : top_k;
    s->top_p = top_p > 0.0f && top_p <= 1.0f ? top_p : 1.0f;
    s->min_p = min_p >= 0.0f && min_p < 1.0f ? min_p : 0.0f;
    /* splitmix64 of the seed: nearby seeds give unrelated streams, and never 0 */
    uint64_t z = seed + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    s->state = (z ^ (z >> 31)) | 1;
}
/* Uniform in [0, 1) from the top 24 bits of xorshift64*. */
static float uniform(Q38Sampler *s) {
    uint64_t x = s->state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s->state = x;
    return (float)((x * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}
int32_t q38_sample(Q38Sampler *s, const float *logits, uint32_t n) {
    if (!n) return 0;
    if (q38_sampler_greedy(s)) {   /* first maximum wins, as the CLI's greedy() */
        uint32_t best = 0;
        for (uint32_t j = 1; j < n; j++) if (logits[j] > logits[best]) best = j;
        return (int32_t)best;
    }
    /* 1. top-k by insertion into a descending list: O(n) for small k, since
     * almost every logit fails the first comparison against the k-th. */
    uint32_t k = s->top_k < n ? s->top_k : n, have = 0;
    int32_t id[Q38_SAMPLE_MAX_K]; float v[Q38_SAMPLE_MAX_K];
    for (uint32_t j = 0; j < n; j++) {
        float x = logits[j];
        if (have == k && x <= v[k - 1]) continue;
        uint32_t i = have < k ? have++ : k - 1;
        while (i > 0 && v[i - 1] < x) { v[i] = v[i - 1]; id[i] = id[i - 1]; i--; }
        v[i] = x; id[i] = (int32_t)j;
    }
    /* 2. softmax of logits/T over the k candidates, max subtracted (v[0]). */
    float p[Q38_SAMPLE_MAX_K], sum = 0.0f;
    for (uint32_t i = 0; i < have; i++) { p[i] = expf((v[i] - v[0]) / s->temperature); sum += p[i]; }
    /* 3. min-p (relative to p[0], the max) and top-p (sorted, so a prefix). */
    uint32_t keep = have; float mass = 0.0f;
    for (uint32_t i = 0; i < have; i++) {
        if (i > 0 && p[i] < s->min_p * p[0]) { keep = i; break; }
        mass += p[i];
        if (mass >= s->top_p * sum) { keep = i + 1; break; }
    }
    float kept = 0.0f;
    for (uint32_t i = 0; i < keep; i++) kept += p[i];
    /* 4. inverse CDF; the last kept token absorbs rounding. */
    float u = uniform(s) * kept;
    for (uint32_t i = 0; i + 1 < keep; i++) { if (u < p[i]) return id[i]; u -= p[i]; }
    return id[keep - 1];
}
