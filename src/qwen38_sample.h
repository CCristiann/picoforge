/* qwen38_sample.h — token sampling from one logits row.
 *
 * Greedy decoding (always the argmax) makes a long reasoning trace fall into
 * loops: once a phrase repeats, repeating it becomes even more likely. The
 * sampler draws instead from the softmax of logits/T, restricted to the
 * top_k most likely tokens, then to the smallest set whose mass reaches
 * top_p, then to tokens with p >= min_p * p_max. T <= 0 is the argmax.
 *
 * Exact speculative decoding comes for free: picoforge's MTP drafts are greedy,
 * i.e. the draft distribution is a point mass on d. Speculative sampling then
 * accepts d with probability p(d) and otherwise draws from p with d removed,
 * which is the same as "draw x ~ p; accept iff x == d; else emit x". So the
 * verify loop only swaps its argmax for a draw from the same filtered p.
 */
#ifndef PICOFORGE_QWEN38_SAMPLE_H
#define PICOFORGE_QWEN38_SAMPLE_H
#include <stdint.h>
#include <stdbool.h>

enum { Q38_SAMPLE_MAX_K = 256 };
typedef struct {
    float temperature;      /* <= 0: greedy */
    uint32_t top_k;         /* 1..Q38_SAMPLE_MAX_K (0 reads as the max) */
    float top_p, min_p;     /* (0,1] and [0,1) */
    uint64_t state;         /* xorshift64* state, never 0 */
} Q38Sampler;

/* Qwen's recommendation for the thinking mode: T 0.6, top-k 20, top-p 0.95. */
void q38_sampler_init(Q38Sampler *, float temperature, uint32_t top_k, float top_p, float min_p, uint64_t seed);
static inline __attribute__((unused)) bool q38_sampler_greedy(const Q38Sampler *s) { return !s || s->temperature <= 0.0f; }
/* One id from a row of n logits (all finite: the caller already checked). */
int32_t q38_sample(Q38Sampler *, const float *logits, uint32_t n);
#endif
