/* test_qwen38_sample.c — the sampler: greedy equivalence, filters, frequencies,
 * and the exact speculative identity used by the MTP verify loop. */
#include "qwen38_sample.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    enum { N = 1000, DRAWS = 200000 };
    static float logits[N];
    for (int j = 0; j < N; j++) logits[j] = -10.0f - (float)(j % 7);
    logits[3] = 2.0f; logits[500] = 1.0f; logits[999] = 0.0f;   /* p ~ 0.665 / 0.245 / 0.090 */
    Q38Sampler s;

    q38_sampler_init(&s, 0.0f, 20, 0.95f, 0.0f, 1);
    CHECK(q38_sample(&s, logits, N) == 3, "greedy must be the argmax");

    /* T 1, no filtering that bites: frequencies match the softmax of the three. */
    q38_sampler_init(&s, 1.0f, 20, 1.0f, 0.0f, 42);
    double z = exp(2.0) + exp(1.0) + exp(0.0), want[3] = {exp(2.0) / z, exp(1.0) / z, 1.0 / z};
    int hits[3] = {0}, other = 0;
    for (int i = 0; i < DRAWS; i++) {
        int32_t t = q38_sample(&s, logits, N);
        if (t == 3) hits[0]++; else if (t == 500) hits[1]++; else if (t == 999) hits[2]++; else other++;
    }
    for (int i = 0; i < 3; i++)
        CHECK(fabs(hits[i] / (double)DRAWS - want[i]) < 0.005, "freq %d: %f vs %f", i, hits[i] / (double)DRAWS, want[i]);
    CHECK(other < DRAWS / 1000, "tail tokens drawn %d times", other);

    /* top-p 0.8 keeps {3, 500} (0.665 < 0.8 <= 0.910); min-p 0.2 drops 999 (0.135 of max). */
    q38_sampler_init(&s, 1.0f, 20, 0.8f, 0.0f, 7);
    int bad = 0; for (int i = 0; i < 20000; i++) { int32_t t = q38_sample(&s, logits, N); bad += t != 3 && t != 500; }
    CHECK(bad == 0, "top-p leaked %d", bad);
    q38_sampler_init(&s, 1.0f, 20, 1.0f, 0.2f, 7);
    bad = 0; for (int i = 0; i < 20000; i++) { int32_t t = q38_sample(&s, logits, N); bad += t != 3 && t != 500; }
    CHECK(bad == 0, "min-p leaked %d", bad);
    q38_sampler_init(&s, 1.0f, 1, 1.0f, 0.0f, 7);
    CHECK(q38_sample(&s, logits, N) == 3, "top-k 1 must be the argmax");

    /* Exact speculation with a greedy draft d = 500: "draw x; accept iff x == d,
     * else emit x" must give the target distribution itself. */
    q38_sampler_init(&s, 1.0f, 20, 1.0f, 0.0f, 9);
    int emit[3] = {0};
    for (int i = 0; i < DRAWS; i++) {
        int32_t x = q38_sample(&s, logits, N), out = x == 500 ? 500 : x;
        emit[out == 3 ? 0 : out == 500 ? 1 : 2]++;
    }
    for (int i = 0; i < 3; i++)
        CHECK(fabs(emit[i] / (double)DRAWS - want[i]) < 0.005, "spec %d: %f vs %f", i, emit[i] / (double)DRAWS, want[i]);

    printf(failures ? "test_qwen38_sample: %d failures\n" : "test_qwen38_sample: ok%.0d\n", failures);
    return failures != 0;
}
