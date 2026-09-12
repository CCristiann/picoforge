/* generate.c — sampling, the chat template, and the generation loop.
 *
 * This is where the engine stops being a function of a prompt and becomes a
 * process: prefill once, then one forward pass per token, each reading a
 * cache that grows underneath it.
 */
#include "picoforge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* xorshift64*. Small, fast, and entirely adequate for picking tokens — this
 * is not a place that needs cryptographic randomness, it needs reproducible
 * randomness, which a named seed gives. */
static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_float(uint64_t *s) {
    return (float)(rng_next(s) >> 40) / 16777216.0f;   /* 24 bits, in [0, 1) */
}

/* Pick one token. Temperature, then top-k, then top-p, in that order —
 * the order HuggingFace uses, and it matters: applying top-p first would
 * measure the nucleus on an unsharpened distribution.
 *
 * No full sort. Sorting 151936 logits per token to keep 20 of them would
 * dominate the sampling cost; a single pass maintaining a k-sized sorted
 * array does the same job. Temperature is monotonic, so it cannot change
 * WHICH tokens are in the top k — only their weights — so the selection
 * runs on raw logits and the softmax comes after. */
int sample(const float *logits, int vocab, float temperature, float top_p,
           int top_k, uint64_t *rng) {
    if (temperature <= 0.0f) {                          /* greedy */
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    if (top_k <= 0 || top_k > vocab) top_k = vocab;
    int *idx = malloc((size_t)top_k * sizeof *idx);
    float *val = malloc((size_t)top_k * sizeof *val);
    if (!idx || !val) die("out of memory sampling from %d logits", vocab);

    int m = 0;
    for (int i = 0; i < vocab; i++) {
        float v = logits[i];
        if (m == top_k && v <= val[m - 1]) continue;
        int j = (m < top_k) ? m++ : top_k - 1;
        while (j > 0 && val[j - 1] < v) {
            val[j] = val[j - 1];
            idx[j] = idx[j - 1];
            j--;
        }
        val[j] = v;
        idx[j] = i;
    }

    float mx = val[0], sum = 0.0f;
    for (int i = 0; i < m; i++) {
        val[i] = expf((val[i] - mx) / temperature);
        sum += val[i];
    }
    for (int i = 0; i < m; i++) val[i] /= sum;

    /* Nucleus: keep the shortest prefix whose mass reaches top_p. Because the
     * array is sorted descending, that prefix is the most probable set. */
    float cum = 0.0f;
    int cut = m;
    for (int i = 0; i < m; i++) {
        cum += val[i];
        if (cum >= top_p) { cut = i + 1; break; }
    }

    float r = rng_float(rng) * cum;                     /* renormalised draw */
    float acc = 0.0f;
    int chosen = idx[cut - 1];
    for (int i = 0; i < cut; i++) {
        acc += val[i];
        if (r < acc) { chosen = idx[i]; break; }
    }

    free(idx); free(val);
    return chosen;
}

/* Qwen3's chat template, for the single-turn case.
 *
 * The real template is Jinja and handles tools, multi-turn history and
 * thinking-mode bookkeeping. This reproduces exactly what transformers
 * renders for one user message, which is what the engine can currently do.
 *
 * The oddity is the thinking block. Qwen3 always emits a <think> section;
 * asking for thinking OFF does not remove it, it pre-fills it as empty so
 * the model has nothing to continue. */
void chat_format(char *out, int cap, const char *user, bool thinking) {
    int n = snprintf(out, (size_t)cap,
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
                     user, thinking ? "" : "<think>\n\n</think>\n\n");
    if (n < 0 || n >= cap) die("chat prompt does not fit in %d bytes", cap);
}

static bool is_eos(const Qwen3Config *cfg, int id) {
    for (int i = 0; i < cfg->n_eos; i++) if (cfg->eos_ids[i] == id) return true;
    return false;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int generate(const Tokenizer *tok, const Weights *w, const Qwen3Config *cfg,
             RunState *s, GpuModel *gpu, const char *prompt, int max_new,
             float temperature, float top_p, int top_k, uint64_t seed,
             int *out_ids, bool quiet) {
    int cap = s->max_seq;
    int *tokens = malloc((size_t)cap * sizeof *tokens);
    if (!tokens) die("out of memory for %d tokens", cap);

    int n = tokenizer_encode(tok, prompt, (int)strlen(prompt), tokens, cap);
    if (n >= cap) die("prompt fills the whole %d-position context", cap);

    /* One row of logits is all generation ever reads. On the GPU path it has
     * to be copied out of device memory; on the CPU path it is already in
     * RunState, so the pointer just aliases it. */
    float *logits = s->logits;
    float *gpu_logits = NULL;
    if (gpu) {
        gpu_logits = malloc((size_t)cfg->vocab_size * sizeof *gpu_logits);
        if (!gpu_logits) die("out of memory for the logits row");
        logits = gpu_logits;
    }

    /* Prefill: the whole prompt in one call, logits for the last row only.
     * Compute-bound — every token is processed against the same weights, so
     * the weights are read once and used n times. */
    double t0 = now();
    if (gpu) gpu_forward(gpu, tokens, n, 0, n - 1, logits);
    else     forward(tokens, n, 0, n - 1, w, cfg, s);
    double t_prefill = now() - t0;

    uint64_t rng = seed;
    char piece[512];
    int generated = 0;
    double t_decode = 0.0;

    for (int pos = n; generated < max_new && pos < cap; pos++) {
        int next = sample(logits, cfg->vocab_size, temperature, top_p, top_k, &rng);
        if (is_eos(cfg, next)) break;
        if (out_ids) out_ids[generated] = next;

        /* Bytes straight out, not a string: one token can be a fragment of a
         * multi-byte character, and the terminal reassembles them. */
        if (!quiet) {
            int len = tokenizer_decode(tok, &next, 1, piece, (int)sizeof piece);
            fwrite(piece, 1, (size_t)len, stdout);
            fflush(stdout);
        }
        generated++;

        /* Decode: one token against the whole model. Every weight is read to
         * produce a single token, so this is bound by memory bandwidth and
         * nothing else. */
        double t1 = now();
        if (gpu) gpu_forward(gpu, &next, 1, pos, 0, logits);
        else     forward(&next, 1, pos, 0, w, cfg, s);
        t_decode += now() - t1;
    }

    if (!quiet) {
        printf("\n\n--- %d prompt tokens, %d generated ---\n", n, generated);
        printf("prefill : %.3f s  (%.1f tok/s)\n", t_prefill, (double)n / t_prefill);
        if (generated > 0) {
            double per_tok = t_decode / generated;
            /* The ceiling every decode optimisation is measured against: one
             * pass over the unique weights per token, at the machine's
             * bandwidth. 1.19 GB is Qwen3-0.6B's unique weights in bf16. */
            printf("decode  : %.3f s  (%.1f tok/s, %.1f GB/s of the 307 available)\n",
                   t_decode, 1.0 / per_tok, 1.19 / per_tok);
        }
    }
    free(tokens);
    free(gpu_logits);
    return generated;
}
