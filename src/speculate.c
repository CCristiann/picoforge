/* speculate.c — speculative decoding, greedy, on the GPU.
 *
 * A decode step reads the whole model to produce one token. Step 4.1c measured
 * that reading it for 8 tokens costs no more (12.9 vs 13.2 ms at depth 512).
 * So: guess the next few tokens cheaply, run them all through the model in one
 * pass, and keep the prefix the model agrees with.
 *
 * Lossless by construction, under greedy. Row i of the verify pass holds the
 * logits that follow the accepted tokens plus drafts 1..i, so its argmax is
 * exactly what plain greedy would emit there IF those drafts were right. A
 * draft is kept only when it equals that argmax; at the first disagreement the
 * argmax itself is kept and the rest of the draft is discarded. Every emitted
 * token is the model's own argmax -- the drafts only decide how many arrive
 * per pass, never which.
 *
 * Two drafters. Prompt lookup costs nothing: find the most recent earlier
 * place where the last few tokens occurred and propose what followed. It wins
 * on text that repeats its context and proposes nothing otherwise. A draft
 * MODEL (Qwen3-0.6B for Qwen3-30B-A3B: same tokenizer) proposes k tokens of
 * its own greedy continuation, paying k of its own decode steps for them.
 *
 * Rejected drafts leave K/V rows past the accepted position, in the target's
 * cache and in the drafter's. Nothing reads them -- attention at position p
 * reads rows <= p -- and the next pass writes over them, so rolling back is
 * free in both: the caches are indexed by position.
 */
#include "picoforge.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
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

/* Up to max_draft tokens continuing ctx[0..n), written to draft. Longest
 * match first (3 tokens, then 2, then 1), most recent occurrence first: a
 * longer match is more likely to continue the same way, and a recent one is
 * more likely to be the thing being repeated. */
static int lookup_draft(const int *ctx, int n, int max_draft, int *draft) {
    for (int ng = 3; ng >= 1; ng--) {
        if (n < ng + 1) continue;
        const int *tail = ctx + n - ng;
        for (int start = n - ng - 1; start >= 0; start--) {
            if (memcmp(ctx + start, tail, (size_t)ng * sizeof *ctx) != 0) continue;
            int k = 0;
            while (k < max_draft && start + ng + k < n) {
                draft[k] = ctx[start + ng + k];
                k++;
            }
            return k;
        }
    }
    return 0;
}

/* k greedy tokens from the draft model continuing ctx[0..n), written to
 * ctx[n..n+k). The drafter's cache holds ctx[0..*dpos); what it has not seen
 * of the accepted text goes in first as one pass. It is fed its own drafts
 * 1..k-1 to produce 2..k, so afterwards its cache holds ctx[0..n+k-1). */
static void model_draft(GpuModel *d, int *ctx, int n, int *dpos, int k, float *logits, int V) {
    gpu_forward(d, ctx + *dpos, n - *dpos, *dpos, n - *dpos - 1, logits);
    for (int j = 0; j < k; j++) {
        ctx[n + j] = argmax(logits, V);
        if (j + 1 < k) gpu_forward(d, ctx + n + j, 1, n + j, 0, logits);
    }
    *dpos = n + (k > 0 ? k - 1 : 0);
}

int generate_speculative(const Tokenizer *tok, const Qwen3Config *cfg, GpuModel *gpu,
                         GpuModel *drafter, int max_seq, int max_rows, const char *prompt,
                         int max_new, int max_draft, int *out_ids, SpecStats *stats) {
    if (max_draft + 1 > max_rows)
        die("a draft of %d needs %d logit rows; the GPU model has %d", max_draft,
            max_draft + 1, max_rows);
    int *tokens = malloc((size_t)max_seq * sizeof *tokens);
    float *logits = malloc((size_t)max_rows * (size_t)cfg->vocab_size * sizeof *logits);
    if (!tokens || !logits) die("out of memory for speculative decoding");
    const size_t V = (size_t)cfg->vocab_size;

    int pos = tokenizer_encode(tok, prompt, (int)strlen(prompt), tokens, max_seq);
    if (pos >= max_seq - 1) die("prompt fills the whole %d-position context", max_seq);
    gpu_forward(gpu, tokens, pos, 0, pos - 1, logits);
    int dpos = 0;                 /* tokens[0..dpos) are in the drafter's cache */

    /* Invariant: tokens[0..pos) are in the cache; `next` is decided and not. */
    int next = argmax(logits, cfg->vocab_size);
    int generated = 0;
    SpecStats s = {0};
    double t0 = now();

    while (generated < max_new && !is_eos(cfg, next) && pos + 1 < max_seq) {
        tokens[pos] = next;
        out_ids[generated++] = next;
        if (generated == max_new) break;

        /* Drafts go straight into tokens[] after `next`, so the verify pass
         * is one contiguous slice: [next, d_1 .. d_k] at position pos. */
        int room = max_seq - pos - 2;
        int budget = max_draft < max_new - generated ? max_draft : max_new - generated;
        if (budget > room) budget = room;
        double t1 = now();
        int k = 0;
        if (budget > 0 && drafter) {
            model_draft(drafter, tokens, pos + 1, &dpos, budget, logits, cfg->vocab_size);
            k = budget;
        } else if (budget > 0) {
            k = lookup_draft(tokens, pos + 1, budget, tokens + pos + 1);
        }
        double t2 = now();
        s.draft_s += t2 - t1;

        gpu_forward(gpu, tokens + pos, k + 1, pos, 0, logits);
        s.verify_s += now() - t2;
        s.passes++;
        s.drafted += k;

        int i = 0;
        bool stop = false;
        for (; i < k; i++) {
            const int t = argmax(logits + (size_t)i * V, cfg->vocab_size);
            if (t != tokens[pos + 1 + i]) break;
            if (is_eos(cfg, t) || generated == max_new) { stop = true; break; }
            out_ids[generated++] = t;
        }
        s.accepted += i;
        /* The drafter's rows past the accepted text hold rejected drafts. */
        if (dpos > pos + 1 + i) dpos = pos + 1 + i;
        if (stop) break;
        next = argmax(logits + (size_t)i * V, cfg->vocab_size);
        pos += 1 + i;
    }

    s.decode_s = now() - t0;
    s.generated = generated;
    if (stats) *stats = s;
    free(tokens);
    free(logits);
    return generated;
}
