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

#include <math.h>
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
 * 1..k-1 to produce 2..k, so afterwards its cache holds ctx[0..n+k-1).
 * Returns the mean seconds of its ordinary passes -- one or two tokens -- or
 * -1 if there were none: a catch-up pass after plain steps is a different
 * cost, and charging it to every draft token taught the scheduler that
 * drafting costs twice what it does. */
static double model_draft(GpuModel *d, int *ctx, int n, int *dpos, int k, float *logits, int V) {
    double sum = 0.0;
    int passes = 0;
    const int catch_up = n - *dpos;
    double t = now();
    gpu_forward(d, ctx + *dpos, catch_up, *dpos, catch_up - 1, logits);
    if (catch_up <= 2) { sum += now() - t; passes++; }
    for (int j = 0; j < k; j++) {
        ctx[n + j] = argmax(logits, V);
        if (j + 1 < k) {
            t = now();
            gpu_forward(d, ctx + n + j, 1, n + j, 0, logits);
            sum += now() - t;
            passes++;
        }
    }
    *dpos = n + (k > 0 ? k - 1 : 0);
    return passes ? sum / passes : -1.0;
}

/* ------------------------------------------------------ adaptive draft length
 * A fixed draft length is wrong for most text: measured on Qwen3-30B-A3B, the
 * best k ran from 2 to 6 across five prompts, and k = 8 on a story the drafter
 * could not predict halved throughput. So k is chosen before every pass, from
 * three running estimates, none trained, all measured on this machine and
 * this text as generation goes:
 *
 *   alpha     the chance a draft token is accepted given its predecessors
 *             were, from accepted and rejected drafts (Beta(1,1) prior,
 *             exponentially forgotten so a change of topic shows quickly)
 *   draft_s   seconds of drafting per draft token
 *   verify_s  seconds for a target pass of n tokens, per n. Unmeasured sizes
 *             come from the nearest measured one scaled by a line fitted to
 *             nothing yet: 6% more per extra token (the 30B's verify of 8
 *             cost 1.43 decode steps) until real passes replace it.
 *
 * With acceptance alpha, a draft of k yields (1 - alpha^(k+1)) / (1 - alpha)
 * tokens in expectation (the accepted prefix plus the target's own token), for
 * k * draft_s + verify_s[k+1] seconds; the k with the most tokens per second
 * wins. k = 0 is plain decoding. On a MoE the verify cost already carries the
 * experts the text's routing touches, because it is timed, not assumed. */
typedef struct {
    double acc, rej;              /* forgotten counts of accepted / rejected drafts */
    double draft_s;               /* per draft token, < 0 until measured            */
    double verify_s[64];          /* per pass size n, < 0 until measured            */
    int    zero_streak;
} Scheduler;

static void sched_init(Scheduler *sc) {
    sc->acc = sc->rej = 0.0;
    sc->draft_s = -1.0;
    for (int n = 0; n < 64; n++) sc->verify_s[n] = -1.0;
    sc->zero_streak = 0;
}

static double sched_verify(const Scheduler *sc, int n) {
    if (sc->verify_s[n] > 0) return sc->verify_s[n];
    for (int d = 1; d < 64; d++)                       /* nearest measured size */
        for (int m = n - d; m <= n + d; m += 2 * d)
            if (m >= 1 && m < 64 && sc->verify_s[m] > 0)
                return sc->verify_s[m] * (1.0 + 0.06 * (n - 1)) / (1.0 + 0.06 * (m - 1));
    return -1.0;
}

static int sched_choose(Scheduler *sc, int max_k) {
    const double alpha = (sc->acc + 1.0) / (sc->acc + sc->rej + 2.0);
    const double v1 = sched_verify(sc, 1);
    if (v1 < 0 || sc->draft_s < 0) return max_k < 2 ? max_k : 2;   /* nothing measured: probe */
    int best = 0;
    double best_rate = 1.0 / v1;
    for (int k = 1; k <= max_k; k++) {
        const double tokens = (1.0 - pow(alpha, k + 1)) / (1.0 - alpha);
        const double rate = tokens / (k * sc->draft_s + sched_verify(sc, k + 1));
        if (rate > best_rate) { best_rate = rate; best = k; }
    }
    /* Never stop measuring: after 8 plain steps, try one draft so that a text
     * turning predictable again is noticed. */
    if (best == 0 && max_k > 0 && ++sc->zero_streak >= 8) { sc->zero_streak = 0; return 1; }
    if (best > 0) sc->zero_streak = 0;
    return best;
}

static void sched_update(Scheduler *sc, int k, int accepted, double pass_s, double verify_s) {
    const double keep = 0.85, ema = 0.3;
    sc->acc = keep * sc->acc + accepted;
    sc->rej = keep * sc->rej + (accepted < k ? 1.0 : 0.0);
    if (pass_s > 0)
        sc->draft_s = sc->draft_s < 0 ? pass_s : (1 - ema) * sc->draft_s + ema * pass_s;
    const int n = k + 1;
    if (n < 64) sc->verify_s[n] = sc->verify_s[n] < 0 ? verify_s : (1 - ema) * sc->verify_s[n] + ema * verify_s;
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
    /* The drafter reads the prompt up front, as the target just did: a prefill
     * is not a draft, and timing it as one inflates what drafting costs. */
    int dpos = 0;                 /* tokens[0..dpos) are in the drafter's cache */
    if (drafter) {
        /* NULL: `logits` still holds the target's prediction for `next`. Writing
         * the drafter's there made its guess the first emitted token. */
        gpu_forward(drafter, tokens, pos, 0, pos - 1, NULL);
        dpos = pos;
    }
    const bool adaptive = max_draft < 0;
    if (adaptive) max_draft = -max_draft;
    Scheduler sched;
    sched_init(&sched);

    /* Invariant: tokens[0..pos) are in the cache; `next` is decided and not. */
    int next = argmax(logits, cfg->vocab_size);
    int generated = 0;
    SpecStats s = {0};
    double t0 = now();

    while (generated < max_new && !is_eos(cfg, next) && pos + 1 < max_seq) {
        tokens[pos] = next;
        out_ids[generated++] = next;
        if (s.groups < (int)(sizeof s.group_len / sizeof s.group_len[0])) s.group_len[s.groups++] = 1;
        if (generated == max_new) break;

        /* Drafts go straight into tokens[] after `next`, so the verify pass
         * is one contiguous slice: [next, d_1 .. d_k] at position pos. */
        int room = max_seq - pos - 2;
        int budget = max_draft < max_new - generated ? max_draft : max_new - generated;
        if (budget > room) budget = room;
        if (adaptive && budget > 0) budget = sched_choose(&sched, budget);
        double t1 = now();
        int k = 0;
        double pass_s = -1.0;
        if (budget > 0 && drafter) {
            pass_s = model_draft(drafter, tokens, pos + 1, &dpos, budget, logits, cfg->vocab_size);
            k = budget;
        } else if (budget > 0) {
            k = lookup_draft(tokens, pos + 1, budget, tokens + pos + 1);
        }
        double t2 = now();
        s.draft_s += t2 - t1;

        gpu_forward(gpu, tokens + pos, k + 1, pos, 0, logits);
        const double t3 = now();
        s.verify_s += t3 - t2;
        if (k < (int)(sizeof s.k_hist / sizeof s.k_hist[0])) s.k_hist[k]++;
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
        /* Prompt lookup drafts in microseconds; its pass cost is the clock's. */
        if (adaptive) sched_update(&sched, k, i, drafter ? pass_s : (k ? (t2 - t1) / k : -1.0), t3 - t2);
        if (s.groups > 0 && s.groups <= (int)(sizeof s.group_len / sizeof s.group_len[0]))
            s.group_len[s.groups - 1] = (short)(s.group_len[s.groups - 1] + i);
        /* The drafter's rows past the accepted text hold rejected drafts. */
        if (dpos > pos + 1 + i) dpos = pos + 1 + i;
        if (stop) break;
        next = argmax(logits + (size_t)i * V, cfg->vocab_size);
        pos += 1 + i;
    }

    if (adaptive && getenv("PICOFORGE_SCHED_DEBUG")) {
        fprintf(stderr, "sched: alpha %.3f  draft %.2f ms/token  verify ms:",
                (sched.acc + 1.0) / (sched.acc + sched.rej + 2.0), sched.draft_s * 1e3);
        for (int n = 1; n <= max_draft + 1; n++) fprintf(stderr, " n%d=%.1f%s", n,
                sched_verify(&sched, n) * 1e3, sched.verify_s[n] > 0 ? "" : "*");
        fprintf(stderr, "\n");
    }
    s.decode_s = now() - t0;
    s.generated = generated;
    if (stats) *stats = s;
    free(tokens);
    free(logits);
    return generated;
}
