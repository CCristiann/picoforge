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

static int sched_choose(Scheduler *sc, int max_k, const double *expert_seconds) {
    const double alpha = (sc->acc + 1.0) / (sc->acc + sc->rej + 2.0);
    const double v1 = expert_seconds && expert_seconds[1] > 0 ? expert_seconds[1] : sched_verify(sc, 1);
    if (v1 < 0 || sc->draft_s < 0) return max_k < 2 ? max_k : 2;   /* nothing measured: probe */
    int best = 0;
    double best_rate = 1.0 / v1;
    for (int k = 1; k <= max_k; k++) {
        const double tokens = (1.0 - pow(alpha, k + 1)) / (1.0 - alpha);
        const double verify = expert_seconds && expert_seconds[k + 1] > 0
                            ? expert_seconds[k + 1] : sched_verify(sc, k + 1);
        const double rate = tokens / (k * sc->draft_s + verify);
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
                         int max_new, int max_draft, int *out_ids, SpecStats *stats,
                         const SpecOptions *options) {
    const bool adaptive = max_draft < 0;
    const bool experts = options && options->expert_cost;
    const char *trace_path = options ? options->trace_path : NULL;
    if (experts && !adaptive) die("expert cost requires adaptive drafting");
    if ((experts || trace_path) && !cfg->num_experts) die("expert cost and spec trace need a MoE target");
    if (max_draft < -31 || max_draft > 31 || max_new < 1)
        die("speculation needs 0..31 drafts and at least one output token");
    if (adaptive) max_draft = -max_draft;
    if (max_draft + 1 > max_rows)
        die("a draft of %d needs %d logit rows; the GPU model has %d", max_draft,
            max_draft + 1, max_rows);
    int *tokens = malloc((size_t)max_seq * sizeof *tokens);
    float *logits = malloc((size_t)max_rows * (size_t)cfg->vocab_size * sizeof *logits);
    if (!tokens || !logits) die("out of memory for speculative decoding");
    const size_t V = (size_t)cfg->vocab_size;

    SpecStats s = {0};
    int pos = tokenizer_encode(tok, prompt, (int)strlen(prompt), tokens, max_seq);
    if (pos == 0) die("prompt must contain at least one token");
    if (pos >= max_seq) die("prompt fills the whole %d-position context", max_seq);
    FILE *trace_file = trace_path ? fopen(trace_path, "wx") : NULL;
    if (trace_path && !trace_file) die("cannot create spec trace %s (must not exist)", trace_path);
    s.prompt_tokens = pos;
    double prefill_start = now();
    gpu_forward(gpu, tokens, pos, 0, pos - 1, logits);
    s.prefill_s = now() - prefill_start;
    /* The drafter reads the prompt up front, as the target just did: a prefill
     * is not a draft, and timing it as one inflates what drafting costs. */
    int dpos = 0;                 /* tokens[0..dpos) are in the drafter's cache */
    if (drafter) {
        /* NULL: `logits` still holds the target's prediction for `next`. Writing
         * the drafter's there made its guess the first emitted token. */
        prefill_start = now();
        gpu_forward(drafter, tokens, pos, 0, pos - 1, NULL);
        s.draft_prefill_s = now() - prefill_start;
        dpos = pos;
    }
    Scheduler sched;
    sched_init(&sched);
    ExpertCost *cost = (experts || trace_file) ? expert_cost_create(cfg) : NULL;
    /* Start tracing after prefill: its chunked passes retain only the last
     * chunk, and prompt first-use time is not a decode cost observation. */
    if (cost) gpu_set_routing_trace(gpu, true);
    int history_tokens = 0;

    /* Invariant: tokens[0..pos) are in the cache; `next` is decided and not. */
    int next = argmax(logits, cfg->vocab_size);
    int generated = 0;
    double t0 = now();

    while (generated < max_new && !is_eos(cfg, next) && pos < max_seq) {
        tokens[pos] = next;
        out_ids[generated++] = next;
        int group = -1;
        if (s.groups < (int)(sizeof s.group_len / sizeof s.group_len[0])) {
            group = s.groups++;
            s.group_len[group] = 1;
        }
        if (generated == max_new || pos + 1 == max_seq) break;

        /* Drafts go straight into tokens[] after `next`, so the verify pass
         * is one contiguous slice: [next, d_1 .. d_k] at position pos. */
        int room = max_seq - pos - 1;
        int budget = max_draft < max_new - generated ? max_draft : max_new - generated;
        if (budget > room) budget = room;
        double predicted_experts[33] = {0}, predicted_seconds[33] = {0};
        double prediction_start = cost ? now() : 0.0;
        if (cost) {
            expert_cost_predict(cost, budget + 1, predicted_experts);
            for (int n = 1; n <= budget + 1; n++)
                predicted_seconds[n] = expert_cost_seconds(cost, predicted_experts[n]);
        }
        if (adaptive && budget > 0)
            budget = sched_choose(&sched, budget, experts ? predicted_seconds : NULL);
        double prediction_s = cost ? now() - prediction_start : 0.0;
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
        const double baseline_prediction = cost ? sched_verify(&sched, k + 1) : -1.0;

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
        if (adaptive || cost) sched_update(&sched, k, i, drafter ? pass_s : (k ? (t2 - t1) / k : -1.0), t3 - t2);
        if (cost) {
            prediction_start = now();
            const uint32_t *routes = gpu_routing_trace(gpu);
            const double actual_experts = expert_cost_distinct(cost, routes, k + 1);
            const double predicted = predicted_seconds[k + 1];
            s.route_abs_error += fabs(predicted_experts[k + 1] - actual_experts);
            if (predicted > 0 && baseline_prediction > 0) {
                s.expert_predictions++;
                s.expert_abs_error_s += fabs(predicted - (t3 - t2));
                s.baseline_abs_error_s += fabs(baseline_prediction - (t3 - t2));
            }
            /* Price the ENTIRE verify, including rejected drafts; retain only
             * next + accepted drafts as history for the following prediction. */
            expert_cost_observe(cost, actual_experts, t3 - t2);
            expert_cost_accept(cost, routes, i + 1);
            prediction_s += now() - prediction_start;
            s.expert_cost_s += prediction_s;
            if (trace_file) {
                fprintf(trace_file, "{\"pass\":%d,\"position\":%d,\"k\":%d,\"accepted\":%d,"
                        "\"history_tokens\":%d,\"predicted_experts\":%.9f,\"actual_experts\":%.9f,"
                        "\"predicted_verify_s\":%.9f,\"baseline_verify_s\":%.9f,"
                        "\"verify_s\":%.9f,\"predictor_s\":%.9f,\"routes\":[",
                        s.passes, pos, k, i, history_tokens, predicted_experts[k + 1], actual_experts,
                        predicted, baseline_prediction, t3 - t2, prediction_s);
                for (int l = 0; l < cfg->num_hidden_layers; l++) {
                    fprintf(trace_file, l ? ",[" : "[");
                    if (config_is_moe_layer(cfg, l)) {
                        for (int t = 0; t <= k; t++) {
                            fprintf(trace_file, t ? ",[" : "[");
                            for (int j = 0; j < cfg->num_experts_per_tok; j++)
                                fprintf(trace_file, j ? ",%u" : "%u", routes[
                                    ((size_t)l * 32 + (size_t)t) * (size_t)cfg->num_experts_per_tok + (size_t)j]);
                            fputc(']', trace_file);
                        }
                    }
                    fputc(']', trace_file);
                }
                fputs("]}\n", trace_file);
                if (ferror(trace_file)) die("cannot write spec trace %s", trace_path);
            }
            history_tokens += i + 1;
            if (history_tokens > 32) history_tokens = 32;
        }
        if (group >= 0) s.group_len[group] = (short)(s.group_len[group] + i);
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
    if (trace_file && fclose(trace_file)) die("cannot finish spec trace %s", trace_path);
    if (cost) gpu_set_routing_trace(gpu, false);
    expert_cost_free(cost);
    free(tokens);
    free(logits);
    return generated;
}
