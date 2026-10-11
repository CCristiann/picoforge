/* expert_cost.c — an experimental, training-free forecast of MoE verify cost.
 *
 * Accepted input routes supply per-expert frequencies. The union forecast
 * assumes independent draws from those frequencies; it is a hypothesis, not
 * the actual routing of a draft. All verified routes (including rejected
 * drafts) train a small online time-vs-union regression. No model weights,
 * logits or acceptance decisions are changed by this module. */
#include "picoforge.h"

#include <math.h>
#include <stdlib.h>

enum { ROUTE_WINDOW = 32, TRACE_ROWS = 32 };

struct ExpertCost {
    int layers, experts, topk, moe_layers;
    int used, cursor, samples;
    bool *moe;
    unsigned char *seen;
    unsigned int *counts;
    uint32_t *history;
    double weight, x, y, xx, xy;
};

ExpertCost *expert_cost_create(const Qwen3Config *cfg) {
    if (cfg->num_experts < 1 || cfg->num_experts_per_tok < 1 ||
        cfg->num_experts_per_tok > cfg->num_experts || cfg->num_hidden_layers < 1)
        die("expert cost needs a MoE target");
    ExpertCost *ec = calloc(1, sizeof *ec);
    if (!ec) die("out of memory for expert cost");
    ec->layers = cfg->num_hidden_layers;
    ec->experts = cfg->num_experts;
    ec->topk = cfg->num_experts_per_tok;
    ec->moe = calloc((size_t)ec->layers, sizeof *ec->moe);
    ec->seen = calloc((size_t)ec->experts, sizeof *ec->seen);
    ec->counts = calloc((size_t)ec->layers * (size_t)ec->experts, sizeof *ec->counts);
    ec->history = calloc(ROUTE_WINDOW * (size_t)ec->layers * (size_t)ec->topk, sizeof *ec->history);
    if (!ec->moe || !ec->seen || !ec->counts || !ec->history)
        die("out of memory for expert history");
    for (int l = 0; l < ec->layers; l++) {
        ec->moe[l] = config_is_moe_layer(cfg, l);
        ec->moe_layers += ec->moe[l] ? 1 : 0;
    }
    if (!ec->moe_layers) die("expert cost needs at least one MoE layer");
    return ec;
}

void expert_cost_free(ExpertCost *ec) {
    if (!ec) return;
    free(ec->moe);
    free(ec->seen);
    free(ec->counts);
    free(ec->history);
    free(ec);
}

void expert_cost_predict(const ExpertCost *ec, int max_n, double *distinct) {
    if (max_n < 1 || max_n > TRACE_ROWS) die("expert forecast needs 1..32 rows");
    memset(distinct, 0, (size_t)(max_n + 1) * sizeof *distinct);
    const double prior = (double)ec->topk / ec->experts;
    for (int l = 0; l < ec->layers; l++) if (ec->moe[l]) {
        for (int e = 0; e < ec->experts; e++) {
            const double p = (ec->counts[(size_t)l * (size_t)ec->experts + (size_t)e] + prior)
                           / (ec->used + 1.0);
            double absent = 1.0;
            for (int n = 1; n <= max_n; n++) {
                absent *= 1.0 - p;
                distinct[n] += (1.0 - absent) / ec->moe_layers;
            }
        }
    }
    distinct[1] = ec->topk;       /* exactly top-k, independent of the prior */
}

double expert_cost_distinct(ExpertCost *ec, const uint32_t *trace, int n) {
    if (!trace || n < 1 || n > TRACE_ROWS) die("invalid expert trace");
    double total = 0.0;
    for (int l = 0; l < ec->layers; l++) if (ec->moe[l]) {
        memset(ec->seen, 0, (size_t)ec->experts);
        for (int t = 0; t < n; t++) for (int j = 0; j < ec->topk; j++) {
            const uint32_t e = trace[((size_t)l * TRACE_ROWS + (size_t)t) * (size_t)ec->topk + (size_t)j];
            if (e >= (uint32_t)ec->experts) die("invalid expert id in routing trace");
            if (!ec->seen[e]) { ec->seen[e] = 1; total += 1.0; }
        }
    }
    return total / ec->moe_layers;
}

void expert_cost_accept(ExpertCost *ec, const uint32_t *trace, int accepted_rows) {
    if (!trace || accepted_rows < 0 || accepted_rows > TRACE_ROWS) die("invalid accepted route count");
    for (int t = 0; t < accepted_rows; t++) {
        for (int l = 0; l < ec->layers; l++) if (ec->moe[l]) {
            unsigned int *counts = ec->counts + (size_t)l * (size_t)ec->experts;
            uint32_t *old = ec->history + ((size_t)ec->cursor * (size_t)ec->layers + (size_t)l) * (size_t)ec->topk;
            for (int j = 0; j < ec->topk; j++) {
                const uint32_t e = trace[((size_t)l * TRACE_ROWS + (size_t)t) * (size_t)ec->topk + (size_t)j];
                if (e >= (uint32_t)ec->experts) die("invalid accepted expert id");
                if (ec->used == ROUTE_WINDOW) counts[old[j]]--;
                old[j] = e;
                counts[e]++;
            }
        }
        ec->cursor = (ec->cursor + 1) % ROUTE_WINDOW;
        if (ec->used < ROUTE_WINDOW) ec->used++;
    }
}

void expert_cost_observe(ExpertCost *ec, double distinct, double seconds) {
    if (!isfinite(distinct) || !isfinite(seconds) || distinct < ec->topk ||
        distinct > ec->experts || seconds <= 0.0) die("invalid expert cost observation");
    const double keep = 0.95;
    ec->weight = keep * ec->weight + 1.0;
    ec->x = keep * ec->x + distinct;
    ec->y = keep * ec->y + seconds;
    ec->xx = keep * ec->xx + distinct * distinct;
    ec->xy = keep * ec->xy + distinct * seconds;
    ec->samples++;
}

double expert_cost_seconds(const ExpertCost *ec, double distinct) {
    if (ec->samples < 4) return -1.0;
    const double variance = ec->xx - ec->x * ec->x / ec->weight;
    if (variance <= 1e-9 * ec->xx) return -1.0;  /* identical routes cannot price a new expert */
    double slope = (ec->xy - ec->x * ec->y / ec->weight) / variance;
    if (slope < 0.0) slope = 0.0;
    double intercept = (ec->y - slope * ec->x) / ec->weight;
    if (intercept < 0.0) { intercept = 0.0; slope = ec->xy / ec->xx; }
    const double result = intercept + slope * distinct;
    return isfinite(result) && result > 0.0 ? result : -1.0;
}
