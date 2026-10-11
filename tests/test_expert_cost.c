/* Known route sets and an exact cost law, independent of GPU timing noise. */
#include "picoforge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void close_to(double got, double want) {
    if (fabs(got - want) > 1e-9) die("expert cost: got %.12g, expected %.12g", got, want);
}

int main(void) {
    Qwen3Config cfg = {.num_hidden_layers = 3, .num_experts = 8,
                      .num_experts_per_tok = 2, .decoder_sparse_step = 1,
                      .n_mlp_only_layers = 1, .mlp_only_layers = {1}};
    ExpertCost *ec = expert_cost_create(&cfg);
    uint32_t trace[3 * 32 * 2];
    memset(trace, 0xff, sizeof trace);    /* dense layer deliberately invalid as expert ids */
    for (int t = 0; t < 32; t++) {
        trace[t * 2] = 0; trace[t * 2 + 1] = 1;
        trace[(2 * 32 + t) * 2] = (uint32_t)(2 * (t % 4));
        trace[(2 * 32 + t) * 2 + 1] = (uint32_t)(2 * (t % 4) + 1);
    }
    close_to(expert_cost_distinct(ec, trace, 1), 2.0);
    close_to(expert_cost_distinct(ec, trace, 4), 5.0);  /* (2 + 8) / 2 */
    double predicted[33], accepted[33];
    expert_cost_predict(ec, 32, predicted);
    for (int n = 1; n <= 32; n++) close_to(predicted[n], 8.0 * (1.0 - pow(0.75, n)));

    expert_cost_accept(ec, trace, 1);    /* all the other routes were rejected */
    expert_cost_predict(ec, 32, accepted);
    /* One prior uniform token + one accepted {0,1}: p=.625 for 2 experts,
     * .125 for the remaining 6, in BOTH MoE layers. */
    close_to(accepted[2], 2 * (1 - .375 * .375) + 6 * (1 - .875 * .875));
    expert_cost_accept(ec, trace, 0);
    expert_cost_predict(ec, 32, predicted);
    for (int n = 1; n <= 32; n++) close_to(predicted[n], accepted[n]);

    ExpertCost *fresh = expert_cost_create(&cfg);
    expert_cost_accept(fresh, trace, 32);
    expert_cost_accept(ec, trace, 32);   /* completely evicts the earlier accepted row */
    expert_cost_predict(fresh, 32, accepted);
    expert_cost_predict(ec, 32, predicted);
    for (int n = 1; n <= 32; n++) {
        close_to(predicted[n], accepted[n]);
        assert(predicted[n] >= 2.0 && predicted[n] <= 8.0);
        if (n > 1) assert(predicted[n] >= predicted[n - 1]);
    }

    assert(expert_cost_seconds(ec, 4) < 0);
    for (int i = 0; i < 4; i++) {
        const double d = 2.0 + i;
        expert_cost_observe(ec, d, .010 + .002 * d);
        if (i < 3) assert(expert_cost_seconds(ec, 4) < 0);
    }
    for (int d = 2; d <= 8; d++) close_to(expert_cost_seconds(ec, d), .010 + .002 * d);
    for (int i = 0; i < 20; i++) expert_cost_observe(fresh, 2.0, .015);
    assert(expert_cost_seconds(fresh, 4) < 0);  /* no variation, no learned slope */
    expert_cost_free(fresh);
    expert_cost_free(ec);
    puts("PASS: expert unions, dense layers, accepted-only history, eviction and cost calibration");
    return 0;
}
