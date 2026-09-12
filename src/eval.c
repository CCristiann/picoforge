/* eval.c — perplexity, measured by the engine itself.
 *
 * tools/quant/sweep.py already measured every format, in transformers, with
 * exact fp32 arithmetic. This measures the shipped thing: the engine's own
 * tokenizer, its own GPU forward pass, and for Q4 the bf16 narrowing that
 * transformers never did. Two numbers come out of agreeing with the sweep:
 * fp32 against fp32 validates the whole engine end to end, and Q4 against Q4
 * isolates what narrowing costs in perplexity -- the only place that cost is
 * visible at the scale of the model rather than of one logit.
 *
 * Same protocol as the sweep, so the columns line up: windows of CTX tokens,
 * non-overlapping, second half scored, dNLL's standard error across windows.
 */
#include "picoforge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CTX 1024

/* log-softmax of one row, in double. KL accumulated in fp32 once read
 * negative in this project; fp64 is not optional here. */
static void log_softmax64(const float *x, double *out, int n) {
    double mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp((double)x[i] - mx);
    const double lse = mx + log(sum);
    for (int i = 0; i < n; i++) out[i] = (double)x[i] - lse;
}

static GpuModel *open_model(const char *dir, MetalContext *mtl, SafeTensors *st, Qwen3Config *cfg) {
    config_load(dir, cfg);
    st_open(dir, st);
    return gpu_model_create(mtl, st, cfg, CTX, CTX / 2 + 1);
}

void eval_perplexity(const char *model_dir, const char *ref_dir, const char *corpus,
                     const char *csv_path) {
    MetalContext *mtl = metal_init("picoforge.metallib");
    Qwen3Config cfg, rcfg;
    SafeTensors st, rst;
    GpuModel *g = open_model(model_dir, mtl, &st, &cfg);
    GpuModel *r = open_model(ref_dir, mtl, &rst, &rcfg);
    if (cfg.vocab_size != rcfg.vocab_size) die("--ppl: the two models have different vocabularies");

    Tokenizer tok;
    tokenizer_load(model_dir, &tok);
    size_t len;
    char *text = slurp(corpus, &len);
    int cap = (int)len + 16;                  /* never more tokens than bytes */
    int *ids = malloc((size_t)cap * sizeof *ids);
    if (!ids) die("out of memory for the corpus tokens");
    const int n_tok = tokenizer_encode(&tok, text, (int)len, ids, cap);
    const int windows = n_tok / CTX, scored = CTX / 2, V = cfg.vocab_size;

    const size_t rows = (size_t)(scored + 1);
    float  *lq = malloc(rows * (size_t)V * sizeof *lq), *lr = malloc(rows * (size_t)V * sizeof *lr);
    double *pq = malloc((size_t)V * sizeof *pq), *pr = malloc((size_t)V * sizeof *pr);
    double *win_d = malloc((size_t)windows * sizeof *win_d);
    if (!lq || !lr || !pq || !pr || !win_d) die("out of memory for --ppl");

    printf("\n=== perplexity: %s vs %s ===\n", model_dir, ref_dir);
    printf("corpus                : %s, %d tokens, %d windows of %d, second half scored\n",
           corpus, n_tok, windows, CTX);

    double nll_q = 0, nll_r = 0, kl = 0, agree = 0;
    for (int w = 0; w < windows; w++) {
        const int *t = ids + w * CTX;
        /* Row j of the output is the prediction made at position scored-1+j,
         * for the token at scored+j. */
        gpu_forward(g, t, CTX, 0, scored - 1, lq);
        gpu_forward(r, t, CTX, 0, scored - 1, lr);
        double dw = 0;
        for (int j = 0; j < scored; j++) {
            const float *a = lq + (size_t)j * (size_t)V, *b = lr + (size_t)j * (size_t)V;
            log_softmax64(a, pq, V);
            log_softmax64(b, pr, V);
            const int target = t[scored + j];
            nll_q -= pq[target];
            nll_r -= pr[target];
            dw += pr[target] - pq[target];
            double k = 0;
            int am = 0, bm = 0;
            for (int i = 0; i < V; i++) {
                k += exp(pr[i]) * (pr[i] - pq[i]);
                if (a[i] > a[am]) am = i;
                if (b[i] > b[bm]) bm = i;
            }
            kl += k;
            agree += (am == bm);
        }
        win_d[w] = dw / scored;
    }
    const double n = (double)windows * scored;
    double mean_d = 0, var = 0;
    for (int w = 0; w < windows; w++) mean_d += win_d[w] / windows;
    for (int w = 0; w < windows; w++) var += (win_d[w] - mean_d) * (win_d[w] - mean_d);
    const double se = sqrt(var / (windows - 1)) / sqrt((double)windows);
    const double ppl = exp(nll_q / n), ppl_r = exp(nll_r / n);

    printf("perplexity            : %.4f (reference %.4f, %+.3f%%)\n", ppl, ppl_r,
           100.0 * (ppl / ppl_r - 1.0));
    printf("dNLL                  : %+.5f +- %.5f nats (SE across windows)\n", mean_d, se);
    printf("KL(ref || model)      : %.5f nats per token\n", kl / n);
    printf("top-1 agreement       : %.2f%%\n", 100.0 * agree / n);

    FILE *csv = fopen(csv_path, "ab");
    if (!csv) die("cannot append to %s", csv_path);
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0)
        fprintf(csv, "model,reference,ppl,ppl_ref,dppl_pct,dnll,dnll_se,kl,top1,tokens\n");
    fprintf(csv, "%s,%s,%.4f,%.4f,%.3f,%.5f,%.5f,%.5f,%.4f,%.0f\n", model_dir, ref_dir, ppl,
            ppl_r, 100.0 * (ppl / ppl_r - 1.0), mean_d, se, kl / n, agree / n, n);
    fclose(csv);

    free(lq); free(lr); free(pq); free(pr); free(win_d); free(ids); free(text);
    tokenizer_free(&tok);
    gpu_model_free(g); gpu_model_free(r);
    st_close(&st); st_close(&rst);
    metal_shutdown(mtl);
}
