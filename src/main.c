/* main.c — picoforge entry point.
 *
 * Usage: ./picoforge [model_dir]
 */
#include "picoforge.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* The config and the checkpoint are two files that can disagree. Nothing
 * downstream re-checks them, so every dimension the forward pass will index
 * with gets confronted with the shape actually on disk, once, here. A
 * mismatch is a wrong model directory or a truncated download — both of
 * which would otherwise show up as plausible-looking wrong logits. */
static void check_shape(const SafeTensors *st, const char *name, long d0, long d1) {
    const Tensor *t = st_find(st, name);
    int want_ndim = (d1 > 0) ? 2 : 1;
    if (t->ndim != want_ndim || t->shape[0] != d0 || (d1 > 0 && t->shape[1] != d1))
        die("\"%s\": checkpoint has shape [%ld,%ld] (ndim %d), config implies [%ld,%ld]",
            name, t->shape[0], t->ndim > 1 ? t->shape[1] : 0L, t->ndim, d0, d1);
}

/* First few weights of a tensor, printed with enough digits to round-trip
 * exactly (%.9g). These are diffed against NumPy: bf16 -> fp32 is a shift,
 * so agreement must be EXACT, not approximate. Any difference at all means
 * the offsets are wrong, not that the arithmetic drifted. */
static void fingerprint(const SafeTensors *st, const char *name, int n) {
    const Tensor *t = st_find(st, name);
    if (t->dtype != DT_BF16) die("fingerprint expects a bf16 tensor");
    const uint16_t *src = t->data;
    printf("%-44s", name);
    for (int i = 0; i < n; i++) printf(" %.9g", (double)bf16_to_f32(src[i]));
    printf("\n");
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "models/Qwen3-0.6B";

    Qwen3Config cfg;
    config_load(model_dir, &cfg);
    config_print(&cfg);

    SafeTensors st;
    st_open(model_dir, &st);
    st_summary(&st);

    int q_dim  = cfg.num_attention_heads  * cfg.head_dim;
    int kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
    check_shape(&st, "model.embed_tokens.weight", cfg.vocab_size, cfg.hidden_size);
    check_shape(&st, "model.norm.weight", cfg.hidden_size, 0);
    check_shape(&st, "model.layers.0.self_attn.q_proj.weight", q_dim, cfg.hidden_size);
    check_shape(&st, "model.layers.0.self_attn.k_proj.weight", kv_dim, cfg.hidden_size);
    check_shape(&st, "model.layers.0.self_attn.o_proj.weight", cfg.hidden_size, q_dim);
    check_shape(&st, "model.layers.0.self_attn.q_norm.weight", cfg.head_dim, 0);
    check_shape(&st, "model.layers.0.mlp.gate_proj.weight", cfg.intermediate_size,
                cfg.hidden_size);
    check_shape(&st, "model.layers.0.mlp.down_proj.weight", cfg.hidden_size,
                cfg.intermediate_size);
    printf("shape cross-check     : all %d layers' dimensions agree with config\n",
           cfg.num_hidden_layers);

    printf("\n=== fingerprint (first 4 values, fp32) ===\n");
    fingerprint(&st, "model.embed_tokens.weight", 4);
    fingerprint(&st, "model.layers.0.input_layernorm.weight", 4);
    fingerprint(&st, "model.layers.27.mlp.down_proj.weight", 4);

    /* --forward OUT.bin ID ID ... : run the model on those token ids, dump
     * the raw fp32 logits for tests/test_forward.py to judge. Tokenisation is
     * step 1.6; until then the ids come from the command line. */
    if (argc > 4 && strcmp(argv[2], "--forward") == 0) {
        const char *out_path = argv[3];
        int seq = argc - 4;

        int *tokens = malloc((size_t)seq * sizeof *tokens);
        if (!tokens) die("out of memory for %d tokens", seq);
        for (int i = 0; i < seq; i++) {
            tokens[i] = atoi(argv[4 + i]);
            /* An out-of-range id indexes past the embedding matrix: a read
             * into whatever follows it in the mapping, with no crash and no
             * sign that anything went wrong. */
            if (tokens[i] < 0 || tokens[i] >= cfg.vocab_size)
                die("token id %d is outside [0, %d)", tokens[i], cfg.vocab_size);
        }

        Weights w;
        weights_bind(&st, &cfg, &w);
        RunState state;
        state_alloc(&state, &cfg, seq);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        forward(tokens, seq, &w, &cfg, &state);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec)
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

        printf("\n=== forward ===\n");
        printf("tokens                : %d\n", seq);
        printf("prefill               : %.3f s  (%.1f tok/s, scalar CPU, 1 thread)\n",
               secs, (double)seq / secs);

        /* Top-5 of the last row: what the model thinks comes next. softmax is
         * in place and destructive, so it runs on a copy of that one row —
         * 600 KB, against the 2 seconds a second forward pass would cost. */
        size_t vocab = (size_t)cfg.vocab_size;
        float *probs = malloc(vocab * sizeof *probs);
        if (!probs) die("out of memory for the probability row");
        memcpy(probs, state.logits + (size_t)(seq - 1) * vocab, vocab * sizeof *probs);
        softmax(probs, cfg.vocab_size);

        printf("top-5 next tokens     :");
        for (int rank = 0; rank < 5; rank++) {
            int best = 0;
            for (int i = 1; i < cfg.vocab_size; i++) if (probs[i] > probs[best]) best = i;
            printf(" %d(%.1f%%)", best, (double)probs[best] * 100.0);
            probs[best] = -1.0f;                    /* pop it and look again */
        }
        printf("\n");
        free(probs);

        FILE *f = fopen(out_path, "wb");
        if (!f) die("cannot write %s", out_path);
        size_t n = (size_t)seq * (size_t)cfg.vocab_size;
        if (fwrite(state.logits, sizeof(float), n, f) != n)
            die("short write to %s", out_path);
        fclose(f);
        printf("logits                : %zu floats -> %s\n", n, out_path);

        state_free(&state);
        weights_free(&w);
        free(tokens);
    }

    st_close(&st);
    return 0;
}
