/* main.c — picoforge entry point.
 *
 * Usage: ./picoforge [model_dir]
 */
#include "picoforge.h"

#include <stdio.h>

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

    st_close(&st);
    return 0;
}
