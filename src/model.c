/* model.c — Qwen3's forward pass, composed from the verified primitives.
 *
 * Structurally identical to the oracle, on purpose: same order of operations,
 * same conventions, so that a divergence in the logits can only come from
 * arithmetic and not from a different reading of the architecture.
 *
 * No KV cache yet (step 1.7). This recomputes every key and value for every
 * position on every call, which is exactly what prefill does anyway and is
 * hopelessly wasteful for generation. Correctness first.
 */
#include "picoforge.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* Look up a tensor by a printf-built name and keep only its data pointer.
 * st_find dies if the name is absent, so a checkpoint missing a tensor fails
 * here at bind time rather than as a null dereference mid-forward. */
static const uint16_t *bind(const SafeTensors *st, const char *fmt, ...) {
    char name[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    return st_find(st, name)->data;
}

void weights_bind(const SafeTensors *st, const Qwen3Config *cfg, Weights *w) {
    /* tie_word_embeddings: the config says the LM head IS the embedding
     * matrix. The checkpoint also carries an lm_head.weight copy; the config
     * is the contract, so we ignore the copy and never map it twice. */
    if (!cfg->tie_word_embeddings)
        die("untied embeddings are not implemented (this checkpoint ties them)");

    w->embed      = bind(st, "model.embed_tokens.weight");
    w->final_norm = bind(st, "model.norm.weight");

    w->layers = calloc((size_t)cfg->num_hidden_layers, sizeof *w->layers);
    if (!w->layers) die("out of memory for %d layers", cfg->num_hidden_layers);

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        LayerWeights *L = &w->layers[l];
        L->input_ln     = bind(st, "model.layers.%d.input_layernorm.weight", l);
        L->q_proj       = bind(st, "model.layers.%d.self_attn.q_proj.weight", l);
        L->k_proj       = bind(st, "model.layers.%d.self_attn.k_proj.weight", l);
        L->v_proj       = bind(st, "model.layers.%d.self_attn.v_proj.weight", l);
        L->q_norm       = bind(st, "model.layers.%d.self_attn.q_norm.weight", l);
        L->k_norm       = bind(st, "model.layers.%d.self_attn.k_norm.weight", l);
        L->o_proj       = bind(st, "model.layers.%d.self_attn.o_proj.weight", l);
        L->post_attn_ln = bind(st, "model.layers.%d.post_attention_layernorm.weight", l);
        L->gate_proj    = bind(st, "model.layers.%d.mlp.gate_proj.weight", l);
        L->up_proj      = bind(st, "model.layers.%d.mlp.up_proj.weight", l);
        L->down_proj    = bind(st, "model.layers.%d.mlp.down_proj.weight", l);
    }
}

void weights_free(Weights *w) { free(w->layers); w->layers = NULL; }

static float *xalloc(size_t n, const char *what) {
    float *p = malloc(n * sizeof *p);
    if (!p) die("out of memory for %s (%zu floats)", what, n);
    return p;
}

void state_alloc(RunState *s, const Qwen3Config *cfg, int max_seq, int max_rows) {
    size_t t      = (size_t)max_seq;
    size_t H      = (size_t)cfg->hidden_size;
    size_t q_dim  = (size_t)(cfg->num_attention_heads * cfg->head_dim);
    size_t kv_dim = (size_t)(cfg->num_key_value_heads * cfg->head_dim);
    size_t I      = (size_t)cfg->intermediate_size;
    size_t L      = (size_t)cfg->num_hidden_layers;

    s->max_seq  = max_seq;
    s->max_rows = max_rows;
    s->x        = xalloc(t * H, "residual stream");
    s->xb       = xalloc(t * H, "scratch");
    s->q        = xalloc(t * q_dim, "queries");
    s->kcache   = xalloc(L * t * kv_dim, "key cache");
    s->vcache   = xalloc(L * t * kv_dim, "value cache");
    s->att      = xalloc(t, "attention scores");
    s->attout   = xalloc(t * q_dim, "attention output");
    s->hb       = xalloc(t * I, "mlp gate");
    s->hb2      = xalloc(t * I, "mlp up");
    s->logits   = xalloc((size_t)max_rows * (size_t)cfg->vocab_size, "logits");
}

void state_free(RunState *s) {
    free(s->x);   free(s->xb);     free(s->q);
    free(s->kcache); free(s->vcache);
    free(s->att); free(s->attout); free(s->hb); free(s->hb2); free(s->logits);
}

void forward(const int *tokens, int n, int pos, int logits_from,
             const Weights *w, const Qwen3Config *cfg, RunState *s) {
    const int H      = cfg->hidden_size;
    const int hd     = cfg->head_dim;
    const int n_head = cfg->num_attention_heads;
    const int n_kv   = cfg->num_key_value_heads;
    const int q_dim  = n_head * hd;
    const int kv_dim = n_kv * hd;
    const int group  = n_head / n_kv;        /* Q heads served by one KV head */
    const int I      = cfg->intermediate_size;
    const float eps  = cfg->rms_norm_eps;

    /* 1/sqrt(head_dim). Without it the dot products grow like head_dim, the
     * softmax saturates, and attention collapses onto a single position. */
    const float scale = 1.0f / sqrtf((float)hd);

    if (pos + n > s->max_seq)
        die("context overflow: %d tokens at position %d exceeds the %d-position cache",
            n, pos, s->max_seq);

    for (int t = 0; t < n; t++) {
        const uint16_t *row = w->embed + (size_t)tokens[t] * (size_t)H;
        for (int i = 0; i < H; i++) s->x[t * H + i] = bf16_to_f32(row[i]);
    }

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        const LayerWeights *L = &w->layers[l];
        float *kc = s->kcache + (size_t)l * (size_t)s->max_seq * (size_t)kv_dim;
        float *vc = s->vcache + (size_t)l * (size_t)s->max_seq * (size_t)kv_dim;

        /* ---- attention sub-block ---- */
        for (int t = 0; t < n; t++)
            rmsnorm(s->xb + t * H, s->x + t * H, L->input_ln, H, eps);

        /* K and V are projected straight into the cache at their absolute
         * positions. They are never recomputed again. */
        for (int t = 0; t < n; t++) {
            matmul(s->q + t * q_dim, s->xb + t * H, L->q_proj, H, q_dim);
            matmul(kc + (size_t)(pos + t) * (size_t)kv_dim, s->xb + t * H,
                   L->k_proj, H, kv_dim);
            matmul(vc + (size_t)(pos + t) * (size_t)kv_dim, s->xb + t * H,
                   L->v_proj, H, kv_dim);
        }

        /* Qwen3's per-head QK-norm, then RoPE, at the ABSOLUTE position. This
         * is where a KV cache goes wrong most easily: rotate by t instead of
         * pos + t and every generated token believes it is at the start of
         * the sequence. The cached keys already carry their own rotation, so
         * the error is invisible during prefill and appears only once
         * generation begins.
         *
         * rmsnorm runs in place, which is safe only because it sums all of x
         * before writing any of out. */
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < n_head; h++) {
                float *qh = s->q + t * q_dim + h * hd;
                rmsnorm(qh, qh, L->q_norm, hd, eps);
                rope_apply(qh, hd, pos + t, cfg->rope_theta);
            }
            for (int h = 0; h < n_kv; h++) {
                float *kh = kc + (size_t)(pos + t) * (size_t)kv_dim + h * hd;
                rmsnorm(kh, kh, L->k_norm, hd, eps);
                rope_apply(kh, hd, pos + t, cfg->rope_theta);
            }
        }

        /* Causal attention over the cache. Causality is structural: the key
         * loop stops at the query's own absolute position, so there is no
         * mask to build and none to get wrong.
         *
         * GQA is an index, not a copy: Q head h reads KV head h / group. */
        for (int h = 0; h < n_head; h++) {
            const int kvh = h / group;
            for (int t = 0; t < n; t++) {
                const int abs_t = pos + t;
                const float *qh = s->q + t * q_dim + h * hd;

                for (int j = 0; j <= abs_t; j++) {
                    const float *kh = kc + (size_t)j * (size_t)kv_dim + kvh * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += qh[d] * kh[d];
                    s->att[j] = dot * scale;
                }
                softmax(s->att, abs_t + 1);

                float *out = s->attout + t * q_dim + h * hd;
                for (int d = 0; d < hd; d++) out[d] = 0.0f;
                for (int j = 0; j <= abs_t; j++) {
                    const float *vh = vc + (size_t)j * (size_t)kv_dim + kvh * hd;
                    const float a = s->att[j];
                    for (int d = 0; d < hd; d++) out[d] += a * vh[d];
                }
            }
        }

        for (int t = 0; t < n; t++) {
            matmul(s->xb + t * H, s->attout + t * q_dim, L->o_proj, q_dim, H);
            for (int i = 0; i < H; i++) s->x[t * H + i] += s->xb[t * H + i];
        }

        /* ---- SwiGLU MLP sub-block ----
         * Strictly per-token: attention moves information between positions,
         * the MLP digests it in place. */
        for (int t = 0; t < n; t++)
            rmsnorm(s->xb + t * H, s->x + t * H, L->post_attn_ln, H, eps);

        for (int t = 0; t < n; t++) {
            matmul(s->hb  + t * I, s->xb + t * H, L->gate_proj, H, I);
            matmul(s->hb2 + t * I, s->xb + t * H, L->up_proj,   H, I);
            for (int i = 0; i < I; i++)
                s->hb[t * I + i] = silu(s->hb[t * I + i]) * s->hb2[t * I + i];

            matmul(s->xb + t * H, s->hb + t * I, L->down_proj, I, H);
            for (int i = 0; i < H; i++) s->x[t * H + i] += s->xb[t * H + i];
        }
    }

    /* Final norm, then the LM head — the embedding matrix read the other way
     * round. Only the requested rows: during generation the caller wants the
     * last one, and this matmul is 13% of a prompt-length forward pass. */
    if (logits_from < 0 || logits_from >= n) die("forward: logits_from out of range");
    if (n - logits_from > s->max_rows)
        die("forward: %d logit rows requested, %d allocated", n - logits_from, s->max_rows);

    for (int t = logits_from; t < n; t++) {
        rmsnorm(s->xb + t * H, s->x + t * H, w->final_norm, H, eps);
        matmul(s->logits + (size_t)(t - logits_from) * (size_t)cfg->vocab_size,
               s->xb + t * H, w->embed, H, cfg->vocab_size);
    }
}
