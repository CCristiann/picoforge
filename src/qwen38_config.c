/* GGUF dimensions plus the explicit qwen38-q2-text-f32-v1 semantic profile.
 * No model payload, defaults from Qwen3, or borrowed metadata arrays. */
#include "qwen38_a_internal.h"
#include <math.h>

Qwen38Status q38a_meta(const Gguf *g, const char *name, uint32_t type,
                       Qwen38Bytes *value, Qwen38Error *e) {
    if (!g || !value || !g->header || (g->metadata_count && !g->metadata))
        return q38a_error(e, Q38_INVALID_ARGUMENT, name, "invalid GGUF view");
    const GgufMetadata *found = NULL;
    for (size_t i = 0; i < g->metadata_count; ++i) {
        const GgufMetadata *m = g->metadata+i;
        if (!memchr(m->name, 0, sizeof m->name))
            return q38a_error(e, Q38_FORMAT, "metadata", "unterminated name");
        if (!strcmp(m->name, name)) {
            if (found) return q38a_error(e, Q38_FORMAT, name, "duplicate key");
            found = m;
        }
    }
    if (!found) return q38a_error(e, Q38_MISSING, name, "required metadata");
    if (found->type != type) return q38a_error(e, Q38_TYPE, name, "metadata type");
    if (found->offset > g->header_size || found->bytes > g->header_size-found->offset)
        return q38a_error(e, Q38_FORMAT, name, "metadata span");
    *value = (Qwen38Bytes){(const uint8_t *)g->header+found->offset, found->bytes};
    return Q38_OK;
}
static Qwen38Status validate_fields(Qwen38Config *c, Qwen38Error *e) {
    Qwen38ConfigView *v = &c->view;
    Qwen38AttentionConfig *a = &v->attention;
    Qwen38MoeConfig *m = &v->moe;
    Qwen38NgramConfig *n = &c->ngram;
    if (v->blocks > 4096 || v->mtp_blocks >= v->blocks || v->vocab_size < 2 ||
        n->order < 2 || n->order > 64)
        return q38a_error(e, Q38_RANGE, "config", "layer/order admission");
    v->trunk_blocks = v->blocks-v->mtp_blocks;
    if (a->heads % a->kv_heads || a->rotary_dim % 2 || a->rotary_dim > a->head_dim ||
        a->rotary_dim > a->index_dim || v->gdn.value_heads % v->gdn.key_heads ||
        c->inner % v->gdn.value_heads || m->top_k > m->experts)
        return q38a_error(e, Q38_SHAPE, "config", "head/rotary/expert dimensions");
    v->gdn.value_dim = c->inner/v->gdn.value_heads;
    uint64_t padded = ((uint64_t)m->intermediate+255)/256*256;
    if (padded > UINT32_MAX || padded != m->down_physical)
        return q38a_error(e, Q38_SHAPE, "down", "physical K must round logical K to 256");
    uint64_t heads = (uint64_t)(n->order-1)*n->heads_per_order;
    if (heads > 2000000 || heads*n->head_dim != v->hidden_size || n->eos_id >= v->vocab_size)
        return q38a_error(e, Q38_SHAPE, "PLE", "embedding width or EOS");
    /* Kernel dimensions must eventually narrow to uint32, never wrap first. */
    if ((uint64_t)v->hidden_size*v->hc.streams > UINT32_MAX ||
        (uint64_t)a->heads*a->head_dim*2 > UINT32_MAX ||
        (uint64_t)a->index_heads*a->index_dim > UINT32_MAX ||
        2*(uint64_t)v->gdn.key_heads*v->gdn.key_dim+c->inner > UINT32_MAX)
        return q38a_error(e, Q38_OVERFLOW, "config", "derived dimension exceeds uint32");
    n->head_count = (size_t)heads; n->multiplier_count = n->order;
    return Q38_OK;
}
Qwen38Status q38a_array(const Gguf *g, const char *name, uint32_t type,
                        Qwen38Bytes *value, size_t *count, Qwen38Error *e) {
    Qwen38Status s = q38a_meta(g, name, 9, value, e);
    if (s) return s;
    if (value->len < 12) return q38a_error(e, Q38_FORMAT, name, "array header");
    if (q38a_le(value->data, 4) != type) return q38a_error(e, Q38_TYPE, name, "array element type");
    uint64_t n = q38a_le(value->data+4, 8);
    if (n > 2000000) return q38a_error(e, Q38_RANGE, name, "array admission limit");
    *count = (size_t)n;
    value->data += 12; value->len -= 12;
    return Q38_OK;
}
static Qwen38Status uint_field(const Gguf *g, const char *key, uint32_t *out, Qwen38Error *e) {
    Qwen38Bytes b;
    Qwen38Status s = q38a_meta(g, key, 4, &b, e);
    if (s) return s;
    if (b.len != 4) return q38a_error(e, Q38_FORMAT, key, "uint32 width");
    *out = (uint32_t)q38a_le(b.data, 4);
    return Q38_OK;
}
static Qwen38Status float_field(const Gguf *g, const char *key, float *out, Qwen38Error *e) {
    Qwen38Bytes b;
    Qwen38Status s = q38a_meta(g, key, 6, &b, e);
    if (s) return s;
    if (b.len != 4) return q38a_error(e, Q38_FORMAT, key, "float32 width");
    uint32_t bits = (uint32_t)q38a_le(b.data, 4);
    memcpy(out, &bits, 4);
    if (!isfinite(*out)) return q38a_error(e, Q38_NONFINITE, key, "finite value required");
    if (*out <= 0) return q38a_error(e, Q38_RANGE, key, "positive value required");
    return Q38_OK;
}
static Qwen38Status array64(const Gguf *g, const char *key, size_t expected,
                            uint64_t **out, Qwen38Error *e) {
    Qwen38Bytes b; size_t n;
    Qwen38Status s = q38a_array(g, key, 10, &b, &n, e);
    if (s) return s;
    if (n != expected || b.len != n*8)
        return q38a_error(e, Q38_SHAPE, key, "array length");
    *out = calloc(n ? n : 1, sizeof **out);
    if (!*out) return q38a_error(e, Q38_OOM, key, "metadata copy");
    for (size_t i = 0; i < n; ++i) (*out)[i] = q38a_le(b.data+8*i, 8);
    return Q38_OK;
}
static Qwen38Status load_fields(const Gguf *g, Qwen38Config *c, Qwen38Error *e) {
    Qwen38ConfigView *v = &c->view;
    Qwen38AttentionConfig *a = &v->attention;
    Qwen38MoeConfig *m = &v->moe;
    Qwen38NgramConfig *n = &c->ngram;
    Qwen38Status s; uint32_t tmp;
#define U(key, dst) do { s = uint_field(g, "qwen4exp." key, &tmp, e); \
    if (s) return s; if (!tmp) return q38a_error(e, Q38_RANGE, key, "positive dimension"); \
    (dst) = tmp; } while (0)
    U("block_count", v->blocks); U("context_length", v->max_positions);
    U("embedding_length", v->hidden_size); U("vocab_size", v->vocab_size);
    s = uint_field(g, "qwen4exp.nextn_predict_layers", &v->mtp_blocks, e);
    if (s) return s;
    U("attention.head_count", a->heads); U("attention.head_count_kv", a->kv_heads);
    U("attention.key_length", a->head_dim); U("attention.value_length", tmp);
    if (tmp != a->head_dim) return q38a_error(e, Q38_UNSUPPORTED, "attention", "unequal K/V dimensions");
    U("attention.indexer.head_count", a->index_heads);
    U("attention.indexer.key_length", a->index_dim); U("attention.indexer.top_k", a->token_budget);
    U("expert_count", m->experts); U("expert_used_count", m->top_k);
    U("expert_feed_forward_length", m->intermediate);
    U("expert_shared_feed_forward_length", m->shared_intermediate);
    U("ssm.conv_kernel", v->gdn_conv_kernel); U("ssm.state_size", v->gdn.key_dim);
    U("ssm.group_count", v->gdn.key_heads); U("ssm.time_step_rank", v->gdn.value_heads);
    U("ssm.inner_size", c->inner); U("full_attention_interval", c->interval);
    U("hyper_connection.count", v->hc.streams); U("hyper_connection.low_rank", v->hc.low_rank);
    U("rope.dimension_count", a->rotary_dim);
    U("ple.ngram_size", n->order); U("ple.heads_per_ngram", n->heads_per_order);
    U("ple.conv_kernel", n->conv_kernel); U("ple.row_dimension", n->head_dim);
    U("embedding_length_per_layer_input", tmp);
    if (tmp != n->head_dim) return q38a_error(e, Q38_SHAPE, "PLE", "row dimensions disagree");
    U("ple.row_count", n->row_count); U("ple.vocab_divisor", c->ple_divisor);
#undef U
    s = uint_field(g, "qwen4exp.ple.eos_token_id", &n->eos_id, e);
    if (s) return s;
    s = float_field(g, "qwen4exp.attention.layer_norm_rms_epsilon", &a->rms_eps, e);
    if (s) return s;
    s = float_field(g, "qwen4exp.rope.freq_base", &a->rope_theta, e);
    if (s) return s;
    s = uint_field(g, "ds4.qwen4.down.logical_input", &tmp, e);
    if (s) return s;
    if (tmp != m->intermediate) return q38a_error(e, Q38_SHAPE, "down", "logical K mismatch");
    s = uint_field(g, "ds4.qwen4.down.physical_input", &m->down_physical, e);
    if (s) return s;
    /* Fields not exported to GGUF: pinned original HF profile, never Qwen3 defaults.
     * index_kv_heads=1, norm_topk_prob=true, SiLU/sigmoid, no bias, untied head;
     * direct gains, tiled GDN heads, pretransformed A, converted HC weights. */
    v->semantic_profile = "qwen38-q2-text-f32-v1";
    a->index_kv_heads = 1; m->norm_topk_prob = true; v->gdn.l2_eps = 1e-6f;
    m->hidden_size = n->hidden_size = v->hidden_size;
    v->hc.hidden_size = v->hidden_size; n->vocab_size = v->vocab_size;
    n->streams = (uint32_t)v->hc.streams;
    n->rms_eps = v->hc.rms_eps = v->gdn.rms_eps = a->rms_eps;
    return Q38_OK;
}
static Qwen38Status load_arrays(const Gguf *g, Qwen38Config *c, Qwen38Error *e) {
    Qwen38ConfigView *v = &c->view;
    Qwen38NgramConfig *n = &c->ngram;
    uint64_t *ratios = NULL, *ple = NULL;
    Qwen38Status s = array64(g, "qwen4exp.attention.compress_ratios", v->blocks, &ratios, e);
    if (s) return s;
    c->layers = calloc(v->blocks, sizeof *c->layers);
    if (!c->layers) { free(ratios); return q38a_error(e, Q38_OOM, "layers", "config allocation"); }
    for (uint32_t i = 0; i < v->blocks; ++i) {
        bool mtp = i >= v->trunk_blocks, qsa = mtp || (i+1)%c->interval == 0;
        uint64_t r = ratios[i];
        if ((!qsa && r) || (qsa && (!r || r > UINT32_MAX || v->attention.token_budget % r))) {
            s = q38a_error(e, Q38_SHAPE, "compress_ratios", "layer kind/budget mismatch"); break;
        }
        c->layers[i] = (Qwen38Layer){qsa ? Q38_LAYER_QSA : Q38_LAYER_GDN,
            mtp ? Q38_MTP : Q38_TRUNK, (uint32_t)r, false};
        if (qsa && !v->attention.ratio) v->attention.ratio = (uint32_t)r;
    }
    free(ratios);
    if (s) return s;
    /* One PLE site is supported by the frozen profile; not a hidden kernel constant. */
    s = array64(g, "qwen4exp.ple.layers", 1, &ple, e);
    if (s) return s;
    uint64_t layer = ple[0]; free(ple);
    if (layer >= v->trunk_blocks) return q38a_error(e, Q38_RANGE, "ple.layers", "trunk layer required");
    n->layer = (uint32_t)layer; c->layers[layer].has_ple = true;
    s = array64(g, "qwen4exp.ple.layer_multipliers", n->order, &c->multipliers, e);
    if (s) return s;
    s = array64(g, "qwen4exp.ple.head_offsets", n->head_count, &c->offsets, e);
    if (s) return s;
    s = array64(g, "qwen4exp.ple.head_vocab_sizes", n->head_count, &c->vocabs, e);
    if (s) return s;
    for (size_t i = 0; i < n->order; ++i)
        if (!c->multipliers[i] || c->multipliers[i] > (uint64_t)INT64_MAX/(v->vocab_size-1u))
            return q38a_error(e, Q38_RANGE, "PLE multipliers", "signed reference product bound");
    uint64_t end = 0;
    for (size_t i = 0; i < n->head_count; ++i) {
        if (!c->vocabs[i] || c->offsets[i] != end || c->vocabs[i] > n->row_count-end)
            return q38a_error(e, Q38_RANGE, "PLE heads", "contiguous row ranges required");
        end += c->vocabs[i];
    }
    if ((end+c->ple_divisor-1)/c->ple_divisor*c->ple_divisor != n->row_count)
        return q38a_error(e, Q38_SHAPE, "PLE rows", "physical tail padding");
    n->multipliers = c->multipliers; n->head_offsets = c->offsets; n->head_vocab_sizes = c->vocabs;
    return Q38_OK;
}
Qwen38Status qwen38_config_load(const Gguf *g, Qwen38Config **out, Qwen38Error *e) {
    if (out) *out = NULL;
    if (!g || !out) return q38a_error(e, Q38_INVALID_ARGUMENT, "config", "null argument");
    Qwen38Bytes b;
    Qwen38Status s = q38a_meta(g, "general.architecture", 8, &b, e);
    if (s) return s;
    if (b.len != 16 || q38a_le(b.data, 8) != 8 || memcmp(b.data+8, "qwen4exp", 8))
        return q38a_error(e, Q38_UNSUPPORTED, "architecture", "expected qwen4exp");
    Qwen38Config *c = calloc(1, sizeof *c);
    if (!c) return q38a_error(e, Q38_OOM, "config", "allocation");
    s = load_fields(g, c, e);
    if (!s) s = validate_fields(c, e);
    if (!s) s = load_arrays(g, c, e);
    if (s) { qwen38_config_free(c); return s; }
    *out = c;
    return q38a_error(e, Q38_OK, NULL, NULL);
}
void qwen38_config_free(Qwen38Config *c) {
    if (!c) return;
    free(c->layers); free(c->multipliers); free(c->offsets); free(c->vocabs); free(c);
}
const Qwen38ConfigView *qwen38_config_view(const Qwen38Config *c) { return c ? &c->view : NULL; }
Qwen38Status qwen38_config_layer(const Qwen38Config *c, uint32_t i,
                                const Qwen38Layer **out, Qwen38Error *e) {
    if (out) *out = NULL;
    if (!c || !out) return q38a_error(e, Q38_INVALID_ARGUMENT, "layer", "null argument");
    if (i >= c->view.blocks) return q38a_error(e, Q38_RANGE, "layer", "index outside config");
    *out = c->layers+i;
    return q38a_error(e, Q38_OK, NULL, NULL);
}
Qwen38Status qwen38_config_ngram(const Qwen38Config *c, uint32_t i,
                                const Qwen38NgramConfig **out, Qwen38Error *e) {
    if (out) *out = NULL;
    if (!c || !out) return q38a_error(e, Q38_INVALID_ARGUMENT, "ngram", "null argument");
    if (i >= c->view.blocks) return q38a_error(e, Q38_RANGE, "ngram", "layer index");
    if (!c->layers[i].has_ple) return q38a_error(e, Q38_MISSING, "ngram", "layer has no PLE");
    *out = &c->ngram;
    return q38a_error(e, Q38_OK, NULL, NULL);
}
