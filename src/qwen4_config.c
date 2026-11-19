/* GGUF dimensions plus the explicit qwen38-q2-text-f32-v1 semantic profile.
 * No model payload, defaults from Qwen3, or borrowed metadata arrays. */
#include "qwen4_text_internal.h"
#include <math.h>

Qwen4Status qw4a_meta(const Gguf *g, const char *name, uint32_t type,
                       Qwen4Bytes *value, Qwen4Error *e) {
    if (!g || !value || !g->header || (g->metadata_count && !g->metadata))
        return qw4a_error(e, QW4_INVALID_ARGUMENT, name, "invalid GGUF view");
    const GgufMetadata *found = NULL;
    for (size_t i = 0; i < g->metadata_count; ++i) {
        const GgufMetadata *m = g->metadata+i;
        if (!memchr(m->name, 0, sizeof m->name))
            return qw4a_error(e, QW4_FORMAT, "metadata", "unterminated name");
        if (!strcmp(m->name, name)) {
            if (found) return qw4a_error(e, QW4_FORMAT, name, "duplicate key");
            found = m;
        }
    }
    if (!found) return qw4a_error(e, QW4_MISSING, name, "required metadata");
    if (found->type != type) return qw4a_error(e, QW4_TYPE, name, "metadata type");
    if (found->offset > g->header_size || found->bytes > g->header_size-found->offset)
        return qw4a_error(e, QW4_FORMAT, name, "metadata span");
    *value = (Qwen4Bytes){(const uint8_t *)g->header+found->offset, found->bytes};
    return QW4_OK;
}
static Qwen4Status validate_fields(Qwen4Config *c, Qwen4Error *e) {
    Qwen4ConfigView *v = &c->view;
    Qwen4AttentionConfig *a = &v->attention;
    Qwen4MoeConfig *m = &v->moe;
    Qwen4NgramConfig *n = &c->ngram;
    if (v->blocks > 4096 || v->mtp_blocks >= v->blocks || v->vocab_size < 2 ||
        (c->has_ple && (n->order < 2 || n->order > 64)))
        return qw4a_error(e, QW4_RANGE, "config", "layer/order admission");
    v->trunk_blocks = v->blocks-v->mtp_blocks;
    if (a->heads % a->kv_heads || a->rotary_dim % 2 || a->rotary_dim > a->head_dim ||
        a->rotary_dim > a->index_dim || v->gdn.value_heads % v->gdn.key_heads ||
        c->inner % v->gdn.value_heads || m->top_k > m->experts)
        return qw4a_error(e, QW4_SHAPE, "config", "head/rotary/expert dimensions");
    v->gdn.value_dim = c->inner/v->gdn.value_heads;
    if (v->gdn.value_dim != v->gdn.key_dim)
        return qw4a_error(e, QW4_UNSUPPORTED, "GDN", "profile requires equal key/value dimensions");
    uint64_t padded = m->trunk_type[2] == 10 ? ((uint64_t)m->intermediate+255)/256*256 : m->intermediate;
    if (padded > UINT32_MAX || padded != m->down_physical)   /* R5.3: only Q2_K pads K (to 256) */
        return qw4a_error(e, QW4_SHAPE, "down", "physical K must round logical K to 256");
    uint64_t heads = c->has_ple ? (uint64_t)(n->order-1)*n->heads_per_order : 0;
    if (c->has_ple && (heads > 2000000 || heads*n->head_dim != v->hidden_size || n->eos_id >= v->vocab_size))
        return qw4a_error(e, QW4_SHAPE, "PLE", "embedding width or EOS");
    /* Kernel dimensions must eventually narrow to uint32, never wrap first. */
    if ((uint64_t)v->hidden_size*v->hc.streams > UINT32_MAX ||
        (uint64_t)a->heads*a->head_dim > UINT32_MAX/2 ||
        (uint64_t)a->index_heads*a->index_dim > UINT32_MAX ||
        (uint64_t)v->gdn.key_heads*v->gdn.key_dim > (UINT32_MAX-c->inner)/2 ||
        (v->mtp_blocks && v->hidden_size > UINT32_MAX/2))
        return qw4a_error(e, QW4_OVERFLOW, "config", "derived dimension exceeds uint32");
    n->head_count = (size_t)heads; n->multiplier_count = n->order;
    if (sizeof *c+(size_t)v->blocks*sizeof(Qwen4Layer)+n->head_count*16+n->order*8 > 4*1024*1024)
        return qw4a_error(e,QW4_RANGE,"config","4 MiB copied metadata budget");
    return QW4_OK;
}
static Qwen4Status known_fields(const Gguf *g, Qwen4Error *e) {
    static const char *const names[] = {
        "block_count","nextn_predict_layers","context_length","embedding_length","vocab_size",
        "attention.head_count","attention.head_count_kv","attention.key_length","attention.value_length",
        "attention.layer_norm_rms_epsilon","attention.indexer.head_count","attention.indexer.key_length",
        "attention.indexer.top_k","expert_count","expert_used_count","expert_feed_forward_length",
        "expert_shared_feed_forward_length","ssm.conv_kernel","ssm.state_size","ssm.group_count",
        "ssm.time_step_rank","ssm.inner_size","full_attention_interval","hyper_connection.count",
        "hyper_connection.low_rank","rope.dimension_count","rope.freq_base","attention.compress_ratios",
        "ple.layers","ple.ngram_size","ple.heads_per_ngram","ple.conv_kernel","ple.eos_token_id",
        "embedding_length_per_layer_input","ple.seed","ple.vocab_base","ple.vocab_divisor","ple.row_count",
        "ple.row_dimension","ple.layer_multipliers","ple.head_offsets","ple.head_vocab_sizes",
        "ssm.output_gate_type"
    };
    for (size_t i = 0; i < g->metadata_count; ++i) {
        const char *key = g->metadata[i].name;
        if (strncmp(key,"qwen4exp.",9)) continue;
        bool known = false;
        for (size_t j = 0; j < sizeof names/sizeof *names; ++j) if (!strcmp(key+9,names[j])) known = true;
        if (!known) return qw4a_error(e,QW4_UNSUPPORTED,"config","unknown computational metadata");
    }
    return QW4_OK;
}
Qwen4Status qw4a_array(const Gguf *g, const char *name, uint32_t type,
                        Qwen4Bytes *value, size_t *count, Qwen4Error *e) {
    Qwen4Status s = qw4a_meta(g, name, 9, value, e);
    if (s) return s;
    if (value->len < 12) return qw4a_error(e, QW4_FORMAT, name, "array header");
    if (qw4a_le(value->data, 4) != type) return qw4a_error(e, QW4_TYPE, name, "array element type");
    uint64_t n = qw4a_le(value->data+4, 8);
    if (n > 2000000) return qw4a_error(e, QW4_RANGE, name, "array admission limit");
    *count = (size_t)n;
    value->data += 12; value->len -= 12;
    return QW4_OK;
}
static const GgufTensor *tensor_named(const Gguf *g, const char *name) {
    for (size_t i = 0; i < g->tensor_count; ++i) if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}
static Qwen4Status uint_field(const Gguf *g, const char *key, uint32_t *out, Qwen4Error *e) {
    Qwen4Bytes b;
    Qwen4Status s = qw4a_meta(g, key, 4, &b, e);
    if (s) return s;
    if (b.len != 4) return qw4a_error(e, QW4_FORMAT, key, "uint32 width");
    *out = (uint32_t)qw4a_le(b.data, 4);
    return QW4_OK;
}
static Qwen4Status float_field(const Gguf *g, const char *key, float *out, Qwen4Error *e) {
    Qwen4Bytes b;
    Qwen4Status s = qw4a_meta(g, key, 6, &b, e);
    if (s) return s;
    if (b.len != 4) return qw4a_error(e, QW4_FORMAT, key, "float32 width");
    uint32_t bits = (uint32_t)qw4a_le(b.data, 4);
    memcpy(out, &bits, 4);
    if (!isfinite(*out)) return qw4a_error(e, QW4_NONFINITE, key, "finite value required");
    if (*out <= 0) return qw4a_error(e, QW4_RANGE, key, "positive value required");
    return QW4_OK;
}
static Qwen4Status array64(const Gguf *g, const char *key, size_t expected,
                            uint64_t **out, Qwen4Error *e) {
    Qwen4Bytes b; size_t n;
    Qwen4Status s = qw4a_array(g, key, 10, &b, &n, e);
    if (s) return s;
    if (n != expected || b.len != n*8)
        return qw4a_error(e, QW4_SHAPE, key, "array length");
    *out = calloc(n ? n : 1, sizeof **out);
    if (!*out) return qw4a_error(e, QW4_OOM, key, "metadata copy");
    for (size_t i = 0; i < n; ++i) (*out)[i] = qw4a_le(b.data+8*i, 8);
    return QW4_OK;
}
static bool has_key(const Gguf *g, const char *key) {
    for (size_t i = 0; i < g->metadata_count; ++i) if (!strcmp(g->metadata[i].name, key)) return true;
    return false;
}
/* GDN output gate. Upstream's output_gate_type falls back to hidden_act (SiLU);
 * Flash-Next sets "sigmoid" and its GGUF carries no key, so an absent key means
 * sigmoid. The key name mirrors the HF field; provisional until a converter
 * for a SiLU checkpoint fixes it. */
static Qwen4Status gate_field(const Gguf *g, bool *silu, Qwen4Error *e) {
    *silu = false;
    if (!has_key(g, "qwen4exp.ssm.output_gate_type")) return QW4_OK;
    Qwen4Bytes b;
    Qwen4Status s = qw4a_meta(g, "qwen4exp.ssm.output_gate_type", 8, &b, e);
    if (s) return s;
    if (b.len < 8 || qw4a_le(b.data, 8) != b.len-8) return qw4a_error(e, QW4_FORMAT, "output_gate_type", "string length");
    size_t n = b.len-8; const uint8_t *v = b.data+8;
    if (n == 4 && !memcmp(v, "silu", 4)) *silu = true;
    else if (!(n == 7 && !memcmp(v, "sigmoid", 7)))
        return qw4a_error(e, QW4_UNSUPPORTED, "output_gate_type", "expected sigmoid or silu");
    return QW4_OK;
}
static Qwen4Status load_fields(const Gguf *g, Qwen4Config *c, Qwen4Error *e) {
    Qwen4ConfigView *v = &c->view;
    Qwen4AttentionConfig *a = &v->attention;
    Qwen4MoeConfig *m = &v->moe;
    Qwen4NgramConfig *n = &c->ngram;
    Qwen4Status s; uint32_t tmp;
#define U(key, dst) do { s = uint_field(g, "qwen4exp." key, &tmp, e); \
    if (s) return s; if (!tmp) return qw4a_error(e, QW4_RANGE, key, "positive dimension"); \
    (dst) = tmp; } while (0)
    U("block_count", v->blocks); U("context_length", v->max_positions);
    U("embedding_length", v->hidden_size); U("vocab_size", v->vocab_size);
    s = uint_field(g, "qwen4exp.nextn_predict_layers", &v->mtp_blocks, e);
    if (s) return s;
    U("attention.head_count", a->heads); U("attention.head_count_kv", a->kv_heads);
    U("attention.key_length", a->head_dim); U("attention.value_length", tmp);
    if (tmp != a->head_dim) return qw4a_error(e, QW4_UNSUPPORTED, "attention", "unequal K/V dimensions");
    U("attention.indexer.head_count", a->index_heads);
    U("attention.indexer.key_length", a->index_dim); U("attention.indexer.top_k", a->token_budget);
    U("expert_count", m->experts); U("expert_used_count", m->top_k);
    U("expert_feed_forward_length", m->intermediate);
    U("expert_shared_feed_forward_length", m->shared_intermediate);
    U("ssm.conv_kernel", v->gdn_conv_kernel); U("ssm.state_size", v->gdn.key_dim);
    U("ssm.group_count", v->gdn.key_heads); U("ssm.time_step_rank", v->gdn.value_heads);
    U("ssm.inner_size", c->inner);
    U("hyper_connection.count", v->hc.streams); U("hyper_connection.low_rank", v->hc.low_rank);
    U("rope.dimension_count", a->rotary_dim);
    /* PLE is optional upstream (ple_layer_ids empty): without ple.layers no
     * ple.* field is read and no layer has PLE. */
    c->has_ple = has_key(g, "qwen4exp.ple.layers");
    if (c->has_ple) {
        U("ple.ngram_size", n->order); U("ple.heads_per_ngram", n->heads_per_order);
        U("ple.conv_kernel", n->conv_kernel); U("ple.row_dimension", n->head_dim);
        U("embedding_length_per_layer_input", tmp);
        if (tmp != n->head_dim) return qw4a_error(e, QW4_SHAPE, "PLE", "row dimensions disagree");
        U("ple.row_count", n->row_count); U("ple.vocab_divisor", c->ple_divisor);
        s = uint_field(g, "qwen4exp.ple.eos_token_id", &n->eos_id, e);
        if (s) return s;
    }
#undef U
    s = float_field(g, "qwen4exp.attention.layer_norm_rms_epsilon", &a->rms_eps, e);
    if (s) return s;
    s = float_field(g, "qwen4exp.rope.freq_base", &a->rope_theta, e);
    if (s) return s;
    s = gate_field(g, &v->gdn.silu_gate, e);
    if (s) return s;
    /* R5.3: two profiles of trunk experts, picked by the file's layer-0 tensors and nothing else.
     * Q2: IQ2_XXS gate/up, Q2_K down whose K is padded to 256 (the ds4.qwen4.down.* fields say so).
     * Q4: Q4_K gate/up, MXFP4 down at the logical K: the formats the MTP layer already uses. */
    const GgufTensor *gate = tensor_named(g, "blk.0.ffn_gate_exps.weight"), *down = tensor_named(g, "blk.0.ffn_down_exps.weight");
    if (!gate || !down) return qw4a_error(e, QW4_FORMAT, "experts", "layer 0 routed expert tensors");
    if (gate->type == 16 && down->type == 10) {
        s = uint_field(g, "ds4.qwen4.down.logical_input", &tmp, e);
        if (s) return s;
        if (tmp != m->intermediate) return qw4a_error(e, QW4_SHAPE, "down", "logical K mismatch");
        s = uint_field(g, "ds4.qwen4.down.physical_input", &m->down_physical, e);
        if (s) return s;
        m->trunk_type[0] = m->trunk_type[1] = 16; m->trunk_type[2] = 10;
        v->semantic_profile = "qwen38-q2-text-f32-v1";
    } else if (gate->type == 12 && down->type == 39) {
        m->down_physical = m->intermediate;
        m->trunk_type[0] = m->trunk_type[1] = 12; m->trunk_type[2] = 39;
        v->semantic_profile = "qwen38-q4-text-f32-v1";
    } else {
        return qw4a_error(e, QW4_UNSUPPORTED, "experts", "routed expert formats are neither the Q2 nor the Q4 profile");
    }
    /* Fields not exported to GGUF: pinned original HF profile, never Qwen3 defaults.
     * index_kv_heads=1, norm_topk_prob=true, SiLU/sigmoid, no bias, untied head;
     * direct gains, tiled GDN heads, pretransformed A, converted HC weights. */
    a->index_kv_heads = 1; m->norm_topk_prob = true; v->gdn.l2_eps = 1e-6f;
    m->hidden_size = n->hidden_size = v->hidden_size;
    v->hc.hidden_size = v->hidden_size; n->vocab_size = v->vocab_size;
    n->streams = (uint32_t)v->hc.streams;
    n->rms_eps = v->hc.rms_eps = v->gdn.rms_eps = a->rms_eps;
    return QW4_OK;
}
static Qwen4Status load_arrays(const Gguf *g, Qwen4Config *c, Qwen4Error *e) {
    Qwen4ConfigView *v = &c->view;
    Qwen4NgramConfig *n = &c->ngram;
    uint64_t *ratios = NULL, *ple = NULL;
    Qwen4Status s = array64(g, "qwen4exp.attention.compress_ratios", v->blocks, &ratios, e);
    if (s) return s;
    c->layers = calloc(v->blocks, sizeof *c->layers);
    if (!c->layers) { free(ratios); return qw4a_error(e, QW4_OOM, "layers", "config allocation"); }
    /* The ratios name the layer kinds (0 = GDN, r = QSA with blocks of r keys),
     * so any upstream layer_types list loads, not only the every-4th pattern of
     * full_attention_interval. MTP layers are QSA. */
    for (uint32_t i = 0; i < v->blocks; ++i) {
        uint64_t r = ratios[i];
        bool mtp = i >= v->trunk_blocks, qsa = r != 0;
        if ((mtp && !qsa) || (qsa && (r > UINT32_MAX || v->attention.token_budget % r))) {
            s = qw4a_error(e, QW4_SHAPE, "compress_ratios", "layer kind/budget mismatch"); break;
        }
        c->layers[i] = (Qwen4Layer){qsa ? QW4_LAYER_QSA : QW4_LAYER_GDN,
            mtp ? QW4_MTP : QW4_TRUNK, (uint32_t)r, false};
        if (qsa && !v->attention.ratio) v->attention.ratio = (uint32_t)r;
    }
    free(ratios);
    if (s) return s;
    if (!c->has_ple) { c->ngram.layer = UINT32_MAX; return QW4_OK; }
    /* One PLE site is supported by the frozen profile; not a hidden kernel constant. */
    s = array64(g, "qwen4exp.ple.layers", 1, &ple, e);
    if (s) return s;
    uint64_t layer = ple[0]; free(ple);
    if (layer >= v->trunk_blocks) return qw4a_error(e, QW4_RANGE, "ple.layers", "trunk layer required");
    n->layer = (uint32_t)layer; c->layers[layer].has_ple = true;
    s = array64(g, "qwen4exp.ple.layer_multipliers", n->order, &c->multipliers, e);
    if (s) return s;
    s = array64(g, "qwen4exp.ple.head_offsets", n->head_count, &c->offsets, e);
    if (s) return s;
    s = array64(g, "qwen4exp.ple.head_vocab_sizes", n->head_count, &c->vocabs, e);
    if (s) return s;
    for (size_t i = 0; i < n->order; ++i)
        if (!c->multipliers[i] || c->multipliers[i] > (uint64_t)INT64_MAX/(v->vocab_size-1u))
            return qw4a_error(e, QW4_RANGE, "PLE multipliers", "signed reference product bound");
    uint64_t end = 0;
    for (size_t i = 0; i < n->head_count; ++i) {
        if (!c->vocabs[i] || c->offsets[i] != end || c->vocabs[i] > n->row_count-end)
            return qw4a_error(e, QW4_RANGE, "PLE heads", "contiguous row ranges required");
        end += c->vocabs[i];
    }
    if ((end+c->ple_divisor-1)/c->ple_divisor*c->ple_divisor != n->row_count)
        return qw4a_error(e, QW4_SHAPE, "PLE rows", "physical tail padding");
    n->multipliers = c->multipliers; n->head_offsets = c->offsets; n->head_vocab_sizes = c->vocabs;
    return QW4_OK;
}
Qwen4Status qwen4_config_load(const Gguf *g, Qwen4Config **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!g || !out) return qw4a_error(e, QW4_INVALID_ARGUMENT, "config", "null argument");
    Qwen4Bytes b;
    Qwen4Status s = qw4a_meta(g, "general.architecture", 8, &b, e);
    if (s) return s;
    if (b.len != 16 || qw4a_le(b.data, 8) != 8 || memcmp(b.data+8, "qwen4exp", 8))
        return qw4a_error(e, QW4_UNSUPPORTED, "architecture", "expected qwen4exp");
    Qwen4Config *c = calloc(1, sizeof *c);
    if (!c) return qw4a_error(e, QW4_OOM, "config", "allocation");
    s = known_fields(g, e);
    if (!s) s = load_fields(g, c, e);
    if (!s) s = validate_fields(c, e);
    if (!s) s = load_arrays(g, c, e);
    if (s) { qwen4_config_free(c); return s; }
    *out = c;
    return qw4a_error(e, QW4_OK, NULL, NULL);
}
void qwen4_config_free(Qwen4Config *c) {
    if (!c) return;
    free(c->layers); free(c->multipliers); free(c->offsets); free(c->vocabs); free(c);
}
const Qwen4ConfigView *qwen4_config_view(const Qwen4Config *c) { return c ? &c->view : NULL; }
Qwen4Status qwen4_config_layer(const Qwen4Config *c, uint32_t i,
                                const Qwen4Layer **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!c || !out) return qw4a_error(e, QW4_INVALID_ARGUMENT, "layer", "null argument");
    if (i >= c->view.blocks) return qw4a_error(e, QW4_RANGE, "layer", "index outside config");
    *out = c->layers+i;
    return qw4a_error(e, QW4_OK, NULL, NULL);
}
Qwen4Status qwen4_config_ngram(const Qwen4Config *c, uint32_t i,
                                const Qwen4NgramConfig **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!c || !out) return qw4a_error(e, QW4_INVALID_ARGUMENT, "ngram", "null argument");
    if (i >= c->view.blocks) return qw4a_error(e, QW4_RANGE, "ngram", "layer index");
    if (!c->layers[i].has_ple) return qw4a_error(e, QW4_MISSING, "ngram", "layer has no PLE");
    *out = &c->ngram;
    return qw4a_error(e, QW4_OK, NULL, NULL);
}
