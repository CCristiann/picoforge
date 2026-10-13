/* G1 CPU graph: composition of A/B/C/D/E/F, never a second operator.
 * Roles are resolved once at create (A); every product is B's; norms,
 * activations and HC are C's primitives. See qwen38_forward.h for the order. */
#include "qwen38_forward.h"
#include "qwen38_c_internal.h"
#include "qwen38_moe.h"
#include "qwen38_ngram.h"
#include "qwen38_attention.h"
#include <stdlib.h>

typedef struct { Q38PackedCpuView down, up, inject; float *gain; } G1Hc;
typedef struct {
    G1Hc attn, ffn;               /* attn only for QSA: C binds GDN's HC-attn */
    Q38CLayer gdn;                /* C composition: HC-attn+GDN+out+combine */
    Q38CBlockState gdn_state;
    void *gdn_memory[5];          /* conv/GDN metadata+payload, decoded constants */
    Qwen38Moe *moe;
    Qwen38MoeWork *moe_work;
    Qwen38NgramStore *ple_store;  /* E: disk rows, history and PLE Delta */
    Qwen38NgramState *ple_state;
    Qwen38PleWorkspace *ple_work;
    Qwen38NgramJob *ple_job;      /* pending until the whole append succeeds */
    Qwen38PleWeights ple;
    Q38PackedCpuView qsa[6];      /* D: Q/gate, K, V, index Q, index K, Wout */
    float *qsa_gamma[4];          /* q, k, index q, index k norm gains */
    Qwen38AttentionState *att;
    Qwen38AttentionWorkspace *att_work;
    Qwen38AttentionTicket *ticket; /* pending until the whole append succeeds */
    float *ple_constant[4];       /* gains key/query/conv [H,E], taps [H*E,K] */
} G1Layer;
struct Qwen38Forward {
    Qwen38ConfigView c;
    const Gguf *gguf;
    const Qwen38Config *config;
    const Qwen38Weights *weights;
    const GpuWeightStore *store;
    uint32_t max_tokens, capacity;
    uint64_t epoch, next_pos;
    bool valid;
    Q38PackedCpuView embed, head;
    G1Hc final;
    G1Layer *layer;
    Qwen38Memory c_work;          /* one C workspace, re-viewed per GDN layer */
    /* One arena: residual R[T,H,E], HC scratch, then per-block outputs. */
    float *arena, *r, *normed, *low, *act, *gate, *il, *inject, *mixed, *block;
    float *qg, *k, *v, *iq, *ik, *attn;
};

static Qwen38Status fail(Qwen38Error *e, Qwen38Status code, const char *detail) {
    if (e) {
        memset(e, 0, sizeof *e); e->code = code;
        if (code != Q38_OK) {
            snprintf(e->subject, sizeof e->subject, "qwen38/G1");
            snprintf(e->detail, sizeof e->detail, "%s", detail);
        }
    }
    return code;
}
#define TRY(call) do { Qwen38Status g1_rc = (call); if (g1_rc != Q38_OK) return g1_rc; } while (0)

/* Dense token-major views over graph scratch: [rows, cols] and [rows, a, b]. */
static Qwen38F32Mut mat(float *p, size_t rows, size_t cols) {
    return (Qwen38F32Mut){.data=p, .capacity_bytes=rows*cols*4, .rank=2,
        .dim={rows, cols}, .stride_bytes={cols*4, 4}};
}
static Qwen38F32Mut cube(float *p, size_t rows, size_t a, size_t b) {
    return (Qwen38F32Mut){.data=p, .capacity_bytes=rows*a*b*4, .rank=3,
        .dim={rows, a, b}, .stride_bytes={a*b*4, b*4, 4}};
}

/* A resolves the role, the store lends the mapped bytes, B checks the layout. */
static Qwen38Status bind(const Qwen38Forward *f, uint32_t layer, Qwen38Role role,
        Q38PackedCpuView *out, Qwen38Error *e) {
    const Qwen38Tensor *t;
    TRY(qwen38_weights_get(f->weights, layer, role, &t, e));
    if (!t->gpu) return fail(e, Q38_MISSING, "graph weight is not resident");
    TRY(qwen38_weights_cpu_span(f->store, t->gpu, &out->span, e));
    return q38_packed_layout_init(&t->storage, (size_t)t->logical_shape[0], &out->layout, e);
}
/* Small vectors (norm gains) decoded once by B into owned FP32. */
static Qwen38Status constant(const Qwen38Forward *f, uint32_t layer, Qwen38Role role,
        float **out, Qwen38Error *e) {
    Q38PackedCpuView v;
    TRY(bind(f, layer, role, &v, e));
    size_t n = v.layout.k_storage * v.layout.n_out;
    if (!(*out = malloc(n * 4))) return fail(e, Q38_OOM, "graph constant");
    for (size_t row = 0; row < v.layout.n_out; ++row)
        TRY(q38_packed_decode_row(&v, 0, row, *out + row*v.layout.k_storage, v.layout.k_storage, e));
    return Q38_OK;
}
static Qwen38Status bind_hc(const Qwen38Forward *f, uint32_t layer, const Qwen38Role roles[4],
        G1Hc *hc, Qwen38Error *e) {
    TRY(constant(f, layer, roles[0], &hc->gain, e));
    TRY(bind(f, layer, roles[1], &hc->down, e));
    TRY(bind(f, layer, roles[2], &hc->up, e));
    return roles[3] == Q38_ROLE_COUNT ? Q38_OK : bind(f, layer, roles[3], &hc->inject, e);
}

/* HC read (C's primitives, same order as q38c_hc_cpu): N=RMS(R), G from the
 * low-rank pair, mixed=mean_H(sigmoid(G)*N); inject weights only if bound. */
static Qwen38Status hc_read(const Qwen38Forward *f, const G1Hc *hc, size_t t, Qwen38Error *e) {
    const Qwen38HcShape *s = &f->c.hc;
    size_t h = s->streams, w = s->hidden_size, x = h*w;
    Qwen38F32 gain = q38c_ro(mat(hc->gain, h, w));
    TRY(qwen38_rmsnorm_groups(q38c_ro(cube(f->r, t, h, w)), gain, cube(f->normed, t, h, w), s->rms_eps, e));
    TRY(q38_packed_matmul(&hc->down, 0, q38c_ro(mat(f->normed, t, x)), mat(f->low, t, s->low_rank), e));
    TRY(qwen38_hc_lowrank_activate(s, q38c_ro(mat(f->low, t, s->low_rank)), mat(f->act, t, s->low_rank), e));
    TRY(q38_packed_matmul(&hc->up, 0, q38c_ro(mat(f->act, t, s->low_rank)), mat(f->gate, t, x), e));
    TRY(qwen38_hc_read(s, q38c_ro(cube(f->normed, t, h, w)), q38c_ro(cube(f->gate, t, h, w)), mat(f->mixed, t, w), e));
    if (!hc->inject.span.data) return Q38_OK;
    TRY(q38_packed_matmul(&hc->inject, 0, q38c_ro(mat(f->normed, t, x)), mat(f->il, t, h), e));
    return qwen38_hc_inject_weights(s, q38c_ro(mat(f->il, t, h)), mat(f->inject, t, h), e);
}

static void emit(Qwen38GraphTrace trace, void *user, uint32_t layer, Qwen38GraphStage stage, Qwen38F32Mut v) {
    if (trace) trace(user, layer, stage, q38c_ro(v));
}

static const Qwen38Role attn_roles[4] = {Q38_HC_ATTN_NORM, Q38_HC_ATTN_DOWN, Q38_HC_ATTN_UP, Q38_HC_ATTN_INJECT};
static const Qwen38Role ffn_roles[4] = {Q38_HC_FFN_NORM, Q38_HC_FFN_DOWN, Q38_HC_FFN_UP, Q38_HC_FFN_INJECT};
static const Qwen38Role final_roles[4] = {Q38_OUTPUT_HC_NORM, Q38_OUTPUT_HC_DOWN, Q38_OUTPUT_HC_UP, Q38_ROLE_COUNT};

static Qwen38Status arena(Qwen38Forward *f, Qwen38Error *e) {
    size_t t = f->max_tokens, h = f->c.hc.streams, w = f->c.hidden_size, r = f->c.hc.low_rank;
    const Qwen38AttentionConfig *a = &f->c.attention;
    size_t q = (size_t)a->heads*a->head_dim, kv = (size_t)a->kv_heads*a->head_dim;
    float **slot[] = {&f->r, &f->normed, &f->low, &f->act, &f->gate, &f->il, &f->inject, &f->mixed, &f->block,
        &f->qg, &f->k, &f->v, &f->iq, &f->ik, &f->attn};
    size_t width[] = {h*w, h*w, r, r, h*w, h, h, w, w, 2*q, kv, kv,
        (size_t)a->index_heads*a->index_dim, a->index_dim, q}, total = 0;
    for (size_t i = 0; i < sizeof width / sizeof *width; ++i) total += (t*width[i] + 15) & ~(size_t)15;
    if (!(f->arena = aligned_alloc(64, total*4))) return fail(e, Q38_OOM, "graph scratch");
    for (size_t i = 0, offset = 0; i < sizeof width / sizeof *width; ++i) {
        *slot[i] = f->arena + offset; offset += (t*width[i] + 15) & ~(size_t)15;
    }
    return Q38_OK;
}

static Qwen38Status aligned(void **out, size_t bytes, Qwen38Error *e) {
    *out = bytes ? aligned_alloc(64, (bytes + 63) & ~(size_t)63) : NULL;
    if (bytes && !*out) return fail(e, Q38_OOM, "graph backing");
    if (bytes) memset(*out, 0, bytes);
    return Q38_OK;
}
/* C plans/binds the GDN block; states live in graph-owned aligned backings. */
static Qwen38Status create_gdn(Qwen38Forward *f, uint32_t i, size_t *work, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    TRY(q38c_layer_plan(f->config, f->weights, i, &l->gdn, e));
    Qwen38CpuSpan spans[Q38CW_COUNT];
    for (size_t j = 0; j < Q38CW_COUNT; ++j) {
        Q38PackedCpuView v;
        TRY(bind(f, i, q38c_layer_roles[j], &v, e));
        spans[j] = v.span;
    }
    TRY(aligned(&l->gdn_memory[4], l->gdn.constant_bytes, e));
    TRY(q38c_layer_bind(&l->gdn, spans, (Qwen38Memory){l->gdn_memory[4], l->gdn.constant_bytes}, e));
    Qwen38StateSize cs, gs; Qwen38ConvState *conv; Qwen38GdnState *gdn;
    TRY(qwen38_conv_state_bytes(&l->gdn.conv, &cs, e));
    TRY(qwen38_gdn_state_bytes(&f->c.gdn, &gs, e));
    size_t bytes[4] = {cs.metadata_bytes, cs.payload_bytes, gs.metadata_bytes, gs.payload_bytes};
    for (size_t j = 0; j < 4; ++j) TRY(aligned(&l->gdn_memory[j], bytes[j], e));
    Qwen38StateKey key = {f->config, i, f->epoch};
    TRY(qwen38_conv_state_init(&l->gdn.conv, key, (Qwen38Memory){l->gdn_memory[0], bytes[0]},
        (Qwen38Memory){l->gdn_memory[1], bytes[1]}, &conv, e));
    TRY(qwen38_gdn_state_init(&f->c.gdn, key, (Qwen38Memory){l->gdn_memory[2], bytes[2]},
        (Qwen38Memory){l->gdn_memory[3], bytes[3]}, &gdn, e));
    TRY(q38c_block_init(&l->gdn, conv, gdn, &l->gdn_state, e));
    size_t need;
    TRY(q38c_work_size(&l->gdn, f->max_tokens, &need, e));
    if (need > *work) *work = need;
    return Q38_OK;
}
/* F owns routing and experts; every product inside goes through B. */
static Qwen38Status create_moe(Qwen38Forward *f, uint32_t i, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    Qwen38MoeWeights w;
    Q38PackedCpuView *views[] = {&w.router, &w.gate, &w.up, &w.down, &w.shared_gate, &w.shared_up,
        &w.shared_down, &w.shared_router};
    const Qwen38Role roles[] = {Q38_MOE_ROUTER, Q38_MOE_GATE, Q38_MOE_UP, Q38_MOE_DOWN,
        Q38_SHARED_GATE, Q38_SHARED_UP, Q38_SHARED_DOWN, Q38_SHARED_ROUTER};
    for (size_t j = 0; j < 8; ++j) TRY(bind(f, i, roles[j], views[j], e));
    TRY(qwen38_moe_create(&f->c.moe, &w, &l->moe, e));
    size_t bytes;
    TRY(qwen38_moe_work_size(l->moe, f->max_tokens, &bytes, e));
    if (bytes > 32u << 20) return fail(e, Q38_CAPACITY, "MoE workspace exceeds the 32 MiB budget");
    return qwen38_moe_work_create(l->moe, f->max_tokens, bytes, &l->moe_work, e);
}
/* E reads its disk-only table through its own fd; B projects, C norms/convs. */
static Qwen38Status create_ple(Qwen38Forward *f, uint32_t i, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    const Qwen38NgramConfig *nc;
    size_t h = f->c.hc.streams, w = f->c.hidden_size;
    TRY(qwen38_config_ngram(f->config, i, &nc, e));
    TRY(bind(f, i, Q38_PLE_KEY, &l->ple.key, e));
    TRY(bind(f, i, Q38_PLE_VALUE, &l->ple.value, e));
    const Qwen38Role roles[] = {Q38_PLE_KEY_NORM, Q38_PLE_QUERY_NORM, Q38_PLE_CONV_NORM, Q38_PLE_CONV};
    Qwen38F32 *views[] = {&l->ple.gamma_key, &l->ple.gamma_query, &l->ple.gamma_conv, &l->ple.taps};
    for (size_t j = 0; j < 4; ++j) {
        TRY(constant(f, i, roles[j], &l->ple_constant[j], e));
        *views[j] = q38c_ro(j < 3 ? mat(l->ple_constant[j], h, w) : mat(l->ple_constant[j], h*w, nc->conv_kernel));
    }
    TRY(qwen38_ngram_open(f->gguf, nc, f->max_tokens, 32u << 20, &l->ple_store, e));
    TRY(qwen38_ngram_state_create(l->ple_store, (Qwen38StateKey){f->config, i, f->epoch}, &l->ple_state, e));
    return qwen38_ple_workspace_create(l->ple_store, 224u << 20, &l->ple_work, e);
}
/* D owns the FP32 K/V/indexer cache; the graph owns projections and Wout. */
static Qwen38Status create_qsa(Qwen38Forward *f, uint32_t i, uint32_t ratio, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    const Qwen38Role roles[] = {Q38_ATTN_QG, Q38_ATTN_K, Q38_ATTN_V, Q38_INDEX_Q, Q38_INDEX_K, Q38_ATTN_OUT};
    const Qwen38Role gains[] = {Q38_ATTN_Q_NORM, Q38_ATTN_K_NORM, Q38_INDEX_Q_NORM, Q38_INDEX_K_NORM};
    for (size_t j = 0; j < 6; ++j) TRY(bind(f, i, roles[j], &l->qsa[j], e));
    for (size_t j = 0; j < 4; ++j) TRY(constant(f, i, gains[j], &l->qsa_gamma[j], e));
    Qwen38AttentionConfig c = f->c.attention;
    c.ratio = ratio;                       /* the layer record overrides the profile */
    Qwen38AttentionPlan plan;
    TRY(qwen38_attention_plan(&c, f->capacity, f->max_tokens, &plan, e));
    return qwen38_attention_create(&c, &plan, (Qwen38StateKey){f->config, i, f->epoch},
        plan.cache_bytes + plan.workspace_bytes, &l->att, &l->att_work, e);
}

static Qwen38Status create(Qwen38Forward *f, Qwen38Error *e) {
    TRY(bind(f, QWEN38_GLOBAL_LAYER, Q38_TOKEN_EMBD, &f->embed, e));
    TRY(bind(f, QWEN38_GLOBAL_LAYER, Q38_LM_HEAD, &f->head, e));
    if (f->embed.layout.k_storage != f->c.hidden_size || f->embed.layout.n_out != f->c.vocab_size ||
        f->head.layout.k_logical != f->c.hidden_size || f->head.layout.n_out != f->c.vocab_size)
        return fail(e, Q38_SHAPE, "embedding/LM head disagree with the config");
    TRY(bind_hc(f, QWEN38_GLOBAL_LAYER, final_roles, &f->final, e));
    if (!(f->layer = calloc(f->c.trunk_blocks, sizeof *f->layer))) return fail(e, Q38_OOM, "graph layers");
    for (uint32_t i = 0; i < f->c.trunk_blocks; ++i) {
        const Qwen38Layer *kind;
        TRY(qwen38_config_layer(f->config, i, &kind, e));
        if (kind->domain != Q38_TRUNK) return fail(e, Q38_UNSUPPORTED, "trunk layer outside the trunk domain");
        /* GDN layers: C's composition binds its own HC-attn (see layer steps). */
        if (kind->kind == Q38_LAYER_QSA) {
            TRY(bind_hc(f, i, attn_roles, &f->layer[i].attn, e));
            TRY(create_qsa(f, i, kind->compress_ratio, e));
        } else TRY(create_gdn(f, i, &f->c_work.bytes, e));
        TRY(bind_hc(f, i, ffn_roles, &f->layer[i].ffn, e));
        TRY(create_moe(f, i, e));
        if (kind->has_ple) TRY(create_ple(f, i, e));
    }
    TRY(aligned(&f->c_work.data, f->c_work.bytes, e));
    return arena(f, e);
}

Qwen38Status qwen38_forward_create(const Gguf *g, const Qwen38Config *config,
        const Qwen38Weights *weights, const GpuWeightStore *store, uint32_t max_tokens,
        uint32_t capacity, Qwen38Forward **out, Qwen38Error *e) {
    if (out) *out = NULL;
    if (!g || !config || !weights || !store || !out) return fail(e, Q38_INVALID_ARGUMENT, "missing graph input");
    const Qwen38ConfigView *view = qwen38_config_view(config);
    if (!view) return fail(e, Q38_INVALID_ARGUMENT, "missing config view");
    if (!max_tokens || max_tokens > capacity || capacity > view->max_positions)
        return fail(e, Q38_RANGE, "graph needs 0 < max_tokens <= capacity <= max_positions");
    Qwen38Forward *f = calloc(1, sizeof *f);
    if (!f) return fail(e, Q38_OOM, "graph");
    *f = (Qwen38Forward){.c=*view, .gguf=g, .config=config, .weights=weights, .store=store,
        .max_tokens=max_tokens, .capacity=capacity, .epoch=1, .valid=true};
    Qwen38Status rc = create(f, e);
    if (rc != Q38_OK) { qwen38_forward_destroy(f); return rc; }
    *out = f;
    return fail(e, Q38_OK, NULL);
}

void qwen38_forward_destroy(Qwen38Forward *f) {
    if (!f) return;
    for (uint32_t i = 0; f->layer && i < f->c.trunk_blocks; ++i) {
        G1Layer *l = &f->layer[i];
        free(l->attn.gain); free(l->ffn.gain);
        qwen38_moe_work_destroy(l->moe_work); qwen38_moe_destroy(l->moe);
        for (size_t j = 0; j < 5; ++j) free(l->gdn_memory[j]);
        if (l->ple_job) { qwen38_ngram_abort(l->ple_job, NULL); qwen38_ngram_job_release(l->ple_job, NULL); }
        if (l->ple_work) qwen38_ple_workspace_destroy(l->ple_work, NULL);
        if (l->ple_state) qwen38_ngram_state_destroy(l->ple_state, NULL);
        if (l->ple_store) qwen38_ngram_close(l->ple_store, NULL);
        for (size_t j = 0; j < 4; ++j) { free(l->ple_constant[j]); free(l->qsa_gamma[j]); }
        if (l->ticket) qwen38_attention_finish(l->ticket, false, NULL);
        if (l->att) qwen38_attention_destroy(l->att, l->att_work, NULL);
    }
    free(f->c_work.data); free(f->final.gain); free(f->layer); free(f->arena); free(f);
}

Qwen38StateInfo qwen38_forward_info(const Qwen38Forward *f) {
    if (!f) return (Qwen38StateInfo){0};
    return (Qwen38StateInfo){.key={f->config, QWEN38_GLOBAL_LAYER, f->epoch},
        .next_pos=f->next_pos, .valid=f->valid};
}

/* GDN: C's verified composition runs HC-attn, projections, conv, scan, gated
 * norm, Wout and the inject combine on R in place, at the session position. */
static Qwen38Status gdn_cpu(Qwen38Forward *f, uint32_t i, size_t t, Qwen38GraphTrace trace,
        void *user, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    Q38CWork work;
    TRY(q38c_work_init(&l->gdn, f->max_tokens, f->c_work, &work, e));
    TRY(q38c_block_cpu(&l->gdn, &work, cube(f->r, t, f->c.hc.streams, f->c.hidden_size), &l->gdn_state,
        f->next_pos, e));
    emit(trace, user, i, Q38G_ATTN_MIXED, q38c_trace(&work, Q38CT_MIX, t));
    emit(trace, user, i, Q38G_ATTN_INJECT, q38c_trace(&work, Q38CT_INJECT, t));
    emit(trace, user, i, Q38G_ATTN_CONTRIBUTION, q38c_trace(&work, Q38CT_OUT, t));
    return Q38_OK;
}

/* PLE before HC-attn: E produces Delta from the n-gram rows; the graph adds
 * R+Delta exactly once. E's history stays a candidate until finish(). */
static Qwen38Status ple_cpu(Qwen38Forward *f, uint32_t i, const int32_t *tokens, size_t t,
        Qwen38GraphTrace trace, void *user, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    size_t h = f->c.hc.streams, w = f->c.hidden_size, n = t*h*w;
    TRY(qwen38_ngram_prepare(l->ple_store, l->ple_state, tokens, t, f->next_pos, &l->ple_job, e));
    TRY(qwen38_ngram_read(l->ple_job, e));
    float *delta = f->normed;              /* HC has not started: scratch is free */
    TRY(qwen38_ple_cpu(l->ple_job, &l->ple, q38c_ro(cube(f->r, t, h, w)), cube(delta, t, h, w), l->ple_work, e));
    emit(trace, user, i, Q38G_PLE_DELTA, cube(delta, t, h, w));
    for (size_t j = 0; j < n; ++j) {
        f->r[j] += delta[j];
        if (!isfinite(f->r[j])) return fail(e, Q38_NONFINITE, "PLE residual overflow");
    }
    return Q38_OK;
}

/* QSA: HC-attn, B projections, D's indexer+sparse attention (cache append
 * stays a ticket until finish), flatten, B's Wout, inject combine. */
static Qwen38Status qsa_cpu(Qwen38Forward *f, uint32_t i, size_t t, Qwen38GraphTrace trace,
        void *user, Qwen38Error *e) {
    G1Layer *l = &f->layer[i];
    const Qwen38AttentionConfig *a = &f->c.attention;
    size_t h = f->c.hc.streams, w = f->c.hidden_size, hq = a->heads, hk = a->kv_heads, d = a->head_dim;
    TRY(hc_read(f, &l->attn, t, e));
    float *out[] = {f->qg, f->k, f->v, f->iq, f->ik};
    for (size_t j = 0; j < 5; ++j)
        TRY(q38_packed_matmul(&l->qsa[j], 0, q38c_ro(mat(f->mixed, t, w)), mat(out[j], t, l->qsa[j].layout.n_out), e));
    Qwen38F32Mut qg = {.data=f->qg, .capacity_bytes=t*hq*2*d*4, .rank=4, .dim={t, hq, 2, d},
        .stride_bytes={hq*2*d*4, 2*d*4, d*4, 4}};
    Qwen38F32 g[4];
    for (size_t j = 0; j < 4; ++j) {
        size_t n = j < 2 ? d : a->index_dim;
        g[j] = (Qwen38F32){.data=l->qsa_gamma[j], .capacity_bytes=n*4, .rank=1, .dim={n}, .stride_bytes={4}};
    }
    Qwen38AttentionInput in = {q38c_ro(qg), q38c_ro(cube(f->k, t, hk, d)), q38c_ro(cube(f->v, t, hk, d)),
        q38c_ro(cube(f->iq, t, a->index_heads, a->index_dim)), q38c_ro(mat(f->ik, t, a->index_dim)),
        g[0], g[1], g[2], g[3]};
    TRY(qwen38_attention_begin(l->att, f->next_pos, (uint32_t)t, f->epoch, &l->ticket, e));
    TRY(qwen38_attention_cpu(l->ticket, &in, cube(f->attn, t, hq, d), l->att_work, e));
    TRY(q38_packed_matmul(&l->qsa[5], 0, q38c_ro(mat(f->attn, t, hq*d)), mat(f->block, t, w), e));
    emit(trace, user, i, Q38G_ATTN_MIXED, mat(f->mixed, t, w));
    emit(trace, user, i, Q38G_ATTN_INJECT, mat(f->inject, t, h));
    emit(trace, user, i, Q38G_ATTN_CONTRIBUTION, mat(f->block, t, w));
    return qwen38_hc_add(&f->c.hc, cube(f->r, t, h, w), q38c_ro(mat(f->block, t, w)), q38c_ro(mat(f->inject, t, h)), e);
}

static Qwen38Status layer_cpu(Qwen38Forward *f, uint32_t i, const int32_t *tokens, size_t t,
        Qwen38GraphTrace trace, void *user, Qwen38Error *e) {
    const Qwen38Layer *kind;
    size_t h = f->c.hc.streams, w = f->c.hidden_size;
    TRY(qwen38_config_layer(f->config, i, &kind, e));
    if (kind->has_ple) TRY(ple_cpu(f, i, tokens, t, trace, user, e));
    if (kind->kind == Q38_LAYER_QSA) TRY(qsa_cpu(f, i, t, trace, user, e));
    else TRY(gdn_cpu(f, i, t, trace, user, e));
    emit(trace, user, i, Q38G_ATTN_RESIDUAL, cube(f->r, t, h, w));
    G1Layer *l = &f->layer[i];               /* HC-ffn -> MoE -> inject */
    TRY(hc_read(f, &l->ffn, t, e));
    emit(trace, user, i, Q38G_FFN_MIXED, mat(f->mixed, t, w));
    emit(trace, user, i, Q38G_FFN_INJECT, mat(f->inject, t, h));
    TRY(qwen38_moe_forward(l->moe, q38c_ro(mat(f->mixed, t, w)), mat(f->block, t, w), NULL, l->moe_work, e));
    emit(trace, user, i, Q38G_MOE, mat(f->block, t, w));
    TRY(qwen38_hc_add(&f->c.hc, cube(f->r, t, h, w), q38c_ro(mat(f->block, t, w)), q38c_ro(mat(f->inject, t, h)), e));
    emit(trace, user, i, Q38G_OUTPUT, cube(f->r, t, h, w));
    return Q38_OK;
}

static Qwen38Status run(Qwen38Forward *f, const int32_t *tokens, size_t t, uint32_t layers,
        Qwen38F32Mut logits, Qwen38GraphTrace trace, void *user, Qwen38Error *e) {
    size_t h = f->c.hc.streams, w = f->c.hidden_size, rows = logits.dim[0];
    for (size_t i = 0; i < t; ++i) {          /* embed, then repeat over H streams */
        float *row = f->r + i*h*w;
        TRY(q38_packed_decode_row(&f->embed, 0, (size_t)tokens[i], row, w, e));
        for (size_t s = 1; s < h; ++s) memcpy(row + s*w, row, w*4);
    }
    Qwen38F32Mut first = mat(f->r, t, w); first.stride_bytes[0] = h*w*4;
    first.capacity_bytes = t*h*w*4;
    emit(trace, user, QWEN38_GLOBAL_LAYER, Q38G_EMBEDDING, first);
    for (uint32_t i = 0; i < layers; ++i) TRY(layer_cpu(f, i, tokens, t, trace, user, e));
    TRY(hc_read(f, &f->final, t, e));          /* output mixer: no inject */
    emit(trace, user, QWEN38_GLOBAL_LAYER, Q38G_FINAL, mat(f->mixed, t, w));
    return q38_packed_matmul(&f->head, 0, q38c_ro(mat(f->mixed + (t-rows)*w, rows, w)), logits, e);
}

/* Deferred commits: E's candidate history becomes the state only once every
 * layer succeeded; otherwise abort. Either way release the job. */
static Qwen38Status finish(Qwen38Forward *f, bool commit, Qwen38Status rc, Qwen38Error *e) {
    Qwen38Error local;
    for (uint32_t i = 0; i < f->c.trunk_blocks; ++i) {
        G1Layer *l = &f->layer[i];
        if (l->ticket) {
            Qwen38Status s = qwen38_attention_finish(l->ticket, commit, &local);
            l->ticket = NULL;
            if (rc == Q38_OK && s != Q38_OK) { rc = s; if (e) *e = local; }
        }
        if (!l->ple_job) continue;
        Qwen38Status s = commit ? qwen38_ngram_commit(l->ple_state, l->ple_job, &local)
                                : qwen38_ngram_abort(l->ple_job, &local);
        Qwen38Status r = qwen38_ngram_job_release(l->ple_job, s == Q38_OK ? &local : NULL);
        l->ple_job = NULL;
        if (rc == Q38_OK && (s != Q38_OK || r != Q38_OK)) { rc = s != Q38_OK ? s : r; if (e) *e = local; }
    }
    return rc;
}

Qwen38Status qwen38_forward_cpu(Qwen38Forward *f, const int32_t *tokens, size_t count,
        uint32_t layers, Qwen38F32Mut logits, Qwen38GraphTrace trace, void *user, Qwen38Error *e) {
    if (!f || (count && !tokens)) return fail(e, Q38_INVALID_ARGUMENT, "missing graph call input");
    if (!f->valid) return fail(e, Q38_BAD_STATE, "graph session requires reset");
    if (layers > f->c.trunk_blocks) return fail(e, Q38_RANGE, "more layers than the trunk");
    if (layers < f->c.trunk_blocks && f->next_pos) return fail(e, Q38_BAD_STATE, "prefix runs start a fresh session");
    if (count > f->max_tokens || count > f->capacity - f->next_pos) return fail(e, Q38_CAPACITY, "append exceeds chunk/context");
    TRY(q38c_view(q38c_ro(logits), e));
    if (logits.rank != 2 || logits.dim[0] > count || logits.dim[1] != f->c.vocab_size || logits.stride_bytes[0] != f->c.vocab_size*4)
        return fail(e, Q38_SHAPE, "logits must be dense [rows<=count, vocab]");
    for (size_t i = 0; i < count; ++i)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= f->c.vocab_size) return fail(e, Q38_RANGE, "token outside vocabulary");
    if (!count) return fail(e, Q38_OK, NULL);
    Qwen38Status rc = run(f, tokens, count, layers, logits, trace, user, e);
    rc = finish(f, rc == Q38_OK && layers == f->c.trunk_blocks, rc, e);
    /* Prefix slices and failures leave layer states mid-stream: reset first. */
    if (rc != Q38_OK || layers < f->c.trunk_blocks) f->valid = false;
    else f->next_pos += count;
    return rc == Q38_OK ? fail(e, Q38_OK, NULL) : rc;
}

Qwen38Status qwen38_forward_reset(Qwen38Forward *f, Qwen38Error *e) {
    if (!f) return fail(e, Q38_INVALID_ARGUMENT, "missing graph");
    if (f->epoch == UINT64_MAX) return fail(e, Q38_OVERFLOW, "epoch exhausted");
    uint64_t epoch = f->epoch + 1;
    f->valid = false;              /* a partial reset leaves mixed epochs: stay invalid */
    for (uint32_t i = 0; i < f->c.trunk_blocks; ++i)
        if (f->layer[i].gdn.bound) TRY(q38c_block_reset(&f->layer[i].gdn_state, epoch, e));
    for (uint32_t i = 0; i < f->c.trunk_blocks; ++i)
        if (f->layer[i].ple_state) TRY(qwen38_ngram_state_reset(f->layer[i].ple_state, epoch, e));
    for (uint32_t i = 0; i < f->c.trunk_blocks; ++i)
        if (f->layer[i].att) TRY(qwen38_attention_reset(f->layer[i].att, epoch, e));
    f->epoch = epoch; f->next_pos = 0; f->valid = true;
    return fail(e, Q38_OK, NULL);
}
