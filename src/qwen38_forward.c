/* G1 CPU graph: composition of A/B/C/D/E/F, never a second operator.
 * Roles are resolved once at create (A); every product is B's; norms,
 * activations and HC are C's primitives. See qwen38_forward.h for the order. */
#include "qwen38_forward.h"
#include "qwen38_c_internal.h"
#include <stdlib.h>

typedef struct { Q38PackedCpuView down, up, inject; float *gain; } G1Hc;
typedef struct { G1Hc attn, ffn; } G1Layer;
struct Qwen38Forward {
    Qwen38ConfigView c;
    const Qwen38Config *config;
    const Qwen38Weights *weights;
    const GpuWeightStore *store;
    uint32_t max_tokens, capacity;
    uint64_t epoch, next_pos;
    bool valid;
    Q38PackedCpuView embed, head;
    G1Hc final;
    G1Layer *layer;
    /* One arena: residual R[T,H,E], HC scratch, then per-block outputs. */
    float *arena, *r, *normed, *low, *act, *gate, *il, *inject, *mixed;
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
    float **slot[] = {&f->r, &f->normed, &f->low, &f->act, &f->gate, &f->il, &f->inject, &f->mixed};
    size_t width[] = {h*w, h*w, r, r, h*w, h, h, w}, total = 0;
    for (size_t i = 0; i < sizeof width / sizeof *width; ++i) total += (t*width[i] + 15) & ~(size_t)15;
    if (!(f->arena = aligned_alloc(64, total*4))) return fail(e, Q38_OOM, "graph scratch");
    for (size_t i = 0, offset = 0; i < sizeof width / sizeof *width; ++i) {
        *slot[i] = f->arena + offset; offset += (t*width[i] + 15) & ~(size_t)15;
    }
    return Q38_OK;
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
        if (kind->kind == Q38_LAYER_QSA) TRY(bind_hc(f, i, attn_roles, &f->layer[i].attn, e));
        TRY(bind_hc(f, i, ffn_roles, &f->layer[i].ffn, e));
    }
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
    *f = (Qwen38Forward){.c=*view, .config=config, .weights=weights, .store=store,
        .max_tokens=max_tokens, .capacity=capacity, .epoch=1, .valid=true};
    Qwen38Status rc = create(f, e);
    if (rc != Q38_OK) { qwen38_forward_destroy(f); return rc; }
    *out = f;
    return fail(e, Q38_OK, NULL);
}

void qwen38_forward_destroy(Qwen38Forward *f) {
    if (!f) return;
    for (uint32_t i = 0; f->layer && i < f->c.trunk_blocks; ++i) {
        free(f->layer[i].attn.gain); free(f->layer[i].ffn.gain);
    }
    free(f->final.gain); free(f->layer); free(f->arena); free(f);
}

Qwen38StateInfo qwen38_forward_info(const Qwen38Forward *f) {
    if (!f) return (Qwen38StateInfo){0};
    return (Qwen38StateInfo){.key={f->config, QWEN38_GLOBAL_LAYER, f->epoch},
        .next_pos=f->next_pos, .valid=f->valid};
}

/* Placeholder until the block steps land: refuses, never fakes success. */
static Qwen38Status layer_cpu(Qwen38Forward *f, uint32_t i, size_t t, Qwen38GraphTrace trace,
        void *user, Qwen38Error *e) {
    (void)f; (void)i; (void)t; (void)trace; (void)user;
    return fail(e, Q38_UNSUPPORTED, "graph layers are not wired yet");
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
    for (uint32_t i = 0; i < layers; ++i) TRY(layer_cpu(f, i, t, trace, user, e));
    TRY(hc_read(f, &f->final, t, e));          /* output mixer: no inject */
    emit(trace, user, QWEN38_GLOBAL_LAYER, Q38G_FINAL, mat(f->mixed, t, w));
    return q38_packed_matmul(&f->head, 0, q38c_ro(mat(f->mixed + (t-rows)*w, rows, w)), logits, e);
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
    /* Prefix slices and failures leave layer states mid-stream: reset first. */
    if (rc != Q38_OK || layers < f->c.trunk_blocks) f->valid = false;
    else f->next_pos += count;
    return rc == Q38_OK ? fail(e, Q38_OK, NULL) : rc;
}

Qwen38Status qwen38_forward_reset(Qwen38Forward *f, Qwen38Error *e) {
    if (!f) return fail(e, Q38_INVALID_ARGUMENT, "missing graph");
    if (f->epoch == UINT64_MAX) return fail(e, Q38_OVERFLOW, "epoch exhausted");
    f->epoch++; f->next_pos = 0; f->valid = true;
    return fail(e, Q38_OK, NULL);
}
