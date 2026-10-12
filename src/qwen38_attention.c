/* QSA CPU core, FP32 and no fast-math. See integration.md and spec D. */
#include "qwen38_d_internal.h"
#include "qwen38_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct Qwen38AttentionTicket {
    Qwen38AttentionState *state;
    uint32_t tokens;
    uint64_t epoch;
    bool active, computed;
};
struct Qwen38AttentionState {
    Qwen38AttentionConfig config;
    Qwen38AttentionPlan plan;
    Qwen38StateInfo info;
    Qwen38AttentionTicket ticket;
    Qwen38AttentionWorkspace *workspace;
    float *k, *v, *ik, *blocks;
};
struct Qwen38AttentionWorkspace {
    Qwen38AttentionState *owner;
    float *scores, *scratch;
    uint32_t *heap, *blocks, *ids;
};

static Qwen38Status report(Qwen38Error *e, Qwen38Status code) {
    if (e) { memset(e, 0, sizeof(*e)); e->code = code; }
    return code;
}
static bool add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b; return true;
}
static bool mul_size(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b; return true;
}
static size_t aligned_metadata(size_t n) { return (n + 15) & ~(size_t)15; }
static Qwen38Status config_valid(const Qwen38AttentionConfig *c) {
    if (!c) return Q38_INVALID_ARGUMENT;
    if (!c->heads || !c->kv_heads || !c->head_dim || !c->index_heads || !c->index_dim
        || !c->ratio || !c->token_budget || !c->rotary_dim) return Q38_SHAPE;
    if (c->index_kv_heads != 1) return Q38_UNSUPPORTED;
    if (c->heads % c->kv_heads || c->token_budget % c->ratio || c->rotary_dim % 2
        || c->rotary_dim > c->head_dim || c->rotary_dim > c->index_dim) return Q38_SHAPE;
    if (!isfinite(c->rms_eps) || c->rms_eps <= 0 || !isfinite(c->rope_theta)
        || c->rope_theta <= 0) return Q38_RANGE;
    return Q38_OK;
}

Qwen38Status qwen38_attention_plan(const Qwen38AttentionConfig *c,
    uint32_t capacity, uint32_t max_tokens, Qwen38AttentionPlan *p, Qwen38Error *e) {
    if (!p) return report(e, Q38_INVALID_ARGUMENT);
    memset(p, 0, sizeof(*p));
    Qwen38Status code = config_valid(c);
    if (code != Q38_OK) return report(e, code);
    if (!capacity || !max_tokens || max_tokens > capacity) return report(e, Q38_RANGE);
    size_t kd, cache, blocks, bytes, scratch, temp, selected;
    /* Cache K/V + raw index keys + complete block keys. No eviction. */
    if (!mul_size(c->kv_heads, c->head_dim, &kd)
        || !mul_size(kd, 2, &cache) || !add_size(cache, c->index_dim, &cache)
        || !mul_size(cache, capacity, &cache)
        || !mul_size(capacity / c->ratio, c->index_dim, &blocks)
        || !add_size(cache, blocks, &cache) || !mul_size(cache, sizeof(float), &bytes)
        || !add_size(bytes, aligned_metadata(sizeof(Qwen38AttentionState)), &p->cache_bytes))
        return report(e, Q38_OVERFLOW);
    /* Reserve one token of C2 prep, then reduction + selection.
     * No allocation per token and no dense attention matrix. */
    if (!mul_size(c->heads, c->head_dim, &scratch) || !mul_size(scratch, 2, &scratch)
        || !add_size(scratch, kd, &scratch)
        || !mul_size(c->index_heads, c->index_dim, &temp)
        || !add_size(scratch, temp, &scratch) || !add_size(scratch, c->index_dim, &scratch)
        || !add_size(scratch, c->head_dim, &scratch) || !add_size(scratch, 16, &scratch)
        || !add_size(scratch, capacity / c->ratio, &scratch)
        || !add_size(c->token_budget, c->ratio - 1, &selected)
        || !mul_size(c->token_budget / c->ratio, 2, &temp)
        || !add_size(selected, temp, &selected)
        || !mul_size(selected, sizeof(uint32_t), &selected)
        || !mul_size(scratch, sizeof(float), &bytes) || !add_size(bytes, selected, &bytes)
        || !add_size(bytes, aligned_metadata(sizeof(Qwen38AttentionWorkspace)), &p->workspace_bytes)) {
        memset(p, 0, sizeof(*p)); return report(e, Q38_OVERFLOW);
    }
    p->capacity = capacity; p->max_tokens = max_tokens; p->alignment = 16;
    return report(e, Q38_OK);
}

Qwen38Status qwen38_attention_create(const Qwen38AttentionConfig *c,
    const Qwen38AttentionPlan *p, Qwen38StateKey key, size_t budget,
    Qwen38AttentionState **state, Qwen38AttentionWorkspace **workspace, Qwen38Error *e) {
    if (state) *state = NULL;
    if (workspace) *workspace = NULL;
    if (!state || !workspace || !p || !key.model || !key.epoch) return report(e, Q38_INVALID_ARGUMENT);
    Qwen38AttentionPlan expected;
    Qwen38Status code = qwen38_attention_plan(c, p->capacity, p->max_tokens, &expected, e);
    if (code != Q38_OK) return code;
    if (p->cache_bytes != expected.cache_bytes || p->workspace_bytes != expected.workspace_bytes
        || p->alignment != expected.alignment) return report(e, Q38_SHAPE);
    size_t total;
    if (!add_size(p->cache_bytes, p->workspace_bytes, &total)) return report(e, Q38_OVERFLOW);
    if (total > budget) return report(e, Q38_CAPACITY);
    Qwen38AttentionState *s = calloc(1, p->cache_bytes);
    Qwen38AttentionWorkspace *w = calloc(1, p->workspace_bytes);
    if (!s || !w) { free(s); free(w); return report(e, Q38_OOM); }
    s->config = *c; s->plan = *p; s->info.key = key; s->info.valid = true;
    s->workspace = w; s->ticket.state = s; w->owner = s;
    s->k = (float *)((unsigned char *)s + aligned_metadata(sizeof(*s)));
    size_t kd = (size_t)p->capacity * c->kv_heads * c->head_dim;
    s->v = s->k + kd; s->ik = s->v + kd;
    s->blocks = s->ik + (size_t)p->capacity * c->index_dim;
    w->scores = (float *)((unsigned char *)w + aligned_metadata(sizeof(*w)));
    w->scratch = w->scores + p->capacity / c->ratio;
    /* Reserved prep follows scratch; IDs at the end are naturally aligned. */
    size_t prep = 2 * (size_t)c->heads * c->head_dim + (size_t)c->kv_heads * c->head_dim
                + (size_t)c->index_heads * c->index_dim + c->index_dim;
    w->heap = (uint32_t *)(w->scratch + c->head_dim + 16 + prep);
    w->blocks = w->heap + c->token_budget / c->ratio;
    w->ids = w->blocks + c->token_budget / c->ratio;
    *state = s; *workspace = w;
    return report(e, Q38_OK);
}

Qwen38StateInfo qwen38_attention_state_info(const Qwen38AttentionState *s) {
    return s ? s->info : (Qwen38StateInfo){0};
}
Qwen38Status qwen38_attention_begin(Qwen38AttentionState *s, uint64_t pos,
    uint32_t tokens, uint64_t epoch, Qwen38AttentionTicket **ticket, Qwen38Error *e) {
    if (ticket) *ticket = NULL;
    if (!s || !ticket) return report(e, Q38_INVALID_ARGUMENT);
    if (epoch != s->info.key.epoch) return report(e, Q38_STALE);
    if (s->info.busy) return report(e, Q38_BUSY);
    if (!s->info.valid || pos != s->info.next_pos) return report(e, Q38_BAD_STATE);
    if (tokens > s->plan.max_tokens || tokens > s->plan.capacity - pos)
        return report(e, Q38_CAPACITY);
    s->ticket = (Qwen38AttentionTicket){s, tokens, epoch, true, tokens == 0};
    s->info.busy = true; *ticket = &s->ticket;
    return report(e, Q38_OK);
}
static Qwen38Status ticket_valid(Qwen38AttentionTicket *t) {
    if (!t || !t->state) return Q38_INVALID_ARGUMENT;
    if (!t->active || t->epoch != t->state->info.key.epoch) return Q38_STALE;
    if (!t->state->info.busy || !t->state->info.valid) return Q38_BAD_STATE;
    return Q38_OK;
}
Qwen38Status qwen38_attention_finish(Qwen38AttentionTicket *t, bool success, Qwen38Error *e) {
    if (!t || !t->state) return report(e, Q38_INVALID_ARGUMENT);
    if (!t->active || t->epoch != t->state->info.key.epoch) return report(e, Q38_STALE);
    Qwen38AttentionState *s = t->state;
    if (success && (!t->computed || !s->info.valid)) return report(e, Q38_BAD_STATE);
    if (success) s->info.next_pos += t->tokens;
    else s->info.valid = false; /* Even another layer's failure requires replay. */
    s->info.busy = false; t->active = false;
    return report(e, Q38_OK);
}
Qwen38Status qwen38_attention_reset(Qwen38AttentionState *s, uint64_t epoch, Qwen38Error *e) {
    if (!s) return report(e, Q38_INVALID_ARGUMENT);
    if (s->info.busy) return report(e, Q38_BUSY);
    if (!epoch || epoch <= s->info.key.epoch) return report(e, Q38_STALE);
    s->info.key.epoch = epoch; s->info.next_pos = 0; s->info.valid = true;
    /* Old bytes become unreachable. Prefix length, not zero-fill, owns validity. */
    return report(e, Q38_OK);
}
Qwen38Status qwen38_attention_destroy(Qwen38AttentionState *s,
    Qwen38AttentionWorkspace *w, Qwen38Error *e) {
    if (!s || !w) return report(e, Q38_INVALID_ARGUMENT);
    if (s->workspace != w || w->owner != s) return report(e, Q38_OWNER);
    if (s->info.busy) return report(e, Q38_BUSY);
    free(w); free(s); return report(e, Q38_OK);
}
const float *q38d_cache_keys(const Qwen38AttentionState *s) { return s->k; }
const float *q38d_cache_raw_index(const Qwen38AttentionState *s) { return s->ik; }
const float *q38d_cache_blocks(const Qwen38AttentionState *s) { return s->blocks; }

/* D's internal head/feature axes are dense; token pitch may include padding. */
static Qwen38Status view(Qwen38F32 v, size_t rank, const size_t *shape, size_t *span) {
    if (v.rank != rank) return Q38_SHAPE;
    size_t row = sizeof(float);
    for (size_t i = rank; i-- > 0;) {
        if (v.dim[i] != shape[i]) return Q38_SHAPE;
        if (((i > 0 || rank == 1) && v.stride_bytes[i] != row) || v.stride_bytes[i] % sizeof(float))
            return Q38_STRIDE;
        if (i > 0 && !mul_size(row, shape[i], &row)) return Q38_OVERFLOW;
    }
    if (v.stride_bytes[0] < row) return Q38_STRIDE;
    *span = 0;
    if (!shape[0]) return Q38_OK;
    if (!mul_size(shape[0] - 1, v.stride_bytes[0], span) || !add_size(*span, row, span))
        return Q38_OVERFLOW;
    if (*span > v.capacity_bytes) return Q38_CAPACITY;
    if (!v.data) return Q38_INVALID_ARGUMENT;
    if ((uintptr_t)v.data % _Alignof(float)) return Q38_STRIDE;
    if (*span > UINTPTR_MAX - (uintptr_t)v.data) return Q38_OVERFLOW;
    return Q38_OK;
}
static bool overlap(const void *a, size_t an, const void *b, size_t bn) {
    if (!an || !bn) return false;
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}
static const float *row_at(Qwen38F32 v, size_t row) {
    return (const float *)((const unsigned char *)v.data + row * v.stride_bytes[0]);
}
static Qwen38Status copy_rows(float *dst, Qwen38F32 v, uint32_t rows, size_t width) {
    for (uint32_t i = 0; i < rows; ++i) {
        const float *src = row_at(v, i);
        for (size_t d = 0; d < width; ++d) {
            if (!isfinite(src[d])) return Q38_NONFINITE;
            dst[(size_t)i * width + d] = src[d];
        }
    }
    return Q38_OK;
}

static Qwen38Status attend_row(Qwen38AttentionState *s, uint32_t pos,
    const float *q, const float *gate, const float *iq, float *out) {
    const Qwen38AttentionConfig *c = &s->config;
    Qwen38AttentionWorkspace *w = s->workspace;
    uint32_t n = (pos + 1) / c->ratio;
    Qwen38Status code = q38d_scores(iq, s->blocks, c->index_heads, c->index_dim, n, w->scores);
    if (code != Q38_OK) return code;
    code = q38d_select(w->scores, n, c->token_budget / c->ratio, w->heap, w->blocks);
    if (code != Q38_OK) return code;
    uint32_t selected = n < c->token_budget / c->ratio ? n : c->token_budget / c->ratio;
    uint32_t count = 0;
    for (uint32_t b = 0; b < selected; ++b)
        for (uint32_t r = 0; r < c->ratio; ++r) w->ids[count++] = w->blocks[b] * c->ratio + r;
    for (uint32_t tail = n * c->ratio; tail <= pos; ++tail) w->ids[count++] = tail;
    return q38d_attend(c,q,gate,s->k,s->v,pos+1,w->ids,count,out,w->scratch);
}

Qwen38Status q38d_append_prepared(Qwen38AttentionTicket *t, const Q38DPrepared *in,
    Qwen38F32Mut out, Qwen38AttentionWorkspace *w, Qwen38Error *e) {
    Qwen38Status code = ticket_valid(t);
    if (code != Q38_OK) return report(e, code);
    Qwen38AttentionState *s = t->state;
    if (!w || s->workspace != w || w->owner != s) return report(e, Q38_OWNER);
    if (!in) return report(e, Q38_INVALID_ARGUMENT);
    if (t->computed && t->tokens) return report(e, Q38_BAD_STATE);
    const Qwen38AttentionConfig *c = &s->config;
    uint32_t begin = (uint32_t)s->info.next_pos, end = begin + t->tokens;
    uint32_t old_blocks = begin / c->ratio, new_blocks = end / c->ratio - old_blocks;
    Qwen38F32 ov = {out.data, out.capacity_bytes, out.rank, {0}, {0}};
    memcpy(ov.dim, out.dim, sizeof(out.dim)); memcpy(ov.stride_bytes, out.stride_bytes, sizeof(out.stride_bytes));
    Qwen38F32 views[] = {in->q, in->gate_factor, in->k, in->v, in->iq, in->ik, in->blocks, ov};
    size_t shapes[8][3] = {{t->tokens,c->heads,c->head_dim}, {t->tokens,c->heads,c->head_dim},
        {t->tokens,c->kv_heads,c->head_dim}, {t->tokens,c->kv_heads,c->head_dim},
        {t->tokens,c->index_heads,c->index_dim}, {t->tokens,c->index_dim,0},
        {new_blocks,c->index_dim,0}, {t->tokens,c->heads,c->head_dim}};
    size_t spans[8];
    for (size_t i = 0; i < 8; ++i) {
        code = view(views[i], (i == 5 || i == 6) ? 2 : 3, shapes[i], &spans[i]);
        if (code != Q38_OK) return report(e, code);
        if (overlap(views[i].data, spans[i], s, s->plan.cache_bytes)
            || overlap(views[i].data, spans[i], w, s->plan.workspace_bytes)) return report(e, Q38_OWNER);
        for (size_t j = 0; j < i; ++j)
            if (overlap(views[i].data, spans[i], views[j].data, spans[j])) return report(e, Q38_INVALID_ARGUMENT);
    }
    if (!t->tokens) return report(e, Q38_OK);
    size_t kd = (size_t)c->kv_heads * c->head_dim;
    /* From this point any failure invalidates the session, including output. */
    code = copy_rows(s->k + begin * kd, in->k, t->tokens, kd);
    if (code == Q38_OK) code = copy_rows(s->v + begin * kd, in->v, t->tokens, kd);
    if (code == Q38_OK) code = copy_rows(s->ik + (size_t)begin * c->index_dim, in->ik, t->tokens, c->index_dim);
    if (code == Q38_OK) code = copy_rows(s->blocks + (size_t)old_blocks * c->index_dim, in->blocks, new_blocks, c->index_dim);
    for (uint32_t i = 0; i < t->tokens && code == Q38_OK; ++i) {
        float *dest = (float *)((unsigned char *)out.data + (size_t)i * out.stride_bytes[0]);
        code = attend_row(s,begin+i,row_at(in->q,i),row_at(in->gate_factor,i),row_at(in->iq,i),dest);
    }
    if (code != Q38_OK) s->info.valid = false;
    else t->computed = true;
    return report(e, code);
}

static Qwen38F32 head_view(const float *x, uint32_t h, uint32_t d, size_t step) {
    return (Qwen38F32){x, ((h-1)*step+d)*4, 3, {1,h,d,0}, {h*step*4,step*4,4,0}};
}
void q38d_last_query(const Qwen38AttentionWorkspace *w, const float **q,
    const float **gate, const float **iq, const float **scores, const uint32_t **ids) {
    const Qwen38AttentionConfig *c=&w->owner->config;
    size_t hd=(size_t)c->heads*c->head_dim, kd=(size_t)c->kv_heads*c->head_dim;
    *q=w->scratch+c->head_dim+16; *gate=*q+hd; *iq=*gate+hd+kd;
    *scores=w->scores; *ids=w->ids;
}
static Qwen38F32Mut head_out(float *x, uint32_t h, uint32_t d) {
    return (Qwen38F32Mut){x,(size_t)h*d*4,3,{1,h,d,0},{(size_t)h*d*4,(size_t)d*4,4,0}};
}
static Qwen38Status norm_rope(Qwen38F32 x, Qwen38F32 gain, float *out,
    const Qwen38AttentionConfig *c, uint32_t pos) {
    uint32_t h = (uint32_t)x.dim[1], d = (uint32_t)x.dim[2];
    Qwen38Status code = qwen38_rmsnorm_groups(x,gain,head_out(out,h,d),c->rms_eps,NULL);
    if (code == Q38_OK) code = q38d_rope(out,h,d,c->rotary_dim,pos,c->rope_theta);
    return code;
}

Qwen38Status qwen38_attention_cpu(Qwen38AttentionTicket *t,
    const Qwen38AttentionInput *in, Qwen38F32Mut out,
    Qwen38AttentionWorkspace *w, Qwen38Error *e) {
    Qwen38Status code = ticket_valid(t);
    if (code != Q38_OK) return report(e,code);
    Qwen38AttentionState *s = t->state;
    if (!w || w != s->workspace || w->owner != s) return report(e,Q38_OWNER);
    if (!in) return report(e,Q38_INVALID_ARGUMENT);
    if (t->computed && t->tokens) return report(e,Q38_BAD_STATE);
    const Qwen38AttentionConfig *c = &s->config;
    Qwen38F32 ov = {out.data,out.capacity_bytes,out.rank,{0},{0}};
    memcpy(ov.dim,out.dim,sizeof(out.dim)); memcpy(ov.stride_bytes,out.stride_bytes,sizeof(out.stride_bytes));
    Qwen38F32 inputs[] = {in->qg,in->k,in->v,in->iq,in->ik,
                         in->gamma_q,in->gamma_k,in->gamma_iq,in->gamma_ik,ov};
    size_t dims[10][4] = {{t->tokens,c->heads,2,c->head_dim}, {t->tokens,c->kv_heads,c->head_dim,0},
        {t->tokens,c->kv_heads,c->head_dim,0}, {t->tokens,c->index_heads,c->index_dim,0},
        {t->tokens,c->index_dim,0,0}, {c->head_dim,0,0,0}, {c->head_dim,0,0,0},
        {c->index_dim,0,0,0}, {c->index_dim,0,0,0}, {t->tokens,c->heads,c->head_dim,0}};
    const size_t ranks[] = {4,3,3,3,2,1,1,1,1,3};
    size_t spans[10];
    /* Structural admission is complete before any output or cache write.
     * Read-only inputs may share backing (e.g. one gain for Q and K). */
    for (size_t i=0;i<10;++i) {
        code = view(inputs[i],ranks[i],dims[i],&spans[i]);
        if (code != Q38_OK) return report(e,code);
        if (overlap(inputs[i].data,spans[i],s,s->plan.cache_bytes)
            || overlap(inputs[i].data,spans[i],w,s->plan.workspace_bytes)) return report(e,Q38_OWNER);
    }
    for (size_t i=0;i<9;++i)
        if (overlap(inputs[i].data,spans[i],ov.data,spans[9])) return report(e,Q38_INVALID_ARGUMENT);
    if (!t->tokens) return report(e,Q38_OK);
    size_t hd=(size_t)c->heads*c->head_dim, kd=(size_t)c->kv_heads*c->head_dim;
    float *q=w->scratch+c->head_dim+16, *gate=q+hd;
    float *iq=gate+hd+kd, *pool=iq+(size_t)c->index_heads*c->index_dim;
    uint32_t begin=(uint32_t)s->info.next_pos;
    for (uint32_t i=0;i<t->tokens && code==Q38_OK;++i) {
        uint32_t pos=begin+i;
        const float *qg=row_at(in->qg,i);
        code=norm_rope(head_view(qg,c->heads,c->head_dim,2*(size_t)c->head_dim),in->gamma_q,q,c,pos);
        if (code==Q38_OK) code=qwen38_activate(Q38_SIGMOID,
            head_view(qg+c->head_dim,c->heads,c->head_dim,2*(size_t)c->head_dim),
            head_out(gate,c->heads,c->head_dim),NULL);
        if (code==Q38_OK) code=norm_rope(head_view(row_at(in->k,i),c->kv_heads,c->head_dim,c->head_dim),
            in->gamma_k,s->k+pos*kd,c,pos);
        if (code==Q38_OK) code=norm_rope(head_view(row_at(in->iq,i),c->index_heads,c->index_dim,c->index_dim),
            in->gamma_iq,iq,c,pos);
        Qwen38F32 vr=in->v, ir=in->ik;
        vr.data=row_at(in->v,i); ir.data=row_at(in->ik,i);
        if (code==Q38_OK) code=copy_rows(s->v+pos*kd,vr,1,kd);
        if (code==Q38_OK) code=copy_rows(s->ik+(size_t)pos*c->index_dim,ir,1,c->index_dim);
        if (code==Q38_OK && (pos+1)%c->ratio==0) {
            uint32_t b=pos/c->ratio, start=b*c->ratio;
            code=q38d_pool(s->ik+(size_t)start*c->index_dim,c->ratio,c->index_dim,pool);
            if (code==Q38_OK) code=norm_rope(head_view(pool,1,c->index_dim,c->index_dim),
                in->gamma_ik,s->blocks+(size_t)b*c->index_dim,c,start);
        }
        float *dest=(float *)((unsigned char *)out.data+(size_t)i*out.stride_bytes[0]);
        if (code==Q38_OK) code=attend_row(s,pos,q,gate,iq,dest);
    }
    if (code!=Q38_OK) s->info.valid=false;
    else t->computed=true;
    return report(e,code);
}

float q38d_rope_frequency(float theta, uint32_t i, uint32_t rotary) {
    return 1.0f / powf(theta, (float)(2 * i) / (float)rotary);
}
Qwen38Status q38d_rope(float *x, uint32_t heads, uint32_t dim,
                      uint32_t rotary, uint32_t pos, float theta) {
    if (!x || !heads || !dim || !rotary || rotary % 2 || rotary > dim)
        return Q38_SHAPE;
    if (!isfinite(theta) || theta <= 0) return Q38_RANGE;
    for (uint32_t h = 0; h < heads; ++h) {
        float *row = x + (size_t)h * dim;
        for (uint32_t i = 0; i < dim; ++i)
            if (!isfinite(row[i])) return Q38_NONFINITE;
        for (uint32_t i = 0; i < rotary / 2; ++i) {
            /* Match default HF frequency construction: reciprocal pow,
             * then F32 position multiplication, not a half-angle variant. */
            float inv = q38d_rope_frequency(theta,i,rotary);
            float angle = (float)pos * inv;
            float c = cosf(angle), s = sinf(angle);
            float a = row[i], b = row[i + rotary / 2];
            row[i] = a * c - b * s;
            row[i + rotary / 2] = b * c + a * s;
            if (!isfinite(row[i]) || !isfinite(row[i + rotary / 2]))
                return Q38_NONFINITE;
        }
    }
    return Q38_OK;
}

Qwen38Status q38d_pool(const float *raw, uint32_t ratio, uint32_t dim, float *pooled) {
    if (!raw || !pooled || !ratio || !dim) return Q38_SHAPE;
    for (uint32_t d = 0; d < dim; ++d) {
        float sum = 0;
        for (uint32_t r = 0; r < ratio; ++r) {
            float x = raw[(size_t)r * dim + d];
            if (!isfinite(x)) return Q38_NONFINITE;
            sum += x;
        }
        if (!isfinite(sum)) return Q38_NONFINITE;
        pooled[d] = sum / (float)ratio;
    }
    return Q38_OK;
}

Qwen38Status q38d_scores(const float *iq, const float *blocks,
                        uint32_t heads, uint32_t dim, uint32_t count, float *scores) {
    if (!heads || !dim || !iq || (count && (!blocks || !scores))) return Q38_SHAPE;
    for (size_t i = 0; i < (size_t)heads * dim; ++i)
        if (!isfinite(iq[i])) return Q38_NONFINITE;
    for (uint32_t b = 0; b < count; ++b) {
        float score = 0;
        for (uint32_t h = 0; h < heads; ++h) {
            float dot = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                float q = iq[(size_t)h * dim + d], k = blocks[(size_t)b * dim + d];
                if (!isfinite(q) || !isfinite(k)) return Q38_NONFINITE;
                dot += q * k;
            }
            if (!isfinite(dot)) return Q38_NONFINITE;
            score += fmaxf(dot, 0);
        }
        if (!isfinite(score)) return Q38_NONFINITE;
        scores[b] = score; /* Unscaled, unlike HF's positive 1/sqrt(Di). */
    }
    return Q38_OK;
}

static bool worse(const float *s, uint32_t a, uint32_t b) {
    return s[a] < s[b] || (s[a] == s[b] && a > b);
}

static void sift(const float *s, uint32_t *heap, uint32_t n, uint32_t root) {
    while (root < n / 2) {
        uint32_t child = 2 * root + 1;
        if (child + 1 < n && worse(s, heap[child + 1], heap[child])) ++child;
        if (!worse(s, heap[child], heap[root])) break;
        uint32_t swap = heap[root]; heap[root] = heap[child]; heap[child] = swap;
        root = child;
    }
}

static int id_order(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

Qwen38Status q38d_select(const float *scores, uint32_t count, uint32_t budget,
                        uint32_t *heap, uint32_t *selected) {
    if (!budget || (count && (!scores || !heap || !selected))) return Q38_SHAPE;
    /* Preflight every visible score. Never inspect future/poisoned slots. */
    for (uint32_t i = 0; i < count; ++i)
        if (!isfinite(scores[i])) return Q38_NONFINITE;
    uint32_t n = count < budget ? count : budget;
    if (!n) return Q38_OK;
    for (uint32_t i = 0; i < n; ++i) heap[i] = i;
    for (uint32_t i = n / 2; i > 0; --i) sift(scores, heap, n, i - 1);
    for (uint32_t i = n; i < count; ++i)
        if (worse(scores, heap[0], i)) { heap[0] = i; sift(scores, heap, n, 0); }
    memcpy(selected, heap, (size_t)n * sizeof(*heap));
    qsort(selected, n, sizeof(*selected), id_order);
    return Q38_OK;
}

Qwen38Status q38d_attend(const Qwen38AttentionConfig *c, const float *q,
                        const float *gate, const float *k, const float *v,
                        uint32_t visible, const uint32_t *ids, uint32_t count,
                        float *out, float *scratch) {
    if (!c || !q || !gate || !k || !v || !ids || !out || !scratch)
        return Q38_INVALID_ARGUMENT;
    if (!c->heads || !c->kv_heads || c->heads % c->kv_heads || !c->head_dim || !count)
        return Q38_SHAPE;
    for (uint32_t i = 0; i < count; ++i)
        if (ids[i] >= visible || (i && ids[i] <= ids[i - 1])) return Q38_RANGE;
    uint32_t dim = c->head_dim, group = c->heads / c->kv_heads;
    float scale = 1.0f / sqrtf((float)dim);
    float *acc = scratch, *logits = scratch + dim;
    for (uint32_t h = 0; h < c->heads; ++h) {
        const float *query = q + (size_t)h * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            float g = gate[(size_t)h * dim + d];
            if (!isfinite(query[d]) || !isfinite(g)) return Q38_NONFINITE;
            if (g < 0 || g > 1) return Q38_RANGE;
            acc[d] = 0;
        }
        float maximum = -INFINITY, denominator = 0;
        for (uint32_t base = 0; base < count;) {
            uint32_t tile = count - base < 16 ? count - base : 16;
            float next_max = maximum;
            for (uint32_t i = 0; i < tile; ++i) {
                const float *key = k + ((size_t)ids[base + i] * c->kv_heads + h / group) * dim;
                float dot = 0;
                for (uint32_t d = 0; d < dim; ++d) {
                    if (!isfinite(key[d])) return Q38_NONFINITE;
                    dot += query[d] * key[d];
                }
                logits[i] = dot * scale;
                if (!isfinite(logits[i])) return Q38_NONFINITE;
                next_max = fmaxf(next_max, logits[i]);
            }
            /* First tile has old denominator zero: no -inf - (-inf). */
            float rescale = denominator == 0 ? 0 : expf(maximum - next_max);
            denominator *= rescale;
            for (uint32_t d = 0; d < dim; ++d) acc[d] *= rescale;
            for (uint32_t i = 0; i < tile; ++i) {
                float weight = expf(logits[i] - next_max);
                denominator += weight;
                const float *value = v + ((size_t)ids[base + i] * c->kv_heads + h / group) * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    if (!isfinite(value[d])) return Q38_NONFINITE;
                    acc[d] += weight * value[d];
                }
            }
            maximum = next_max;
            base += tile;
        }
        if (!isfinite(denominator) || denominator <= 0) return Q38_NONFINITE;
        for (uint32_t d = 0; d < dim; ++d) {
            float y = (acc[d] / denominator) * gate[(size_t)h * dim + d];
            if (!isfinite(acc[d]) || !isfinite(y)) return Q38_NONFINITE;
            out[(size_t)h * dim + d] = y;
        }
    }
    return Q38_OK;
}
