/* F v1: routing on caller-owned projected logits. Packed products belong to B. */
#include "qwen38_f_internal.h"
#include "qwen38_ops.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct { uintptr_t first, end; } MoeSpan;
enum { LOGITS, PROBS, SCORES, IDS, GATHER, DOWN, GATE, UP, ACT, SHARED, SG, SA,
       OFFSETS, CURSORS, P2R, R2P, NBUF };
struct Qwen38Moe { Qwen38MoeConfig c; Q38PackedCpuView w[8]; size_t refs; bool alive; };
struct Qwen38MoeWork {
    Qwen38Moe *owner;
    size_t cap, micro, bytes, count[NBUF];
    void *buf[NBUF], *memory;
    bool busy, failed;
    Q38MoeObserver observer; void *observer_ctx;
};
_Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "requires 32-bit elements");

static Qwen38Status moe_error(Qwen38Error *err, Qwen38Status code, const char *detail) {
    if (err) {
        memset(err, 0, sizeof *err);
        err->code = code;
        if (code != Q38_OK) {
            snprintf(err->subject, sizeof err->subject, "qwen38_moe");
            snprintf(err->detail, sizeof err->detail, "%s", detail);
        }
    }
    return code;
}

static Qwen38Status moe_config(const Qwen38MoeConfig *c, Qwen38Error *err) {
    if (!c) return moe_error(err, Q38_INVALID_ARGUMENT, "null config");
    if (!c->hidden_size || !c->experts || !c->top_k || c->top_k > c->experts ||
        !c->intermediate || !c->shared_intermediate || c->down_physical < c->intermediate)
        return moe_error(err, Q38_RANGE, "invalid MoE dimensions");
    return Q38_OK;
}

static Qwen38F32Mut moe_matrix(float *p, size_t rows, size_t cols) {
    return (Qwen38F32Mut){.data=p,.capacity_bytes=rows*cols*4,.rank=2,
        .dim={rows,cols,1,1},.stride_bytes={cols*4,4,0,0}};
}

Qwen38Status qwen38_moe_create(const Qwen38MoeConfig *c, const Qwen38MoeWeights *weights,
                             Qwen38Moe **out, Qwen38Error *err) {
    if (out) *out=NULL;
    if (!out || !weights) return moe_error(err,Q38_INVALID_ARGUMENT,"null create argument");
    Qwen38Status st=moe_config(c,err); if (st) return st;
    Q38PackedCpuView w[]={weights->router,weights->gate,weights->up,weights->down,
        weights->shared_gate,weights->shared_up,weights->shared_down,weights->shared_router};
    size_t h=c->hidden_size, i=c->intermediate, s=c->shared_intermediate;
    size_t in[]={h,h,h,i,h,h,s,h}, n[]={c->experts,i,i,h,s,s,h,1};
    for (size_t a=0;a<8;a++) {
        Q38PackedLayout l=w[a].layout;
        size_t experts=(a>=1 && a<=3)?c->experts:1;
        if (l.k_logical!=in[a] || l.n_out!=n[a] || l.n_experts!=experts ||
            l.k_storage!=(a==3?c->down_physical:in[a]))
            return moe_error(err,Q38_SHAPE,"MoE weight shape mismatch");
        if ((a==0 || a==7) && l.gguf_type!=0)
            return moe_error(err,Q38_FORMAT,"router must be F32");
        /* B validates its entire descriptor/span even for M=0, without reading
         * weight bytes. No duplicate packed-layout validation in F. */
        Qwen38F32Mut y=moe_matrix(NULL,0,n[a]);
        Qwen38F32 rx={.data=NULL,.rank=2,.dim={0,in[a],1,1},.stride_bytes={in[a]*4,4,0,0}};
        st=q38_packed_matmul(&w[a],0,rx,y,err); if (st) return st;
    }
    Qwen38Moe *m=calloc(1,sizeof *m);
    if (!m) return moe_error(err,Q38_OOM,"MoE allocation");
    m->c=*c; memcpy(m->w,w,sizeof w); m->refs=1; m->alive=true; *out=m;
    return moe_error(err,Q38_OK,"");
}
static void moe_release(Qwen38Moe *m) { if (m && --m->refs==0) free(m); }
void qwen38_moe_destroy(Qwen38Moe *m) { if (m && m->alive) { m->alive=false; moe_release(m); } }

/* Counts are checked before multiplication; each arena offset is size_t aligned.
 * Microbatching limits activation memory independently of admitted token_cap. */
static Qwen38Status moe_work_plan(const Qwen38Moe *m, size_t cap, Qwen38MoeWork *p,
                                 Qwen38Error *err) {
    if (!m || !m->alive) return moe_error(err,Q38_BAD_STATE,"inactive MoE");
    size_t t=cap<32?cap:32, k=m->c.top_k, e=m->c.experts, h=m->c.hidden_size;
    size_t i=m->c.intermediate>m->c.shared_intermediate?m->c.intermediate:m->c.shared_intermediate;
    if (k && t>SIZE_MAX/k) return moe_error(err,Q38_OVERFLOW,"pair count overflow");
    size_t pairs=t*k;
    size_t a[]={t,t,t,t,pairs,pairs,t,t,t,t,t,t,e+1,e,pairs,pairs};
    size_t b[]={e,e,k,k,h,h,i,i,i,h,1,1,1,1,1,1};
    p->bytes=0; p->cap=cap; p->micro=t;
    for (size_t j=0;j<NBUF;j++) {
        size_t unit=j<OFFSETS?4:sizeof(size_t);
        if (a[j]>SIZE_MAX/b[j] || a[j]*b[j]>(SIZE_MAX-7)/unit)
            return moe_error(err,Q38_OVERFLOW,"scratch size overflow");
        size_t bytes=(a[j]*b[j]*unit+7)&~(size_t)7;
        if (p->bytes>SIZE_MAX-bytes) return moe_error(err,Q38_OVERFLOW,"arena overflow");
        p->count[j]=bytes; p->bytes+=bytes;
    }
    if (p->bytes>SIZE_MAX-sizeof *p) return moe_error(err,Q38_OVERFLOW,"workspace overflow");
    return Q38_OK;
}
Qwen38Status qwen38_moe_work_size(const Qwen38Moe *m,size_t cap,size_t *bytes,Qwen38Error *err) {
    if (!bytes) return moe_error(err,Q38_INVALID_ARGUMENT,"null size output");
    *bytes=0; Qwen38MoeWork p={0}; Qwen38Status st=moe_work_plan(m,cap,&p,err);
    if (st) return st;
    *bytes=sizeof p+p.bytes; return moe_error(err,Q38_OK,"");
}
Qwen38Status qwen38_moe_work_create(const Qwen38Moe *m,size_t cap,size_t budget,
                                   Qwen38MoeWork **out,Qwen38Error *err) {
    if (out) *out=NULL;
    if (!out) return moe_error(err,Q38_INVALID_ARGUMENT,"null workspace output");
    Qwen38MoeWork p={0}; Qwen38Status st=moe_work_plan(m,cap,&p,err); if (st) return st;
    if (budget<sizeof p+p.bytes) return moe_error(err,Q38_CAPACITY,"workspace budget too small");
    Qwen38MoeWork *w=malloc(sizeof *w); if (!w) return moe_error(err,Q38_OOM,"workspace metadata");
    *w=p; w->memory=calloc(1,p.bytes);
    if (!w->memory) { free(w); return moe_error(err,Q38_OOM,"workspace arena"); }
    size_t offset=0;
    for (size_t j=0;j<NBUF;j++) { w->buf[j]=(char *)w->memory+offset; offset+=w->count[j]; }
    w->owner=(Qwen38Moe *)m; w->owner->refs++; *out=w;
    return moe_error(err,Q38_OK,"");
}
Qwen38Status qwen38_moe_work_reset(Qwen38MoeWork *w,Qwen38Error *err) {
    if (!w) return moe_error(err,Q38_INVALID_ARGUMENT,"null workspace");
    if (w->busy) return moe_error(err,Q38_BUSY,"workspace executing");
    if (!w->owner->alive) return moe_error(err,Q38_BAD_STATE,"destroyed owner");
    memset(w->memory,0,w->bytes); w->failed=false; return moe_error(err,Q38_OK,"");
}
void qwen38_moe_work_destroy(Qwen38MoeWork *w) {
    if (w && !w->busy) { moe_release(w->owner); free(w->memory); free(w); }
}
void q38_moe_observe(Qwen38MoeWork *w,Q38MoeObserver fn,void *ctx) {
    if (w && !w->busy) { w->observer=fn; w->observer_ctx=ctx; }
}

static Qwen38Status moe_memory(const void *data, size_t cap, size_t bytes, size_t alignment,
                              MoeSpan *span, Qwen38Error *err) {
    span->first = span->end = 0;
    if (!bytes) return Q38_OK;
    if (cap < bytes) return moe_error(err, Q38_CAPACITY, "view capacity too small");
    if (!data) return moe_error(err, Q38_INVALID_ARGUMENT, "null nonempty view");
    uintptr_t first = (uintptr_t)data;
    if (first % alignment) return moe_error(err, Q38_STRIDE, "unaligned view");
    if (bytes > UINTPTR_MAX - first)
        return moe_error(err, Q38_OVERFLOW, "address extent overflow");
    *span = (MoeSpan){first, first + bytes};
    return Q38_OK;
}

/* Bounding spans include row padding: aliasing through a gap is rejected too.
 * Empty batches validate shape/stride but neither inspect nor require data. */
static Qwen38Status moe_span(const void *data, size_t cap, size_t rows, size_t cols,
                            size_t pitch, MoeSpan *span, Qwen38Error *err) {
    span->first = span->end = 0;
    if (cols > SIZE_MAX / 4) return moe_error(err, Q38_OVERFLOW, "row bytes overflow");
    size_t row_bytes = cols * 4;
    if (pitch % 4 || pitch < row_bytes)
        return moe_error(err, Q38_STRIDE, "row stride must cover contiguous features");
    if (!rows) return Q38_OK;
    if (rows - 1 > (SIZE_MAX - row_bytes) / pitch)
        return moe_error(err, Q38_OVERFLOW, "view extent overflow");
    size_t bytes = (rows - 1) * pitch + row_bytes;
    return moe_memory(data, cap, bytes, 4, span, err);
}

static Qwen38Status moe_f32(Qwen38F32 v, size_t rows, size_t cols,
                           MoeSpan *span, Qwen38Error *err) {
    if (v.rank != 2 || v.dim[0] != rows || v.dim[1] != cols)
        return moe_error(err, Q38_SHAPE, "expected rank-2 token-major view");
    if (v.stride_bytes[1] != sizeof(float))
        return moe_error(err, Q38_STRIDE, "features must be contiguous");
    return moe_span(v.data, v.capacity_bytes, rows, cols, v.stride_bytes[0], span, err);
}

static Qwen38F32 moe_const(Qwen38F32Mut v) {
    Qwen38F32 out = {.data = v.data, .capacity_bytes = v.capacity_bytes, .rank = v.rank};
    memcpy(out.dim, v.dim, sizeof out.dim);
    memcpy(out.stride_bytes, v.stride_bytes, sizeof out.stride_bytes);
    return out;
}

static Qwen38Status moe_disjoint(const MoeSpan *spans, size_t n, Qwen38Error *err) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < i; j++)
            if (spans[i].first < spans[j].end && spans[j].first < spans[i].end)
                return moe_error(err, Q38_INVALID_ARGUMENT, "overlapping views");
    return Q38_OK;
}

Qwen38Status qwen38_moe_route(const Qwen38MoeConfig *c, Qwen38F32 logits,
                            Qwen38MoeRoutes routes, Qwen38F32Mut probabilities,
                            Qwen38Error *err) {
    Qwen38Status st = moe_config(c, err);
    if (st != Q38_OK) return st;
    const size_t t_count = logits.dim[0], e_count = c->experts, k = c->top_k;
    MoeSpan spans[4];
    st = moe_f32(logits, t_count, e_count, &spans[0], err);
    if (st != Q38_OK) return st;
    st = moe_f32(moe_const(probabilities), t_count, e_count, &spans[1], err);
    if (st != Q38_OK) return st;
    st = moe_f32(moe_const(routes.weights), t_count, k, &spans[2], err);
    if (st != Q38_OK) return st;
    if (routes.ids.rows != t_count || routes.ids.cols != k)
        return moe_error(err, Q38_SHAPE, "route ID shape mismatch");
    st = moe_span(routes.ids.data, routes.ids.capacity_bytes, t_count, k,
                  routes.ids.row_stride_bytes, &spans[3], err);
    if (st != Q38_OK) return st;
    st = moe_disjoint(spans, 4, err);
    if (st != Q38_OK) return st;
    /* Preflight all input rows so NaN/Inf cannot publish a partial route. */
    for (size_t t = 0; t < t_count; t++) {
        const float *z = logits.data + t * (logits.stride_bytes[0] / 4);
        for (size_t e = 0; e < e_count; e++) {
            if (!isfinite(z[e])) {
                st = moe_error(err, Q38_NONFINITE, "nonfinite router logit");
                if (err) err->row = t;
                return st;
            }
        }
    }
    for (size_t t = 0; t < t_count; t++) {
        const float *z = logits.data + t * (logits.stride_bytes[0] / 4);
        float *p = probabilities.data + t * (probabilities.stride_bytes[0] / 4);
        float *w = routes.weights.data + t * (routes.weights.stride_bytes[0] / 4);
        uint32_t *ids = routes.ids.data + t * (routes.ids.row_stride_bytes / 4);
        float maximum = z[0], sum = 0.0f;
        for (size_t e = 1; e < e_count; e++) if (z[e] > maximum) maximum = z[e];
        for (size_t e = 0; e < e_count; e++) {
            /* Finite extreme inputs can give -Inf here; exp(-Inf)=0 is correct. */
            p[e] = expf(z[e] - maximum);
            sum += p[e];
        }
        for (size_t e = 0; e < e_count; e++) p[e] /= sum;
        float selected_sum = 0.0f;
        for (size_t slot = 0; slot < k; slot++) {
            size_t best = 0;
            bool found = false;
            for (size_t e = 0; e < e_count; e++) {
                bool used = false;
                for (size_t j = 0; j < slot; j++) used |= ids[j] == e;
                if (!used && (!found || p[e] > p[best])) { best = e; found = true; }
            }
            ids[slot] = (uint32_t)best; /* ascending scan + strict > resolves ties */
            w[slot] = p[best];
            selected_sum += w[slot];
        }
        if (c->norm_topk_prob)
            for (size_t slot = 0; slot < k; slot++) w[slot] /= selected_sum;
    }
    return moe_error(err, Q38_OK, "");
}

Qwen38Status q38_moe_group_pairs(const Qwen38MoeConfig *c, Qwen38U32Mut ids,
                               Q38MoeGrouping g, Qwen38Error *err) {
    Qwen38Status st = moe_config(c, err);
    if (st != Q38_OK) return st;
    if (ids.cols != c->top_k) return moe_error(err, Q38_SHAPE, "group ID shape mismatch");
    const size_t e_count = c->experts, k = c->top_k;
    if (ids.rows > SIZE_MAX / k || e_count == SIZE_MAX)
        return moe_error(err, Q38_OVERFLOW, "group count overflow");
    const size_t pairs = ids.rows * k;
    MoeSpan spans[5];
    st = moe_span(ids.data, ids.capacity_bytes, ids.rows, ids.cols,
                  ids.row_stride_bytes, &spans[0], err);
    if (st != Q38_OK) return st;
    const void *data[] = {g.offsets, g.cursors, g.pair_to_row, g.row_to_pair};
    const size_t counts[] = {e_count+1, e_count, pairs, pairs};
    const size_t caps[] = {g.offsets_capacity_bytes, g.cursors_capacity_bytes,
                          g.pair_to_row_capacity_bytes, g.row_to_pair_capacity_bytes};
    for (size_t a = 0; a < 4; a++) {
        if (counts[a] > SIZE_MAX / sizeof(size_t))
            return moe_error(err, Q38_OVERFLOW, "group array bytes overflow");
        st = moe_memory(data[a], caps[a], counts[a]*sizeof(size_t), _Alignof(size_t),
                        &spans[a+1], err);
        if (st != Q38_OK) return st;
    }
    st = moe_disjoint(spans, 5, err);
    if (st != Q38_OK) return st;
    for (size_t t = 0; t < ids.rows; t++) {
        const uint32_t *row = ids.data + t*(ids.row_stride_bytes/4);
        for (size_t s = 0; s < k; s++) {
            if (row[s] >= e_count) return moe_error(err, Q38_RANGE, "expert ID out of range");
            for (size_t j = 0; j < s; j++)
                if (row[j] == row[s]) return moe_error(err, Q38_INVALID_ARGUMENT, "duplicate expert ID");
        }
    }
    memset(g.offsets, 0, (e_count+1)*sizeof(size_t));
    for (size_t t = 0; t < ids.rows; t++)
        for (size_t s = 0; s < k; s++)
            g.offsets[ids.data[t*(ids.row_stride_bytes/4)+s]+1]++;
    for (size_t e = 0; e < e_count; e++) {
        g.offsets[e+1] += g.offsets[e];
        g.cursors[e] = g.offsets[e];
    }
    for (size_t t = 0; t < ids.rows; t++) {
        for (size_t s = 0; s < k; s++) {
            size_t pair = t*k+s;
            uint32_t expert = ids.data[t*(ids.row_stride_bytes/4)+s];
            size_t row = g.cursors[expert]++;
            g.pair_to_row[pair] = row;
            g.row_to_pair[row] = pair;
        }
    }
    return moe_error(err, Q38_OK, "");
}

static void moe_trace(Qwen38MoeWork *w,size_t stage,size_t expert,size_t token,
                      Qwen38F32 in,Qwen38F32Mut out) {
    if (w->observer) w->observer(w->observer_ctx,stage,expert,token,in,moe_const(out));
}
static Qwen38Status moe_linear(Qwen38MoeWork *w,size_t stage,size_t expert,size_t token,
                               Qwen38F32 x,Qwen38F32Mut y,Qwen38Error *err) {
    Qwen38Status st=q38_packed_matmul(&w->owner->w[stage],expert,x,y,err);
    if (!st) moe_trace(w,stage,expert,token,x,y);
    return st;
}
#define MOE_TRY(call) do { Qwen38Status result=(call); if (result) return result; } while (0)
/* One expert group has at most micro tokens, since IDs are distinct per token.
 * Gate/up/activation scratch is reused for every group and the shared branch. */
static Qwen38Status moe_expert(Qwen38MoeWork *w,size_t expert,size_t token,
                               Qwen38F32 x,Qwen38F32Mut y,bool shared,Qwen38Error *err) {
    size_t n=x.dim[0], i=shared?w->owner->c.shared_intermediate:w->owner->c.intermediate;
    Qwen38F32Mut g=moe_matrix(w->buf[GATE],n,i), u=moe_matrix(w->buf[UP],n,i);
    Qwen38F32Mut a=moe_matrix(w->buf[ACT],n,i);
    MOE_TRY(moe_linear(w,shared?4:1,expert,token,x,g,err));
    MOE_TRY(moe_linear(w,shared?5:2,expert,token,x,u,err));
    MOE_TRY(qwen38_activate(Q38_SILU,moe_const(g),a,err));
    moe_trace(w,8,shared?SIZE_MAX:expert,token,moe_const(g),a);
    for (size_t j=0;j<n*i;j++) {
        a.data[j]*=u.data[j];
        if (!isfinite(a.data[j])) return moe_error(err,Q38_NONFINITE,"gate/up overflow");
    }
    moe_trace(w,9,shared?SIZE_MAX:expert,token,moe_const(u),a);
    return moe_linear(w,shared?6:3,expert,token,moe_const(a),y,err);
}
static Qwen38Status moe_batch(Qwen38MoeWork *w,Qwen38F32 x,Qwen38F32Mut y,
                              size_t token,Qwen38Error *err) {
    const Qwen38MoeConfig *c=&w->owner->c;
    size_t t=x.dim[0], e=c->experts,k=c->top_k,h=c->hidden_size,pairs=t*k;
    Qwen38F32Mut z=moe_matrix(w->buf[LOGITS],t,e),p=moe_matrix(w->buf[PROBS],t,e);
    Qwen38MoeRoutes routes={{w->buf[IDS],t*k*4,t,k,k*4},moe_matrix(w->buf[SCORES],t,k)};
    MOE_TRY(moe_linear(w,0,0,token,x,z,err));
    MOE_TRY(qwen38_moe_route(c,moe_const(z),routes,p,err));
    Q38MoeGrouping g={w->buf[OFFSETS],w->count[OFFSETS],w->buf[CURSORS],w->count[CURSORS],
        w->buf[P2R],w->buf[R2P],w->count[P2R],w->count[R2P]};
    MOE_TRY(q38_moe_group_pairs(c,routes.ids,g,err));
    float *gather=w->buf[GATHER],*down=w->buf[DOWN];
    for (size_t row=0;row<pairs;row++) {
        size_t original=g.row_to_pair[row]/k;
        memcpy(gather+row*h,x.data+original*(x.stride_bytes[0]/4),h*4);
    }
    for (size_t expert=0;expert<e;expert++) {
        size_t first=g.offsets[expert], n=g.offsets[expert+1]-first;
        if (!n) continue;
        Qwen38F32 gx=moe_const(moe_matrix(gather+first*h,n,h));
        MOE_TRY(moe_expert(w,expert,token,gx,moe_matrix(down+first*h,n,h),false,err));
    }
    Qwen38F32Mut shared=moe_matrix(w->buf[SHARED],t,h);
    MOE_TRY(moe_expert(w,0,token,x,shared,true,err));
    Qwen38F32Mut sg=moe_matrix(w->buf[SG],t,1),sa=moe_matrix(w->buf[SA],t,1);
    MOE_TRY(moe_linear(w,7,0,token,x,sg,err));
    MOE_TRY(qwen38_activate(Q38_SIGMOID,moe_const(sg),sa,err));
    moe_trace(w,10,SIZE_MAX,token,moe_const(sg),sa);
    for (size_t row=0;row<t;row++) for (size_t col=0;col<h;col++) {
        float sum=0;
        for (size_t slot=0;slot<k;slot++) {
            size_t pair=row*k+slot;
            float term=routes.weights.data[pair]*down[g.pair_to_row[pair]*h+col];
            sum+=term;
            if (!isfinite(term) || !isfinite(sum))
                return moe_error(err,Q38_NONFINITE,"routed reduction overflow");
        }
        y.data[row*(y.stride_bytes[0]/4)+col]=sum;
    }
    moe_trace(w,11,SIZE_MAX,token,x,y);
    for (size_t row=0;row<t;row++) for (size_t col=0;col<h;col++) {
        float term=sa.data[row]*shared.data[row*h+col];
        float *out=y.data+row*(y.stride_bytes[0]/4)+col;
        *out+=term;
        if (!isfinite(term) || !isfinite(*out)) return moe_error(err,Q38_NONFINITE,"shared sum overflow");
    }
    moe_trace(w,12,SIZE_MAX,token,x,y);
    return Q38_OK;
}

Qwen38Status qwen38_moe_forward(const Qwen38Moe *m,Qwen38F32 x,Qwen38F32Mut y,
                               Qwen38MoeRoutes *routes,Qwen38MoeWork *w,Qwen38Error *err) {
    if (!m || !w) return moe_error(err,Q38_INVALID_ARGUMENT,"null forward handle");
    if (w->owner!=m) return moe_error(err,Q38_OWNER,"workspace belongs to another MoE");
    if (!m->alive || w->failed) return moe_error(err,Q38_BAD_STATE,"inactive MoE/workspace; reset required");
    if (w->busy) return moe_error(err,Q38_BUSY,"workspace executing");
    size_t t=x.dim[0],h=m->c.hidden_size,k=m->c.top_k;
    if (t>w->cap) return moe_error(err,Q38_CAPACITY,"batch exceeds token capacity");
    MoeSpan spans[4]; size_t ns=2;
    MOE_TRY(moe_f32(x,t,h,&spans[0],err));
    MOE_TRY(moe_f32(moe_const(y),t,h,&spans[1],err));
    if (routes) {
        MOE_TRY(moe_f32(moe_const(routes->weights),t,k,&spans[2],err));
        if (routes->ids.rows!=t || routes->ids.cols!=k)
            return moe_error(err,Q38_SHAPE,"optional route shape mismatch");
        MOE_TRY(moe_span(routes->ids.data,routes->ids.capacity_bytes,t,k,
                         routes->ids.row_stride_bytes,&spans[3],err));
        ns=4;
    }
    MOE_TRY(moe_disjoint(spans,ns,err));
    for (size_t a=0;a<8;a++) {
        const Q38PackedCpuView *v=&m->w[a];
        MoeSpan weights={(uintptr_t)v->span.data,(uintptr_t)v->span.data+v->layout.tensor_bytes};
        for (size_t b=0;b<ns;b++) {
            MoeSpan both[]={spans[b],weights}; MOE_TRY(moe_disjoint(both,2,err));
        }
    }
    for (size_t row=0;row<t;row++) for (size_t col=0;col<h;col++)
        if (!isfinite(x.data[row*(x.stride_bytes[0]/4)+col]))
            return moe_error(err,Q38_NONFINITE,"nonfinite input");
    w->busy=true;
    for (size_t start=0;start<t;) {
        size_t n=t-start<w->micro?t-start:w->micro;
        Qwen38F32 bx=x; Qwen38F32Mut by=y;
        bx.data+=start*(x.stride_bytes[0]/4); bx.capacity_bytes-=start*x.stride_bytes[0]; bx.dim[0]=n;
        by.data+=start*(y.stride_bytes[0]/4); by.capacity_bytes-=start*y.stride_bytes[0]; by.dim[0]=n;
        Qwen38Status st=moe_batch(w,bx,by,start,err);
        if (st) { w->busy=false; w->failed=true; return st; }
        if (routes) for (size_t row=0;row<n;row++) {
            memcpy(routes->ids.data+(start+row)*(routes->ids.row_stride_bytes/4),
                   (uint32_t *)w->buf[IDS]+row*k,k*4);
            memcpy(routes->weights.data+(start+row)*(routes->weights.stride_bytes[0]/4),
                   (float *)w->buf[SCORES]+row*k,k*4);
        }
        start+=n;
    }
    w->busy=false; return moe_error(err,Q38_OK,"");
}
