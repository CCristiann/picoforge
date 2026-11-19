/* PLE core on already normalized FP32 inputs. Projections/norm/conv belong
 * to B/C; this fused gate does not export a second activation primitive. */
#include "qwen4_ngram_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static Qwen4Status result(Qwen4Error *e, Qwen4Status code, const char *detail) {
    if (e) {
        *e=(Qwen4Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"PLE normalized gate");
        snprintf(e->detail,sizeof e->detail,"%s",detail);
    }
    return code;
}

static Qwen4F32 readonly(Qwen4F32Mut v) {
    Qwen4F32 r={.data=v.data,.capacity_bytes=v.capacity_bytes,.rank=v.rank};
    for (size_t i=0; i<4; i++) { r.dim[i]=v.dim[i]; r.stride_bytes[i]=v.stride_bytes[i]; }
    return r;
}

static Qwen4Status view_span(Qwen4F32 v, size_t *span) {
    if (!v.rank || v.rank>4) return QW4_SHAPE;
    size_t extent=sizeof(float);
    for (size_t j=v.rank; j-->0;) {
        if ((!v.dim[j] && j) || v.dim[j]>UINT32_MAX) return QW4_SHAPE;
        size_t stride=v.stride_bytes[j];
        if (stride%sizeof(float) || stride<extent || (j==v.rank-1 && stride!=sizeof(float)))
            return QW4_STRIDE;
        if (v.dim[j]) {
            if (v.dim[j]-1>(SIZE_MAX-extent)/stride) return QW4_OVERFLOW;
            extent+=(v.dim[j]-1)*stride;
        }
    }
    *span=v.dim[0] ? extent : 0;
    if (*span>v.capacity_bytes) return QW4_CAPACITY;
    if (*span && (!v.data || (uintptr_t)v.data%_Alignof(float))) return QW4_INVALID_ARGUMENT;
    if (*span>UINTPTR_MAX-(uintptr_t)v.data) return QW4_OVERFLOW;
    return QW4_OK;
}

static const float *at(Qwen4F32 v, size_t t, size_t s) {
    return (const float *)((const unsigned char *)v.data+t*v.stride_bytes[0]+
        (v.rank==3 ? s*v.stride_bytes[1] : 0));
}

/* Kept observable by the private test harness: sign near zero is a separate
 * acceptance criterion, not something a large residual can conceal. */
float qw4e_ple_scaled_dot(const float *key, const float *query, size_t width) {
    float dot=0.0f;
    for (size_t d=0; d<width; d++) dot+=key[d]*query[d];
    return dot/sqrtf((float)width);
}

Qwen4Status qwen4_ple_gate(Qwen4F32 key, Qwen4F32 query, Qwen4F32 value,
    Qwen4F32Mut gate, Qwen4F32Mut output, Qwen4Error *e) {
    const Qwen4F32 views[]={key,query,value,readonly(gate),readonly(output)};
    if (key.rank!=3 || query.rank!=3 || value.rank!=2 || gate.rank!=2 || output.rank!=3)
        return result(e,QW4_SHAPE,"expected key/query/u[T,S,D], value[T,D], g[T,S]");
    const size_t tokens=key.dim[0], streams=key.dim[1], width=key.dim[2];
    if (query.dim[0]!=tokens || query.dim[1]!=streams || query.dim[2]!=width ||
        value.dim[0]!=tokens || value.dim[1]!=width || gate.dim[0]!=tokens ||
        gate.dim[1]!=streams || output.dim[0]!=tokens || output.dim[1]!=streams || output.dim[2]!=width)
        return result(e,QW4_SHAPE,"incompatible dimensions");
    size_t spans[5];
    for (size_t i=0; i<5; i++) {
        Qwen4Status code=view_span(views[i],spans+i);
        if (code!=QW4_OK) return result(e,code,"invalid view");
        for (size_t j=0; j<i; j++) {
            uintptr_t a=(uintptr_t)views[i].data, b=(uintptr_t)views[j].data;
            if (spans[i] && spans[j] && (a<=b ? b-a<spans[i] : a-b<spans[j]))
                return result(e,QW4_INVALID_ARGUMENT,"views must be disjoint");
        }
    }
    /* Scan input first so bad input cannot publish a prefix of valid gates. */
    for (size_t t=0; t<tokens; t++) {
        for (size_t d=0; d<width; d++)
            if (!isfinite(at(value,t,0)[d])) return result(e,QW4_NONFINITE,"value");
        for (size_t s=0; s<streams; s++) for (size_t d=0; d<width; d++)
            if (!isfinite(at(key,t,s)[d]) || !isfinite(at(query,t,s)[d]))
                return result(e,QW4_NONFINITE,"key/query");
    }
    for (size_t t=0; t<tokens; t++) for (size_t s=0; s<streams; s++) {
        float dot=qw4e_ple_scaled_dot(at(key,t,s),at(query,t,s),width);
        if (!isfinite(dot)) return result(e,QW4_NONFINITE,"dot overflow; discard all output");
        float root=sqrtf(fmaxf(fabsf(dot),1e-6f));
        /* Sigmoid of signed root: bounded exp, with sign(0)=0 exactly.
         * This is the fused PLE expression, not a replacement for C's API. */
        float decay=expf(-root);
        float g=dot==0.0f ? 0.5f : dot>0.0f ? 1.0f/(1.0f+decay) : decay/(1.0f+decay);
        float *gp=(float *)((unsigned char *)gate.data+t*gate.stride_bytes[0]);
        float *up=(float *)((unsigned char *)output.data+t*output.stride_bytes[0]+s*output.stride_bytes[1]);
        gp[s]=g;
        const float *v=at(value,t,0);
        for (size_t d=0; d<width; d++) up[d]=g*v[d];
    }
    return result(e,QW4_OK,"");
}

Qwen4Status qwen4_ple_workspace_create(Qwen4NgramStore *s, size_t budget,
    Qwen4PleWorkspace **out, Qwen4Error *e) {
    if (out) *out=NULL;
    if (!s || !out) return result(e,QW4_INVALID_ARGUMENT,"workspace arguments");
    size_t per=5*s->channels+s->config.hidden_size+s->config.streams;
    if (s->max_tokens>(SIZE_MAX-sizeof(Qwen4PleWorkspace))/4/per)
        return result(e,QW4_OVERFLOW,"workspace geometry");
    size_t bytes=s->max_tokens*per*4;
    if (sizeof(Qwen4PleWorkspace)>budget || bytes>budget-sizeof(Qwen4PleWorkspace))
        return result(e,QW4_CAPACITY,"workspace budget");
    Qwen4PleWorkspace *w=NULL;
    Qwen4Status code=qw4e_allocate(s,sizeof *w,(void **)&w,e); if (code) return code;
    w->store=s; w->bytes=bytes;
    code=qw4e_allocate(s,bytes,(void **)&w->data,e);
    if (code) { qw4e_free(s,w,sizeof *w); return code; }
    size_t n=s->max_tokens*s->channels;
    w->key=w->data; w->query=w->key+n; w->u=w->query+n;
    w->z=w->u+n; w->conv=w->z+n; w->value=w->conv+n;
    w->gate=w->value+s->max_tokens*s->config.hidden_size;
    s->workspaces++; *out=w; return result(e,QW4_OK,"");
}
Qwen4Status qwen4_ple_workspace_destroy(Qwen4PleWorkspace *w, Qwen4Error *e) {
    if (!w) return result(e,QW4_INVALID_ARGUMENT,"null workspace");
    if (w->job) return result(e,QW4_BUSY,"workspace referenced until job_release");
    Qwen4NgramStore *s=w->store;
    qw4e_free(s,w->data,w->bytes); qw4e_free(s,w,sizeof *w); s->workspaces--;
    return result(e,QW4_OK,"");
}
static Qwen4F32Mut dense(float *p, size_t t, size_t s, size_t d, size_t rank) {
    return (Qwen4F32Mut){.data=p,.rank=rank,.capacity_bytes=t*s*d*4,
        .dim={t,rank==2 ? s*d : s,d},.stride_bytes={s*d*4,rank==2 ? 4 : d*4,4}};
}
static bool range_overlap(const void *p, size_t n, const void *q, size_t m) {
    uintptr_t a=(uintptr_t)p,b=(uintptr_t)q;
    return n && m && (a<=b ? b-a<n : a-b<m);
}
static Qwen4Status gain_shape(Qwen4F32 *g, size_t streams, size_t width) {
    size_t span; Qwen4Status code=view_span(*g,&span); if (code) return code;
    if (g->rank==1 && g->dim[0]==streams*width) {
        g->rank=2; g->dim[0]=streams; g->dim[1]=width;
        g->stride_bytes[0]=width*4; g->stride_bytes[1]=4;
    }
    if ((g->rank==1 && g->dim[0]==width) ||
        (g->rank==2 && g->dim[0]==streams && g->dim[1]==width)) return QW4_OK;
    return QW4_SHAPE;
}
Qwen4Status qwen4_ple_cpu(Qwen4NgramJob *j, const Qwen4PleWeights *weights,
    Qwen4F32 residual, Qwen4F32Mut delta, Qwen4PleWorkspace *w, Qwen4Error *e) {
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (j->state->gpu_session) return result(e,QW4_OWNER,"GPU state cannot use CPU PLE");
    if (!w || !weights) return result(e,QW4_INVALID_ARGUMENT,"PLE arguments");
    if (w->store!=j->store) return result(e,QW4_OWNER,"workspace owner");
    if (w->job) return result(e,QW4_BUSY,"workspace already referenced");
    if (j->phase!=(j->count ? QW4E_READY : QW4E_COMPUTED))
        return result(e,QW4_BAD_STATE,"PLE requires READY");
    Qwen4NgramStore *s=j->store;
    size_t t=j->count, h=s->config.streams, d=s->config.hidden_size, c=s->channels;
    size_t rspan=0,yspan=0;
    if (residual.rank!=3 || delta.rank!=3 || residual.dim[0]!=t || delta.dim[0]!=t ||
        residual.dim[1]!=h || delta.dim[1]!=h || residual.dim[2]!=d || delta.dim[2]!=d)
        return result(e,QW4_SHAPE,"residual/Delta [T,S,D]");
    code=view_span(residual,&rspan); if (!code) code=view_span(readonly(delta),&yspan);
    if (code) return result(e,code,"residual/Delta view");
    Qwen4F32 gains[]={weights->gamma_key,weights->gamma_query,weights->gamma_conv};
    for (size_t i=0;i<3;i++) { code=gain_shape(gains+i,h,d); if (code) return result(e,code,"gain shape"); }
    size_t tapspan;
    code=view_span(weights->taps,&tapspan);
    if (code) return result(e,code,"tap view");
    if (weights->taps.rank!=2 || weights->taps.dim[0]!=c || weights->taps.dim[1]!=s->config.conv_kernel)
        return result(e,QW4_SHAPE,"taps [S*D,K]");
    Qwen4F32 rows;
    code=qwen4_ngram_rows(j,&rows,e); if (code) return code;
    const QW4PackedCpuView *matrices[]={&weights->key,&weights->value};
    for (size_t i=0;i<2;i++) {
        const QW4PackedLayout *l=&matrices[i]->layout;
        if (l->k_logical!=s->width || l->n_out!=(i ? d : c) || l->n_experts!=1)
            return result(e,QW4_SHAPE,"projection shape");
        code=qw4_packed_matmul(matrices[i],0,readonly(dense(NULL,0,1,s->width,2)),dense(NULL,0,1,l->n_out,2),e);
        if (code) return code; /* B validates layout/span, without payload reads. */
    }
    const void *protected[]={residual.data,rows.data,weights->key.span.data,weights->value.span.data,
        gains[0].data,gains[1].data,gains[2].data,weights->taps.data,s,s->arrays,j,j->state,
        j->state->previous,j->next,j->ids,j->candidate.allocation,j->state->conv.allocation,w,weights};
    size_t sizes[]={rspan,rows.capacity_bytes,weights->key.layout.tensor_bytes,weights->value.layout.tensor_bytes,
        0,0,0,tapspan,sizeof *s,(s->config.order+2*s->config.head_count)*8,sizeof *j,sizeof *j->state,
        (s->config.order-1u)*4,(s->config.order-1u)*4,t*s->config.head_count*8,
        j->candidate.bytes,j->state->conv.bytes,sizeof *w,sizeof *weights};
    for (size_t i=0;i<3;i++) { code=view_span(gains[i],sizes+4+i); if (code) return result(e,code,"gain view"); }
    for (size_t i=0;i<sizeof sizes/sizeof *sizes;i++)
        if (range_overlap(delta.data,yspan,protected[i],sizes[i]) ||
            range_overlap(w->data,w->bytes,protected[i],sizes[i]))
            return result(e,QW4_INVALID_ARGUMENT,"PLE input/output alias");
    if (range_overlap(delta.data,yspan,w->data,w->bytes)) return result(e,QW4_INVALID_ARGUMENT,"Delta aliases workspace");
    if (!t) return result(e,QW4_OK,"");
    w->job=j; j->workspace=w;
    Qwen4F32Mut key=dense(w->key,t,h,d,3), query=dense(w->query,t,h,d,3);
    Qwen4F32Mut value=dense(w->value,t,1,d,2), gate=dense(w->gate,t,1,h,2);
    Qwen4F32Mut u=dense(w->u,t,h,d,3), z=dense(w->z,t,h,d,3), conv=dense(w->conv,t,1,c,2);
    code=qw4_packed_matmul(&weights->key,0,rows,dense(w->key,t,1,c,2),e);
    if (!code) code=qw4_packed_matmul(&weights->value,0,rows,value,e);
    if (!code) code=qwen4_rmsnorm_groups(readonly(key),gains[0],key,s->config.rms_eps,e);
    if (!code) code=qwen4_rmsnorm_groups(residual,gains[1],query,s->config.rms_eps,e);
    if (!code) code=qwen4_ple_gate(readonly(key),readonly(query),readonly(value),gate,u,e);
    if (!code) code=qwen4_rmsnorm_groups(readonly(u),gains[2],z,s->config.rms_eps,e);
    if (!code) code=qwen4_conv_causal(&s->conv,readonly(dense(w->z,t,1,c,2)),weights->taps,conv,
        j->candidate.handle,j->position,e);
    if (!code) for (size_t ti=0;ti<t && !code;ti++) for (size_t si=0;si<h && !code;si++) for (size_t di=0;di<d;di++) {
        size_t i=(ti*h+si)*d+di;
        float v=w->u[i]+w->conv[i];
        if (!isfinite(v)) { code=result(e,QW4_NONFINITE,"Delta overflow"); break; }
        delta.data[ti*(delta.stride_bytes[0]/4)+si*(delta.stride_bytes[1]/4)+di]=v;
    }
    j->phase=code ? QW4E_ABORTED : QW4E_COMPUTED;
    return code ? code : result(e,QW4_OK,"");
}
