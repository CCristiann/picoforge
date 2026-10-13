/* PLE core on already normalized FP32 inputs. Projections/norm/conv belong
 * to B/C; this fused gate does not export a second activation primitive. */
#include "qwen38_ngram_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static Qwen38Status result(Qwen38Error *e, Qwen38Status code, const char *detail) {
    if (e) {
        *e=(Qwen38Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"PLE normalized gate");
        snprintf(e->detail,sizeof e->detail,"%s",detail);
    }
    return code;
}

static Qwen38F32 readonly(Qwen38F32Mut v) {
    Qwen38F32 r={.data=v.data,.capacity_bytes=v.capacity_bytes,.rank=v.rank};
    for (size_t i=0; i<4; i++) { r.dim[i]=v.dim[i]; r.stride_bytes[i]=v.stride_bytes[i]; }
    return r;
}

static Qwen38Status view_span(Qwen38F32 v, size_t *span) {
    if (!v.rank || v.rank>4) return Q38_SHAPE;
    size_t extent=sizeof(float);
    for (size_t j=v.rank; j-->0;) {
        if ((!v.dim[j] && j) || v.dim[j]>UINT32_MAX) return Q38_SHAPE;
        size_t stride=v.stride_bytes[j];
        if (stride%sizeof(float) || stride<extent || (j==v.rank-1 && stride!=sizeof(float)))
            return Q38_STRIDE;
        if (v.dim[j]) {
            if (v.dim[j]-1>(SIZE_MAX-extent)/stride) return Q38_OVERFLOW;
            extent+=(v.dim[j]-1)*stride;
        }
    }
    *span=v.dim[0] ? extent : 0;
    if (*span>v.capacity_bytes) return Q38_CAPACITY;
    if (*span && (!v.data || (uintptr_t)v.data%_Alignof(float))) return Q38_INVALID_ARGUMENT;
    if (*span>UINTPTR_MAX-(uintptr_t)v.data) return Q38_OVERFLOW;
    return Q38_OK;
}

static const float *at(Qwen38F32 v, size_t t, size_t s) {
    return (const float *)((const unsigned char *)v.data+t*v.stride_bytes[0]+
        (v.rank==3 ? s*v.stride_bytes[1] : 0));
}

/* Kept observable by the private test harness: sign near zero is a separate
 * acceptance criterion, not something a large residual can conceal. */
float q38e_ple_scaled_dot(const float *key, const float *query, size_t width) {
    float dot=0.0f;
    for (size_t d=0; d<width; d++) dot+=key[d]*query[d];
    return dot/sqrtf((float)width);
}

Qwen38Status qwen38_ple_gate(Qwen38F32 key, Qwen38F32 query, Qwen38F32 value,
    Qwen38F32Mut gate, Qwen38F32Mut output, Qwen38Error *e) {
    const Qwen38F32 views[]={key,query,value,readonly(gate),readonly(output)};
    if (key.rank!=3 || query.rank!=3 || value.rank!=2 || gate.rank!=2 || output.rank!=3)
        return result(e,Q38_SHAPE,"expected key/query/u[T,S,D], value[T,D], g[T,S]");
    const size_t tokens=key.dim[0], streams=key.dim[1], width=key.dim[2];
    if (query.dim[0]!=tokens || query.dim[1]!=streams || query.dim[2]!=width ||
        value.dim[0]!=tokens || value.dim[1]!=width || gate.dim[0]!=tokens ||
        gate.dim[1]!=streams || output.dim[0]!=tokens || output.dim[1]!=streams || output.dim[2]!=width)
        return result(e,Q38_SHAPE,"incompatible dimensions");
    size_t spans[5];
    for (size_t i=0; i<5; i++) {
        Qwen38Status code=view_span(views[i],spans+i);
        if (code!=Q38_OK) return result(e,code,"invalid view");
        for (size_t j=0; j<i; j++) {
            uintptr_t a=(uintptr_t)views[i].data, b=(uintptr_t)views[j].data;
            if (spans[i] && spans[j] && (a<=b ? b-a<spans[i] : a-b<spans[j]))
                return result(e,Q38_INVALID_ARGUMENT,"views must be disjoint");
        }
    }
    /* Scan input first so bad input cannot publish a prefix of valid gates. */
    for (size_t t=0; t<tokens; t++) {
        for (size_t d=0; d<width; d++)
            if (!isfinite(at(value,t,0)[d])) return result(e,Q38_NONFINITE,"value");
        for (size_t s=0; s<streams; s++) for (size_t d=0; d<width; d++)
            if (!isfinite(at(key,t,s)[d]) || !isfinite(at(query,t,s)[d]))
                return result(e,Q38_NONFINITE,"key/query");
    }
    for (size_t t=0; t<tokens; t++) for (size_t s=0; s<streams; s++) {
        float dot=q38e_ple_scaled_dot(at(key,t,s),at(query,t,s),width);
        if (!isfinite(dot)) return result(e,Q38_NONFINITE,"dot overflow; discard all output");
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
    return result(e,Q38_OK,"");
}

Qwen38Status qwen38_ple_workspace_create(Qwen38NgramStore *s, size_t budget,
    Qwen38PleWorkspace **out, Qwen38Error *e) {
    if (out) *out=NULL;
    if (!s || !out) return result(e,Q38_INVALID_ARGUMENT,"workspace arguments");
    size_t per=5*s->channels+s->config.hidden_size+s->config.streams;
    if (s->max_tokens>(SIZE_MAX-sizeof(Qwen38PleWorkspace))/4/per)
        return result(e,Q38_OVERFLOW,"workspace geometry");
    size_t bytes=s->max_tokens*per*4;
    if (sizeof(Qwen38PleWorkspace)>budget || bytes>budget-sizeof(Qwen38PleWorkspace))
        return result(e,Q38_CAPACITY,"workspace budget");
    Qwen38PleWorkspace *w=NULL;
    Qwen38Status code=q38e_allocate(s,sizeof *w,(void **)&w,e); if (code) return code;
    w->store=s; w->bytes=bytes;
    code=q38e_allocate(s,bytes,(void **)&w->data,e);
    if (code) { q38e_free(s,w,sizeof *w); return code; }
    size_t n=s->max_tokens*s->channels;
    w->key=w->data; w->query=w->key+n; w->u=w->query+n;
    w->z=w->u+n; w->conv=w->z+n; w->value=w->conv+n;
    w->gate=w->value+s->max_tokens*s->config.hidden_size;
    s->workspaces++; *out=w; return result(e,Q38_OK,"");
}
Qwen38Status qwen38_ple_workspace_destroy(Qwen38PleWorkspace *w, Qwen38Error *e) {
    if (!w) return result(e,Q38_INVALID_ARGUMENT,"null workspace");
    if (w->job) return result(e,Q38_BUSY,"workspace referenced until job_release");
    Qwen38NgramStore *s=w->store;
    q38e_free(s,w->data,w->bytes); q38e_free(s,w,sizeof *w); s->workspaces--;
    return result(e,Q38_OK,"");
}
static Qwen38F32Mut dense(float *p, size_t t, size_t s, size_t d, size_t rank) {
    return (Qwen38F32Mut){.data=p,.rank=rank,.capacity_bytes=t*s*d*4,
        .dim={t,rank==2 ? s*d : s,d},.stride_bytes={s*d*4,rank==2 ? 4 : d*4,4}};
}
static bool range_overlap(const void *p, size_t n, const void *q, size_t m) {
    uintptr_t a=(uintptr_t)p,b=(uintptr_t)q;
    return n && m && (a<=b ? b-a<n : a-b<m);
}
static Qwen38Status gain_shape(Qwen38F32 *g, size_t streams, size_t width) {
    size_t span; Qwen38Status code=view_span(*g,&span); if (code) return code;
    if (g->rank==1 && g->dim[0]==streams*width) {
        g->rank=2; g->dim[0]=streams; g->dim[1]=width;
        g->stride_bytes[0]=width*4; g->stride_bytes[1]=4;
    }
    if ((g->rank==1 && g->dim[0]==width) ||
        (g->rank==2 && g->dim[0]==streams && g->dim[1]==width)) return Q38_OK;
    return Q38_SHAPE;
}
Qwen38Status qwen38_ple_cpu(Qwen38NgramJob *j, const Qwen38PleWeights *weights,
    Qwen38F32 residual, Qwen38F32Mut delta, Qwen38PleWorkspace *w, Qwen38Error *e) {
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (j->state->gpu_session) return result(e,Q38_OWNER,"GPU state cannot use CPU PLE");
    if (!w || !weights) return result(e,Q38_INVALID_ARGUMENT,"PLE arguments");
    if (w->store!=j->store) return result(e,Q38_OWNER,"workspace owner");
    if (w->job) return result(e,Q38_BUSY,"workspace already referenced");
    if (j->phase!=(j->count ? Q38E_READY : Q38E_COMPUTED))
        return result(e,Q38_BAD_STATE,"PLE requires READY");
    Qwen38NgramStore *s=j->store;
    size_t t=j->count, h=s->config.streams, d=s->config.hidden_size, c=s->channels;
    size_t rspan=0,yspan=0;
    if (residual.rank!=3 || delta.rank!=3 || residual.dim[0]!=t || delta.dim[0]!=t ||
        residual.dim[1]!=h || delta.dim[1]!=h || residual.dim[2]!=d || delta.dim[2]!=d)
        return result(e,Q38_SHAPE,"residual/Delta [T,S,D]");
    code=view_span(residual,&rspan); if (!code) code=view_span(readonly(delta),&yspan);
    if (code) return result(e,code,"residual/Delta view");
    Qwen38F32 gains[]={weights->gamma_key,weights->gamma_query,weights->gamma_conv};
    for (size_t i=0;i<3;i++) { code=gain_shape(gains+i,h,d); if (code) return result(e,code,"gain shape"); }
    size_t tapspan;
    code=view_span(weights->taps,&tapspan);
    if (code) return result(e,code,"tap view");
    if (weights->taps.rank!=2 || weights->taps.dim[0]!=c || weights->taps.dim[1]!=s->config.conv_kernel)
        return result(e,Q38_SHAPE,"taps [S*D,K]");
    Qwen38F32 rows;
    code=qwen38_ngram_rows(j,&rows,e); if (code) return code;
    const Q38PackedCpuView *matrices[]={&weights->key,&weights->value};
    for (size_t i=0;i<2;i++) {
        const Q38PackedLayout *l=&matrices[i]->layout;
        if (l->k_logical!=s->width || l->n_out!=(i ? d : c) || l->n_experts!=1)
            return result(e,Q38_SHAPE,"projection shape");
        code=q38_packed_matmul(matrices[i],0,readonly(dense(NULL,0,1,s->width,2)),dense(NULL,0,1,l->n_out,2),e);
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
            return result(e,Q38_INVALID_ARGUMENT,"PLE input/output alias");
    if (range_overlap(delta.data,yspan,w->data,w->bytes)) return result(e,Q38_INVALID_ARGUMENT,"Delta aliases workspace");
    if (!t) return result(e,Q38_OK,"");
    w->job=j; j->workspace=w;
    Qwen38F32Mut key=dense(w->key,t,h,d,3), query=dense(w->query,t,h,d,3);
    Qwen38F32Mut value=dense(w->value,t,1,d,2), gate=dense(w->gate,t,1,h,2);
    Qwen38F32Mut u=dense(w->u,t,h,d,3), z=dense(w->z,t,h,d,3), conv=dense(w->conv,t,1,c,2);
    code=q38_packed_matmul(&weights->key,0,rows,dense(w->key,t,1,c,2),e);
    if (!code) code=q38_packed_matmul(&weights->value,0,rows,value,e);
    if (!code) code=qwen38_rmsnorm_groups(readonly(key),gains[0],key,s->config.rms_eps,e);
    if (!code) code=qwen38_rmsnorm_groups(residual,gains[1],query,s->config.rms_eps,e);
    if (!code) code=qwen38_ple_gate(readonly(key),readonly(query),readonly(value),gate,u,e);
    if (!code) code=qwen38_rmsnorm_groups(readonly(u),gains[2],z,s->config.rms_eps,e);
    if (!code) code=qwen38_conv_causal(&s->conv,readonly(dense(w->z,t,1,c,2)),weights->taps,conv,
        j->candidate.handle,j->position,e);
    if (!code) for (size_t ti=0;ti<t && !code;ti++) for (size_t si=0;si<h && !code;si++) for (size_t di=0;di<d;di++) {
        size_t i=(ti*h+si)*d+di;
        float v=w->u[i]+w->conv[i];
        if (!isfinite(v)) { code=result(e,Q38_NONFINITE,"Delta overflow"); break; }
        delta.data[ti*(delta.stride_bytes[0]/4)+si*(delta.stride_bytes[1]/4)+di]=v;
    }
    j->phase=code ? Q38E_ABORTED : Q38E_COMPUTED;
    return code ? code : result(e,Q38_OK,"");
}
