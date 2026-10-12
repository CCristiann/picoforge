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
