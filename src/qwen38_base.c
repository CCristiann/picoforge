#include "qwen38_c_internal.h"

static Qwen38Status positive_eps(float eps, Qwen38Error *e) {
    if (!isfinite(eps)) return q38c_error(e,Q38_NONFINITE,"nonfinite normalization epsilon");
    if (eps<=0) return q38c_error(e,Q38_RANGE,"normalization epsilon must be positive");
    return Q38_OK;
}
static Qwen38Status activate_scaled(Qwen38Activation kind, Qwen38F32 x, Qwen38F32Mut out,
        float divisor, float multiplier, bool inplace, Qwen38Error *e) {
    Qwen38F32 y=q38c_ro(out);
    Q38C_TRY(q38c_view(x,e)); Q38C_TRY(q38c_dims(y,x.rank,x.dim,e));
    Q38C_TRY(q38c_alias(y,x,inplace,e));
    Q38C_TRY(q38c_finite(x,e));
    for (size_t i=0;i<q38c_count(x);++i) {
        float v=q38c_get(x,i)/divisor, result;
        if (kind==Q38_SIGMOID) result=q38c_sigmoid(v);
        else if (kind==Q38_SILU) result=q38c_silu(v);
        else result=fmaxf(v,0.0f)+log1pf(expf(-fabsf(v)));
        result *= multiplier;
        if (!isfinite(result)) return q38c_error(e,Q38_NONFINITE,"activation overflow");
        out.data[q38c_offset(y,i)]=result;
    }
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_activate(Qwen38Activation kind, Qwen38F32 x, Qwen38F32Mut out, Qwen38Error *e) {
    if (kind!=Q38_SIGMOID && kind!=Q38_SILU && kind!=Q38_SOFTPLUS)
        return q38c_error(e,Q38_UNSUPPORTED,"unknown activation");
    /* Generic activation has disjoint views; HC's explicit in-place APIs below
     * opt in separately, as required by integration.md's alias contract. */
    return activate_scaled(kind,x,out,1.0f,1.0f,false,e);
}
Qwen38Status qwen38_rmsnorm_groups(Qwen38F32 x, Qwen38F32 gain, Qwen38F32Mut out, float eps, Qwen38Error *e) {
    Q38C_TRY(positive_eps(eps,e));
    Q38C_TRY(q38c_dims(x,3,x.dim,e)); Qwen38F32 y=q38c_ro(out);
    Q38C_TRY(q38c_dims(y,3,x.dim,e));
    size_t tokens=x.dim[0], groups=x.dim[1], width=x.dim[2];
    if (gain.rank==1) { Q38C_TRY(q38c_dims(gain,1,(size_t[]){width},e)); }
    else { Q38C_TRY(q38c_dims(gain,2,(size_t[]){groups,width},e)); }
    Q38C_TRY(q38c_alias(y,x,true,e)); Q38C_TRY(q38c_alias(y,gain,false,e));
    if (!tokens) return q38c_error(e,Q38_OK,NULL);
    Q38C_TRY(q38c_finite(x,e)); Q38C_TRY(q38c_finite(gain,e));
    for (size_t t=0;t<tokens;++t) for (size_t g=0;g<groups;++g) {
        size_t xi=t*(x.stride_bytes[0]/4)+g*(x.stride_bytes[1]/4);
        size_t yi=t*(y.stride_bytes[0]/4)+g*(y.stride_bytes[1]/4);
        size_t wi=gain.rank==1 ? 0 : g*(gain.stride_bytes[0]/4);
        float sum=0.0f;
        for (size_t j=0;j<width;++j) sum += x.data[xi+j]*x.data[xi+j];
        float variance=sum/(float)width+eps;
        if (!isfinite(sum) || !isfinite(variance)) return q38c_error(e,Q38_NONFINITE,"RMS reduction overflow");
        float scale=1.0f/sqrtf(variance);
        for (size_t j=0;j<width;++j) {
            float result=(x.data[xi+j]*scale)*gain.data[wi+j];
            if (!isfinite(result)) return q38c_error(e,Q38_NONFINITE,"RMS gain overflow");
            out.data[yi+j]=result;
        }
    }
    return q38c_error(e,Q38_OK,NULL);
}
static Qwen38Status hc_shape(const Qwen38HcShape *s, Qwen38Error *e) {
    if (!s) return q38c_error(e,Q38_INVALID_ARGUMENT,"null HC shape");
    if (!s->streams || !s->hidden_size || !s->low_rank) return q38c_error(e,Q38_SHAPE,"zero HC dimension");
    if (s->streams>UINT32_MAX || s->hidden_size>UINT32_MAX || s->low_rank>UINT32_MAX)
        return q38c_error(e,Q38_RANGE,"HC dimension exceeds uint32");
    return positive_eps(s->rms_eps,e);
}
Qwen38Status qwen38_hc_lowrank_activate(const Qwen38HcShape *s, Qwen38F32 lo, Qwen38F32Mut out, Qwen38Error *e) {
    Q38C_TRY(hc_shape(s,e)); Q38C_TRY(q38c_dims(lo,2,(size_t[]){lo.dim[0],s->low_rank},e));
    return activate_scaled(Q38_SILU,lo,out,(float)s->streams,1.0f,true,e);
}
Qwen38Status qwen38_hc_inject_weights(const Qwen38HcShape *s, Qwen38F32 logits, Qwen38F32Mut out, Qwen38Error *e) {
    Q38C_TRY(hc_shape(s,e)); Q38C_TRY(q38c_dims(logits,2,(size_t[]){logits.dim[0],s->streams},e));
    return activate_scaled(Q38_SIGMOID,logits,out,(float)s->streams,2.0f,true,e);
}
Qwen38Status qwen38_hc_read(const Qwen38HcShape *s, Qwen38F32 normed,
        Qwen38F32 logits, Qwen38F32Mut out, Qwen38Error *e) {
    Q38C_TRY(hc_shape(s,e)); size_t tokens=normed.dim[0]; Qwen38F32 y=q38c_ro(out);
    Q38C_TRY(q38c_dims(normed,3,(size_t[]){tokens,s->streams,s->hidden_size},e));
    Q38C_TRY(q38c_dims(logits,3,normed.dim,e)); Q38C_TRY(q38c_dims(y,2,(size_t[]){tokens,s->hidden_size},e));
    Q38C_TRY(q38c_alias(y,normed,false,e)); Q38C_TRY(q38c_alias(y,logits,false,e));
    Q38C_TRY(q38c_finite(normed,e)); Q38C_TRY(q38c_finite(logits,e));
    for (size_t t=0;t<tokens;++t) for (size_t j=0;j<s->hidden_size;++j) {
        float sum=0;
        for (size_t h=0;h<s->streams;++h) {
            size_t i=(t*s->streams+h)*s->hidden_size+j;
            sum += q38c_get(normed,i)*q38c_sigmoid(q38c_get(logits,i));
        }
        if (!isfinite(sum)) return q38c_error(e,Q38_NONFINITE,"HC read overflow");
        out.data[t*(y.stride_bytes[0]/4)+j]=sum/(float)s->streams;
    }
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_hc_add(const Qwen38HcShape *s, Qwen38F32Mut residual,
        Qwen38F32 block, Qwen38F32 inject, Qwen38Error *e) {
    Q38C_TRY(hc_shape(s,e)); Qwen38F32 r=q38c_ro(residual); size_t tokens=r.dim[0];
    Q38C_TRY(q38c_dims(r,3,(size_t[]){tokens,s->streams,s->hidden_size},e));
    Q38C_TRY(q38c_dims(block,2,(size_t[]){tokens,s->hidden_size},e));
    Q38C_TRY(q38c_dims(inject,2,(size_t[]){tokens,s->streams},e));
    Q38C_TRY(q38c_alias(r,block,false,e)); Q38C_TRY(q38c_alias(r,inject,false,e));
    Q38C_TRY(q38c_finite(r,e)); Q38C_TRY(q38c_finite(block,e)); Q38C_TRY(q38c_finite(inject,e));
    for (size_t t=0;t<tokens;++t) for (size_t h=0;h<s->streams;++h) for (size_t j=0;j<s->hidden_size;++j) {
        size_t ri=t*(r.stride_bytes[0]/4)+h*(r.stride_bytes[1]/4)+j;
        float result=residual.data[ri]+q38c_get(inject,t*s->streams+h)*q38c_get(block,t*s->hidden_size+j);
        if (!isfinite(result)) return q38c_error(e,Q38_NONFINITE,"HC residual overflow");
        residual.data[ri]=result;
    }
    return q38c_error(e,Q38_OK,NULL);
}
