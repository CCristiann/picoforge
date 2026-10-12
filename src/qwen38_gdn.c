#include "qwen38_c_internal.h"

struct Qwen38GdnState { Q38CState base; Qwen38GdnShape shape; size_t elements; };
static Qwen38Status gdn_shape(const Qwen38GdnShape *s, size_t *channels, size_t *elements, Qwen38Error *e) {
    if (!s) return q38c_error(e,Q38_INVALID_ARGUMENT,"null GDN shape");
    if (!s->key_heads || !s->value_heads || !s->key_dim || !s->value_dim || s->value_heads%s->key_heads)
        return q38c_error(e,Q38_SHAPE,"invalid GDN dimensions/head ratio");
    if (s->key_heads>UINT32_MAX || s->value_heads>UINT32_MAX || s->key_dim>UINT32_MAX || s->value_dim>UINT32_MAX)
        return q38c_error(e,Q38_RANGE,"GDN dimension exceeds uint32");
    if (!isfinite(s->rms_eps) || !isfinite(s->l2_eps)) return q38c_error(e,Q38_NONFINITE,"nonfinite GDN epsilon");
    if (s->rms_eps<=0 || s->l2_eps<=0) return q38c_error(e,Q38_RANGE,"GDN epsilon must be positive");
    size_t keys, values;
    if (!q38c_mul(s->key_heads,s->key_dim,&keys) || !q38c_mul(keys,2,&keys) ||
        !q38c_mul(s->value_heads,s->value_dim,&values) || !q38c_add(keys,values,channels) ||
        !q38c_mul(values,s->key_dim,elements)) return q38c_error(e,Q38_OVERFLOW,"GDN shape overflows");
    if (*channels>UINT32_MAX) return q38c_error(e,Q38_RANGE,"QKV width exceeds uint32");
    return Q38_OK;
}
Qwen38Status qwen38_gdn_state_bytes(const Qwen38GdnShape *s, Qwen38StateSize *out, Qwen38Error *e) {
    if (out) *out=(Qwen38StateSize){0}; size_t channels, elements;
    Q38C_TRY(gdn_shape(s,&channels,&elements,e));
    return q38c_size(sizeof(Qwen38GdnState),elements,out,e);
}
Qwen38Status qwen38_gdn_state_init(const Qwen38GdnShape *shape, Qwen38StateKey key,
        Qwen38Memory metadata, Qwen38Memory payload, Qwen38GdnState **out, Qwen38Error *e) {
    if (!out) return q38c_error(e,Q38_INVALID_ARGUMENT,"null GDN state output");
    *out=NULL; Qwen38StateSize size; size_t channels, elements;
    Q38C_TRY(gdn_shape(shape,&channels,&elements,e)); Q38C_TRY(qwen38_gdn_state_bytes(shape,&size,e));
    Q38C_TRY(q38c_backings(key,metadata,payload,size,e));
    Qwen38GdnState *s=metadata.data;
    *s=(Qwen38GdnState){.base={.self=s,.magic=Q38C_GDN_MAGIC,.info={.key=key,.valid=true},
        .metadata=metadata,.payload=payload},.shape=*shape,.elements=elements};
    memset(payload.data,0,size.payload_bytes); *out=s;
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38StateInfo qwen38_gdn_state_info(const Qwen38GdnState *s) {
    if (!s || s->base.self!=s || s->base.magic!=Q38C_GDN_MAGIC) return (Qwen38StateInfo){0};
    return s->base.info;
}
Qwen38Status qwen38_gdn_state_reset(Qwen38GdnState *s, uint64_t epoch, Qwen38Error *e) {
    if (!s) return q38c_error(e,Q38_INVALID_ARGUMENT,"null GDN state");
    return q38c_state_reset(&s->base,Q38C_GDN_MAGIC,epoch,s->elements*sizeof(float),e);
}
static Qwen38F32 gdn_payload(const Qwen38GdnState *s) {
    return (Qwen38F32){.data=s->base.payload.data,.capacity_bytes=s->base.payload.bytes,.rank=3,
        .dim={s->shape.value_heads,s->shape.value_dim,s->shape.key_dim},
        .stride_bytes={s->shape.value_dim*s->shape.key_dim*4,s->shape.key_dim*4,4}};
}
Qwen38Status qwen38_gdn_state_export(const Qwen38GdnState *s, Qwen38F32Mut out, Qwen38Error *e) {
    Q38C_TRY(q38c_state_check(s ? &s->base : NULL,Q38C_GDN_MAGIC,true,e));
    Qwen38F32 payload=gdn_payload(s), y=q38c_ro(out);
    Q38C_TRY(q38c_dims(y,3,payload.dim,e)); Q38C_TRY(q38c_state_alias(&s->base,y,e));
    Q38C_TRY(q38c_finite(payload,e));
    for (size_t i=0;i<s->elements;++i) out.data[q38c_offset(y,i)]=payload.data[i];
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gdn_prepare(const Qwen38GdnShape *s, Qwen38F32 qkv,
        Qwen38F32 a, Qwen38F32 b, Qwen38F32 decay_weight, Qwen38F32 dt,
        Qwen38F32Mut prepared, Qwen38F32Mut log_decay, Qwen38F32Mut beta, Qwen38Error *e) {
    size_t channels,elements; Q38C_TRY(gdn_shape(s,&channels,&elements,e)); size_t tokens=qkv.dim[0];
    Qwen38F32 outputs[]={q38c_ro(prepared),q38c_ro(log_decay),q38c_ro(beta)};
    Qwen38F32 inputs[]={qkv,a,b,decay_weight,dt};
    Q38C_TRY(q38c_dims(qkv,2,(size_t[]){tokens,channels},e));
    Q38C_TRY(q38c_dims(a,2,(size_t[]){tokens,s->value_heads},e)); Q38C_TRY(q38c_dims(b,2,a.dim,e));
    Q38C_TRY(q38c_dims(decay_weight,1,(size_t[]){s->value_heads},e)); Q38C_TRY(q38c_dims(dt,1,decay_weight.dim,e));
    for (size_t i=0;i<3;++i) Q38C_TRY(q38c_dims(outputs[i],2,inputs[i].dim,e));
    for (size_t i=0;i<3;++i) {
        for (size_t j=0;j<5;++j) Q38C_TRY(q38c_alias(outputs[i],inputs[j],i==j,e));
        for (size_t j=0;j<i;++j) Q38C_TRY(q38c_alias(outputs[i],outputs[j],false,e));
    }
    if (!tokens) return q38c_error(e,Q38_OK,NULL);
    for (size_t i=0;i<5;++i) Q38C_TRY(q38c_finite(inputs[i],e));
    for (size_t h=0;h<s->value_heads;++h)
        if (decay_weight.data[h]>0) return q38c_error(e,Q38_RANGE,"ssm_a must already be nonpositive");
    size_t key_width=s->key_heads*s->key_dim;
    float q_scale=1.0f/sqrtf((float)s->key_dim);
    for (size_t t=0;t<tokens;++t) {
        const float *src=qkv.data+t*(qkv.stride_bytes[0]/4);
        float *dst=prepared.data+t*(prepared.stride_bytes[0]/4);
        for (size_t h=0;h<2*s->key_heads;++h) {
            size_t begin=h*s->key_dim; float sum=0;
            for (size_t j=0;j<s->key_dim;++j) sum += src[begin+j]*src[begin+j];
            float squared_norm=sum+s->l2_eps;
            if (!isfinite(squared_norm)) return q38c_error(e,Q38_NONFINITE,"L2 reduction overflow");
            float inv=1.0f/sqrtf(squared_norm), scale=h<s->key_heads ? q_scale : 1.0f;
            for (size_t j=0;j<s->key_dim;++j) dst[begin+j]=(src[begin+j]*inv)*scale;
        }
        memmove(dst+2*key_width,src+2*key_width,s->value_heads*s->value_dim*sizeof(float));
        for (size_t h=0;h<s->value_heads;++h) {
            float value=q38c_get(a,t*s->value_heads+h)+dt.data[h];
            if (!isfinite(value)) return q38c_error(e,Q38_NONFINITE,"GDN time-step overflow");
            float softplus=fmaxf(value,0.0f)+log1pf(expf(-fabsf(value)));
            float log_g=decay_weight.data[h]*softplus;
            if (!isfinite(log_g)) return q38c_error(e,Q38_NONFINITE,"GDN log-decay overflow");
            log_decay.data[q38c_offset(outputs[1],t*s->value_heads+h)]=log_g;
            beta.data[q38c_offset(outputs[2],t*s->value_heads+h)]=q38c_sigmoid(q38c_get(b,t*s->value_heads+h));
        }
    }
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gdn_norm_gate(const Qwen38GdnShape *s, Qwen38F32 o,
        Qwen38F32 z, Qwen38F32 gain, Qwen38F32Mut out, Qwen38Error *e) {
    size_t channels,elements; Q38C_TRY(gdn_shape(s,&channels,&elements,e));
    Qwen38F32 y=q38c_ro(out);
    Q38C_TRY(q38c_dims(o,3,(size_t[]){o.dim[0],s->value_heads,s->value_dim},e));
    Q38C_TRY(q38c_dims(z,3,o.dim,e)); Q38C_TRY(q38c_dims(y,3,o.dim,e));
    Q38C_TRY(q38c_dims(gain,1,(size_t[]){s->value_dim},e));
    Q38C_TRY(q38c_alias(y,z,false,e)); Q38C_TRY(q38c_finite(z,e));
    /* RMS validates all other aliases and finite inputs before writing out. */
    Q38C_TRY(qwen38_rmsnorm_groups(o,gain,out,s->rms_eps,e));
    for (size_t i=0;i<q38c_count(y);++i) out.data[q38c_offset(y,i)] *= q38c_sigmoid(q38c_get(z,i));
    return q38c_error(e,Q38_OK,NULL);
}
static bool gdn_same(Qwen38GdnShape a, Qwen38GdnShape b) {
    return a.key_heads==b.key_heads && a.value_heads==b.value_heads && a.key_dim==b.key_dim &&
        a.value_dim==b.value_dim && a.rms_eps==b.rms_eps && a.l2_eps==b.l2_eps;
}
Qwen38Status qwen38_gdn_scan(const Qwen38GdnShape *shape, Qwen38F32 prepared,
        Qwen38F32 log_decay, Qwen38F32 beta, Qwen38F32Mut out,
        Qwen38GdnState *s, uint64_t start, Qwen38Error *e) {
    size_t channels,elements; Q38C_TRY(gdn_shape(shape,&channels,&elements,e));
    Q38C_TRY(q38c_state_check(s ? &s->base : NULL,Q38C_GDN_MAGIC,true,e));
    if (!gdn_same(*shape,s->shape)) return q38c_error(e,Q38_SHAPE,"state/GDN shape mismatch");
    size_t tokens=prepared.dim[0], hk=shape->key_heads, hv=shape->value_heads, dk=shape->key_dim, dv=shape->value_dim;
    Qwen38F32 y=q38c_ro(out), inputs[]={prepared,log_decay,beta};
    Q38C_TRY(q38c_dims(prepared,2,(size_t[]){tokens,channels},e));
    Q38C_TRY(q38c_dims(log_decay,2,(size_t[]){tokens,hv},e)); Q38C_TRY(q38c_dims(beta,2,log_decay.dim,e));
    Q38C_TRY(q38c_dims(y,3,(size_t[]){tokens,hv,dv},e));
    for (size_t i=0;i<3;++i) {
        Q38C_TRY(q38c_alias(y,inputs[i],false,e)); Q38C_TRY(q38c_state_alias(&s->base,inputs[i],e));
    }
    Q38C_TRY(q38c_state_alias(&s->base,y,e));
    if (start!=s->base.info.next_pos) return q38c_error(e,Q38_BAD_STATE,"noncontiguous GDN append");
    if (tokens>UINT64_MAX-start) return q38c_error(e,Q38_OVERFLOW,"position overflows");
    if (!tokens) return q38c_error(e,Q38_OK,NULL);
    for (size_t i=0;i<3;++i) Q38C_TRY(q38c_finite(inputs[i],e));
    for (size_t i=0;i<tokens*hv;++i)
        if (q38c_get(log_decay,i)>0 || q38c_get(beta,i)<0 || q38c_get(beta,i)>1)
            return q38c_error(e,Q38_RANGE,"invalid log-decay or beta domain");
    if (q38c_finite(gdn_payload(s),e)!=Q38_OK) { s->base.info.valid=false; return Q38_NONFINITE; }
    float *state=s->base.payload.data; s->base.info.busy=true;
    for (size_t t=0;t<tokens;++t) {
        const float *qkv=prepared.data+t*(prepared.stride_bytes[0]/4);
        for (size_t h=0;h<hv;++h) {
            const float *q=qkv+(h%hk)*dk, *k=q+hk*dk, *value=qkv+2*hk*dk+h*dv;
            float decay=expf(q38c_get(log_decay,t*hv+h)), rate=q38c_get(beta,t*hv+h);
            for (size_t v=0;v<dv;++v) {
                float *row=state+(h*dv+v)*dk, prediction=0;
                /* Decay is applied BEFORE predicting the current value. */
                for (size_t j=0;j<dk;++j) { row[j] *= decay; prediction += row[j]*k[j]; }
                float delta=(value[v]-prediction)*rate, result=0;
                if (!isfinite(prediction) || !isfinite(delta)) goto nonfinite;
                for (size_t j=0;j<dk;++j) {
                    row[j] += delta*k[j];
                    if (!isfinite(row[j])) goto nonfinite;
                    result += row[j]*q[j]; /* Reads the just-updated state. */
                }
                if (!isfinite(result)) goto nonfinite;
                out.data[t*(y.stride_bytes[0]/4)+h*(y.stride_bytes[1]/4)+v]=result;
            }
        }
    }
    s->base.info.next_pos=start+tokens; s->base.info.busy=false;
    return q38c_error(e,Q38_OK,NULL);
nonfinite:
    s->base.info.valid=false; s->base.info.busy=false;
    return q38c_error(e,Q38_NONFINITE,"GDN accumulation overflow; reset/replay required");
}

static bool key_same(Qwen38StateKey a, Qwen38StateKey b) {
    return a.model==b.model && a.layer==b.layer && a.epoch==b.epoch;
}
static Qwen38Status block_states(const Q38CBlockState *s, bool valid, Qwen38Error *e) {
    if (!s || !s->owner || !s->conv || !s->gdn) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C block state");
    if (s->busy) return q38c_error(e,Q38_BUSY,"C block in use");
    if (valid && !s->valid) return q38c_error(e,Q38_BAD_STATE,"C block requires reset/replay");
    Q38C_TRY(q38c_state_check((const Q38CState *)s->conv,Q38C_CONV_MAGIC,valid,e));
    Q38C_TRY(q38c_state_check((const Q38CState *)s->gdn,Q38C_GDN_MAGIC,valid,e));
    Qwen38StateInfo c=qwen38_conv_state_info(s->conv), g=qwen38_gdn_state_info(s->gdn);
    const Q38CState *cc=(const Q38CState *)s->conv, *gg=(const Q38CState *)s->gdn;
    Qwen38Memory backings[]={cc->metadata,cc->payload,gg->metadata,gg->payload};
    for (size_t i=0;i<4;++i) {
        if (q38c_overlap(backings[i].data,backings[i].bytes,s,sizeof(*s)) ||
            q38c_overlap(backings[i].data,backings[i].bytes,s->owner,sizeof(*s->owner)))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"C state aliases composition metadata");
        for (size_t j=0;j<i;++j)
            if (q38c_overlap(backings[i].data,backings[i].bytes,backings[j].data,backings[j].bytes))
                return q38c_error(e,Q38_INVALID_ARGUMENT,"C state backings overlap");
        for (size_t j=0;j<Q38CW_COUNT;++j)
            if (q38c_overlap(backings[i].data,backings[i].bytes,s->owner->weight[j].span.data,s->owner->weight[j].span.bytes))
                return q38c_error(e,Q38_INVALID_ARGUMENT,"C state aliases packed weights");
    }
    if (!key_same(c.key,s->key) || !key_same(g.key,s->key) || s->key.model!=s->owner->model || s->key.layer!=s->owner->layer)
        return q38c_error(e,Q38_STALE,"C block key/epoch differs");
    return Q38_OK;
}
Qwen38Status q38c_block_init(const Q38CLayer *l, Qwen38ConvState *c, Qwen38GdnState *g,
        Q38CBlockState *out, Qwen38Error *e) {
    if (!out) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C block output");
    *out=(Q38CBlockState){0};
    if (!l || !l->bound) return q38c_error(e,Q38_BAD_STATE,"unbound C layer");
    Q38CBlockState next={.owner=l,.conv=c,.gdn=g,.key=qwen38_conv_state_info(c).key,.valid=true};
    Q38C_TRY(block_states(&next,true,e));
    if (qwen38_conv_state_info(c).next_pos!=qwen38_gdn_state_info(g).next_pos)
        return q38c_error(e,Q38_BAD_STATE,"C histories have different positions");
    *out=next; return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_block_reset(Q38CBlockState *s, uint64_t epoch, Qwen38Error *e) {
    Q38C_TRY(block_states(s,false,e));
    if (s->key.epoch==UINT64_MAX) return q38c_error(e,Q38_OVERFLOW,"C block epoch exhausted");
    if (epoch<=s->key.epoch) return q38c_error(e,Q38_STALE,"C block reset needs newer epoch");
    Q38C_TRY(qwen38_conv_state_reset(s->conv,epoch,e));
    Q38C_TRY(qwen38_gdn_state_reset(s->gdn,epoch,e));
    s->key.epoch=epoch; s->valid=true; return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_block_cpu(const Q38CLayer *l, Q38CWork *w, Qwen38F32Mut residual,
        Q38CBlockState *s, uint64_t start, Qwen38Error *e) {
    Qwen38F32 r=q38c_ro(residual); Q38C_TRY(q38c_work_check(l,w,r,e));
    Q38C_TRY(block_states(s,true,e));
    if (s->owner!=l) return q38c_error(e,Q38_OWNER,"foreign C block");
    size_t t=r.dim[0], hv=l->config.gdn.value_heads, dv=l->config.gdn.value_dim;
    if (t>UINT64_MAX-start) return q38c_error(e,Q38_OVERFLOW,"C block position overflow");
    for (size_t j=0;j<2;++j) {
        const Q38CState *state=j ? (const Q38CState *)s->gdn : (const Q38CState *)s->conv;
        Q38C_TRY(q38c_state_alias(state,r,e));
        for (size_t i=0;i<Q38CT_COUNT;++i) Q38C_TRY(q38c_state_alias(state,q38c_ro(q38c_trace(w,(Q38CStage)i,t)),e));
        for (size_t i=0;i<Q38CW_COUNT;++i) Q38C_TRY(q38c_state_alias(state,l->constant[i],e));
    }
    /* Public zero-token calls check both opaque shapes and exact positions,
     * without touching activations/history; no duplicate state admission. */
    Q38C_TRY(qwen38_conv_causal(&l->conv,q38c_ro(q38c_trace(w,Q38CT_QKV,0)),l->constant[Q38CW_CONV],
        q38c_trace(w,Q38CT_CONV,0),s->conv,start,e));
    Q38C_TRY(qwen38_gdn_scan(&l->config.gdn,q38c_ro(q38c_trace(w,Q38CT_PREP,0)),
        q38c_ro(q38c_trace(w,Q38CT_LOGG,0)),q38c_ro(q38c_trace(w,Q38CT_BETA,0)),
        q38c_group(q38c_trace(w,Q38CT_SCAN,0),hv,dv),s->gdn,start,e));
    if (!t) return q38c_error(e,Q38_OK,NULL);
    Q38C_TRY(q38c_finite(r,e)); s->busy=true; Qwen38Status rc;
#define STEP(call) do { rc=(call); if (rc!=Q38_OK) goto failed; } while (0)
#define V(stage) q38c_trace(w,Q38CT_##stage,t)
#define R(stage) q38c_ro(V(stage))
    STEP(q38c_hc_cpu(l,w,r,e));
    STEP(q38_packed_matmul(&l->weight[Q38CW_QKV],0,R(MIX),V(QKV),e));
    STEP(q38_packed_matmul(&l->weight[Q38CW_Z],0,R(MIX),V(Z),e));
    STEP(q38_packed_matmul(&l->weight[Q38CW_AP],0,R(MIX),V(ALPHA),e));
    STEP(q38_packed_matmul(&l->weight[Q38CW_BP],0,R(MIX),V(BR),e));
    STEP(qwen38_conv_causal(&l->conv,R(QKV),l->constant[Q38CW_CONV],V(CONV),s->conv,start,e));
    STEP(qwen38_gdn_prepare(&l->config.gdn,R(CONV),R(ALPHA),R(BR),l->constant[Q38CW_A],l->constant[Q38CW_DT],
        V(PREP),V(LOGG),V(BETA),e));
    STEP(qwen38_gdn_scan(&l->config.gdn,R(PREP),R(LOGG),R(BETA),q38c_group(V(SCAN),hv,dv),s->gdn,start,e));
    STEP(qwen38_gdn_norm_gate(&l->config.gdn,q38c_ro(q38c_group(V(SCAN),hv,dv)),q38c_ro(q38c_group(V(Z),hv,dv)),
        l->constant[Q38CW_GN],q38c_group(V(NORM),hv,dv),e));
    STEP(q38_packed_matmul(&l->weight[Q38CW_OUT],0,R(NORM),V(OUT),e));
    STEP(qwen38_hc_add(&l->config.hc,residual,R(OUT),R(INJECT),e));
#undef STEP
#undef V
#undef R
    s->busy=false; return q38c_error(e,Q38_OK,NULL);
failed:
    s->valid=false; s->busy=false; /* Includes a conv advanced before a later stage failed. */
    return rc;
}
