#include "qwen4_cpu_internal.h"

struct Qwen4GdnState { QW4CState base; Qwen4GdnShape shape; size_t elements; };
static Qwen4Status gdn_shape(const Qwen4GdnShape *s, size_t *channels, size_t *elements, Qwen4Error *e) {
    if (!s) return qw4c_error(e,QW4_INVALID_ARGUMENT,"null GDN shape");
    if (!s->key_heads || !s->value_heads || !s->key_dim || !s->value_dim || s->value_heads%s->key_heads)
        return qw4c_error(e,QW4_SHAPE,"invalid GDN dimensions/head ratio");
    if (s->key_heads>UINT32_MAX || s->value_heads>UINT32_MAX || s->key_dim>UINT32_MAX || s->value_dim>UINT32_MAX)
        return qw4c_error(e,QW4_RANGE,"GDN dimension exceeds uint32");
    if (!isfinite(s->rms_eps) || !isfinite(s->l2_eps)) return qw4c_error(e,QW4_NONFINITE,"nonfinite GDN epsilon");
    if (s->rms_eps<=0 || s->l2_eps<=0) return qw4c_error(e,QW4_RANGE,"GDN epsilon must be positive");
    size_t keys, values;
    if (!qw4c_mul(s->key_heads,s->key_dim,&keys) || !qw4c_mul(keys,2,&keys) ||
        !qw4c_mul(s->value_heads,s->value_dim,&values) || !qw4c_add(keys,values,channels) ||
        !qw4c_mul(values,s->key_dim,elements)) return qw4c_error(e,QW4_OVERFLOW,"GDN shape overflows");
    if (*channels>UINT32_MAX) return qw4c_error(e,QW4_RANGE,"QKV width exceeds uint32");
    return QW4_OK;
}
Qwen4Status qwen4_gdn_state_bytes(const Qwen4GdnShape *s, Qwen4StateSize *out, Qwen4Error *e) {
    if (out) *out=(Qwen4StateSize){0}; size_t channels, elements;
    QW4C_TRY(gdn_shape(s,&channels,&elements,e));
    return qw4c_size(sizeof(Qwen4GdnState),elements,out,e);
}
Qwen4Status qwen4_gdn_state_init(const Qwen4GdnShape *shape, Qwen4StateKey key,
        Qwen4Memory metadata, Qwen4Memory payload, Qwen4GdnState **out, Qwen4Error *e) {
    if (!out) return qw4c_error(e,QW4_INVALID_ARGUMENT,"null GDN state output");
    *out=NULL; Qwen4StateSize size; size_t channels, elements;
    QW4C_TRY(gdn_shape(shape,&channels,&elements,e)); QW4C_TRY(qwen4_gdn_state_bytes(shape,&size,e));
    QW4C_TRY(qw4c_backings(key,metadata,payload,size,e));
    Qwen4GdnState *s=metadata.data;
    *s=(Qwen4GdnState){.base={.self=s,.magic=QW4C_GDN_MAGIC,.info={.key=key,.valid=true},
        .metadata=metadata,.payload=payload},.shape=*shape,.elements=elements};
    memset(payload.data,0,size.payload_bytes); *out=s;
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4StateInfo qwen4_gdn_state_info(const Qwen4GdnState *s) {
    if (!s || s->base.self!=s || s->base.magic!=QW4C_GDN_MAGIC) return (Qwen4StateInfo){0};
    return s->base.info;
}
Qwen4Status qwen4_gdn_state_reset(Qwen4GdnState *s, uint64_t epoch, Qwen4Error *e) {
    if (!s) return qw4c_error(e,QW4_INVALID_ARGUMENT,"null GDN state");
    return qw4c_state_reset(&s->base,QW4C_GDN_MAGIC,epoch,s->elements*sizeof(float),e);
}
static Qwen4F32 gdn_payload(const Qwen4GdnState *s) {
    return (Qwen4F32){.data=s->base.payload.data,.capacity_bytes=s->base.payload.bytes,.rank=3,
        .dim={s->shape.value_heads,s->shape.value_dim,s->shape.key_dim},
        .stride_bytes={s->shape.value_dim*s->shape.key_dim*4,s->shape.key_dim*4,4}};
}
Qwen4Status qwen4_gdn_state_export(const Qwen4GdnState *s, Qwen4F32Mut out, Qwen4Error *e) {
    QW4C_TRY(qw4c_state_check(s ? &s->base : NULL,QW4C_GDN_MAGIC,true,e));
    Qwen4F32 payload=gdn_payload(s), y=qw4c_ro(out);
    QW4C_TRY(qw4c_dims(y,3,payload.dim,e)); QW4C_TRY(qw4c_state_alias(&s->base,y,e));
    QW4C_TRY(qw4c_finite(payload,e));
    for (size_t i=0;i<s->elements;++i) out.data[qw4c_offset(y,i)]=payload.data[i];
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_gdn_prepare(const Qwen4GdnShape *s, Qwen4F32 qkv,
        Qwen4F32 a, Qwen4F32 b, Qwen4F32 decay_weight, Qwen4F32 dt,
        Qwen4F32Mut prepared, Qwen4F32Mut log_decay, Qwen4F32Mut beta, Qwen4Error *e) {
    size_t channels,elements; QW4C_TRY(gdn_shape(s,&channels,&elements,e)); size_t tokens=qkv.dim[0];
    Qwen4F32 outputs[]={qw4c_ro(prepared),qw4c_ro(log_decay),qw4c_ro(beta)};
    Qwen4F32 inputs[]={qkv,a,b,decay_weight,dt};
    QW4C_TRY(qw4c_dims(qkv,2,(size_t[]){tokens,channels},e));
    QW4C_TRY(qw4c_dims(a,2,(size_t[]){tokens,s->value_heads},e)); QW4C_TRY(qw4c_dims(b,2,a.dim,e));
    QW4C_TRY(qw4c_dims(decay_weight,1,(size_t[]){s->value_heads},e)); QW4C_TRY(qw4c_dims(dt,1,decay_weight.dim,e));
    for (size_t i=0;i<3;++i) QW4C_TRY(qw4c_dims(outputs[i],2,inputs[i].dim,e));
    for (size_t i=0;i<3;++i) {
        for (size_t j=0;j<5;++j) QW4C_TRY(qw4c_alias(outputs[i],inputs[j],i==j,e));
        for (size_t j=0;j<i;++j) QW4C_TRY(qw4c_alias(outputs[i],outputs[j],false,e));
    }
    if (!tokens) return qw4c_error(e,QW4_OK,NULL);
    for (size_t i=0;i<5;++i) QW4C_TRY(qw4c_finite(inputs[i],e));
    for (size_t h=0;h<s->value_heads;++h)
        if (decay_weight.data[h]>0) return qw4c_error(e,QW4_RANGE,"ssm_a must already be nonpositive");
    size_t key_width=s->key_heads*s->key_dim;
    float q_scale=1.0f/sqrtf((float)s->key_dim);
    for (size_t t=0;t<tokens;++t) {
        const float *src=qkv.data+t*(qkv.stride_bytes[0]/4);
        float *dst=prepared.data+t*(prepared.stride_bytes[0]/4);
        for (size_t h=0;h<2*s->key_heads;++h) {
            size_t begin=h*s->key_dim; float sum=0;
            for (size_t j=0;j<s->key_dim;++j) sum += src[begin+j]*src[begin+j];
            float squared_norm=sum+s->l2_eps;
            if (!isfinite(squared_norm)) return qw4c_error(e,QW4_NONFINITE,"L2 reduction overflow");
            float inv=1.0f/sqrtf(squared_norm), scale=h<s->key_heads ? q_scale : 1.0f;
            for (size_t j=0;j<s->key_dim;++j) dst[begin+j]=(src[begin+j]*inv)*scale;
        }
        memmove(dst+2*key_width,src+2*key_width,s->value_heads*s->value_dim*sizeof(float));
        for (size_t h=0;h<s->value_heads;++h) {
            float value=qw4c_get(a,t*s->value_heads+h)+dt.data[h];
            if (!isfinite(value)) return qw4c_error(e,QW4_NONFINITE,"GDN time-step overflow");
            float softplus=fmaxf(value,0.0f)+log1pf(expf(-fabsf(value)));
            float log_g=decay_weight.data[h]*softplus;
            if (!isfinite(log_g)) return qw4c_error(e,QW4_NONFINITE,"GDN log-decay overflow");
            log_decay.data[qw4c_offset(outputs[1],t*s->value_heads+h)]=log_g;
            beta.data[qw4c_offset(outputs[2],t*s->value_heads+h)]=qw4c_sigmoid(qw4c_get(b,t*s->value_heads+h));
        }
    }
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_gdn_norm_gate(const Qwen4GdnShape *s, Qwen4F32 o,
        Qwen4F32 z, Qwen4F32 gain, Qwen4F32Mut out, Qwen4Error *e) {
    size_t channels,elements; QW4C_TRY(gdn_shape(s,&channels,&elements,e));
    Qwen4F32 y=qw4c_ro(out);
    QW4C_TRY(qw4c_dims(o,3,(size_t[]){o.dim[0],s->value_heads,s->value_dim},e));
    QW4C_TRY(qw4c_dims(z,3,o.dim,e)); QW4C_TRY(qw4c_dims(y,3,o.dim,e));
    QW4C_TRY(qw4c_dims(gain,1,(size_t[]){s->value_dim},e));
    QW4C_TRY(qw4c_alias(y,z,false,e)); QW4C_TRY(qw4c_finite(z,e));
    /* RMS validates all other aliases and finite inputs before writing out. */
    QW4C_TRY(qwen4_rmsnorm_groups(o,gain,out,s->rms_eps,e));
    for (size_t i=0;i<qw4c_count(y);++i) {
        float g=qw4c_get(z,i);
        out.data[qw4c_offset(y,i)] *= s->silu_gate ? qw4c_silu(g) : qw4c_sigmoid(g);
    }
    return qw4c_error(e,QW4_OK,NULL);
}
static bool gdn_same(Qwen4GdnShape a, Qwen4GdnShape b) {
    return a.key_heads==b.key_heads && a.value_heads==b.value_heads && a.key_dim==b.key_dim &&
        a.value_dim==b.value_dim && a.rms_eps==b.rms_eps && a.l2_eps==b.l2_eps && a.silu_gate==b.silu_gate;
}
Qwen4Status qwen4_gdn_scan(const Qwen4GdnShape *shape, Qwen4F32 prepared,
        Qwen4F32 log_decay, Qwen4F32 beta, Qwen4F32Mut out,
        Qwen4GdnState *s, uint64_t start, Qwen4Error *e) {
    size_t channels,elements; QW4C_TRY(gdn_shape(shape,&channels,&elements,e));
    QW4C_TRY(qw4c_state_check(s ? &s->base : NULL,QW4C_GDN_MAGIC,true,e));
    if (!gdn_same(*shape,s->shape)) return qw4c_error(e,QW4_SHAPE,"state/GDN shape mismatch");
    size_t tokens=prepared.dim[0], hk=shape->key_heads, hv=shape->value_heads, dk=shape->key_dim, dv=shape->value_dim;
    Qwen4F32 y=qw4c_ro(out), inputs[]={prepared,log_decay,beta};
    QW4C_TRY(qw4c_dims(prepared,2,(size_t[]){tokens,channels},e));
    QW4C_TRY(qw4c_dims(log_decay,2,(size_t[]){tokens,hv},e)); QW4C_TRY(qw4c_dims(beta,2,log_decay.dim,e));
    QW4C_TRY(qw4c_dims(y,3,(size_t[]){tokens,hv,dv},e));
    for (size_t i=0;i<3;++i) {
        QW4C_TRY(qw4c_alias(y,inputs[i],false,e)); QW4C_TRY(qw4c_state_alias(&s->base,inputs[i],e));
    }
    QW4C_TRY(qw4c_state_alias(&s->base,y,e));
    if (start!=s->base.info.next_pos) return qw4c_error(e,QW4_BAD_STATE,"noncontiguous GDN append");
    if (tokens>UINT64_MAX-start) return qw4c_error(e,QW4_OVERFLOW,"position overflows");
    if (!tokens) return qw4c_error(e,QW4_OK,NULL);
    for (size_t i=0;i<3;++i) QW4C_TRY(qw4c_finite(inputs[i],e));
    for (size_t i=0;i<tokens*hv;++i)
        if (qw4c_get(log_decay,i)>0 || qw4c_get(beta,i)<0 || qw4c_get(beta,i)>1)
            return qw4c_error(e,QW4_RANGE,"invalid log-decay or beta domain");
    if (qw4c_finite(gdn_payload(s),e)!=QW4_OK) { s->base.info.valid=false; return QW4_NONFINITE; }
    float *state=s->base.payload.data; s->base.info.busy=true;
    for (size_t t=0;t<tokens;++t) {
        const float *qkv=prepared.data+t*(prepared.stride_bytes[0]/4);
        for (size_t h=0;h<hv;++h) {
            const float *q=qkv+(h%hk)*dk, *k=q+hk*dk, *value=qkv+2*hk*dk+h*dv;
            float decay=expf(qw4c_get(log_decay,t*hv+h)), rate=qw4c_get(beta,t*hv+h);
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
    return qw4c_error(e,QW4_OK,NULL);
nonfinite:
    s->base.info.valid=false; s->base.info.busy=false;
    return qw4c_error(e,QW4_NONFINITE,"GDN accumulation overflow; reset/replay required");
}

static bool key_same(Qwen4StateKey a, Qwen4StateKey b) {
    return a.model==b.model && a.layer==b.layer && a.epoch==b.epoch;
}
static Qwen4Status block_states(const QW4CBlockState *s, bool valid, Qwen4Error *e) {
    if (!s || !s->owner || !s->conv || !s->gdn) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C block state");
    if (s->busy) return qw4c_error(e,QW4_BUSY,"C block in use");
    if (valid && !s->valid) return qw4c_error(e,QW4_BAD_STATE,"C block requires reset/replay");
    QW4C_TRY(qw4c_state_check((const QW4CState *)s->conv,QW4C_CONV_MAGIC,valid,e));
    QW4C_TRY(qw4c_state_check((const QW4CState *)s->gdn,QW4C_GDN_MAGIC,valid,e));
    Qwen4StateInfo c=qwen4_conv_state_info(s->conv), g=qwen4_gdn_state_info(s->gdn);
    const QW4CState *cc=(const QW4CState *)s->conv, *gg=(const QW4CState *)s->gdn;
    Qwen4Memory backings[]={cc->metadata,cc->payload,gg->metadata,gg->payload};
    for (size_t i=0;i<4;++i) {
        if (qw4c_overlap(backings[i].data,backings[i].bytes,s,sizeof(*s)) ||
            qw4c_overlap(backings[i].data,backings[i].bytes,s->owner,sizeof(*s->owner)))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"C state aliases composition metadata");
        if (qw4c_overlap(backings[i].data,backings[i].bytes,s->owner->constant_memory.data,s->owner->constant_memory.bytes))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"C state aliases decoded constants");
        for (size_t j=0;j<i;++j)
            if (qw4c_overlap(backings[i].data,backings[i].bytes,backings[j].data,backings[j].bytes))
                return qw4c_error(e,QW4_INVALID_ARGUMENT,"C state backings overlap");
        for (size_t j=0;j<QW4CW_COUNT;++j)
            if (qw4c_overlap(backings[i].data,backings[i].bytes,s->owner->weight[j].span.data,s->owner->weight[j].span.bytes))
                return qw4c_error(e,QW4_INVALID_ARGUMENT,"C state aliases packed weights");
    }
    if (!key_same(c.key,s->key) || !key_same(g.key,s->key) || s->key.model!=s->owner->model || s->key.layer!=s->owner->layer)
        return qw4c_error(e,QW4_STALE,"C block key/epoch differs");
    return QW4_OK;
}
Qwen4Status qw4c_block_init(const QW4CLayer *l, Qwen4ConvState *c, Qwen4GdnState *g,
        QW4CBlockState *out, Qwen4Error *e) {
    if (!out) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C block output");
    *out=(QW4CBlockState){0};
    if (!l || !l->bound) return qw4c_error(e,QW4_BAD_STATE,"unbound C layer");
    QW4CBlockState next={.owner=l,.conv=c,.gdn=g,.key=qwen4_conv_state_info(c).key,.valid=true};
    QW4C_TRY(block_states(&next,true,e));
    if (qwen4_conv_state_info(c).next_pos!=qwen4_gdn_state_info(g).next_pos)
        return qw4c_error(e,QW4_BAD_STATE,"C histories have different positions");
    *out=next; return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_block_reset(QW4CBlockState *s, uint64_t epoch, Qwen4Error *e) {
    QW4C_TRY(block_states(s,false,e));
    if (s->key.epoch==UINT64_MAX) return qw4c_error(e,QW4_OVERFLOW,"C block epoch exhausted");
    if (epoch<=s->key.epoch) return qw4c_error(e,QW4_STALE,"C block reset needs newer epoch");
    QW4C_TRY(qwen4_conv_state_reset(s->conv,epoch,e));
    QW4C_TRY(qwen4_gdn_state_reset(s->gdn,epoch,e));
    s->key.epoch=epoch; s->valid=true; return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_block_cpu(const QW4CLayer *l, QW4CWork *w, Qwen4F32Mut residual,
        QW4CBlockState *s, uint64_t start, Qwen4Error *e) {
    Qwen4F32 r=qw4c_ro(residual); QW4C_TRY(qw4c_work_check(l,w,r,e));
    QW4C_TRY(block_states(s,true,e));
    if (s->owner!=l) return qw4c_error(e,QW4_OWNER,"foreign C block");
    size_t t=r.dim[0], hv=l->config.gdn.value_heads, dv=l->config.gdn.value_dim;
    if (t>UINT64_MAX-start) return qw4c_error(e,QW4_OVERFLOW,"C block position overflow");
    for (size_t j=0;j<2;++j) {
        const QW4CState *state=j ? (const QW4CState *)s->gdn : (const QW4CState *)s->conv;
        QW4C_TRY(qw4c_state_alias(state,r,e));
        for (size_t i=0;i<QW4CT_COUNT;++i) QW4C_TRY(qw4c_state_alias(state,qw4c_ro(qw4c_trace(w,(QW4CStage)i,t)),e));
        for (size_t i=0;i<QW4CW_COUNT;++i) QW4C_TRY(qw4c_state_alias(state,l->constant[i],e));
    }
    /* Public zero-token calls check both opaque shapes and exact positions,
     * without touching activations/history; no duplicate state admission. */
    QW4C_TRY(qwen4_conv_causal(&l->conv,qw4c_ro(qw4c_trace(w,QW4CT_QKV,0)),l->constant[QW4CW_CONV],
        qw4c_trace(w,QW4CT_CONV,0),s->conv,start,e));
    QW4C_TRY(qwen4_gdn_scan(&l->config.gdn,qw4c_ro(qw4c_trace(w,QW4CT_PREP,0)),
        qw4c_ro(qw4c_trace(w,QW4CT_LOGG,0)),qw4c_ro(qw4c_trace(w,QW4CT_BETA,0)),
        qw4c_group(qw4c_trace(w,QW4CT_SCAN,0),hv,dv),s->gdn,start,e));
    if (!t) return qw4c_error(e,QW4_OK,NULL);
    QW4C_TRY(qw4c_finite(r,e)); s->busy=true; Qwen4Status rc;
#define STEP(call) do { rc=(call); if (rc!=QW4_OK) goto failed; } while (0)
#define V(stage) qw4c_trace(w,QW4CT_##stage,t)
#define R(stage) qw4c_ro(V(stage))
    STEP(qw4c_hc_cpu(l,w,r,e));
    STEP(qw4_packed_matmul(&l->weight[QW4CW_QKV],0,R(MIX),V(QKV),e));
    STEP(qw4_packed_matmul(&l->weight[QW4CW_Z],0,R(MIX),V(Z),e));
    STEP(qw4_packed_matmul(&l->weight[QW4CW_AP],0,R(MIX),V(ALPHA),e));
    STEP(qw4_packed_matmul(&l->weight[QW4CW_BP],0,R(MIX),V(BR),e));
    STEP(qwen4_conv_causal(&l->conv,R(QKV),l->constant[QW4CW_CONV],V(CONV),s->conv,start,e));
    STEP(qwen4_gdn_prepare(&l->config.gdn,R(CONV),R(ALPHA),R(BR),l->constant[QW4CW_A],l->constant[QW4CW_DT],
        V(PREP),V(LOGG),V(BETA),e));
    STEP(qwen4_gdn_scan(&l->config.gdn,R(PREP),R(LOGG),R(BETA),qw4c_group(V(SCAN),hv,dv),s->gdn,start,e));
    STEP(qwen4_gdn_norm_gate(&l->config.gdn,qw4c_ro(qw4c_group(V(SCAN),hv,dv)),qw4c_ro(qw4c_group(V(Z),hv,dv)),
        l->constant[QW4CW_GN],qw4c_group(V(NORM),hv,dv),e));
    STEP(qw4_packed_matmul(&l->weight[QW4CW_OUT],0,R(NORM),V(OUT),e));
    STEP(qwen4_hc_add(&l->config.hc,residual,R(OUT),R(INJECT),e));
#undef STEP
#undef V
#undef R
    s->busy=false; return qw4c_error(e,QW4_OK,NULL);
failed:
    s->valid=false; s->busy=false; /* Includes a conv advanced before a later stage failed. */
    return rc;
}
