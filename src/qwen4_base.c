#include "qwen4_cpu_internal.h"

const Qwen4Role qw4c_layer_roles[QW4CW_COUNT]={QW4_HC_ATTN_NORM,QW4_HC_ATTN_DOWN,
    QW4_HC_ATTN_UP,QW4_HC_ATTN_INJECT,QW4_GDN_QKV,QW4_GDN_GATE,QW4_GDN_ALPHA,QW4_GDN_BETA,
    QW4_GDN_A,QW4_GDN_DT,QW4_GDN_CONV,QW4_GDN_NORM,QW4_GDN_OUT};
static const size_t constant_slots[]={QW4CW_HN,QW4CW_A,QW4CW_DT,QW4CW_CONV,QW4CW_GN};
Qwen4Status qw4c_layer_plan(const Qwen4Config *config, const Qwen4Weights *weights,
        uint32_t layer, QW4CLayer *out, Qwen4Error *e) {
    if (!out) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C layer output");
    *out=(QW4CLayer){0};
    const Qwen4ConfigView *view=qwen4_config_view(config); const Qwen4Layer *kind;
    if (!view || !weights) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing A config/catalog");
    QW4C_TRY(qwen4_config_layer(config,layer,&kind,e));
    if (kind->domain!=QW4_TRUNK || kind->kind!=QW4_LAYER_GDN)
        return qw4c_error(e,QW4_UNSUPPORTED,"C4 needs a trunk GDN layer");
    QW4CLayer plan={.config=*view,.model=config,.layer=layer};
    for (size_t i=0;i<QW4CW_COUNT;++i) {
        const Qwen4Tensor *t;
        QW4C_TRY(qwen4_weights_get(weights,layer,qw4c_layer_roles[i],&t,e));
        plan.tensor[i]=*t;
        QW4C_TRY(qw4_packed_layout_init(&t->storage,(size_t)t->logical_shape[0],&plan.weight[i].layout,e));
    }
    plan.conv=(Qwen4ConvShape){plan.weight[QW4CW_QKV].layout.n_out,view->gdn_conv_kernel,1,true};
    Qwen4StateSize size;
    QW4C_TRY(qwen4_conv_state_bytes(&plan.conv,&size,e));
    QW4C_TRY(qwen4_gdn_state_bytes(&view->gdn,&size,e));
    for (size_t j=0;j<sizeof(constant_slots)/sizeof(*constant_slots);++j) {
        QW4PackedLayout p=plan.weight[constant_slots[j]].layout; size_t bytes;
        if (!qw4c_mul(p.k_storage,p.n_out,&bytes) || !qw4c_mul(bytes,4,&bytes) ||
            !qw4c_add(plan.constant_bytes,bytes,&plan.constant_bytes))
            return qw4c_error(e,QW4_OVERFLOW,"C constants overflow");
    }
    *out=plan; return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_layer_bind(QW4CLayer *layer, const Qwen4CpuSpan spans[QW4CW_COUNT],
        Qwen4Memory memory, Qwen4Error *e) {
    if (!layer || !layer->model || !spans) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C bindings");
    if (layer->bound) return qw4c_error(e,QW4_BUSY,"C bindings are immutable after setup");
    QW4C_TRY(qw4c_backing(memory,layer->constant_bytes,e));
    if (qw4c_overlap(memory.data,memory.bytes,layer,sizeof(*layer)))
        return qw4c_error(e,QW4_INVALID_ARGUMENT,"constants alias layer metadata");
    QW4CLayer next=*layer;
    for (size_t i=0;i<QW4CW_COUNT;++i) {
        if (!spans[i].data || spans[i].bytes<next.weight[i].layout.tensor_bytes)
            return qw4c_error(e,QW4_CAPACITY,"packed C weight span too short");
        if (spans[i].bytes>UINTPTR_MAX-(uintptr_t)spans[i].data)
            return qw4c_error(e,QW4_OVERFLOW,"packed C span overflows");
        if (qw4c_overlap(memory.data,memory.bytes,spans[i].data,spans[i].bytes))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"constants alias packed weights");
        next.weight[i].span=spans[i];
    }
    size_t offset=0;
    for (size_t j=0;j<sizeof(constant_slots)/sizeof(*constant_slots);++j) {
        size_t i=constant_slots[j]; QW4PackedLayout p=next.weight[i].layout;
        float *dst=(float *)memory.data+offset;
        for (size_t row=0;row<p.n_out;++row)
            QW4C_TRY(qw4_packed_decode_row(&next.weight[i],0,row,dst+row*p.k_storage,p.k_storage,e));
        size_t bytes=p.n_out*p.k_storage*4;
        next.constant[i]=(Qwen4F32){.data=dst,.capacity_bytes=bytes,.rank=1,
            .dim={p.k_storage},.stride_bytes={4}};
        if (i==QW4CW_CONV) next.constant[i]=(Qwen4F32){.data=dst,.capacity_bytes=bytes,.rank=2,
            .dim={p.n_out,p.k_storage},.stride_bytes={p.k_storage*4,4}};
        QW4C_TRY(qw4c_finite(next.constant[i],e)); offset+=p.n_out*p.k_storage;
    }
    for (size_t h=0;h<next.config.gdn.value_heads;++h)
        if (qw4c_get(next.constant[QW4CW_A],h)>0) return qw4c_error(e,QW4_RANGE,"positive transformed A");
    next.constant_memory=memory; next.bound=true; *layer=next;
    return qw4c_error(e,QW4_OK,NULL);
}

static Qwen4Status positive_eps(float eps, Qwen4Error *e) {
    if (!isfinite(eps)) return qw4c_error(e,QW4_NONFINITE,"nonfinite normalization epsilon");
    if (eps<=0) return qw4c_error(e,QW4_RANGE,"normalization epsilon must be positive");
    return QW4_OK;
}
static Qwen4Status activate_scaled(Qwen4Activation kind, Qwen4F32 x, Qwen4F32Mut out,
        float divisor, float multiplier, bool inplace, Qwen4Error *e) {
    Qwen4F32 y=qw4c_ro(out);
    QW4C_TRY(qw4c_view(x,e)); QW4C_TRY(qw4c_dims(y,x.rank,x.dim,e));
    QW4C_TRY(qw4c_alias(y,x,inplace,e));
    QW4C_TRY(qw4c_finite(x,e));
    for (size_t i=0;i<qw4c_count(x);++i) {
        float v=qw4c_get(x,i)/divisor, result;
        if (kind==QW4_SIGMOID) result=qw4c_sigmoid(v);
        else if (kind==QW4_SILU) result=qw4c_silu(v);
        else result=fmaxf(v,0.0f)+log1pf(expf(-fabsf(v)));
        result *= multiplier;
        if (!isfinite(result)) return qw4c_error(e,QW4_NONFINITE,"activation overflow");
        out.data[qw4c_offset(y,i)]=result;
    }
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_activate(Qwen4Activation kind, Qwen4F32 x, Qwen4F32Mut out, Qwen4Error *e) {
    if (kind!=QW4_SIGMOID && kind!=QW4_SILU && kind!=QW4_SOFTPLUS)
        return qw4c_error(e,QW4_UNSUPPORTED,"unknown activation");
    /* Generic activation has disjoint views (x and out never overlap); only HC's explicit
     * in-place APIs below may alias, and they say so. */
    return activate_scaled(kind,x,out,1.0f,1.0f,false,e);
}
Qwen4Status qwen4_rmsnorm_groups(Qwen4F32 x, Qwen4F32 gain, Qwen4F32Mut out, float eps, Qwen4Error *e) {
    QW4C_TRY(positive_eps(eps,e));
    QW4C_TRY(qw4c_dims(x,3,x.dim,e)); Qwen4F32 y=qw4c_ro(out);
    QW4C_TRY(qw4c_dims(y,3,x.dim,e));
    size_t tokens=x.dim[0], groups=x.dim[1], width=x.dim[2];
    if (gain.rank==1) { QW4C_TRY(qw4c_dims(gain,1,(size_t[]){width},e)); }
    else { QW4C_TRY(qw4c_dims(gain,2,(size_t[]){groups,width},e)); }
    QW4C_TRY(qw4c_alias(y,x,true,e)); QW4C_TRY(qw4c_alias(y,gain,false,e));
    if (!tokens) return qw4c_error(e,QW4_OK,NULL);
    QW4C_TRY(qw4c_finite(x,e)); QW4C_TRY(qw4c_finite(gain,e));
    for (size_t t=0;t<tokens;++t) for (size_t g=0;g<groups;++g) {
        size_t xi=t*(x.stride_bytes[0]/4)+g*(x.stride_bytes[1]/4);
        size_t yi=t*(y.stride_bytes[0]/4)+g*(y.stride_bytes[1]/4);
        size_t wi=gain.rank==1 ? 0 : g*(gain.stride_bytes[0]/4);
        float sum=0.0f;
        for (size_t j=0;j<width;++j) sum += x.data[xi+j]*x.data[xi+j];
        float variance=sum/(float)width+eps;
        if (!isfinite(sum) || !isfinite(variance)) return qw4c_error(e,QW4_NONFINITE,"RMS reduction overflow");
        float scale=1.0f/sqrtf(variance);
        for (size_t j=0;j<width;++j) {
            float result=(x.data[xi+j]*scale)*gain.data[wi+j];
            if (!isfinite(result)) return qw4c_error(e,QW4_NONFINITE,"RMS gain overflow");
            out.data[yi+j]=result;
        }
    }
    return qw4c_error(e,QW4_OK,NULL);
}
static Qwen4Status hc_shape(const Qwen4HcShape *s, Qwen4Error *e) {
    if (!s) return qw4c_error(e,QW4_INVALID_ARGUMENT,"null HC shape");
    if (!s->streams || !s->hidden_size || !s->low_rank) return qw4c_error(e,QW4_SHAPE,"zero HC dimension");
    if (s->streams>UINT32_MAX || s->hidden_size>UINT32_MAX || s->low_rank>UINT32_MAX)
        return qw4c_error(e,QW4_RANGE,"HC dimension exceeds uint32");
    return positive_eps(s->rms_eps,e);
}
Qwen4Status qwen4_hc_lowrank_activate(const Qwen4HcShape *s, Qwen4F32 lo, Qwen4F32Mut out, Qwen4Error *e) {
    QW4C_TRY(hc_shape(s,e)); QW4C_TRY(qw4c_dims(lo,2,(size_t[]){lo.dim[0],s->low_rank},e));
    return activate_scaled(QW4_SILU,lo,out,(float)s->streams,1.0f,true,e);
}
Qwen4Status qwen4_hc_inject_weights(const Qwen4HcShape *s, Qwen4F32 logits, Qwen4F32Mut out, Qwen4Error *e) {
    QW4C_TRY(hc_shape(s,e)); QW4C_TRY(qw4c_dims(logits,2,(size_t[]){logits.dim[0],s->streams},e));
    return activate_scaled(QW4_SIGMOID,logits,out,(float)s->streams,2.0f,true,e);
}
Qwen4Status qwen4_hc_read(const Qwen4HcShape *s, Qwen4F32 normed,
        Qwen4F32 logits, Qwen4F32Mut out, Qwen4Error *e) {
    QW4C_TRY(hc_shape(s,e)); size_t tokens=normed.dim[0]; Qwen4F32 y=qw4c_ro(out);
    QW4C_TRY(qw4c_dims(normed,3,(size_t[]){tokens,s->streams,s->hidden_size},e));
    QW4C_TRY(qw4c_dims(logits,3,normed.dim,e)); QW4C_TRY(qw4c_dims(y,2,(size_t[]){tokens,s->hidden_size},e));
    QW4C_TRY(qw4c_alias(y,normed,false,e)); QW4C_TRY(qw4c_alias(y,logits,false,e));
    QW4C_TRY(qw4c_finite(normed,e)); QW4C_TRY(qw4c_finite(logits,e));
    for (size_t t=0;t<tokens;++t) for (size_t j=0;j<s->hidden_size;++j) {
        float sum=0;
        for (size_t h=0;h<s->streams;++h) {
            size_t i=(t*s->streams+h)*s->hidden_size+j;
            sum += qw4c_get(normed,i)*qw4c_sigmoid(qw4c_get(logits,i));
        }
        if (!isfinite(sum)) return qw4c_error(e,QW4_NONFINITE,"HC read overflow");
        out.data[t*(y.stride_bytes[0]/4)+j]=sum/(float)s->streams;
    }
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_hc_add(const Qwen4HcShape *s, Qwen4F32Mut residual,
        Qwen4F32 block, Qwen4F32 inject, Qwen4Error *e) {
    QW4C_TRY(hc_shape(s,e)); Qwen4F32 r=qw4c_ro(residual); size_t tokens=r.dim[0];
    QW4C_TRY(qw4c_dims(r,3,(size_t[]){tokens,s->streams,s->hidden_size},e));
    QW4C_TRY(qw4c_dims(block,2,(size_t[]){tokens,s->hidden_size},e));
    QW4C_TRY(qw4c_dims(inject,2,(size_t[]){tokens,s->streams},e));
    QW4C_TRY(qw4c_alias(r,block,false,e)); QW4C_TRY(qw4c_alias(r,inject,false,e));
    QW4C_TRY(qw4c_finite(r,e)); QW4C_TRY(qw4c_finite(block,e)); QW4C_TRY(qw4c_finite(inject,e));
    for (size_t t=0;t<tokens;++t) for (size_t h=0;h<s->streams;++h) for (size_t j=0;j<s->hidden_size;++j) {
        size_t ri=t*(r.stride_bytes[0]/4)+h*(r.stride_bytes[1]/4)+j;
        float result=residual.data[ri]+qw4c_get(inject,t*s->streams+h)*qw4c_get(block,t*s->hidden_size+j);
        if (!isfinite(result)) return qw4c_error(e,QW4_NONFINITE,"HC residual overflow");
        residual.data[ri]=result;
    }
    return qw4c_error(e,QW4_OK,NULL);
}

const char *const qw4c_stage_names[QW4CT_COUNT]={"normed","low","activated","gate_logits",
    "injection_logits","injection","mixed","qkv_raw","convolved","prepared","z","alpha",
    "beta_raw","log_decay","beta","scan","gated_norm","contribution"};
static size_t stage_width(const QW4CLayer *l, size_t stage) {
    size_t e=l->config.hidden_size, h=l->config.hc.streams, r=l->config.hc.low_rank;
    size_t hv=l->config.gdn.value_heads, inner=hv*l->config.gdn.value_dim;
    const size_t widths[]={h*e,r,r,h*e,h,h,e,l->conv.channels,l->conv.channels,l->conv.channels,
        inner,hv,hv,hv,hv,inner,inner,e};
    return widths[stage];
}
Qwen4Status qw4c_work_size(const QW4CLayer *l, size_t tokens, size_t *out, Qwen4Error *e) {
    if (!l || !l->model || !out) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C workspace plan");
    *out=0;
    if (tokens>UINT32_MAX) return qw4c_error(e,QW4_RANGE,"C token capacity exceeds uint32");
    size_t total=0;
    for (size_t i=0;i<QW4CT_COUNT;++i) { size_t bytes;
        if (stage_width(l,i)>UINT32_MAX) return qw4c_error(e,QW4_RANGE,"C stage width exceeds uint32");
        if (!qw4c_mul(tokens,stage_width(l,i),&bytes) || !qw4c_mul(bytes,4,&bytes) ||
            !qw4c_add(bytes,63,&bytes) || !qw4c_add(total,bytes&~(size_t)63,&total))
            return qw4c_error(e,QW4_OVERFLOW,"C workspace size overflows");
    }
    Qwen4StateSize cs,gs; QW4C_TRY(qwen4_conv_state_bytes(&l->conv,&cs,e));
    QW4C_TRY(qwen4_gdn_state_bytes(&l->config.gdn,&gs,e));
    size_t extra=l->constant_bytes, sizes[]={cs.metadata_bytes,cs.payload_bytes,gs.metadata_bytes,gs.payload_bytes};
    for (size_t i=0;i<4;++i) if (!qw4c_add(extra,sizes[i],&extra))
        return qw4c_error(e,QW4_OVERFLOW,"C state budget overflows");
    if (extra>1024u*1024u*1024u || total>1024u*1024u*1024u-extra)
        return qw4c_error(e,QW4_CAPACITY,"C workspace plus layer state exceeds 1 GiB");
    *out=total; return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_work_init(const QW4CLayer *l, size_t tokens, Qwen4Memory memory,
        QW4CWork *out, Qwen4Error *e) {
    if (!out) return qw4c_error(e,QW4_INVALID_ARGUMENT,"missing C workspace output");
    *out=(QW4CWork){0}; size_t bytes; QW4C_TRY(qw4c_work_size(l,tokens,&bytes,e));
    QW4C_TRY(qw4c_backing(memory,bytes,e));
    QW4CWork w={.owner=l,.capacity_tokens=tokens,.memory=memory}; size_t offset=0;
    for (size_t i=0;i<QW4CT_COUNT;++i) {
        size_t width=stage_width(l,i), used=tokens*width*4;
        void *p=offset ? (unsigned char *)memory.data+offset : memory.data;
        w.trace[i]=(Qwen4F32Mut){.data=p,.capacity_bytes=used,.rank=2,
            .dim={tokens,width},.stride_bytes={width*4,4}};
        offset+=(used+63)&~(size_t)63;
    }
    *out=w; return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_work_check(const QW4CLayer *l, const QW4CWork *w, Qwen4F32 r, Qwen4Error *e) {
    if (!l || !l->bound || !w || w->owner!=l) return qw4c_error(e,QW4_OWNER,"unbound or foreign C workspace");
    size_t t=r.dim[0], h=l->config.hc.streams, width=l->config.hidden_size;
    QW4C_TRY(qw4c_dims(r,3,(size_t[]){t,h,width},e));
    if (r.stride_bytes[1]!=width*4) return qw4c_error(e,QW4_STRIDE,"C layer requires contiguous residual streams");
    if (t>w->capacity_tokens) return qw4c_error(e,QW4_CAPACITY,"C workspace token capacity exceeded");
    if (qw4c_overlap(r.data,qw4c_span(r),l,sizeof(*l)) || qw4c_overlap(r.data,qw4c_span(r),w,sizeof(*w)))
        return qw4c_error(e,QW4_INVALID_ARGUMENT,"residual aliases C metadata");
    /* Validate ALL destinations before any stage can publish a partial result. */
    for (size_t i=0;i<QW4CT_COUNT;++i) {
        Qwen4F32 v=qw4c_ro(qw4c_trace(w,(QW4CStage)i,t));
        QW4C_TRY(qw4c_dims(v,2,(size_t[]){t,stage_width(l,i)},e)); QW4C_TRY(qw4c_alias(v,r,false,e));
        if (qw4c_overlap(v.data,qw4c_span(v),l,sizeof(*l)) || qw4c_overlap(v.data,qw4c_span(v),w,sizeof(*w)))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"C workspace aliases metadata");
        if (qw4c_overlap(v.data,qw4c_span(v),l->constant_memory.data,l->constant_memory.bytes))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"C workspace aliases constants");
        for (size_t j=0;j<QW4CW_COUNT;++j)
            if (qw4c_overlap(v.data,qw4c_span(v),l->weight[j].span.data,l->weight[j].span.bytes))
                return qw4c_error(e,QW4_INVALID_ARGUMENT,"C workspace aliases weights");
        for (size_t j=0;j<i;++j) QW4C_TRY(qw4c_alias(v,qw4c_ro(qw4c_trace(w,(QW4CStage)j,t)),false,e));
    }
    if (qw4c_overlap(r.data,qw4c_span(r),l->constant_memory.data,l->constant_memory.bytes))
        return qw4c_error(e,QW4_INVALID_ARGUMENT,"residual aliases C constants");
    for (size_t j=0;j<QW4CW_COUNT;++j)
        if (qw4c_overlap(r.data,qw4c_span(r),l->weight[j].span.data,l->weight[j].span.bytes))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"residual aliases C weights");
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qw4c_hc_cpu(const QW4CLayer *l, QW4CWork *w, Qwen4F32 residual, Qwen4Error *e) {
    QW4C_TRY(qw4c_work_check(l,w,residual,e)); size_t t=residual.dim[0];
    if (!t) return qw4c_error(e,QW4_OK,NULL);
    Qwen4HcShape h=l->config.hc; Qwen4F32 gain=l->constant[QW4CW_HN];
    gain.rank=2; gain.dim[0]=h.streams; gain.dim[1]=h.hidden_size;
    gain.stride_bytes[0]=h.hidden_size*4; gain.stride_bytes[1]=4;
#define V(stage) qw4c_trace(w,QW4CT_##stage,t)
#define R(stage) qw4c_ro(V(stage))
    QW4C_TRY(qwen4_rmsnorm_groups(residual,gain,qw4c_group(V(N),h.streams,h.hidden_size),h.rms_eps,e));
    QW4C_TRY(qw4_packed_matmul(&l->weight[QW4CW_HD],0,R(N),V(LOW),e));
    QW4C_TRY(qwen4_hc_lowrank_activate(&h,R(LOW),V(ACT),e));
    QW4C_TRY(qw4_packed_matmul(&l->weight[QW4CW_HU],0,R(ACT),V(HG),e));
    QW4C_TRY(qwen4_hc_read(&h,qw4c_ro(qw4c_group(V(N),h.streams,h.hidden_size)),
        qw4c_ro(qw4c_group(V(HG),h.streams,h.hidden_size)),V(MIX),e));
    QW4C_TRY(qw4_packed_matmul(&l->weight[QW4CW_HI],0,R(N),V(IL),e));
    QW4C_TRY(qwen4_hc_inject_weights(&h,R(IL),V(INJECT),e));
#undef V
#undef R
    return qw4c_error(e,QW4_OK,NULL);
}
