#include "qwen38_c_internal.h"

const Qwen38Role q38c_layer_roles[Q38CW_COUNT]={Q38_HC_ATTN_NORM,Q38_HC_ATTN_DOWN,
    Q38_HC_ATTN_UP,Q38_HC_ATTN_INJECT,Q38_GDN_QKV,Q38_GDN_GATE,Q38_GDN_ALPHA,Q38_GDN_BETA,
    Q38_GDN_A,Q38_GDN_DT,Q38_GDN_CONV,Q38_GDN_NORM,Q38_GDN_OUT};
static const size_t constant_slots[]={Q38CW_HN,Q38CW_A,Q38CW_DT,Q38CW_CONV,Q38CW_GN};
Qwen38Status q38c_layer_plan(const Qwen38Config *config, const Qwen38Weights *weights,
        uint32_t layer, Q38CLayer *out, Qwen38Error *e) {
    if (!out) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C layer output");
    *out=(Q38CLayer){0};
    const Qwen38ConfigView *view=qwen38_config_view(config); const Qwen38Layer *kind;
    if (!view || !weights) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing A config/catalog");
    Q38C_TRY(qwen38_config_layer(config,layer,&kind,e));
    if (kind->domain!=Q38_TRUNK || kind->kind!=Q38_LAYER_GDN)
        return q38c_error(e,Q38_UNSUPPORTED,"C4 needs a trunk GDN layer");
    Q38CLayer plan={.config=*view,.model=config,.layer=layer};
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        const Qwen38Tensor *t;
        Q38C_TRY(qwen38_weights_get(weights,layer,q38c_layer_roles[i],&t,e));
        plan.tensor[i]=*t;
        Q38C_TRY(q38_packed_layout_init(&t->storage,(size_t)t->logical_shape[0],&plan.weight[i].layout,e));
    }
    plan.conv=(Qwen38ConvShape){plan.weight[Q38CW_QKV].layout.n_out,view->gdn_conv_kernel,1,true};
    Qwen38StateSize size;
    Q38C_TRY(qwen38_conv_state_bytes(&plan.conv,&size,e));
    Q38C_TRY(qwen38_gdn_state_bytes(&view->gdn,&size,e));
    for (size_t j=0;j<sizeof(constant_slots)/sizeof(*constant_slots);++j) {
        Q38PackedLayout p=plan.weight[constant_slots[j]].layout; size_t bytes;
        if (!q38c_mul(p.k_storage,p.n_out,&bytes) || !q38c_mul(bytes,4,&bytes) ||
            !q38c_add(plan.constant_bytes,bytes,&plan.constant_bytes))
            return q38c_error(e,Q38_OVERFLOW,"C constants overflow");
    }
    *out=plan; return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_layer_bind(Q38CLayer *layer, const Qwen38CpuSpan spans[Q38CW_COUNT],
        Qwen38Memory memory, Qwen38Error *e) {
    if (!layer || !layer->model || !spans) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C bindings");
    if (layer->bound) return q38c_error(e,Q38_BUSY,"C bindings are immutable after setup");
    Q38C_TRY(q38c_backing(memory,layer->constant_bytes,e));
    if (q38c_overlap(memory.data,memory.bytes,layer,sizeof(*layer)))
        return q38c_error(e,Q38_INVALID_ARGUMENT,"constants alias layer metadata");
    Q38CLayer next=*layer;
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        if (!spans[i].data || spans[i].bytes<next.weight[i].layout.tensor_bytes)
            return q38c_error(e,Q38_CAPACITY,"packed C weight span too short");
        if (spans[i].bytes>UINTPTR_MAX-(uintptr_t)spans[i].data)
            return q38c_error(e,Q38_OVERFLOW,"packed C span overflows");
        if (q38c_overlap(memory.data,memory.bytes,spans[i].data,spans[i].bytes))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"constants alias packed weights");
        next.weight[i].span=spans[i];
    }
    size_t offset=0;
    for (size_t j=0;j<sizeof(constant_slots)/sizeof(*constant_slots);++j) {
        size_t i=constant_slots[j]; Q38PackedLayout p=next.weight[i].layout;
        float *dst=(float *)memory.data+offset;
        for (size_t row=0;row<p.n_out;++row)
            Q38C_TRY(q38_packed_decode_row(&next.weight[i],0,row,dst+row*p.k_storage,p.k_storage,e));
        size_t bytes=p.n_out*p.k_storage*4;
        next.constant[i]=(Qwen38F32){.data=dst,.capacity_bytes=bytes,.rank=1,
            .dim={p.k_storage},.stride_bytes={4}};
        if (i==Q38CW_CONV) next.constant[i]=(Qwen38F32){.data=dst,.capacity_bytes=bytes,.rank=2,
            .dim={p.n_out,p.k_storage},.stride_bytes={p.k_storage*4,4}};
        Q38C_TRY(q38c_finite(next.constant[i],e)); offset+=p.n_out*p.k_storage;
    }
    for (size_t h=0;h<next.config.gdn.value_heads;++h)
        if (q38c_get(next.constant[Q38CW_A],h)>0) return q38c_error(e,Q38_RANGE,"positive transformed A");
    next.constant_memory=memory; next.bound=true; *layer=next;
    return q38c_error(e,Q38_OK,NULL);
}

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

const char *const q38c_stage_names[Q38CT_COUNT]={"normed","low","activated","gate_logits",
    "injection_logits","injection","mixed","qkv_raw","convolved","prepared","z","alpha",
    "beta_raw","log_decay","beta","scan","gated_norm","contribution"};
static size_t stage_width(const Q38CLayer *l, size_t stage) {
    size_t e=l->config.hidden_size, h=l->config.hc.streams, r=l->config.hc.low_rank;
    size_t hv=l->config.gdn.value_heads, inner=hv*l->config.gdn.value_dim;
    const size_t widths[]={h*e,r,r,h*e,h,h,e,l->conv.channels,l->conv.channels,l->conv.channels,
        inner,hv,hv,hv,hv,inner,inner,e};
    return widths[stage];
}
Qwen38Status q38c_work_size(const Q38CLayer *l, size_t tokens, size_t *out, Qwen38Error *e) {
    if (!l || !l->model || !out) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C workspace plan");
    *out=0;
    if (tokens>UINT32_MAX) return q38c_error(e,Q38_RANGE,"C token capacity exceeds uint32");
    size_t total=0;
    for (size_t i=0;i<Q38CT_COUNT;++i) { size_t bytes;
        if (stage_width(l,i)>UINT32_MAX) return q38c_error(e,Q38_RANGE,"C stage width exceeds uint32");
        if (!q38c_mul(tokens,stage_width(l,i),&bytes) || !q38c_mul(bytes,4,&bytes) ||
            !q38c_add(bytes,63,&bytes) || !q38c_add(total,bytes&~(size_t)63,&total))
            return q38c_error(e,Q38_OVERFLOW,"C workspace size overflows");
    }
    Qwen38StateSize cs,gs; Q38C_TRY(qwen38_conv_state_bytes(&l->conv,&cs,e));
    Q38C_TRY(qwen38_gdn_state_bytes(&l->config.gdn,&gs,e));
    size_t extra=l->constant_bytes, sizes[]={cs.metadata_bytes,cs.payload_bytes,gs.metadata_bytes,gs.payload_bytes};
    for (size_t i=0;i<4;++i) if (!q38c_add(extra,sizes[i],&extra))
        return q38c_error(e,Q38_OVERFLOW,"C state budget overflows");
    if (extra>512u*1024u*1024u || total>512u*1024u*1024u-extra)
        return q38c_error(e,Q38_CAPACITY,"C workspace plus layer state exceeds 512 MiB");
    *out=total; return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_work_init(const Q38CLayer *l, size_t tokens, Qwen38Memory memory,
        Q38CWork *out, Qwen38Error *e) {
    if (!out) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing C workspace output");
    *out=(Q38CWork){0}; size_t bytes; Q38C_TRY(q38c_work_size(l,tokens,&bytes,e));
    Q38C_TRY(q38c_backing(memory,bytes,e));
    Q38CWork w={.owner=l,.capacity_tokens=tokens,.memory=memory}; size_t offset=0;
    for (size_t i=0;i<Q38CT_COUNT;++i) {
        size_t width=stage_width(l,i), used=tokens*width*4;
        void *p=offset ? (unsigned char *)memory.data+offset : memory.data;
        w.trace[i]=(Qwen38F32Mut){.data=p,.capacity_bytes=used,.rank=2,
            .dim={tokens,width},.stride_bytes={width*4,4}};
        offset+=(used+63)&~(size_t)63;
    }
    *out=w; return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_work_check(const Q38CLayer *l, const Q38CWork *w, Qwen38F32 r, Qwen38Error *e) {
    if (!l || !l->bound || !w || w->owner!=l) return q38c_error(e,Q38_OWNER,"unbound or foreign C workspace");
    size_t t=r.dim[0], h=l->config.hc.streams, width=l->config.hidden_size;
    Q38C_TRY(q38c_dims(r,3,(size_t[]){t,h,width},e));
    if (r.stride_bytes[1]!=width*4) return q38c_error(e,Q38_STRIDE,"C layer requires contiguous residual streams");
    if (t>w->capacity_tokens) return q38c_error(e,Q38_CAPACITY,"C workspace token capacity exceeded");
    if (q38c_overlap(r.data,q38c_span(r),l,sizeof(*l)) || q38c_overlap(r.data,q38c_span(r),w,sizeof(*w)))
        return q38c_error(e,Q38_INVALID_ARGUMENT,"residual aliases C metadata");
    /* Validate ALL destinations before any stage can publish a partial result. */
    for (size_t i=0;i<Q38CT_COUNT;++i) {
        Qwen38F32 v=q38c_ro(q38c_trace(w,(Q38CStage)i,t));
        Q38C_TRY(q38c_dims(v,2,(size_t[]){t,stage_width(l,i)},e)); Q38C_TRY(q38c_alias(v,r,false,e));
        if (q38c_overlap(v.data,q38c_span(v),l,sizeof(*l)) || q38c_overlap(v.data,q38c_span(v),w,sizeof(*w)))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"C workspace aliases metadata");
        if (q38c_overlap(v.data,q38c_span(v),l->constant_memory.data,l->constant_memory.bytes))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"C workspace aliases constants");
        for (size_t j=0;j<Q38CW_COUNT;++j)
            if (q38c_overlap(v.data,q38c_span(v),l->weight[j].span.data,l->weight[j].span.bytes))
                return q38c_error(e,Q38_INVALID_ARGUMENT,"C workspace aliases weights");
        for (size_t j=0;j<i;++j) Q38C_TRY(q38c_alias(v,q38c_ro(q38c_trace(w,(Q38CStage)j,t)),false,e));
    }
    if (q38c_overlap(r.data,q38c_span(r),l->constant_memory.data,l->constant_memory.bytes))
        return q38c_error(e,Q38_INVALID_ARGUMENT,"residual aliases C constants");
    for (size_t j=0;j<Q38CW_COUNT;++j)
        if (q38c_overlap(r.data,q38c_span(r),l->weight[j].span.data,l->weight[j].span.bytes))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"residual aliases C weights");
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status q38c_hc_cpu(const Q38CLayer *l, Q38CWork *w, Qwen38F32 residual, Qwen38Error *e) {
    Q38C_TRY(q38c_work_check(l,w,residual,e)); size_t t=residual.dim[0];
    if (!t) return q38c_error(e,Q38_OK,NULL);
    Qwen38HcShape h=l->config.hc; Qwen38F32 gain=l->constant[Q38CW_HN];
    gain.rank=2; gain.dim[0]=h.streams; gain.dim[1]=h.hidden_size;
    gain.stride_bytes[0]=h.hidden_size*4; gain.stride_bytes[1]=4;
#define V(stage) q38c_trace(w,Q38CT_##stage,t)
#define R(stage) q38c_ro(V(stage))
    Q38C_TRY(qwen38_rmsnorm_groups(residual,gain,q38c_group(V(N),h.streams,h.hidden_size),h.rms_eps,e));
    Q38C_TRY(q38_packed_matmul(&l->weight[Q38CW_HD],0,R(N),V(LOW),e));
    Q38C_TRY(qwen38_hc_lowrank_activate(&h,R(LOW),V(ACT),e));
    Q38C_TRY(q38_packed_matmul(&l->weight[Q38CW_HU],0,R(ACT),V(HG),e));
    Q38C_TRY(qwen38_hc_read(&h,q38c_ro(q38c_group(V(N),h.streams,h.hidden_size)),
        q38c_ro(q38c_group(V(HG),h.streams,h.hidden_size)),V(MIX),e));
    Q38C_TRY(q38_packed_matmul(&l->weight[Q38CW_HI],0,R(N),V(IL),e));
    Q38C_TRY(qwen38_hc_inject_weights(&h,R(IL),V(INJECT),e));
#undef V
#undef R
    return q38c_error(e,Q38_OK,NULL);
}
