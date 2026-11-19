#include "qwen4_cpu_internal.h"

struct Qwen4ConvState { QW4CState base; Qwen4ConvShape shape; size_t history, elements; };
static Qwen4Status conv_shape(const Qwen4ConvShape *s, size_t *history, size_t *elements, Qwen4Error *e) {
    if (!s) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null conv shape");
    if (!s->channels || !s->kernel || !s->dilation) return qw4c_error(e, QW4_SHAPE, "zero conv dimension");
    if (s->channels > UINT32_MAX || s->kernel > UINT32_MAX || s->dilation > UINT32_MAX)
        return qw4c_error(e, QW4_RANGE, "conv dimension exceeds uint32");
    if (!qw4c_mul(s->kernel-1, s->dilation, history) || !qw4c_mul(*history, s->channels, elements))
        return qw4c_error(e, QW4_OVERFLOW, "conv history overflows");
    if (*history > UINT32_MAX) return qw4c_error(e, QW4_RANGE, "conv history exceeds uint32");
    return QW4_OK;
}
Qwen4Status qwen4_conv_state_bytes(const Qwen4ConvShape *s, Qwen4StateSize *out, Qwen4Error *e) {
    if (out) *out = (Qwen4StateSize){0};
    size_t history, elements; QW4C_TRY(conv_shape(s, &history, &elements, e));
    return qw4c_size(sizeof(Qwen4ConvState), elements, out, e);
}
Qwen4Status qwen4_conv_state_init(const Qwen4ConvShape *shape, Qwen4StateKey key,
        Qwen4Memory metadata, Qwen4Memory payload, Qwen4ConvState **out, Qwen4Error *e) {
    if (!out) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null state output");
    *out = NULL; Qwen4StateSize size; size_t history, elements;
    QW4C_TRY(conv_shape(shape, &history, &elements, e));
    QW4C_TRY(qwen4_conv_state_bytes(shape, &size, e));
    QW4C_TRY(qw4c_backings(key, metadata, payload, size, e));
    Qwen4ConvState *s = metadata.data;
    *s = (Qwen4ConvState){.base={.self=s, .magic=QW4C_CONV_MAGIC,
        .info={.key=key, .valid=true}, .metadata=metadata, .payload=payload},
        .shape=*shape, .history=history, .elements=elements};
    if (size.payload_bytes) memset(payload.data, 0, size.payload_bytes);
    *out=s; return qw4c_error(e, QW4_OK, NULL);
}
Qwen4StateInfo qwen4_conv_state_info(const Qwen4ConvState *s) {
    if (!s || s->base.self != s || s->base.magic != QW4C_CONV_MAGIC) return (Qwen4StateInfo){0};
    return s->base.info;
}
Qwen4Status qwen4_conv_state_reset(Qwen4ConvState *s, uint64_t epoch, Qwen4Error *e) {
    if (!s) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null conv state");
    return qw4c_state_reset(&s->base, QW4C_CONV_MAGIC, epoch, s->elements*sizeof(float), e);
}
static bool conv_same(Qwen4ConvShape a, Qwen4ConvShape b) {
    return a.channels==b.channels && a.kernel==b.kernel && a.dilation==b.dilation && a.apply_silu==b.apply_silu;
}
static Qwen4F32 conv_history(const Qwen4ConvState *s) {
    return (Qwen4F32){.data=s->base.payload.data, .capacity_bytes=s->base.payload.bytes, .rank=2,
        .dim={s->history,s->shape.channels}, .stride_bytes={s->shape.channels*sizeof(float),sizeof(float)}};
}
Qwen4Status qwen4_conv_state_export(const Qwen4ConvState *s, Qwen4F32Mut out, Qwen4Error *e) {
    QW4C_TRY(qw4c_state_check(s ? &s->base : NULL, QW4C_CONV_MAGIC, true, e));
    Qwen4F32 y=qw4c_ro(out), history=conv_history(s);
    QW4C_TRY(qw4c_dims(y, 2, history.dim, e)); QW4C_TRY(qw4c_state_alias(&s->base, y, e));
    QW4C_TRY(qw4c_finite(history, e));
    for (size_t i=0; i<s->elements; ++i) out.data[qw4c_offset(y,i)] = history.data[i];
    return qw4c_error(e, QW4_OK, NULL);
}
Qwen4Status qwen4_conv_state_copy(Qwen4ConvState *dst, const Qwen4ConvState *src, Qwen4Error *e) {
    QW4C_TRY(qw4c_state_check(dst ? &dst->base : NULL, QW4C_CONV_MAGIC, true, e));
    QW4C_TRY(qw4c_state_check(src ? &src->base : NULL, QW4C_CONV_MAGIC, true, e));
    Qwen4StateKey a=dst->base.info.key, b=src->base.info.key;
    if (a.model!=b.model || a.layer!=b.layer) return qw4c_error(e, QW4_OWNER, "different state owner");
    if (a.epoch!=b.epoch) return qw4c_error(e, QW4_STALE, "different state epoch");
    if (!conv_same(dst->shape, src->shape)) return qw4c_error(e, QW4_SHAPE, "different conv shape");
    Qwen4Memory d[]={dst->base.metadata,dst->base.payload}, s[]={src->base.metadata,src->base.payload};
    for (size_t i=0;i<2;++i) for (size_t j=0;j<2;++j)
        if (qw4c_overlap(d[i].data,d[i].bytes,s[j].data,s[j].bytes))
            return qw4c_error(e,QW4_INVALID_ARGUMENT,"copy requires separate state backings");
    QW4C_TRY(qw4c_finite(conv_history(src),e));
    if (src->elements) memcpy(dst->base.payload.data,src->base.payload.data,src->elements*sizeof(float));
    dst->base.info.next_pos=src->base.info.next_pos;
    return qw4c_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_conv_causal(const Qwen4ConvShape *shape, Qwen4F32 x,
        Qwen4F32 taps, Qwen4F32Mut out, Qwen4ConvState *s, uint64_t start, Qwen4Error *e) {
    size_t history, elements; QW4C_TRY(conv_shape(shape,&history,&elements,e));
    QW4C_TRY(qw4c_state_check(s ? &s->base : NULL,QW4C_CONV_MAGIC,true,e));
    if (!conv_same(*shape,s->shape)) return qw4c_error(e,QW4_SHAPE,"state/conv shape mismatch");
    Qwen4F32 y=qw4c_ro(out); size_t tcount=x.dim[0], ccount=shape->channels;
    QW4C_TRY(qw4c_dims(x,2,(size_t[]){tcount,ccount},e));
    QW4C_TRY(qw4c_dims(y,2,x.dim,e));
    QW4C_TRY(qw4c_dims(taps,2,(size_t[]){ccount,shape->kernel},e));
    QW4C_TRY(qw4c_alias(y,x,false,e)); QW4C_TRY(qw4c_alias(y,taps,false,e));
    Qwen4F32 views[]={x,taps,y};
    for (size_t i=0;i<3;++i) QW4C_TRY(qw4c_state_alias(&s->base,views[i],e));
    if (start!=s->base.info.next_pos) return qw4c_error(e,QW4_BAD_STATE,"noncontiguous conv append");
    if (tcount>UINT64_MAX-start) return qw4c_error(e,QW4_OVERFLOW,"position overflows");
    if (!tcount) return qw4c_error(e,QW4_OK,NULL); /* No input/weight reads. */
    QW4C_TRY(qw4c_finite(x,e)); QW4C_TRY(qw4c_finite(taps,e));
    if (qw4c_finite(conv_history(s),e)!=QW4_OK) { s->base.info.valid=false; return QW4_NONFINITE; }
    float *h=s->base.payload.data;
    s->base.info.busy=true;
    /* History is an immutable halo for this entire chunk; tap K-1 is current. */
    for (size_t t=0;t<tcount;++t) for (size_t c=0;c<ccount;++c) {
        float sum=0.0f;
        for (size_t j=0;j<shape->kernel;++j) {
            size_t row=t+j*shape->dilation;
            float value=row<history ? h[row*ccount+c] : x.data[(row-history)*(x.stride_bytes[0]/4)+c];
            sum += taps.data[c*(taps.stride_bytes[0]/4)+j] * value;
        }
        if (!isfinite(sum)) {
            s->base.info.valid=false; s->base.info.busy=false;
            return qw4c_error(e,QW4_NONFINITE,"conv accumulation overflow");
        }
        out.data[t*(y.stride_bytes[0]/4)+c]=shape->apply_silu ? qw4c_silu(sum) : sum;
    }
    /* Commit raw input only after every output has consumed the old halo. */
    if (history) {
        size_t keep=tcount<history ? history-tcount : 0;
        if (keep) memmove(h,h+tcount*ccount,keep*ccount*sizeof(float));
        for (size_t r=keep;r<history;++r)
            memcpy(h+r*ccount,x.data+(tcount-(history-r))*(x.stride_bytes[0]/4),ccount*sizeof(float));
    }
    s->base.info.next_pos=start+tcount; s->base.info.busy=false;
    return qw4c_error(e,QW4_OK,NULL);
}
