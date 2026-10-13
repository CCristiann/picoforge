#include "qwen38_c_internal.h"

struct Qwen38ConvState { Q38CState base; Qwen38ConvShape shape; size_t history, elements; };
static Qwen38Status conv_shape(const Qwen38ConvShape *s, size_t *history, size_t *elements, Qwen38Error *e) {
    if (!s) return q38c_error(e, Q38_INVALID_ARGUMENT, "null conv shape");
    if (!s->channels || !s->kernel || !s->dilation) return q38c_error(e, Q38_SHAPE, "zero conv dimension");
    if (s->channels > UINT32_MAX || s->kernel > UINT32_MAX || s->dilation > UINT32_MAX)
        return q38c_error(e, Q38_RANGE, "conv dimension exceeds uint32");
    if (!q38c_mul(s->kernel-1, s->dilation, history) || !q38c_mul(*history, s->channels, elements))
        return q38c_error(e, Q38_OVERFLOW, "conv history overflows");
    if (*history > UINT32_MAX) return q38c_error(e, Q38_RANGE, "conv history exceeds uint32");
    return Q38_OK;
}
Qwen38Status qwen38_conv_state_bytes(const Qwen38ConvShape *s, Qwen38StateSize *out, Qwen38Error *e) {
    if (out) *out = (Qwen38StateSize){0};
    size_t history, elements; Q38C_TRY(conv_shape(s, &history, &elements, e));
    return q38c_size(sizeof(Qwen38ConvState), elements, out, e);
}
Qwen38Status qwen38_conv_state_init(const Qwen38ConvShape *shape, Qwen38StateKey key,
        Qwen38Memory metadata, Qwen38Memory payload, Qwen38ConvState **out, Qwen38Error *e) {
    if (!out) return q38c_error(e, Q38_INVALID_ARGUMENT, "null state output");
    *out = NULL; Qwen38StateSize size; size_t history, elements;
    Q38C_TRY(conv_shape(shape, &history, &elements, e));
    Q38C_TRY(qwen38_conv_state_bytes(shape, &size, e));
    Q38C_TRY(q38c_backings(key, metadata, payload, size, e));
    Qwen38ConvState *s = metadata.data;
    *s = (Qwen38ConvState){.base={.self=s, .magic=Q38C_CONV_MAGIC,
        .info={.key=key, .valid=true}, .metadata=metadata, .payload=payload},
        .shape=*shape, .history=history, .elements=elements};
    if (size.payload_bytes) memset(payload.data, 0, size.payload_bytes);
    *out=s; return q38c_error(e, Q38_OK, NULL);
}
Qwen38StateInfo qwen38_conv_state_info(const Qwen38ConvState *s) {
    if (!s || s->base.self != s || s->base.magic != Q38C_CONV_MAGIC) return (Qwen38StateInfo){0};
    return s->base.info;
}
Qwen38Status qwen38_conv_state_reset(Qwen38ConvState *s, uint64_t epoch, Qwen38Error *e) {
    if (!s) return q38c_error(e, Q38_INVALID_ARGUMENT, "null conv state");
    return q38c_state_reset(&s->base, Q38C_CONV_MAGIC, epoch, s->elements*sizeof(float), e);
}
static bool conv_same(Qwen38ConvShape a, Qwen38ConvShape b) {
    return a.channels==b.channels && a.kernel==b.kernel && a.dilation==b.dilation && a.apply_silu==b.apply_silu;
}
static Qwen38F32 conv_history(const Qwen38ConvState *s) {
    return (Qwen38F32){.data=s->base.payload.data, .capacity_bytes=s->base.payload.bytes, .rank=2,
        .dim={s->history,s->shape.channels}, .stride_bytes={s->shape.channels*sizeof(float),sizeof(float)}};
}
Qwen38Status qwen38_conv_state_export(const Qwen38ConvState *s, Qwen38F32Mut out, Qwen38Error *e) {
    Q38C_TRY(q38c_state_check(s ? &s->base : NULL, Q38C_CONV_MAGIC, true, e));
    Qwen38F32 y=q38c_ro(out), history=conv_history(s);
    Q38C_TRY(q38c_dims(y, 2, history.dim, e)); Q38C_TRY(q38c_state_alias(&s->base, y, e));
    Q38C_TRY(q38c_finite(history, e));
    for (size_t i=0; i<s->elements; ++i) out.data[q38c_offset(y,i)] = history.data[i];
    return q38c_error(e, Q38_OK, NULL);
}
Qwen38Status qwen38_conv_state_copy(Qwen38ConvState *dst, const Qwen38ConvState *src, Qwen38Error *e) {
    Q38C_TRY(q38c_state_check(dst ? &dst->base : NULL, Q38C_CONV_MAGIC, true, e));
    Q38C_TRY(q38c_state_check(src ? &src->base : NULL, Q38C_CONV_MAGIC, true, e));
    Qwen38StateKey a=dst->base.info.key, b=src->base.info.key;
    if (a.model!=b.model || a.layer!=b.layer) return q38c_error(e, Q38_OWNER, "different state owner");
    if (a.epoch!=b.epoch) return q38c_error(e, Q38_STALE, "different state epoch");
    if (!conv_same(dst->shape, src->shape)) return q38c_error(e, Q38_SHAPE, "different conv shape");
    Qwen38Memory d[]={dst->base.metadata,dst->base.payload}, s[]={src->base.metadata,src->base.payload};
    for (size_t i=0;i<2;++i) for (size_t j=0;j<2;++j)
        if (q38c_overlap(d[i].data,d[i].bytes,s[j].data,s[j].bytes))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"copy requires separate state backings");
    Q38C_TRY(q38c_finite(conv_history(src),e));
    if (src->elements) memcpy(dst->base.payload.data,src->base.payload.data,src->elements*sizeof(float));
    dst->base.info.next_pos=src->base.info.next_pos;
    return q38c_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_conv_causal(const Qwen38ConvShape *shape, Qwen38F32 x,
        Qwen38F32 taps, Qwen38F32Mut out, Qwen38ConvState *s, uint64_t start, Qwen38Error *e) {
    size_t history, elements; Q38C_TRY(conv_shape(shape,&history,&elements,e));
    Q38C_TRY(q38c_state_check(s ? &s->base : NULL,Q38C_CONV_MAGIC,true,e));
    if (!conv_same(*shape,s->shape)) return q38c_error(e,Q38_SHAPE,"state/conv shape mismatch");
    Qwen38F32 y=q38c_ro(out); size_t tcount=x.dim[0], ccount=shape->channels;
    Q38C_TRY(q38c_dims(x,2,(size_t[]){tcount,ccount},e));
    Q38C_TRY(q38c_dims(y,2,x.dim,e));
    Q38C_TRY(q38c_dims(taps,2,(size_t[]){ccount,shape->kernel},e));
    Q38C_TRY(q38c_alias(y,x,false,e)); Q38C_TRY(q38c_alias(y,taps,false,e));
    Qwen38F32 views[]={x,taps,y};
    for (size_t i=0;i<3;++i) Q38C_TRY(q38c_state_alias(&s->base,views[i],e));
    if (start!=s->base.info.next_pos) return q38c_error(e,Q38_BAD_STATE,"noncontiguous conv append");
    if (tcount>UINT64_MAX-start) return q38c_error(e,Q38_OVERFLOW,"position overflows");
    if (!tcount) return q38c_error(e,Q38_OK,NULL); /* No input/weight reads. */
    Q38C_TRY(q38c_finite(x,e)); Q38C_TRY(q38c_finite(taps,e));
    if (q38c_finite(conv_history(s),e)!=Q38_OK) { s->base.info.valid=false; return Q38_NONFINITE; }
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
            return q38c_error(e,Q38_NONFINITE,"conv accumulation overflow");
        }
        out.data[t*(y.stride_bytes[0]/4)+c]=shape->apply_silu ? q38c_silu(sum) : sum;
    }
    /* Commit raw input only after every output has consumed the old halo. */
    if (history) {
        size_t keep=tcount<history ? history-tcount : 0;
        if (keep) memmove(h,h+tcount*ccount,keep*ccount*sizeof(float));
        for (size_t r=keep;r<history;++r)
            memcpy(h+r*ccount,x.data+(tcount-(history-r))*(x.stride_bytes[0]/4),ccount*sizeof(float));
    }
    s->base.info.next_pos=start+tcount; s->base.info.busy=false;
    return q38c_error(e,Q38_OK,NULL);
}
