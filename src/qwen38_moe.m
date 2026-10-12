/* Private F post-routing composition. B owns products, C owns activations. */
#import <Metal/Metal.h>
#include "qwen38_f_internal.h"
#include "qwen38_c_internal.h"
#include <stdlib.h>
enum { GX,GD,GG,GU,GA,GS,GSL,GSA,GR2P,GP2R,GWEIGHT,GROUTED,GBUFFERS };
struct Q38FMoeGpu {
    Qwen38MoeConfig c; size_t cap,bytes;
    const GpuWeightStore *store; const GgufBinding *binding[8]; Q38PackedLayout layout[8];
    Q38PackedGpu *b; Q38CGpu *ops;
    void *device,*buffers[GBUFFERS],*pipeline[3];
    size_t *offsets,*cursors,*p2r,*r2p;
    Q38FMoeGpuObserver observer; void *observer_ctx;
};
typedef struct { uint32_t t,h,k,xpitch,ypitch; } FArgs;
static id<MTLBuffer> fb(Q38FMoeGpu *f,size_t i) { return (__bridge id<MTLBuffer>)f->buffers[i]; }
static Qwen38GpuF32 fv(Q38FMoeGpu *f,size_t index,size_t row,size_t n,size_t cols) {
    return (Qwen38GpuF32){.span={f->buffers[index],row*cols*4,n*cols*4},.rank=2,
        .dim={n,cols,1,1},.stride_bytes={cols*4,4,0,0}};
}
size_t q38f_gpu_bytes(const Q38FMoeGpu *f) { return f?f->bytes:0; }
void q38f_gpu_observe(Q38FMoeGpu *f,Q38FMoeGpuObserver observer,void *ctx) {
    if (f) { f->observer=observer; f->observer_ctx=ctx; }
}
static void gpu_trace(Q38FMoeGpu *f,Qwen38GpuCommand cmd,size_t stage,size_t ex,
                      Qwen38GpuF32 x,Qwen38GpuF32 y) {
    if (f->observer) f->observer(f->observer_ctx,cmd,stage,ex,x,y);
}
void q38f_gpu_destroy(Q38FMoeGpu *f) {
    if (!f) return;
    q38_packed_gpu_free(f->b); q38c_gpu_destroy(f->ops);
    for (size_t j=0;j<GBUFFERS;j++) if (f->buffers[j]) CFRelease(f->buffers[j]);
    for (size_t j=0;j<3;j++) if (f->pipeline[j]) CFRelease(f->pipeline[j]);
    if (f->device) CFRelease(f->device);
    free(f->offsets); free(f->cursors); free(f->p2r); free(f->r2p); free(f);
}
Qwen38Status q38f_gpu_create(void *device,void *library,const Qwen38MoeConfig *c,
    const GpuWeightStore *store,const GgufBinding *const weights[8],size_t cap,size_t budget,
    Q38FMoeGpu **out,Qwen38Error *err) {
    if (out) *out=NULL;
    if (!out || !device || !library || !c || !store || !weights)
        return q38c_error(err,Q38_INVALID_ARGUMENT,"missing GPU MoE argument");
    if (!cap || cap>32) return q38c_error(err,Q38_RANGE,"private GPU cap must be 1..32");
    Q38PackedCpuView views[8];
    for (size_t j=0;j<8;j++) {
        Q38C_TRY(qwen38_weights_cpu_span(store,weights[j],&views[j].span,err));
        Q38C_TRY(q38_packed_layout_init(&weights[j]->tensor,j==3?c->intermediate:weights[j]->tensor.shape[0],&views[j].layout,err));
    }
    Qwen38MoeWeights w={views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7]};
    Qwen38Moe *validation=NULL; Q38C_TRY(qwen38_moe_create(c,&w,&validation,err));
    qwen38_moe_destroy(validation);
    size_t p=cap*c->top_k,h=c->hidden_size,i=MAX(c->intermediate,c->shared_intermediate);
    if (p>SIZE_MAX/h) return q38c_error(err,Q38_OVERFLOW,"GPU pair size overflow");
    size_t counts[]={p*h,p*h,cap*i,cap*i,cap*i,cap*h,cap,cap,p,p,p,cap*h};
    size_t bytes=sizeof(Q38FMoeGpu)+(2*(size_t)c->experts+1+2*p)*sizeof(size_t);
    for (size_t j=0;j<GBUFFERS;j++) {
        if (counts[j]>SIZE_MAX/4 || bytes>SIZE_MAX-counts[j]*4)
            return q38c_error(err,Q38_OVERFLOW,"GPU arena overflow");
        bytes+=counts[j]*4;
    }
    if (bytes>budget || bytes>32*1024*1024) return q38c_error(err,Q38_CAPACITY,"GPU scratch budget");
    Q38FMoeGpu *f=calloc(1,sizeof *f); if (!f) return q38c_error(err,Q38_OOM,"GPU context");
    f->c=*c; f->cap=cap; f->bytes=bytes; f->store=store;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device; id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    f->device=(__bridge_retained void *)dev;
    for (size_t j=0;j<8;j++) { f->binding[j]=weights[j]; f->layout[j]=views[j].layout; }
    f->offsets=calloc((size_t)c->experts+1,sizeof(size_t)); f->cursors=calloc(c->experts,sizeof(size_t));
    f->p2r=calloc(p,sizeof(size_t)); f->r2p=calloc(p,sizeof(size_t));
    Qwen38Status st=Q38_OK;
    if (!f->offsets || !f->cursors || !f->p2r || !f->r2p) st=Q38_OOM;
    for (size_t j=0;j<GBUFFERS && !st;j++) {
        id<MTLBuffer> buffer=[dev newBufferWithLength:counts[j]*4 options:MTLResourceStorageModeShared];
        if (!buffer) st=Q38_OOM; else f->buffers[j]=(__bridge_retained void *)buffer;
    }
    if (!st) st=q38_packed_gpu_create(device,library,&f->b,err);
    if (!st) st=q38c_gpu_create(device,library,&f->ops,err);
    NSArray *names=@[@"q38f_gather",@"q38f_product",@"q38f_reduce"];
    for (size_t j=0;j<3 && !st;j++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[j]];
        id<MTLComputePipelineState> pipeline=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (!pipeline) st=Q38_GPU_ERROR; else f->pipeline[j]=(__bridge_retained void *)pipeline;
    }
    if (st) { q38f_gpu_destroy(f); return q38c_error(err,st,"GPU context setup"); }
    *out=f; return q38c_error(err,Q38_OK,NULL);
}

static void dispatch(Q38FMoeGpu *f,id<MTLComputeCommandEncoder> enc,size_t which,size_t count) {
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)f->pipeline[which];
    [enc setComputePipelineState:p];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)32,p.maxTotalThreadsPerThreadgroup),1,1)];
}
static Qwen38Status linear(Q38FMoeGpu *f,Qwen38GpuCommand cmd,size_t stage,size_t expert,
                           Qwen38GpuF32 x,Qwen38GpuF32 y,Qwen38Error *err) {
    Qwen38Status st=q38_packed_encode(f->b,&cmd,f->store,f->binding[stage],&f->layout[stage],expert,x,y,err);
    if (!st) gpu_trace(f,cmd,stage,expert,x,y); return st;
}
static Qwen38Status expert(Q38FMoeGpu *f,Qwen38GpuCommand cmd,size_t index,bool shared,
                           Qwen38GpuF32 x,Qwen38GpuF32 y,Qwen38Error *err) {
    size_t n=x.dim[0],i=shared?f->c.shared_intermediate:f->c.intermediate;
    Qwen38GpuF32 g=fv(f,GG,0,n,i),u=fv(f,GU,0,n,i),a=fv(f,GA,0,n,i);
    Q38C_TRY(linear(f,cmd,shared?4:1,index,x,g,err));
    Q38C_TRY(linear(f,cmd,shared?5:2,index,x,u,err));
    Q38CGpuSpec spec={.activation=Q38_SILU}; Qwen38GpuF32 views[]={g,a};
    Q38C_TRY(q38c_gpu_encode(f->ops,cmd,Q38C_ACT,&spec,views,2,err));
    gpu_trace(f,cmd,8,shared?SIZE_MAX:index,g,a);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    [enc setBuffer:fb(f,GA) offset:0 atIndex:0]; [enc setBuffer:fb(f,GU) offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:2];
    dispatch(f,enc,1,n*i);
    gpu_trace(f,cmd,9,shared?SIZE_MAX:index,u,a);
    return linear(f,cmd,shared?6:3,index,a,y,err);
}
static Qwen38Status external_view(Q38FMoeGpu *f,Qwen38GpuF32 v,size_t t,Qwen38Error *err) {
    Qwen38F32 shape={.rank=v.rank}; memcpy(shape.dim,v.dim,sizeof shape.dim);
    memcpy(shape.stride_bytes,v.stride_bytes,sizeof shape.stride_bytes);
    size_t bytes; Q38C_TRY(q38c_layout(shape,&bytes,err));
    if (v.rank!=2 || v.dim[0]!=t || v.dim[1]!=f->c.hidden_size)
        return q38c_error(err,Q38_SHAPE,"GPU MoE shape");
    if (v.stride_bytes[0]/4>UINT32_MAX) return q38c_error(err,Q38_RANGE,"GPU pitch exceeds uint32");
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)v.span.buffer;
    if (!b) return q38c_error(err,Q38_INVALID_ARGUMENT,"missing GPU buffer");
    if ((__bridge void *)b.device!=f->device) return q38c_error(err,Q38_OWNER,"GPU device mismatch");
    if (v.span.offset_bytes%4) return q38c_error(err,Q38_STRIDE,"GPU offset alignment");
    if (v.span.offset_bytes>b.length || v.span.bytes>b.length-v.span.offset_bytes || bytes>v.span.bytes)
        return q38c_error(err,Q38_CAPACITY,"GPU view outside buffer");
    return Q38_OK;
}
/* Conservative: bounding spans, including padding, and no-copy Shared aliases. */
static bool spans_alias(Qwen38GpuSpan a,Qwen38GpuSpan b) {
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x==y) return a.offset_bytes<b.offset_bytes+b.bytes && b.offset_bytes<a.offset_bytes+a.bytes;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared)
        return q38c_overlap((char *)x.contents+a.offset_bytes,a.bytes,(char *)y.contents+b.offset_bytes,b.bytes);
    return false;
}
Qwen38Status q38f_gpu_encode(Q38FMoeGpu *f,Qwen38GpuCommand cmd,Qwen38GpuF32 x,
    Qwen38GpuF32 y,Qwen38MoeRoutes routes,Qwen38Error *err) {
    if (!f || !cmd.compute_encoder) return q38c_error(err,Q38_INVALID_ARGUMENT,"missing GPU context/encoder");
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=f->device || (__bridge void *)enc.device!=f->device)
        return q38c_error(err,Q38_OWNER,"GPU command device mismatch");
    size_t t=x.dim[0],h=f->c.hidden_size,k=f->c.top_k,e=f->c.experts,p=t*k;
    if (t>f->cap) return q38c_error(err,Q38_CAPACITY,"GPU token capacity");
    Q38C_TRY(external_view(f,x,t,err)); Q38C_TRY(external_view(f,y,t,err));
    id<MTLBuffer> word=(__bridge id<MTLBuffer>)cmd.error_word.buffer;
    if (!word || (__bridge void *)word.device!=f->device || word.storageMode!=MTLStorageModeShared)
        return q38c_error(err,Q38_OWNER,"GPU error word owner/storage");
    Qwen38GpuSpan es=cmd.error_word;
    if (es.offset_bytes%4 || es.bytes<4 || es.offset_bytes>word.length || es.bytes>word.length-es.offset_bytes)
        return q38c_error(err,Q38_CAPACITY,"GPU error word bounds");
    if (spans_alias(x.span,y.span) || spans_alias(x.span,es) || spans_alias(y.span,es))
        return q38c_error(err,Q38_INVALID_ARGUMENT,"GPU input/output/error alias");
    for (size_t a=0;a<8;a++) {
        Qwen38GpuSpan ws={gpu_weights_buffer(f->store,f->binding[a]),f->binding[a]->offset,f->layout[a].tensor_bytes};
        if (spans_alias(x.span,ws) || spans_alias(y.span,ws) || spans_alias(es,ws))
            return q38c_error(err,Q38_INVALID_ARGUMENT,"GPU weights alias");
    }
    Qwen38F32 scores=q38c_ro(routes.weights);
    Q38C_TRY(q38c_dims(scores,2,(size_t[]){t,k},err));
    Q38C_TRY(q38c_finite(scores,err));
    for (size_t row=0;row<t;row++) {
        float sum=0;
        for (size_t slot=0;slot<k;slot++) {
            float value=scores.data[row*(scores.stride_bytes[0]/4)+slot];
            if (value<0 || value>1) return q38c_error(err,Q38_RANGE,"invalid frozen route weight");
            sum+=value;
        }
        if (f->c.norm_topk_prob && fabsf(sum-1)>2e-6f)
            return q38c_error(err,Q38_RANGE,"unnormalized frozen route");
    }
    if (routes.ids.rows!=t) return q38c_error(err,Q38_SHAPE,"GPU route count");
    Q38MoeGrouping g={f->offsets,(e+1)*sizeof(size_t),f->cursors,e*sizeof(size_t),
        f->p2r,f->r2p,f->cap*k*sizeof(size_t),f->cap*k*sizeof(size_t)};
    Q38C_TRY(q38_moe_group_pairs(&f->c,routes.ids,g,err));
    if (!t) return q38c_error(err,Q38_OK,NULL);
    for (size_t j=0;j<p;j++) {
        ((uint32_t *)fb(f,GR2P).contents)[j]=(uint32_t)f->r2p[j];
        ((uint32_t *)fb(f,GP2R).contents)[j]=(uint32_t)f->p2r[j];
        ((float *)fb(f,GWEIGHT).contents)[j]=scores.data[(j/k)*(scores.stride_bytes[0]/4)+j%k];
    }
    FArgs a={(uint32_t)t,(uint32_t)h,(uint32_t)k,(uint32_t)(x.stride_bytes[0]/4),(uint32_t)(y.stride_bytes[0]/4)};
    [enc setBuffer:(__bridge id<MTLBuffer>)x.span.buffer offset:x.span.offset_bytes atIndex:0];
    [enc setBuffer:fb(f,GX) offset:0 atIndex:1]; [enc setBuffer:fb(f,GR2P) offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof a atIndex:3]; [enc setBuffer:word offset:es.offset_bytes atIndex:4];
    dispatch(f,enc,0,p*h);
    for (size_t ex=0;ex<e;ex++) {
        size_t first=f->offsets[ex],n=f->offsets[ex+1]-first;
        if (n) Q38C_TRY(expert(f,cmd,ex,false,fv(f,GX,first,n,h),fv(f,GD,first,n,h),err));
    }
    Q38C_TRY(expert(f,cmd,0,true,x,fv(f,GS,0,t,h),err));
    Q38C_TRY(linear(f,cmd,7,0,x,fv(f,GSL,0,t,1),err));
    Q38CGpuSpec spec={.activation=Q38_SIGMOID};
    Qwen38GpuF32 acts[]={fv(f,GSL,0,t,1),fv(f,GSA,0,t,1)};
    Q38C_TRY(q38c_gpu_encode(f->ops,cmd,Q38C_ACT,&spec,acts,2,err));
    gpu_trace(f,cmd,10,SIZE_MAX,acts[0],acts[1]);
    size_t buffers[]={GD,GS,GSA,GWEIGHT,GP2R};
    for (size_t j=0;j<5;j++) [enc setBuffer:fb(f,buffers[j]) offset:0 atIndex:j];
    [enc setBuffer:(__bridge id<MTLBuffer>)y.span.buffer offset:y.span.offset_bytes atIndex:5];
    [enc setBytes:&a length:sizeof a atIndex:6]; [enc setBuffer:word offset:es.offset_bytes atIndex:7];
    [enc setBuffer:fb(f,GROUTED) offset:0 atIndex:8];
    dispatch(f,enc,2,t*h);
    gpu_trace(f,cmd,11,SIZE_MAX,x,fv(f,GROUTED,0,t,h));
    gpu_trace(f,cmd,12,SIZE_MAX,x,y);
    return q38c_error(err,Q38_OK,NULL);
}
