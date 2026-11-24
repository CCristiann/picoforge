/* Private F post-routing composition. B owns products, C owns activations. */
#import <Metal/Metal.h>
#include "qwen4_moe_internal.h"
#include "qwen4_cpu_internal.h"
#include "qwen4_packed_internal.h"
#include "qwen4_ref.h"
#include "qwen4_profile.h"   /* only the format sentinels, for the R1.9a sub-group marks */
#include <stdlib.h>
enum { GX,GD,GG,GU,GA,GS,GSL,GSA,GR2P,GP2R,GWEIGHT,GROUTED,GBUFFERS };
struct QW4FMoeGpu {
    Qwen4MoeConfig c; size_t cap,bytes;
    const GpuWeightStore *store; const GgufBinding *binding[8]; QW4PackedLayout layout[8];
    QW4PackedGpu *b; QW4CGpu *ops;
    void *device,*buffers[GBUFFERS],*pipeline[3];
    size_t *offsets,*cursors,*p2r,*r2p;
    QW4FMoeGpuObserver observer; void *observer_ctx;
};
typedef struct { uint32_t t,h,k,xpitch,ypitch; } FArgs;
static id<MTLBuffer> fb(QW4FMoeGpu *f,size_t i) { return (__bridge id<MTLBuffer>)f->buffers[i]; }
static Qwen4GpuF32 fv(QW4FMoeGpu *f,size_t index,size_t row,size_t n,size_t cols) {
    return (Qwen4GpuF32){.span={f->buffers[index],row*cols*4,n*cols*4},.rank=2,
        .dim={n,cols,1,1},.stride_bytes={cols*4,4,0,0}};
}
size_t qw4f_gpu_bytes(const QW4FMoeGpu *f) { return f?f->bytes:0; }
void qw4f_gpu_observe(QW4FMoeGpu *f,QW4FMoeGpuObserver observer,void *ctx) {
    if (f) { f->observer=observer; f->observer_ctx=ctx; }
}
static void gpu_trace(QW4FMoeGpu *f,Qwen4GpuCommand cmd,size_t stage,size_t ex,
                      Qwen4GpuF32 x,Qwen4GpuF32 y) {
    if (f->observer) f->observer(f->observer_ctx,cmd,stage,ex,x,y);
}
void qw4f_gpu_destroy(QW4FMoeGpu *f) {
    if (!f) return;
    qw4_packed_gpu_free(f->b); qw4c_gpu_destroy(f->ops);
    for (size_t j=0;j<GBUFFERS;j++) if (f->buffers[j]) CFRelease(f->buffers[j]);
    for (size_t j=0;j<3;j++) if (f->pipeline[j]) CFRelease(f->pipeline[j]);
    if (f->device) CFRelease(f->device);
    free(f->offsets); free(f->cursors); free(f->p2r); free(f->r2p); free(f);
}
Qwen4Status qw4f_gpu_create(void *device,void *library,const Qwen4MoeConfig *c,
    const GpuWeightStore *store,const GgufBinding *const weights[8],size_t cap,size_t budget,
    QW4FMoeGpu **out,Qwen4Error *err) {
    if (out) *out=NULL;
    if (!out || !device || !library || !c || !store || !weights)
        return qw4c_error(err,QW4_INVALID_ARGUMENT,"missing GPU MoE argument");
    if (!cap || cap>32) return qw4c_error(err,QW4_RANGE,"private GPU cap must be 1..32");
    QW4PackedCpuView views[8];
    for (size_t j=0;j<8;j++) {
        QW4C_TRY(qwen4_weights_cpu_span(store,weights[j],&views[j].span,err));
        QW4C_TRY(qw4_packed_layout_init(&weights[j]->tensor,j==3?c->intermediate:weights[j]->tensor.shape[0],&views[j].layout,err));
    }
    Qwen4MoeWeights w={views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7]};
    Qwen4Moe *validation=NULL; QW4C_TRY(qwen4_moe_create(c,&w,&validation,err));
    qwen4_moe_destroy(validation);
    size_t p=cap*c->top_k,h=c->hidden_size,i=MAX(c->intermediate,c->shared_intermediate);
    if (p>SIZE_MAX/h) return qw4c_error(err,QW4_OVERFLOW,"GPU pair size overflow");
    size_t counts[]={p*h,p*h,cap*i,cap*i,cap*i,cap*h,cap,cap,p,p,p,cap*h};
    size_t bytes=sizeof(QW4FMoeGpu)+(2*(size_t)c->experts+1+2*p)*sizeof(size_t);
    for (size_t j=0;j<GBUFFERS;j++) {
        if (counts[j]>SIZE_MAX/4 || bytes>SIZE_MAX-counts[j]*4)
            return qw4c_error(err,QW4_OVERFLOW,"GPU arena overflow");
        bytes+=counts[j]*4;
    }
    if (bytes>budget || bytes>32*1024*1024) return qw4c_error(err,QW4_CAPACITY,"GPU scratch budget");
    QW4FMoeGpu *f=calloc(1,sizeof *f); if (!f) return qw4c_error(err,QW4_OOM,"GPU context");
    f->c=*c; f->cap=cap; f->bytes=bytes; f->store=store;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device; id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    f->device=(__bridge_retained void *)dev;
    for (size_t j=0;j<8;j++) { f->binding[j]=weights[j]; f->layout[j]=views[j].layout; }
    f->offsets=calloc((size_t)c->experts+1,sizeof(size_t)); f->cursors=calloc(c->experts,sizeof(size_t));
    f->p2r=calloc(p,sizeof(size_t)); f->r2p=calloc(p,sizeof(size_t));
    Qwen4Status st=QW4_OK;
    if (!f->offsets || !f->cursors || !f->p2r || !f->r2p) st=QW4_OOM;
    for (size_t j=0;j<GBUFFERS && !st;j++) {
        id<MTLBuffer> buffer=[dev newBufferWithLength:counts[j]*4 options:MTLResourceStorageModeShared];
        if (!buffer) st=QW4_OOM; else f->buffers[j]=(__bridge_retained void *)buffer;
    }
    if (!st) st=qw4_packed_gpu_create(device,library,&f->b,err);
    if (!st) st=qw4c_gpu_create(device,library,&f->ops,err);
    NSArray *names=@[@"qw4f_gather",@"qw4f_product",@"qw4f_reduce"];
    for (size_t j=0;j<3 && !st;j++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[j]];
        id<MTLComputePipelineState> pipeline=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (!pipeline) st=QW4_GPU_ERROR; else f->pipeline[j]=(__bridge_retained void *)pipeline;
    }
    if (st) { qw4f_gpu_destroy(f); return qw4c_error(err,st,"GPU context setup"); }
    *out=f; return qw4c_error(err,QW4_OK,NULL);
}

static void dispatch(QW4FMoeGpu *f,id<MTLComputeCommandEncoder> enc,size_t which,size_t count) {
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)f->pipeline[which];
    [enc setComputePipelineState:p];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)32,p.maxTotalThreadsPerThreadgroup),1,1)];
}
static Qwen4Status linear(QW4FMoeGpu *f,Qwen4GpuCommand cmd,size_t stage,size_t expert,
                           Qwen4GpuF32 x,Qwen4GpuF32 y,Qwen4Error *err) {
    Qwen4Status st=qw4_packed_encode(f->b,&cmd,f->store,f->binding[stage],&f->layout[stage],expert,x,y,err);
    if (!st) gpu_trace(f,cmd,stage,expert,x,y); return st;
}
static Qwen4Status expert(QW4FMoeGpu *f,Qwen4GpuCommand cmd,size_t index,bool shared,
                           Qwen4GpuF32 x,Qwen4GpuF32 y,Qwen4Error *err) {
    size_t n=x.dim[0],i=shared?f->c.shared_intermediate:f->c.intermediate;
    Qwen4GpuF32 g=fv(f,GG,0,n,i),u=fv(f,GU,0,n,i),a=fv(f,GA,0,n,i);
    QW4C_TRY(linear(f,cmd,shared?4:1,index,x,g,err));
    QW4C_TRY(linear(f,cmd,shared?5:2,index,x,u,err));
    QW4CGpuSpec spec={.activation=QW4_SILU}; Qwen4GpuF32 views[]={g,a};
    QW4C_TRY(qw4c_gpu_encode(f->ops,cmd,QW4C_ACT,&spec,views,2,err));
    gpu_trace(f,cmd,8,shared?SIZE_MAX:index,g,a);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    [enc setBuffer:fb(f,GA) offset:0 atIndex:0]; [enc setBuffer:fb(f,GU) offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:2];
    dispatch(f,enc,1,n*i);
    gpu_trace(f,cmd,9,shared?SIZE_MAX:index,u,a);
    return linear(f,cmd,shared?6:3,index,a,y,err);
}
static Qwen4Status external_view(QW4FMoeGpu *f,Qwen4GpuF32 v,size_t t,Qwen4Error *err) {
    Qwen4F32 shape={.rank=v.rank}; memcpy(shape.dim,v.dim,sizeof shape.dim);
    memcpy(shape.stride_bytes,v.stride_bytes,sizeof shape.stride_bytes);
    size_t bytes; QW4C_TRY(qw4c_layout(shape,&bytes,err));
    if (v.rank!=2 || v.dim[0]!=t || v.dim[1]!=f->c.hidden_size)
        return qw4c_error(err,QW4_SHAPE,"GPU MoE shape");
    if (v.stride_bytes[0]/4>UINT32_MAX) return qw4c_error(err,QW4_RANGE,"GPU pitch exceeds uint32");
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)v.span.buffer;
    if (!b) return qw4c_error(err,QW4_INVALID_ARGUMENT,"missing GPU buffer");
    if ((__bridge void *)b.device!=f->device) return qw4c_error(err,QW4_OWNER,"GPU device mismatch");
    if (v.span.offset_bytes%4) return qw4c_error(err,QW4_STRIDE,"GPU offset alignment");
    if (v.span.offset_bytes>b.length || v.span.bytes>b.length-v.span.offset_bytes || bytes>v.span.bytes)
        return qw4c_error(err,QW4_CAPACITY,"GPU view outside buffer");
    return QW4_OK;
}
/* Conservative: bounding spans, including padding, and no-copy Shared aliases. */
static bool spans_alias(Qwen4GpuSpan a,Qwen4GpuSpan b) {
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x==y) return a.offset_bytes<b.offset_bytes+b.bytes && b.offset_bytes<a.offset_bytes+a.bytes;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared)
        return qw4c_overlap((char *)x.contents+a.offset_bytes,a.bytes,(char *)y.contents+b.offset_bytes,b.bytes);
    return false;
}
Qwen4Status qw4f_gpu_encode(QW4FMoeGpu *f,Qwen4GpuCommand cmd,Qwen4GpuF32 x,
    Qwen4GpuF32 y,Qwen4MoeRoutes routes,Qwen4Error *err) {
    if (!f || !cmd.compute_encoder) return qw4c_error(err,QW4_INVALID_ARGUMENT,"missing GPU context/encoder");
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=f->device || (__bridge void *)enc.device!=f->device)
        return qw4c_error(err,QW4_OWNER,"GPU command device mismatch");
    size_t t=x.dim[0],h=f->c.hidden_size,k=f->c.top_k,e=f->c.experts,p=t*k;
    if (t>f->cap) return qw4c_error(err,QW4_CAPACITY,"GPU token capacity");
    QW4C_TRY(external_view(f,x,t,err)); QW4C_TRY(external_view(f,y,t,err));
    id<MTLBuffer> word=(__bridge id<MTLBuffer>)cmd.error_word.buffer;
    if (!word || (__bridge void *)word.device!=f->device || word.storageMode!=MTLStorageModeShared)
        return qw4c_error(err,QW4_OWNER,"GPU error word owner/storage");
    Qwen4GpuSpan es=cmd.error_word;
    if (es.offset_bytes%4 || es.bytes<4 || es.offset_bytes>word.length || es.bytes>word.length-es.offset_bytes)
        return qw4c_error(err,QW4_CAPACITY,"GPU error word bounds");
    if (spans_alias(x.span,y.span) || spans_alias(x.span,es) || spans_alias(y.span,es))
        return qw4c_error(err,QW4_INVALID_ARGUMENT,"GPU input/output/error alias");
    for (size_t a=0;a<8;a++) {
        Qwen4GpuSpan ws={gpu_weights_buffer(f->store,f->binding[a]),f->binding[a]->offset,f->layout[a].tensor_bytes};
        if (spans_alias(x.span,ws) || spans_alias(y.span,ws) || spans_alias(es,ws))
            return qw4c_error(err,QW4_INVALID_ARGUMENT,"GPU weights alias");
    }
    Qwen4F32 scores=qw4c_ro(routes.weights);
    QW4C_TRY(qw4c_dims(scores,2,(size_t[]){t,k},err));
    QW4C_TRY(qw4c_finite(scores,err));
    for (size_t row=0;row<t;row++) {
        float sum=0;
        for (size_t slot=0;slot<k;slot++) {
            float value=scores.data[row*(scores.stride_bytes[0]/4)+slot];
            if (value<0 || value>1) return qw4c_error(err,QW4_RANGE,"invalid frozen route weight");
            sum+=value;
        }
        if (f->c.norm_topk_prob && fabsf(sum-1)>2e-6f)
            return qw4c_error(err,QW4_RANGE,"unnormalized frozen route");
    }
    if (routes.ids.rows!=t) return qw4c_error(err,QW4_SHAPE,"GPU route count");
    QW4MoeGrouping g={f->offsets,(e+1)*sizeof(size_t),f->cursors,e*sizeof(size_t),
        f->p2r,f->r2p,f->cap*k*sizeof(size_t),f->cap*k*sizeof(size_t)};
    QW4C_TRY(qw4_moe_group_pairs(&f->c,routes.ids,g,err));
    if (!t) return qw4c_error(err,QW4_OK,NULL);
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
        if (n) QW4C_TRY(expert(f,cmd,ex,false,fv(f,GX,first,n,h),fv(f,GD,first,n,h),err));
    }
    QW4C_TRY(expert(f,cmd,0,true,x,fv(f,GS,0,t,h),err));
    QW4C_TRY(linear(f,cmd,7,0,x,fv(f,GSL,0,t,1),err));
    QW4CGpuSpec spec={.activation=QW4_SIGMOID};
    Qwen4GpuF32 acts[]={fv(f,GSL,0,t,1),fv(f,GSA,0,t,1)};
    QW4C_TRY(qw4c_gpu_encode(f->ops,cmd,QW4C_ACT,&spec,acts,2,err));
    gpu_trace(f,cmd,10,SIZE_MAX,acts[0],acts[1]);
    size_t buffers[]={GD,GS,GSA,GWEIGHT,GP2R};
    for (size_t j=0;j<5;j++) [enc setBuffer:fb(f,buffers[j]) offset:0 atIndex:j];
    [enc setBuffer:(__bridge id<MTLBuffer>)y.span.buffer offset:y.span.offset_bytes atIndex:5];
    [enc setBytes:&a length:sizeof a atIndex:6]; [enc setBuffer:word offset:es.offset_bytes atIndex:7];
    [enc setBuffer:fb(f,GROUTED) offset:0 atIndex:8];
    dispatch(f,enc,2,t*h);
    gpu_trace(f,cmd,11,SIZE_MAX,x,fv(f,GROUTED,0,t,h));
    gpu_trace(f,cmd,12,SIZE_MAX,x,y);
    return qw4c_error(err,QW4_OK,NULL);
}

/* ---- Device-routed MoE (G2). Same stage order as moe_batch, without grouping:
 * each (token, slot) pair is its own row, its expert read by B from the IDs. */
/* ref_moe: QW4_REF_KERNELS selects `moe` (read at creation): never fuse gate+up
 * or down+reduce. */
struct QW4FRouted { void *device, *route, *iota, *product, *reduce, *route_fast, *shared_act, *route_simd, *mixed_remap, *mixed_select; bool ref_moe; };
void qw4f_routed_destroy(QW4FRouted *r) {
    if (!r) return;
    void *all[]={r->route,r->iota,r->product,r->reduce,r->route_fast,r->shared_act,r->route_simd,r->mixed_remap,r->mixed_select,r->device};
    for (size_t i=0;i<sizeof all/sizeof *all;i++) if (all[i]) CFRelease(all[i]);
    free(r);
}
Qwen4Status qw4f_routed_create(void *device,void *library,QW4FRouted **out,Qwen4Error *err) {
    if (out) *out=NULL;
    if (!out || !device || !library) return qw4c_error(err,QW4_INVALID_ARGUMENT,"missing routed MoE argument");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device; id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    QW4FRouted *r=calloc(1,sizeof *r); if (!r) return qw4c_error(err,QW4_OOM,"routed MoE context");
    r->device=(__bridge_retained void *)dev;
    r->ref_moe=qw4_ref_kernels("moe");
    NSArray *names=@[@"qw4f_route",@"qw4f_iota",@"qw4f_product",@"qw4f_reduce",@"qw4f_mixed_remap",@"qw4f_mixed_select"];
    void **slots[]={&r->route,&r->iota,&r->product,&r->reduce,&r->mixed_remap,&r->mixed_select};
    for (size_t j=0;j<6;j++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[j]];
        id<MTLComputePipelineState> p=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (!p) { qw4f_routed_destroy(r); return qw4c_error(err,QW4_GPU_ERROR,"routed MoE pipeline"); }
        *slots[j]=(__bridge_retained void *)p;
    }
    { /* R1.5d: optional; without a 32-wide simdgroup or 256 threads qw4f_route stays */
        id<MTLFunction> fn=[lib newFunctionWithName:@"qw4_moe_route_fast"];
        id<MTLComputePipelineState> p=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (p && p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=QW4F_ROUTE_THREADS)
            r->route_fast=(__bridge_retained void *)p;
    }
    { /* Decode III: optional; `exact` keeps qw4_moe_route_fast (bit-identical to qw4f_route) */
        id<MTLFunction> fn=[lib newFunctionWithName:@"qw4_moe_route_simd"];
        id<MTLComputePipelineState> p=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (p && r->route_fast && p.threadExecutionWidth==32 && !qw4_ref_kernels("exact"))
            r->route_simd=(__bridge_retained void *)p;
    }
    { /* D1.2: optional; `moe` or `fuse` oracles keep the three separate ops */
        id<MTLFunction> fn=[lib newFunctionWithName:@"qw4f_shared_act"];
        id<MTLComputePipelineState> p=fn?[dev newComputePipelineStateWithFunction:fn error:nil]:nil;
        if (p && !r->ref_moe && !qw4_ref_kernels("fuse")) r->shared_act=(__bridge_retained void *)p;
    }
    *out=r; return qw4c_error(err,QW4_OK,NULL);
}
bool qw4f_routed_fast_route(const QW4FRouted *r,size_t experts,size_t slots) {
    return r && r->route_fast && !r->ref_moe && experts && experts<=QW4F_ROUTE_MAX_EXPERTS && slots && slots<=experts;
}
static void run(id<MTLComputeCommandEncoder> enc,void *pipeline,size_t count) {
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)pipeline;
    [enc setComputePipelineState:p];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)64,p.maxTotalThreadsPerThreadgroup),1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static void at(id<MTLComputeCommandEncoder> enc,Qwen4GpuSpan s,NSUInteger i) {
    [enc setBuffer:(__bridge id<MTLBuffer>)s.buffer offset:s.offset_bytes atIndex:i];
}
static bool dense(Qwen4GpuF32 v,size_t rows,size_t cols) {
    return v.rank==2 && v.dim[0]==rows && v.dim[1]==cols && v.stride_bytes[0]==cols*4 &&
        v.stride_bytes[1]==4 && v.span.buffer && v.span.bytes>=rows*cols*4 && !(v.span.offset_bytes%4);
}
/* picoforge mixed experts: one format's gate+up over every pair (grouped, fused or unfused, as below). */
static Qwen4Status mixed_gate_up(QW4FRouted *r,QW4PackedGpu *b,QW4CGpu *ops,Qwen4GpuCommand *cmd,const GpuWeightStore *store,
    const GgufBinding *gb,const QW4PackedLayout *gl,const GgufBinding *ub,const QW4PackedLayout *ul,Qwen4GpuSpan ids,
    size_t k,size_t p,size_t i,Qwen4GpuF32 x,Qwen4GpuF32 act,const QW4FRoutedWork *w,Qwen4Error *err) {
    const bool grouped=!r->ref_moe && w->group.buffer && gl->gguf_type==ul->gguf_type && qw4_packed_grouped_rows(b,gl->gguf_type,p);
    Qwen4Status st=r->ref_moe ? QW4_UNSUPPORTED : grouped ?
        qw4_packed_encode_grouped_gate_up(b,cmd,store,gb,gl,ub,ul,ids,k,x,act,w->group,err) :
        qw4_packed_encode_gate_up(b,cmd,store,gb,gl,ub,ul,ids,k,x,act,err);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd->compute_encoder;
    if (st==QW4_OK) { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; return QW4_OK; }
    if (grouped || st!=QW4_UNSUPPORTED) return st==QW4_UNSUPPORTED ? qw4c_error(err,QW4_GPU_ERROR,"grouped gate+up refused") : st;
    QW4C_TRY(qw4_packed_encode_indexed(b,cmd,store,gb,gl,ids,k,x,w->gate,err)); [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    QW4C_TRY(qw4_packed_encode_indexed(b,cmd,store,ub,ul,ids,k,x,w->up,err)); [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    QW4CGpuSpec spec={.activation=QW4_SILU}; Qwen4GpuF32 v[]={w->gate,act};
    QW4C_TRY(qw4c_gpu_encode(ops,*cmd,QW4C_ACT,&spec,v,2,err));
    at(enc,act.span,0); at(enc,w->up.span,1); at(enc,cmd->error_word,2); run(enc,r->product,p*i);
    return QW4_OK;
}
/* picoforge mixed experts: one format's down over every pair into rows [0,P) of y. */
static Qwen4Status mixed_down(QW4FRouted *r,QW4PackedGpu *b,Qwen4GpuCommand *cmd,const GpuWeightStore *store,
    const GgufBinding *db,const QW4PackedLayout *dl,Qwen4GpuSpan ids,size_t p,Qwen4GpuF32 act,Qwen4GpuF32 y,
    const QW4FRoutedWork *w,Qwen4Error *err) {
    const bool grouped=!r->ref_moe && w->group.buffer && qw4_packed_grouped_rows(b,dl->gguf_type,p);
    QW4C_TRY(grouped ? qw4_packed_encode_grouped_down(b,cmd,store,db,dl,ids,act,y,w->group,err) :
                       qw4_packed_encode_indexed(b,cmd,store,db,dl,ids,1,act,y,err));
    [(__bridge id<MTLComputeCommandEncoder>)cmd->compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return QW4_OK;
}
Qwen4Status qw4f_routed_encode(QW4FRouted *r,QW4PackedGpu *b,QW4CGpu *ops,Qwen4GpuCommand cmd,
    const Qwen4MoeConfig *c,const GpuWeightStore *store,const GgufBinding *const wb[8],
    const QW4PackedLayout wl[8],Qwen4GpuF32 x,Qwen4GpuF32 y,const QW4FRoutedWork *w,Qwen4Error *err) {
    if (!r || !b || !ops || !c || !store || !wb || !wl || !w || !cmd.compute_encoder)
        return qw4c_error(err,QW4_INVALID_ARGUMENT,"missing routed MoE input");
    if (cmd.device!=r->device) return qw4c_error(err,QW4_OWNER,"routed MoE device");
    size_t t=x.dim[0],h=c->hidden_size,e=c->experts,k=c->top_k,i=c->intermediate,s=c->shared_intermediate,p=t*k;
    if (x.rank!=2 || x.dim[1]!=h || y.rank!=2 || y.dim[0]!=t || y.dim[1]!=h || y.stride_bytes[0]/4>UINT32_MAX)
        return qw4c_error(err,QW4_SHAPE,"routed MoE x/y");
    if (!dense(w->logits,t,e) || !dense(w->probs,t,e) || !dense(w->weights,t,k) || !dense(w->gate,p,i) ||
        !dense(w->up,p,i) || !dense(w->act,p,i) || !dense(w->down,p,h) || !dense(w->sgate,t,s) ||
        !dense(w->sup,t,s) || !dense(w->sact,t,s) || !dense(w->shared,t,h) || !dense(w->slogit,t,1) ||
        !dense(w->ssig,t,1) || !dense(w->routed,t,h) || w->ids.bytes<p*4 || w->identity.bytes<p*4)
        return qw4c_error(err,QW4_SHAPE,"routed MoE workspace views");
    const bool mixed=w->expert_map.buffer!=NULL;
    if (mixed && (!dense(w->act_hot,p,i) || !dense(w->down2,2*p,h) || w->cold_ids.bytes<p*4 || w->hot_ids.bytes<p*4 ||
                  w->expert_map.bytes<e*4 || !w->hot[0] || !w->hot[1] || !w->hot[2]))
        return qw4c_error(err,QW4_SHAPE,"routed MoE mixed-expert views");
    if (mixed && w->remap) return qw4c_error(err,QW4_UNSUPPORTED,"mixed experts cannot be streamed");
    /* M4d: decode-sized mixed layers skip the other format's pairs in the fused kernels and fold
     * both downs into one parallel-slot down+reduce; prefill (grouped) and the exact kernels keep
     * the M4b path (both formats on every pair, then qw4f_mixed_select). */
    const bool fast_mixed=mixed && !r->ref_moe && wl[1].gguf_type==wl[2].gguf_type &&
        w->hot_layout[0].gguf_type==w->hot_layout[1].gguf_type && !qw4_packed_grouped_rows(b,wl[1].gguf_type,p) &&
        qw4_packed_mixed_fast(b,wl[1].gguf_type,w->hot_layout[0].gguf_type,wl[3].gguf_type,w->hot_layout[2].gguf_type,k);
    if (!t) return qw4c_error(err,QW4_OK,NULL);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (w->hc_added) *w->hc_added=false;
#define LIN(slot,in,out) do { QW4C_TRY(qw4_packed_encode(b,&cmd,store,wb[slot],&wl[slot],0,in,out,err)); \
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while (0)
#define IDX(slot,in,out,div) do { QW4C_TRY(qw4_packed_encode_indexed(b,&cmd,store,wb[slot],&wl[slot],xids,div,in,out,err)); \
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while (0)
#define ACT(kind,in,out) do { QW4CGpuSpec spec={.activation=kind}; Qwen4GpuF32 v[]={in,out}; \
    QW4C_TRY(qw4c_gpu_encode(ops,cmd,QW4C_ACT,&spec,v,2,err)); } while (0)
    /* Profile sub-groups (hook set only in profile mode); bytes: what the group's kernels read for these t tokens. */
#define MARK(op,type,bytes) do { if (w->mark) { w->mark(w->mark_ctx,&cmd,op,type,bytes); \
    enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder; } } while (0)
#define WB(slot) wb[slot]->tensor.bytes
#define MUL(a,u,n) do { at(enc,(a).span,0); at(enc,(u).span,1); at(enc,cmd.error_word,2); run(enc,r->product,n); } while (0)
    MARK("moe.shared_in",QW4_PROFILE_MIXED,WB(0)+WB(4)+WB(5)+WB(7));
    { /* R1.4: router, shared gate/up and shared logit all read x: one multi call, one barrier */
        const size_t slot[]={0,4,5,7};
        const GgufBinding *mb[4]; QW4PackedLayout ml[4];
        const Qwen4GpuF32 out[]={w->logits,w->sgate,w->sup,w->slogit};
        for (size_t j=0;j<4;j++) { mb[j]=wb[slot[j]]; ml[j]=wl[slot[j]]; }
        QW4C_TRY(qw4_packed_encode_multi(b,&cmd,store,mb,ml,4,x,out,err));
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    struct { uint32_t t,e,k,norm,pitch; } ra={(uint32_t)t,(uint32_t)e,(uint32_t)k,c->norm_topk_prob,(uint32_t)e};
    MARK("moe.route",QW4_PROFILE_NONE,0);
    at(enc,w->logits.span,0); at(enc,w->probs.span,1); at(enc,w->ids,2); at(enc,w->weights.span,3);
    [enc setBytes:&ra length:sizeof ra atIndex:4]; at(enc,cmd.error_word,5);
    /* Decode III: the shared expert's activation runs beside the router and its down matvec beside
     * gate+up, without the two barriers of their own (same kernels, same bits); `barrier` keeps the
     * old order. */
    bool early_shared=r->shared_act && !w->mark && (!mixed || fast_mixed) && !qw4_ref_kernels("barrier") && !qw4_ref_kernels("exact"); /* profile: own group */
    if (qw4f_routed_fast_route(r,e,k)) { /* R1.5d: one threadgroup per token, same ids, weights and probs */
        const bool simd=r->route_simd!=NULL; /* Decode III: one simdgroup per token */
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)(simd ? r->route_simd : r->route_fast)];
        [enc dispatchThreadgroups:MTLSizeMake(t,1,1) threadsPerThreadgroup:MTLSizeMake(simd ? 32 : QW4F_ROUTE_THREADS,1,1)];
        if (early_shared) { /* Decode III: independent of the router (both read only shared_in's outputs) */
            struct { uint32_t count,tokens; } sa={(uint32_t)(t*s),(uint32_t)t};
            at(enc,w->sgate.span,0); at(enc,w->sup.span,1); at(enc,w->sact.span,2); at(enc,w->slogit.span,3);
            at(enc,w->ssig.span,4); at(enc,cmd.error_word,5); [enc setBytes:&sa length:sizeof sa atIndex:6];
            id<MTLComputePipelineState> ap=(__bridge id<MTLComputePipelineState>)r->shared_act;
            [enc setComputePipelineState:ap];
            [enc dispatchThreads:MTLSizeMake(t*s+t,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)64,ap.maxTotalThreadsPerThreadgroup),1,1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else { early_shared=false; run(enc,r->route,t); }
    Qwen4GpuSpan xids=w->ids;   /* what the expert kernels index: expert ids, or pool slots (R5.1b) */
    if (w->remap) {
        if (w->slots.bytes<p*4) return qw4c_error(err,QW4_SHAPE,"routed MoE slot view");
        QW4C_TRY(w->remap(w->remap_ctx,&cmd,w->ids,w->slots,p,err));
        enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder; xids=w->slots;
    }
    /* R1.5b: gate+up+SwiGLU in one fused dispatch, act bit-identical to the
     * unfused fast sequence below; gate/up are then not written (nothing reads
     * them but ACT and MUL). Fused only when neither `moe` nor `indexed` is
     * selected: under `indexed` B answers UNSUPPORTED, since the fused act
     * would not match the oracle indexed sequence; so does a format without a
     * row dot. Then the unfused sequence. */
    MARK("moe.gate_up",QW4_PROFILE_MIXED,p*(wl[1].expert_bytes+wl[2].expert_bytes));
    if (mixed) {   /* picoforge mixed experts: slots per format, then both formats on every pair */
        struct { uint32_t pairs,experts,filler,pad; } ma={(uint32_t)p,(uint32_t)e,fast_mixed ? 0xFFFFFFFFu : 0u,0};
        at(enc,w->ids,0); at(enc,w->expert_map,1); at(enc,w->cold_ids,2); at(enc,w->hot_ids,3); at(enc,w->identity,4);
        [enc setBytes:&ma length:sizeof ma atIndex:5]; at(enc,cmd.error_word,6); run(enc,r->mixed_remap,p);
        if (fast_mixed) {   /* disjoint pairs, one act: no barrier between the two formats */
            Qwen4Status st=qw4_packed_encode_gate_up(b,&cmd,store,wb[1],&wl[1],wb[2],&wl[2],w->cold_ids,k,x,w->act,err);
            if (!st) st=qw4_packed_encode_gate_up(b,&cmd,store,w->hot[0],&w->hot_layout[0],w->hot[1],&w->hot_layout[1],w->hot_ids,k,x,w->act,err);
            if (st) return st==QW4_UNSUPPORTED ? qw4c_error(err,QW4_GPU_ERROR,"mixed fused gate+up refused") : st;
            if (early_shared) QW4C_TRY(qw4_packed_encode(b,&cmd,store,wb[6],&wl[6],0,w->sact,w->shared,err));
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            QW4C_TRY(mixed_gate_up(r,b,ops,&cmd,store,wb[1],&wl[1],wb[2],&wl[2],w->cold_ids,k,p,i,x,w->act,w,err));
            QW4C_TRY(mixed_gate_up(r,b,ops,&cmd,store,w->hot[0],&w->hot_layout[0],w->hot[1],&w->hot_layout[1],w->hot_ids,k,p,i,x,w->act_hot,w,err));
        }
        enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    }
    /* R3.2: a prefill-sized group of pairs runs one GEMM per expert instead
     * (then down too, below); the per-pair kernels otherwise. */
    const bool grouped=!r->ref_moe && w->group.buffer && qw4_packed_grouped_rows(b,wl[1].gguf_type,p) &&
        qw4_packed_grouped_rows(b,wl[3].gguf_type,p) && wl[1].gguf_type==wl[2].gguf_type;
    Qwen4Status fused=mixed ? QW4_OK : r->ref_moe ? QW4_UNSUPPORTED : grouped ?
        qw4_packed_encode_grouped_gate_up(b,&cmd,store,wb[1],&wl[1],wb[2],&wl[2],xids,k,x,w->act,w->group,err) :
        qw4_packed_encode_gate_up(b,&cmd,store,wb[1],&wl[1],wb[2],&wl[2],xids,k,x,w->act,err);
    if (grouped && fused==QW4_UNSUPPORTED) return qw4c_error(err,QW4_GPU_ERROR,"grouped gate+up refused");
    if (mixed) {
    } else if (fused==QW4_OK) {
        if (early_shared) QW4C_TRY(qw4_packed_encode(b,&cmd,store,wb[6],&wl[6],0,w->sact,w->shared,err));
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else if (fused!=QW4_UNSUPPORTED) return fused;
    else {
        IDX(1,x,w->gate,k); IDX(2,x,w->up,k); ACT(QW4_SILU,w->gate,w->act); MUL(w->act,w->up,p*i);
        if (early_shared) LIN(6,w->sact,w->shared);
    }
    MARK("moe.shared",QW4_PROFILE_MIXED,WB(6));
    if (early_shared) {
    } else if (r->shared_act) { /* D1.2: SILU, MUL and SIGMOID of the shared expert in one dispatch */
        struct { uint32_t count,tokens; } sa={(uint32_t)(t*s),(uint32_t)t};
        at(enc,w->sgate.span,0); at(enc,w->sup.span,1); at(enc,w->sact.span,2); at(enc,w->slogit.span,3);
        at(enc,w->ssig.span,4); at(enc,cmd.error_word,5); [enc setBytes:&sa length:sizeof sa atIndex:6];
        run(enc,r->shared_act,t*s+t);
        LIN(6,w->sact,w->shared);
    } else {
    ACT(QW4_SILU,w->sgate,w->sact); MUL(w->sact,w->sup,t*s);
    LIN(6,w->sact,w->shared);
    ACT(QW4_SIGMOID,w->slogit,w->ssig);
    }
    /* R1.5c: the routed down projection and qw4f_reduce in one fused dispatch
     * after the shared expert (it reads shared and ssig as qw4f_reduce does);
     * y and routed are bit-identical to the unfused fast sequence below and
     * the down scratch is not written (only that sequence reads it). Same
     * rule as gate+up: fused only when neither `moe` nor `indexed` is
     * selected (B answers UNSUPPORTED under `indexed` or for a format without
     * a row dot), else IDX down then qw4f_reduce, as before R1.5c. */
    MARK("moe.down",QW4_PROFILE_MIXED,p*wl[3].expert_bytes);
    const QW4DownReduceViews dv={w->weights,w->shared,w->ssig,w->routed,w->hc_residual,w->hc_inject,w->hc_added};
    fused=r->ref_moe || grouped || mixed ? QW4_UNSUPPORTED : qw4_packed_encode_down_reduce(b,&cmd,store,wb[3],&wl[3],xids,w->act,&dv,y,err);
    if (fast_mixed) {
        fused=qw4_packed_encode_down_reduce_mixed(b,&cmd,store,wb[3],&wl[3],w->cold_ids,w->hot[2],&w->hot_layout[2],
                                                  w->hot_ids,w->act,&dv,y,err);
        if (fused) return fused==QW4_UNSUPPORTED ? qw4c_error(err,QW4_GPU_ERROR,"mixed down+reduce refused") : fused;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else if (mixed) {   /* cold rows [0,P), hot rows [P,2P); identity already holds each pair's row */
        Qwen4GpuF32 cold=w->down2,hot=w->down2;
        cold.dim[0]=hot.dim[0]=p; cold.span.bytes=hot.span.bytes=p*h*4; hot.span.offset_bytes+=p*h*4;
        QW4C_TRY(mixed_down(r,b,&cmd,store,wb[3],&wl[3],w->cold_ids,p,w->act,cold,w,err));
        QW4C_TRY(mixed_down(r,b,&cmd,store,w->hot[2],&w->hot_layout[2],w->hot_ids,p,w->act_hot,hot,w,err));
        enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
        struct { uint32_t pairs,width; } sa={(uint32_t)p,(uint32_t)h};
        at(enc,w->down2.span,0); at(enc,w->identity,1); at(enc,w->down.span,2);
        [enc setBytes:&sa length:sizeof sa atIndex:3]; at(enc,cmd.error_word,4); run(enc,r->mixed_select,p*h);
        at(enc,w->identity,0); run(enc,r->iota,p);
        FArgs fa={(uint32_t)t,(uint32_t)h,(uint32_t)k,(uint32_t)h,(uint32_t)(y.stride_bytes[0]/4)};
        at(enc,w->down.span,0); at(enc,w->shared.span,1); at(enc,w->ssig.span,2); at(enc,w->weights.span,3);
        at(enc,w->identity,4); at(enc,y.span,5); [enc setBytes:&fa length:sizeof fa atIndex:6];
        at(enc,cmd.error_word,7); at(enc,w->routed.span,8); run(enc,r->reduce,t*h);
    } else if (fused==QW4_OK) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    else if (fused!=QW4_UNSUPPORTED) return fused;
    else { /* the identity map is read only by qw4f_reduce */
        at(enc,w->identity,0); run(enc,r->iota,p);
        if (grouped) { /* R3.2: down[pair] by expert GEMMs, then the same qw4f_reduce */
            QW4C_TRY(qw4_packed_encode_grouped_down(b,&cmd,store,wb[3],&wl[3],xids,w->act,w->down,w->group,err));
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else IDX(3,w->act,w->down,1);
        FArgs fa={(uint32_t)t,(uint32_t)h,(uint32_t)k,(uint32_t)h,(uint32_t)(y.stride_bytes[0]/4)};
        at(enc,w->down.span,0); at(enc,w->shared.span,1); at(enc,w->ssig.span,2); at(enc,w->weights.span,3);
        at(enc,w->identity,4); at(enc,y.span,5); [enc setBytes:&fa length:sizeof fa atIndex:6];
        at(enc,cmd.error_word,7); at(enc,w->routed.span,8); run(enc,r->reduce,t*h);
    }
#undef LIN
#undef IDX
#undef ACT
#undef MUL
#undef WB
#undef MARK
    return qw4c_error(err,QW4_OK,NULL);
}
