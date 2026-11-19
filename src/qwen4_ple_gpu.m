/* PLE GPU composition; G0 is the only command completion authority. */
#import <Metal/Metal.h>
#include "qwen4_ple_internal.h"
#include <stdlib.h>
struct QW4EGpu { void *device, *gate, *add; };
static Qwen4Status fail(Qwen4Error *e,Qwen4Status code,const char *s) { return qw4e_error(e,code,s); }
void qw4e_gpu_destroy(QW4EGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    if (g->gate) CFRelease(g->gate);
    if (g->add) CFRelease(g->add);
    free(g);
}
Qwen4Status qw4e_gpu_create(void *device,void *library,QW4EGpu **out,Qwen4Error *e) {
    if (!out) return fail(e,QW4_INVALID_ARGUMENT,"null GPU output");
    *out=NULL;
    if (!device || !library) return fail(e,QW4_INVALID_ARGUMENT,"GPU arguments");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device!=dev) return fail(e,QW4_OWNER,"foreign library");
    QW4EGpu *g=calloc(1,sizeof *g);
    if (!g) return fail(e,QW4_OOM,"GPU allocation");
    g->device=(__bridge_retained void *)dev;
    NSString *names[]={@"qw4e_gate_gpu",@"qw4e_add_gpu"};
    void **slots[]={&g->gate,&g->add};
    for (size_t i=0;i<2;i++) {
        id<MTLFunction> f=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=f ? [dev newComputePipelineStateWithFunction:f error:NULL] : nil;
        if (!p) { qw4e_gpu_destroy(g); return fail(e,QW4_GPU_ERROR,"PLE pipeline"); }
        *slots[i]=(__bridge_retained void *)p;
    }
    *out=g; return fail(e,QW4_OK,NULL);
}
static Qwen4Status span(Qwen4GpuSpan s,id<MTLDevice> dev,size_t need) {
    if (!s.buffer) return QW4_INVALID_ARGUMENT;
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)s.buffer;
    if (b.device!=dev) return QW4_OWNER;
    if (s.offset_bytes%4) return QW4_STRIDE;
    if (s.offset_bytes>b.length || s.bytes>b.length-s.offset_bytes || need>s.bytes) return QW4_CAPACITY;
    return b.hazardTrackingMode==MTLHazardTrackingModeUntracked ? QW4_UNSUPPORTED : QW4_OK;
}
static bool overlap(Qwen4GpuSpan a,size_t na,Qwen4GpuSpan b,size_t nb) {
    if (!na || !nb) return false;
    if (a.buffer==b.buffer) return a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared) {
        uintptr_t p=(uintptr_t)x.contents+a.offset_bytes,q=(uintptr_t)y.contents+b.offset_bytes;
        return p<=q ? q-p<na : p-q<nb;
    }
    return false;
}
static Qwen4Status dense(Qwen4GpuF32 v,id<MTLDevice> dev,size_t *bytes) {
    if (!v.rank || v.rank>4) return QW4_SHAPE;
    size_t n=4;
    for (size_t i=v.rank;i-->0;) {
        if ((!v.dim[i] && i) || v.dim[i]>UINT32_MAX) return QW4_SHAPE;
        if (v.stride_bytes[i]!=n) return QW4_STRIDE;
        if (!qw4c_mul(n,v.dim[i],&n) || n/4>UINT32_MAX) return QW4_OVERFLOW;
    }
    *bytes=n; return span(v.span,dev,n);
}
static bool dims(Qwen4GpuF32 v,size_t rank,const size_t *d) {
    return v.rank==rank && !memcmp(v.dim,d,rank*sizeof *d);
}
static Qwen4Status command_check(QW4EGpu *g,Qwen4GpuCommand cmd) {
    if (!g || !cmd.device || !cmd.compute_encoder) return QW4_INVALID_ARGUMENT;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    if (g->device!=cmd.device || [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder device]!=dev) return QW4_OWNER;
    Qwen4Status code=span(cmd.error_word,dev,4);
    if (!code && [(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared) code=QW4_CAPACITY;
    return code;
}
static void dispatch(void *pipeline,Qwen4GpuCommand cmd,const Qwen4GpuF32 *v,size_t n,
                     const uint32_t args[4],size_t work) {
    if (!work) return;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)pipeline;
    [enc setComputePipelineState:p];
    for (size_t i=0;i<n;i++) [enc setBuffer:(__bridge id<MTLBuffer>)v[i].span.buffer offset:v[i].span.offset_bytes atIndex:i];
    [enc setBytes:args length:16 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:7];
    [enc dispatchThreads:MTLSizeMake(work,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(work,p.maxTotalThreadsPerThreadgroup),1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static Qwen4GpuF32 flat(Qwen4GpuF32 v) {
    v.rank=2; v.dim[1]=v.dim[1]*v.dim[2]; v.stride_bytes[1]=4; return v;
}
Qwen4Status qw4e_add_encode(QW4EGpu *g,Qwen4GpuCommand cmd,Qwen4GpuF32 a,
                           Qwen4GpuF32 b,Qwen4GpuF32 out,Qwen4Error *e) {
    Qwen4Status code=command_check(g,cmd);
    if (code) return fail(e,code,"add command");
    Qwen4GpuF32 v[]={a,b,out}; size_t bytes[3];
    for (size_t i=0;i<3;i++) {
        if (!dims(v[i],a.rank,a.dim)) return fail(e,QW4_SHAPE,"add shape");
        code=dense(v[i],(__bridge id<MTLDevice>)cmd.device,bytes+i);
        if (code) return fail(e,code,"add view");
        if (overlap(v[i].span,bytes[i],cmd.error_word,4)) return fail(e,QW4_INVALID_ARGUMENT,"error alias");
    }
    for (size_t i=0;i<2;i++) if (overlap(out.span,bytes[2],v[i].span,bytes[i]) &&
        !(out.span.buffer==v[i].span.buffer && out.span.offset_bytes==v[i].span.offset_bytes))
        return fail(e,QW4_INVALID_ARGUMENT,"partial add alias");
    dispatch(g->add,cmd,v,3,(uint32_t[]){(uint32_t)(bytes[0]/4),0,0,0},bytes[0]/4);
    return fail(e,QW4_OK,NULL);
}
Qwen4Status qw4e_ple_encode(QW4EGpu *g,QW4PackedGpu *b,QW4CGpu *ops,Qwen4GpuCommand cmd,
    const Qwen4NgramConfig *c,const GpuWeightStore *store,const GgufBinding *const bindings[2],
    const QW4PackedLayout layouts[2],const QW4EGpuWork *w,Qwen4Error *e) {
    Qwen4Status code=command_check(g,cmd);
    if (code) return fail(e,code,"PLE command");
    if (!c || !store || !b || !ops || !bindings || !layouts || !w) return fail(e,QW4_INVALID_ARGUMENT,"PLE arguments");
    size_t t=w->residual.dim[0],h=c->streams,d=c->hidden_size,ch,width,history;
    if (!h || !d || !c->head_count || !c->head_dim || !c->order || !c->conv_kernel ||
        !isfinite(c->rms_eps) || c->rms_eps<=0 || !qw4c_mul(h,d,&ch) ||
        !qw4c_mul(c->head_count,c->head_dim,&width) || !qw4c_mul(c->conv_kernel-1,c->order,&history))
        return fail(e,QW4_SHAPE,"PLE config");
    Qwen4GpuF32 v[]={w->rows,w->residual,w->gamma_key,w->gamma_query,w->gamma_conv,w->taps,
        w->key,w->query,w->value,w->gate,w->u,w->z,w->conv,w->delta,w->history};
    size_t ranks[]={2,3,2,2,2,2,3,3,2,2,3,3,3,3,2};
    size_t shapes[][3]={{t,width,0},{t,h,d},{h,d,0},{h,d,0},{h,d,0},{ch,c->conv_kernel,0},
        {t,h,d},{t,h,d},{t,d,0},{t,h,0},{t,h,d},{t,h,d},{t,h,d},{t,h,d},{history,ch,0}};
    size_t bytes[15];
    for (size_t i=0;i<15;i++) {
        if (!dims(v[i],ranks[i],shapes[i])) return fail(e,QW4_SHAPE,"PLE view shape");
        code=dense(v[i],(__bridge id<MTLDevice>)cmd.device,bytes+i);
        if (code) return fail(e,code,"PLE view");
        if (overlap(v[i].span,bytes[i],cmd.error_word,4)) return fail(e,QW4_INVALID_ARGUMENT,"error alias");
        for (size_t j=0;j<i;j++) if (i>=6 && overlap(v[i].span,bytes[i],v[j].span,bytes[j]))
            return fail(e,QW4_INVALID_ARGUMENT,"PLE alias");
    }
    for (size_t i=0;i<2;i++) if (layouts[i].n_experts!=1 || layouts[i].k_logical!=width ||
        layouts[i].n_out!=(i ? d : ch)) return fail(e,QW4_SHAPE,"projection layout");
    /* Validate both live packed bindings before recording either product,
     * and protect every writable stage/history from resident weight aliases. */
    for (size_t i=0;i<2;i++) {
        Qwen4GpuF32 x=w->rows,y=i ? w->value : flat(w->key);
        x.dim[0]=y.dim[0]=0; x.span.bytes=y.span.bytes=0;
        code=qw4_packed_encode(b,&cmd,store,bindings[i],layouts+i,0,x,y,e);
        if (code) return code;
        Qwen4GpuSpan weight={gpu_weights_buffer(store,bindings[i]),bindings[i]->offset,layouts[i].tensor_bytes};
        for (size_t j=6;j<15;j++) if (overlap(weight,weight.bytes,v[j].span,bytes[j]))
            return fail(e,QW4_INVALID_ARGUMENT,"PLE view aliases weights");
    }
    if (!t) return fail(e,QW4_OK,NULL);
    code=qw4_packed_encode(b,&cmd,store,bindings[0],&layouts[0],0,w->rows,flat(w->key),e);
    if (!code) code=qw4_packed_encode(b,&cmd,store,bindings[1],&layouts[1],0,w->rows,w->value,e);
    if (code) return code;
    [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    QW4CGpuSpec spec={.eps=c->rms_eps,.conv={ch,c->conv_kernel,c->order,true}};
#define OP(op,...) do { Qwen4GpuF32 vv[]={__VA_ARGS__}; code=qw4c_gpu_encode(ops,cmd,QW4C_##op,&spec,vv,sizeof vv/sizeof *vv,e); if (code) return code; } while(0)
    OP(RMS,w->key,w->gamma_key,w->key);
    OP(RMS,w->residual,w->gamma_query,w->query);
    dispatch(g->gate,cmd,(Qwen4GpuF32[]){w->key,w->query,w->value,w->gate,w->u},5,
        (uint32_t[]){(uint32_t)t,(uint32_t)h,(uint32_t)d,0},t*h);
    OP(RMS,w->u,w->gamma_conv,w->z);
    OP(CONV,flat(w->z),w->taps,flat(w->conv),w->history);
#undef OP
    return qw4e_add_encode(g,cmd,w->u,w->conv,w->delta,e);
}
Qwen4Status qw4e_ngram_gpu_bind(Qwen4NgramState *s,Qwen4GpuSession *session,Qwen4Error *e) {
    if (!s || !session) return fail(e,QW4_INVALID_ARGUMENT,"GPU state arguments");
    Qwen4StateInfo g=qwen4_gpu_session_info(session);
    if (s->info.busy || g.busy) return fail(e,QW4_BUSY,"GPU state busy");
    if (s->gpu_session || !s->info.valid || !g.valid || s->info.next_pos || g.next_pos)
        return fail(e,QW4_BAD_STATE,"GPU bind requires fresh state");
    if (s->info.key.model!=g.key.model) return fail(e,QW4_OWNER,"GPU state model");
    if (s->info.key.epoch!=g.key.epoch) return fail(e,QW4_STALE,"GPU state epoch");
    s->gpu_session=session; return fail(e,QW4_OK,NULL);
}
Qwen4Status qw4e_ngram_gpu_commit(Qwen4NgramJob *j,Qwen4Error *e) {
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (!j->state->gpu_session) return fail(e,QW4_OWNER,"not a GPU state");
    Qwen4StateInfo g=qwen4_gpu_session_info(j->state->gpu_session);
    if (g.busy) return fail(e,QW4_BUSY,"GPU append not completed");
    if (g.key.model!=j->key.model) return fail(e,QW4_OWNER,"GPU state model");
    if (g.key.epoch!=j->key.epoch) return fail(e,QW4_STALE,"GPU state epoch");
    if (!g.valid || g.next_pos!=j->position+j->count || j->phase!=(j->count ? QW4E_READY : QW4E_COMPUTED))
        return fail(e,QW4_BAD_STATE,"GPU append did not complete");
    j->phase=QW4E_COMPUTED;
    return qwen4_ngram_commit(j->state,j,e);
}
