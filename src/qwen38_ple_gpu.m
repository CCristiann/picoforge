/* PLE GPU composition; G0 is the only command completion authority. */
#import <Metal/Metal.h>
#include "qwen38_e_internal.h"
#include <stdlib.h>
struct Q38EGpu { void *device, *gate, *add; };
static Qwen38Status fail(Qwen38Error *e,Qwen38Status code,const char *s) { return q38e_error(e,code,s); }
void q38e_gpu_destroy(Q38EGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    if (g->gate) CFRelease(g->gate);
    if (g->add) CFRelease(g->add);
    free(g);
}
Qwen38Status q38e_gpu_create(void *device,void *library,Q38EGpu **out,Qwen38Error *e) {
    if (!out) return fail(e,Q38_INVALID_ARGUMENT,"null GPU output");
    *out=NULL;
    if (!device || !library) return fail(e,Q38_INVALID_ARGUMENT,"GPU arguments");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device!=dev) return fail(e,Q38_OWNER,"foreign library");
    Q38EGpu *g=calloc(1,sizeof *g);
    if (!g) return fail(e,Q38_OOM,"GPU allocation");
    g->device=(__bridge_retained void *)dev;
    NSString *names[]={@"q38e_gate_gpu",@"q38e_add_gpu"};
    void **slots[]={&g->gate,&g->add};
    for (size_t i=0;i<2;i++) {
        id<MTLFunction> f=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=f ? [dev newComputePipelineStateWithFunction:f error:NULL] : nil;
        if (!p) { q38e_gpu_destroy(g); return fail(e,Q38_GPU_ERROR,"PLE pipeline"); }
        *slots[i]=(__bridge_retained void *)p;
    }
    *out=g; return fail(e,Q38_OK,NULL);
}
static Qwen38Status span(Qwen38GpuSpan s,id<MTLDevice> dev,size_t need) {
    if (!s.buffer) return Q38_INVALID_ARGUMENT;
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)s.buffer;
    if (b.device!=dev) return Q38_OWNER;
    if (s.offset_bytes%4) return Q38_STRIDE;
    if (s.offset_bytes>b.length || s.bytes>b.length-s.offset_bytes || need>s.bytes) return Q38_CAPACITY;
    return b.hazardTrackingMode==MTLHazardTrackingModeUntracked ? Q38_UNSUPPORTED : Q38_OK;
}
static bool overlap(Qwen38GpuSpan a,size_t na,Qwen38GpuSpan b,size_t nb) {
    if (!na || !nb) return false;
    if (a.buffer==b.buffer) return a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared) {
        uintptr_t p=(uintptr_t)x.contents+a.offset_bytes,q=(uintptr_t)y.contents+b.offset_bytes;
        return p<=q ? q-p<na : p-q<nb;
    }
    return false;
}
static Qwen38Status dense(Qwen38GpuF32 v,id<MTLDevice> dev,size_t *bytes) {
    if (!v.rank || v.rank>4) return Q38_SHAPE;
    size_t n=4;
    for (size_t i=v.rank;i-->0;) {
        if ((!v.dim[i] && i) || v.dim[i]>UINT32_MAX) return Q38_SHAPE;
        if (v.stride_bytes[i]!=n) return Q38_STRIDE;
        if (!q38c_mul(n,v.dim[i],&n) || n/4>UINT32_MAX) return Q38_OVERFLOW;
    }
    *bytes=n; return span(v.span,dev,n);
}
static bool dims(Qwen38GpuF32 v,size_t rank,const size_t *d) {
    return v.rank==rank && !memcmp(v.dim,d,rank*sizeof *d);
}
static Qwen38Status command_check(Q38EGpu *g,Qwen38GpuCommand cmd) {
    if (!g || !cmd.device || !cmd.compute_encoder) return Q38_INVALID_ARGUMENT;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    if (g->device!=cmd.device || [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder device]!=dev) return Q38_OWNER;
    Qwen38Status code=span(cmd.error_word,dev,4);
    if (!code && [(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared) code=Q38_CAPACITY;
    return code;
}
static void dispatch(void *pipeline,Qwen38GpuCommand cmd,const Qwen38GpuF32 *v,size_t n,
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
static Qwen38GpuF32 flat(Qwen38GpuF32 v) {
    v.rank=2; v.dim[1]=v.dim[1]*v.dim[2]; v.stride_bytes[1]=4; return v;
}
Qwen38Status q38e_add_encode(Q38EGpu *g,Qwen38GpuCommand cmd,Qwen38GpuF32 a,
                           Qwen38GpuF32 b,Qwen38GpuF32 out,Qwen38Error *e) {
    Qwen38Status code=command_check(g,cmd);
    if (code) return fail(e,code,"add command");
    Qwen38GpuF32 v[]={a,b,out}; size_t bytes[3];
    for (size_t i=0;i<3;i++) {
        if (!dims(v[i],a.rank,a.dim)) return fail(e,Q38_SHAPE,"add shape");
        code=dense(v[i],(__bridge id<MTLDevice>)cmd.device,bytes+i);
        if (code) return fail(e,code,"add view");
        if (overlap(v[i].span,bytes[i],cmd.error_word,4)) return fail(e,Q38_INVALID_ARGUMENT,"error alias");
    }
    for (size_t i=0;i<2;i++) if (overlap(out.span,bytes[2],v[i].span,bytes[i]) &&
        !(out.span.buffer==v[i].span.buffer && out.span.offset_bytes==v[i].span.offset_bytes))
        return fail(e,Q38_INVALID_ARGUMENT,"partial add alias");
    dispatch(g->add,cmd,v,3,(uint32_t[]){(uint32_t)(bytes[0]/4),0,0,0},bytes[0]/4);
    return fail(e,Q38_OK,NULL);
}
Qwen38Status q38e_ple_encode(Q38EGpu *g,Q38PackedGpu *b,Q38CGpu *ops,Qwen38GpuCommand cmd,
    const Qwen38NgramConfig *c,const GpuWeightStore *store,const GgufBinding *const bindings[2],
    const Q38PackedLayout layouts[2],const Q38EGpuWork *w,Qwen38Error *e) {
    Qwen38Status code=command_check(g,cmd);
    if (code) return fail(e,code,"PLE command");
    if (!c || !store || !b || !ops || !bindings || !layouts || !w) return fail(e,Q38_INVALID_ARGUMENT,"PLE arguments");
    size_t t=w->residual.dim[0],h=c->streams,d=c->hidden_size,ch,width,history;
    if (!h || !d || !c->head_count || !c->head_dim || !c->order || !c->conv_kernel ||
        !isfinite(c->rms_eps) || c->rms_eps<=0 || !q38c_mul(h,d,&ch) ||
        !q38c_mul(c->head_count,c->head_dim,&width) || !q38c_mul(c->conv_kernel-1,c->order,&history))
        return fail(e,Q38_SHAPE,"PLE config");
    Qwen38GpuF32 v[]={w->rows,w->residual,w->gamma_key,w->gamma_query,w->gamma_conv,w->taps,
        w->key,w->query,w->value,w->gate,w->u,w->z,w->conv,w->delta,w->history};
    size_t ranks[]={2,3,2,2,2,2,3,3,2,2,3,3,3,3,2};
    size_t shapes[][3]={{t,width,0},{t,h,d},{h,d,0},{h,d,0},{h,d,0},{ch,c->conv_kernel,0},
        {t,h,d},{t,h,d},{t,d,0},{t,h,0},{t,h,d},{t,h,d},{t,h,d},{t,h,d},{history,ch,0}};
    size_t bytes[15];
    for (size_t i=0;i<15;i++) {
        if (!dims(v[i],ranks[i],shapes[i])) return fail(e,Q38_SHAPE,"PLE view shape");
        code=dense(v[i],(__bridge id<MTLDevice>)cmd.device,bytes+i);
        if (code) return fail(e,code,"PLE view");
        if (overlap(v[i].span,bytes[i],cmd.error_word,4)) return fail(e,Q38_INVALID_ARGUMENT,"error alias");
        for (size_t j=0;j<i;j++) if (i>=6 && overlap(v[i].span,bytes[i],v[j].span,bytes[j]))
            return fail(e,Q38_INVALID_ARGUMENT,"PLE alias");
    }
    for (size_t i=0;i<2;i++) if (layouts[i].n_experts!=1 || layouts[i].k_logical!=width ||
        layouts[i].n_out!=(i ? d : ch)) return fail(e,Q38_SHAPE,"projection layout");
    /* Validate both live packed bindings before recording either product,
     * and protect every writable stage/history from resident weight aliases. */
    for (size_t i=0;i<2;i++) {
        Qwen38GpuF32 x=w->rows,y=i ? w->value : flat(w->key);
        x.dim[0]=y.dim[0]=0; x.span.bytes=y.span.bytes=0;
        code=q38_packed_encode(b,&cmd,store,bindings[i],layouts+i,0,x,y,e);
        if (code) return code;
        Qwen38GpuSpan weight={gpu_weights_buffer(store,bindings[i]),bindings[i]->offset,layouts[i].tensor_bytes};
        for (size_t j=6;j<15;j++) if (overlap(weight,weight.bytes,v[j].span,bytes[j]))
            return fail(e,Q38_INVALID_ARGUMENT,"PLE view aliases weights");
    }
    if (!t) return fail(e,Q38_OK,NULL);
    code=q38_packed_encode(b,&cmd,store,bindings[0],&layouts[0],0,w->rows,flat(w->key),e);
    if (!code) code=q38_packed_encode(b,&cmd,store,bindings[1],&layouts[1],0,w->rows,w->value,e);
    if (code) return code;
    [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    Q38CGpuSpec spec={.eps=c->rms_eps,.conv={ch,c->conv_kernel,c->order,true}};
#define OP(op,...) do { Qwen38GpuF32 vv[]={__VA_ARGS__}; code=q38c_gpu_encode(ops,cmd,Q38C_##op,&spec,vv,sizeof vv/sizeof *vv,e); if (code) return code; } while(0)
    OP(RMS,w->key,w->gamma_key,w->key);
    OP(RMS,w->residual,w->gamma_query,w->query);
    dispatch(g->gate,cmd,(Qwen38GpuF32[]){w->key,w->query,w->value,w->gate,w->u},5,
        (uint32_t[]){(uint32_t)t,(uint32_t)h,(uint32_t)d,0},t*h);
    OP(RMS,w->u,w->gamma_conv,w->z);
    OP(CONV,flat(w->z),w->taps,flat(w->conv),w->history);
#undef OP
    return q38e_add_encode(g,cmd,w->u,w->conv,w->delta,e);
}
Qwen38Status q38e_ngram_gpu_bind(Qwen38NgramState *s,Qwen38GpuSession *session,Qwen38Error *e) {
    if (!s || !session) return fail(e,Q38_INVALID_ARGUMENT,"GPU state arguments");
    Qwen38StateInfo g=qwen38_gpu_session_info(session);
    if (s->info.busy || g.busy) return fail(e,Q38_BUSY,"GPU state busy");
    if (s->gpu_session || !s->info.valid || !g.valid || s->info.next_pos || g.next_pos)
        return fail(e,Q38_BAD_STATE,"GPU bind requires fresh state");
    if (s->info.key.model!=g.key.model) return fail(e,Q38_OWNER,"GPU state model");
    if (s->info.key.epoch!=g.key.epoch) return fail(e,Q38_STALE,"GPU state epoch");
    s->gpu_session=session; return fail(e,Q38_OK,NULL);
}
Qwen38Status q38e_ngram_gpu_commit(Qwen38NgramJob *j,Qwen38Error *e) {
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (!j->state->gpu_session) return fail(e,Q38_OWNER,"not a GPU state");
    Qwen38StateInfo g=qwen38_gpu_session_info(j->state->gpu_session);
    if (g.busy) return fail(e,Q38_BUSY,"GPU append not completed");
    if (g.key.model!=j->key.model) return fail(e,Q38_OWNER,"GPU state model");
    if (g.key.epoch!=j->key.epoch) return fail(e,Q38_STALE,"GPU state epoch");
    if (!g.valid || g.next_pos!=j->position+j->count || j->phase!=(j->count ? Q38E_READY : Q38E_COMPUTED))
        return fail(e,Q38_BAD_STATE,"GPU append did not complete");
    j->phase=Q38E_COMPUTED;
    return qwen38_ngram_commit(j->state,j,e);
}
