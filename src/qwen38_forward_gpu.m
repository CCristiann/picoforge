/* Ordinary G2 graph. Operators stay in B/C/D/E/F; this file owns resources,
 * graph order, bounded observations and the one G0 transaction per append. */
#import <Metal/Metal.h>
#include "qwen38_forward_gpu.h"
#include "qwen38_g_internal.h"
#include "qwen38_e_internal.h"
#include "qwen38_d_internal.h"
#include "qwen38_f_internal.h"
#include "qwen38_b_internal.h"
#include "qwen38_ngram_internal.h"
#include "qwen38_ref.h"
#include <stdlib.h>
#include <time.h>
static double clock_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}

typedef struct { const GgufBinding *binding; Q38PackedLayout layout; } G2Weight;
typedef struct { G2Weight down,up,inject; Qwen38GpuF32 gain; } G2Hc;
typedef struct {
    Qwen38Layer kind;
    G2Hc attn,ffn;
    Q38CLayer gdn; void *c_constants;
    const GgufBinding *cb[Q38CW_COUNT]; Q38CGpuWork cw;
    G2Weight qsa[6]; Q38DGpuLayer dw; Qwen38AttentionConfig ac;
    const GgufBinding *fb[8]; Q38PackedLayout fl[8];
    Qwen38NgramStore *es; Qwen38NgramState *estate; Qwen38NgramJob *job;
    const Qwen38NgramConfig *nc; const GgufBinding *eb[2]; Q38PackedLayout el[2]; Q38EGpuWork ew;
} G2Layer;
typedef struct { uint32_t layer; Qwen38GraphStage stage; Qwen38GpuF32 view; } G2Trace;
struct Qwen38GpuForward {
    const Gguf *gguf; const Qwen38Config *config; const Qwen38Weights *weights; const GpuWeightStore *store;
    Qwen38ConfigView c; uint32_t max_tokens,capacity,max_logits_rows;
    uint64_t next_pos; /* published only after G0 AND all E commits succeed */
    void *device,*library,*buffers,*histories,*copy; size_t bytes,main_histories;
    bool valid,busy; Qwen38GpuSession *session;
    Q38PackedGpu *b; Q38CGpu *ops; Q38DGpu *d; Q38EGpu *e; Q38FRouted *moe;
    G2Weight embed,head; G2Hc final; G2Layer *layer;
    Qwen38GpuF32 r,normed,low,act,gate,il,inject,mixed,block,logits,tokens,c_arena;
    Q38DGpuLayer dw; Q38EGpuWork ew; Q38FRoutedWork fw;
    G2Trace *traces; size_t trace_count,trace_offset; Qwen38GpuSpan trace_span;
    Qwen38GpuForwardStats stats;
    bool mtp_enabled,mtp_failed; Qwen38GpuSession *mtp_session;
    /* T15c: gpu_argmax is read once at creation (`argmax` oracle: logits row + CPU scan). */
    bool gpu_argmax,want_id; void *argmax; Qwen38GpuF32 argmax_id; float *host_row;
    G2Weight mtp_eh; G2Hc mtp_head;
    Qwen38GpuF32 mtp_egain,mtp_hgain,mtp_en,mtp_hn,mtp_cat,mtp_proj,mtp_r;
    void *mtp_concat,*mtp_sum;
    uint64_t last_start,snapshot_serial,saved_start; size_t last_count,saved_count;
    void *snapshot; size_t snapshot_bytes;
    Qwen38GpuCheckpoint saved;
    Q38Profile *prof; const Qwen38GpuTicket *prof_ticket; bool prof_failed; /* R0.2, trunk appends only */
};
static Qwen38Status fail(Qwen38Error *e,Qwen38Status code,const char *detail) {
    if (e) { memset(e,0,sizeof *e); e->code=code;
        if (code) { snprintf(e->subject,sizeof e->subject,"qwen38/G2"); snprintf(e->detail,sizeof e->detail,"%s",detail); } }
    return code;
}
#define TRY(x) do { Qwen38Status rc_=(x); if (rc_) return rc_; } while(0)
static Qwen38GpuF32 shape(Qwen38GpuSpan span,size_t rank,const size_t *dims) {
    Qwen38GpuF32 v={.span=span,.rank=rank}; size_t pitch=4;
    for (size_t i=rank;i-->0;) { v.dim[i]=dims[i]; v.stride_bytes[i]=pitch; pitch*=dims[i]; }
    v.span.bytes=pitch; return v;
}
static Qwen38GpuF32 time_view(Qwen38GpuF32 v,size_t t) { v.dim[0]=t; v.span.bytes=t*v.stride_bytes[0]; return v; }
static Qwen38GpuF32 flat(Qwen38GpuF32 v) { return shape(v.span,2,(size_t[]){v.dim[0],v.stride_bytes[0]/4}); }
static Qwen38GpuF32 slice(Qwen38GpuF32 v,size_t start,size_t t) {
    v.span.offset_bytes+=start*v.stride_bytes[0]; return time_view(v,t);
}
static void *contents(Qwen38GpuF32 v) { return (char *)[(__bridge id<MTLBuffer>)v.span.buffer contents]+v.span.offset_bytes; }
static Qwen38Status alloc_view(Qwen38GpuForward *f,Qwen38GpuF32 *out,bool state,size_t rank,const size_t *dims,Qwen38Error *e) {
    size_t bytes=4;
    for (size_t i=0;i<rank;i++) if (dims[i]>UINT32_MAX || !q38c_mul(bytes,dims[i],&bytes)) return fail(e,Q38_OVERFLOW,"buffer shape");
    size_t physical=MAX(bytes,4);
    if (bytes/4>UINT32_MAX || physical>((size_t)4<<30)-f->bytes) return fail(e,Q38_CAPACITY,"graph buffers exceed 4 GiB");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)f->device;
    id<MTLBuffer> b=[dev newBufferWithLength:physical options:MTLResourceStorageModeShared];
    if (!b) return fail(e,Q38_OOM,"graph Metal buffer");
    [(__bridge NSMutableArray *)f->buffers addObject:b]; f->bytes+=physical;
    if (state) { memset(b.contents,0,physical); [(__bridge NSMutableArray *)f->histories addObject:b]; }
    *out=shape((Qwen38GpuSpan){(__bridge void *)b,0,bytes},rank,dims); return Q38_OK;
}
#define ALLOC(v,st,...) TRY(alloc_view(f,&(v),st,sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t),(size_t[]){__VA_ARGS__},e))
static Qwen38Status bind_weight(Qwen38GpuForward *f,uint32_t layer,Qwen38Role role,G2Weight *out,Qwen38Error *e) {
    const Qwen38Tensor *t; Qwen38CpuSpan span;
    TRY(qwen38_weights_get(f->weights,layer,role,&t,e));
    if (!t->gpu) return fail(e,Q38_MISSING,"nonresident graph weight");
    TRY(qwen38_weights_cpu_span(f->store,t->gpu,&span,e));
    out->binding=t->gpu; return q38_packed_layout_init(&t->storage,(size_t)t->logical_shape[0],&out->layout,e);
}
static Qwen38Status constant(Qwen38GpuForward *f,uint32_t layer,Qwen38Role role,Qwen38GpuF32 *out,
                            size_t rank,const size_t *dims,Qwen38Error *e) {
    G2Weight w; Q38PackedCpuView v; TRY(bind_weight(f,layer,role,&w,e));
    v.layout=w.layout; TRY(qwen38_weights_cpu_span(f->store,w.binding,&v.span,e));
    size_t n=1; for (size_t i=0;i<rank;i++) if (!q38c_mul(n,dims[i],&n)) return fail(e,Q38_OVERFLOW,"constant shape");
    if (w.layout.n_experts!=1 || n!=w.layout.n_out*w.layout.k_storage) return fail(e,Q38_SHAPE,"constant shape");
    TRY(alloc_view(f,out,false,rank,dims,e));
    for (size_t r=0;r<w.layout.n_out;r++) TRY(q38_packed_decode_row(&v,0,r,(float *)contents(*out)+r*w.layout.k_storage,w.layout.k_storage,e));
    return Q38_OK;
}
static const Qwen38Role attn_roles[]={Q38_HC_ATTN_NORM,Q38_HC_ATTN_DOWN,Q38_HC_ATTN_UP,Q38_HC_ATTN_INJECT};
static const Qwen38Role ffn_roles[]={Q38_HC_FFN_NORM,Q38_HC_FFN_DOWN,Q38_HC_FFN_UP,Q38_HC_FFN_INJECT};
static const Qwen38Role final_roles[]={Q38_OUTPUT_HC_NORM,Q38_OUTPUT_HC_DOWN,Q38_OUTPUT_HC_UP,Q38_ROLE_COUNT};
static Qwen38Status hc_bind(Qwen38GpuForward *f,uint32_t i,const Qwen38Role roles[4],G2Hc *hc,Qwen38Error *e) {
    TRY(constant(f,i,roles[0],&hc->gain,2,(size_t[]){f->c.hc.streams,f->c.hidden_size},e));
    TRY(bind_weight(f,i,roles[1],&hc->down,e)); TRY(bind_weight(f,i,roles[2],&hc->up,e));
    return roles[3]==Q38_ROLE_COUNT ? Q38_OK : bind_weight(f,i,roles[3],&hc->inject,e);
}
static Qwen38Status create_gdn(Qwen38GpuForward *f,uint32_t i,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; TRY(q38c_layer_plan(f->config,f->weights,i,&l->gdn,e));
    Qwen38CpuSpan spans[Q38CW_COUNT];
    for (size_t j=0;j<Q38CW_COUNT;j++) { G2Weight w; TRY(bind_weight(f,i,q38c_layer_roles[j],&w,e)); l->cb[j]=w.binding;
        TRY(qwen38_weights_cpu_span(f->store,w.binding,spans+j,e)); }
    size_t cb=l->gdn.constant_bytes;
    l->c_constants=aligned_alloc(64,(cb+63)&~(size_t)63);
    if (!l->c_constants) return fail(e,Q38_OOM,"GDN constants");
    TRY(q38c_layer_bind(&l->gdn,spans,(Qwen38Memory){l->c_constants,cb},e));
    const size_t slots[]={Q38CW_HN,Q38CW_A,Q38CW_DT,Q38CW_CONV,Q38CW_GN};
    for (size_t j=0;j<5;j++) { size_t k=slots[j]; Qwen38F32 v=l->gdn.constant[k];
        TRY(alloc_view(f,&l->cw.constant[k],false,v.rank,v.dim,e));
        memcpy(contents(l->cw.constant[k]),v.data,q38c_span(v)); }
    l->cw.constant[Q38CW_HN]=shape(l->cw.constant[Q38CW_HN].span,2,
        (size_t[]){f->c.hc.streams,f->c.hidden_size});
    size_t need; TRY(q38c_work_size(&l->gdn,f->max_tokens,&need,e));
    if (!f->c_arena.span.buffer) ALLOC(f->c_arena,false,need/4);
    Q38CWork work;
    TRY(q38c_work_init(&l->gdn,f->max_tokens,(Qwen38Memory){contents(f->c_arena),f->c_arena.span.bytes},&work,e));
    for (size_t j=0;j<Q38CT_COUNT;j++) {
        Qwen38F32Mut v=work.trace[j]; Qwen38GpuSpan s=f->c_arena.span;
        s.offset_bytes+=(size_t)((char *)v.data-(char *)contents(f->c_arena));
        l->cw.trace[j]=shape(s,v.rank,v.dim);
    }
    ALLOC(l->cw.history,true,l->gdn.conv.kernel-1,l->gdn.conv.channels);
    ALLOC(l->cw.state,true,f->c.gdn.value_heads,f->c.gdn.value_dim,f->c.gdn.key_dim);
    return Q38_OK;
}
static Qwen38Status create_qsa(Qwen38GpuForward *f,uint32_t i,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; l->ac=f->c.attention; l->ac.ratio=l->kind.compress_ratio;
    Qwen38AttentionPlan plan; TRY(qwen38_attention_plan(&l->ac,f->capacity,f->max_tokens,&plan,e));
    TRY(hc_bind(f,i,attn_roles,&l->attn,e));
    const Qwen38Role roles[]={Q38_ATTN_QG,Q38_ATTN_K,Q38_ATTN_V,Q38_INDEX_Q,Q38_INDEX_K,Q38_ATTN_OUT};
    for (size_t j=0;j<6;j++) TRY(bind_weight(f,i,roles[j],&l->qsa[j],e));
    const Qwen38Role gains[]={Q38_ATTN_Q_NORM,Q38_ATTN_K_NORM,Q38_INDEX_Q_NORM,Q38_INDEX_K_NORM};
    Qwen38GpuF32 *dst[]={&l->dw.gamma_q,&l->dw.gamma_k,&l->dw.gamma_iq,&l->dw.gamma_ik};
    for (size_t j=0;j<4;j++) TRY(constant(f,i,gains[j],dst[j],1,(size_t[]){j<2 ? l->ac.head_dim : l->ac.index_dim},e));
    ALLOC(l->dw.key_cache,true,f->capacity,l->ac.kv_heads,l->ac.head_dim);
    ALLOC(l->dw.value_cache,true,f->capacity,l->ac.kv_heads,l->ac.head_dim);
    ALLOC(l->dw.index_cache,true,f->capacity,l->ac.index_dim);
    ALLOC(l->dw.block_cache,true,f->capacity/l->ac.ratio,l->ac.index_dim);
    return Q38_OK;
}
static Qwen38Status create_ple(Qwen38GpuForward *f,uint32_t i,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; TRY(qwen38_config_ngram(f->config,i,&l->nc,e));
    const Qwen38NgramConfig *c=l->nc; size_t h=c->streams,d=c->hidden_size;
    const Qwen38Role roles[]={Q38_PLE_KEY,Q38_PLE_VALUE,Q38_PLE_KEY_NORM,Q38_PLE_QUERY_NORM,Q38_PLE_CONV_NORM,Q38_PLE_CONV};
    for (size_t j=0;j<2;j++) { G2Weight w; TRY(bind_weight(f,i,roles[j],&w,e)); l->eb[j]=w.binding; l->el[j]=w.layout; }
    Qwen38GpuF32 *gains[]={&l->ew.gamma_key,&l->ew.gamma_query,&l->ew.gamma_conv,&l->ew.taps};
    for (size_t j=0;j<4;j++) TRY(constant(f,i,roles[j+2],gains[j],2,(size_t[]){j<3 ? h : h*d,j<3 ? d : c->conv_kernel},e));
    ALLOC(l->ew.rows,false,f->max_tokens,(size_t)c->head_count*c->head_dim);
    ALLOC(l->ew.history,true,(c->conv_kernel-1)*c->order,h*d);
    TRY(qwen38_ngram_open(f->gguf,c,f->max_tokens,32u<<20,&l->es,e));
    TRY(qwen38_ngram_state_create(l->es,(Qwen38StateKey){f->config,i,1},&l->estate,e));
    TRY(q38e_ngram_gpu_bind(l->estate,f->session,e));
    if (!f->ew.key.span.buffer) {
        size_t t=f->max_tokens;
        ALLOC(f->ew.key,false,t,h,d); ALLOC(f->ew.query,false,t,h,d);
        ALLOC(f->ew.value,false,t,d); ALLOC(f->ew.gate,false,t,h);
        ALLOC(f->ew.u,false,t,h,d); ALLOC(f->ew.z,false,t,h,d);
        ALLOC(f->ew.conv,false,t,h,d); ALLOC(f->ew.delta,false,t,h,d);
    }
    return Q38_OK;
}
static Qwen38Status scratch(Qwen38GpuForward *f,Qwen38Error *e) {
    size_t t=f->max_tokens,h=f->c.hc.streams,d=f->c.hidden_size,rank=f->c.hc.low_rank;
    ALLOC(f->r,false,t,h,d); ALLOC(f->normed,false,t,h,d); ALLOC(f->gate,false,t,h,d);
    ALLOC(f->low,false,t,rank); ALLOC(f->act,false,t,rank); ALLOC(f->il,false,t,h);
    ALLOC(f->inject,false,t,h); ALLOC(f->mixed,false,t,d); ALLOC(f->block,false,t,d);
    ALLOC(f->tokens,false,t); ALLOC(f->logits,false,f->max_logits_rows,f->c.vocab_size); ALLOC(f->argmax_id,false,1);
    const Qwen38AttentionConfig *a=&f->c.attention;
    ALLOC(f->dw.qg,false,t,a->heads,2,a->head_dim);
    ALLOC(f->dw.k,false,t,a->kv_heads,a->head_dim); ALLOC(f->dw.v,false,t,a->kv_heads,a->head_dim);
    ALLOC(f->dw.iq,false,t,a->index_heads,a->index_dim); ALLOC(f->dw.ik,false,t,a->index_dim);
    ALLOC(f->dw.qn,false,t,a->heads,a->head_dim); ALLOC(f->dw.qr,false,t,a->heads,a->head_dim);
    ALLOC(f->dw.gate,false,t,a->heads,a->head_dim); ALLOC(f->dw.kn,false,t,a->kv_heads,a->head_dim);
    ALLOC(f->dw.iqn,false,t,a->index_heads,a->index_dim); ALLOC(f->dw.iqr,false,t,a->index_heads,a->index_dim);
    ALLOC(f->dw.pooled,false,1,1,a->index_dim); ALLOC(f->dw.pooled_n,false,1,1,a->index_dim);
    ALLOC(f->dw.scores,false,f->capacity); ALLOC(f->dw.out,false,t,a->heads,a->head_dim);
    /* D admits budget + ratio - 1 IDs even when the context is smaller than
     * one pooling block. Capacity alone underallocates that tail at 1/2. */
    size_t tail=f->capacity;
    for (uint32_t i=0;i<f->c.blocks;i++) {
        const Qwen38Layer *kind; TRY(qwen38_config_layer(f->config,i,&kind,e));
        if (kind->kind==Q38_LAYER_QSA) tail=MAX(tail,(size_t)kind->compress_ratio-1);
    }
    Qwen38GpuF32 ids,keep; ALLOC(ids,false,(size_t)a->token_budget+tail); ALLOC(keep,false,f->capacity);
    f->dw.ids=ids.span; f->dw.keep=keep.span;
    Qwen38GpuF32 part; ALLOC(part,false,Q38D_MAX_SPLITS,a->heads,a->head_dim+2); f->dw.part=part.span;
    const Qwen38MoeConfig *m=&f->c.moe; size_t mt=MIN(t,32),p=mt*m->top_k;
    ALLOC(f->fw.logits,false,mt,m->experts); ALLOC(f->fw.probs,false,mt,m->experts); ALLOC(f->fw.weights,false,mt,m->top_k);
    ALLOC(f->fw.gate,false,p,m->intermediate); ALLOC(f->fw.up,false,p,m->intermediate); ALLOC(f->fw.act,false,p,m->intermediate);
    ALLOC(f->fw.down,false,p,d); ALLOC(f->fw.sgate,false,mt,m->shared_intermediate);
    ALLOC(f->fw.sup,false,mt,m->shared_intermediate); ALLOC(f->fw.sact,false,mt,m->shared_intermediate);
    ALLOC(f->fw.shared,false,mt,d); ALLOC(f->fw.slogit,false,mt,1); ALLOC(f->fw.ssig,false,mt,1); ALLOC(f->fw.routed,false,mt,d);
    ALLOC(ids,false,p); ALLOC(keep,false,p); f->fw.ids=ids.span; f->fw.identity=keep.span;
    return Q38_OK;
}
/* T15c: greedy id on the GPU, one threadgroup over the logits row. The idea, not the
 * code: ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_argmax. Ties go to the FIRST
 * index; any non-finite logit sets error bit 1. */
static void *argmax_pipeline(void *device,void *library) {
    id<MTLFunction> fn=[(__bridge id<MTLLibrary>)library newFunctionWithName:@"q38g_argmax"];
    id<MTLComputePipelineState> p=fn ? [(__bridge id<MTLDevice>)device newComputePipelineStateWithFunction:fn error:NULL] : nil;
    return p && p.maxTotalThreadsPerThreadgroup>=1024 ? (__bridge_retained void *)p : NULL;
}
static void argmax_encode(void *pipeline,Qwen38GpuCommand cmd,Qwen38GpuSpan x,Qwen38GpuSpan out,uint32_t n) {
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
    Qwen38GpuSpan s[3]={x,out,cmd.error_word};
    [enc setBuffer:(__bridge id<MTLBuffer>)s[0].buffer offset:s[0].offset_bytes atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)s[1].buffer offset:s[1].offset_bytes atIndex:1];
    [enc setBytes:&n length:sizeof n atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)s[2].buffer offset:s[2].offset_bytes atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
}
/* Test seam: the kernel alone on one row, outside any graph or session. */
Qwen38Status q38g_argmax_probe(void *device,void *library,const float *row,uint32_t n,int32_t *winner,
                               uint32_t *error_bits,Qwen38Error *e) {
    @autoreleasepool {
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    void *pipeline=argmax_pipeline(device,library); if (!pipeline) return fail(e,Q38_GPU_ERROR,"argmax pipeline");
    id<MTLBuffer> x=[dev newBufferWithBytes:row length:MAX((size_t)n*4,4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> err=[dev newBufferWithLength:8 options:MTLResourceStorageModeShared]; memset(err.contents,0,8);
    id<MTLCommandBuffer> cb=[[dev newCommandQueue] commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    Qwen38GpuCommand cmd={.device=device,.compute_encoder=(__bridge void *)enc,.error_word={(__bridge void *)err,0,8}};
    argmax_encode(pipeline,cmd,(Qwen38GpuSpan){(__bridge void *)x,0,(size_t)n*4},(Qwen38GpuSpan){(__bridge void *)out,0,4},n);
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; CFRelease(pipeline);
    if (cb.error) return fail(e,Q38_GPU_ERROR,"argmax probe");
    *winner=*(int32_t *)out.contents; *error_bits=*(uint32_t *)err.contents; return fail(e,Q38_OK,NULL);
    }
}
/* R1.7b: weights stay resident on the session's queue. UNSUPPORTED (old OS, `claim`
 * oracle) is not an error: the engine runs the same without it. */
static Qwen38Status residency(Qwen38GpuForward *f,Qwen38GpuSession *session,Qwen38Error *e) {
    void *shared=f->session!=session ? qwen38_gpu_session_residency(f->session) : NULL;
    if (f->session!=session) {                      /* MTP queue: the main set, or none */
        Qwen38Status rc=shared ? qwen38_gpu_session_share_residency(session,shared,e) : Q38_OK;
        return rc==Q38_UNSUPPORTED ? Q38_OK : rc;
    }
    const GgufPlan *plan=gpu_weights_plan(f->store);
    void **views=calloc(plan->view_count,sizeof *views); if (!views) return fail(e,Q38_OOM,"residency list");
    size_t n=0;
    for (size_t i=0;i<plan->weight_count;i++) {     /* one buffer per view, however many tensors */
        void *b=gpu_weights_buffer(f->store,&plan->weights[i]); bool seen=false;
        for (size_t k=0;k<n&&!seen;k++) seen=views[k]==b;
        if (!seen) views[n++]=b;
    }
    Qwen38Status rc=n ? qwen38_gpu_session_use_residency(session,views,n,4ull<<30,e) : Q38_OK; free(views);
    return rc==Q38_UNSUPPORTED ? Q38_OK : rc;
}
static Qwen38Status create(Qwen38GpuForward *f,void *library,Qwen38Error *e) {
    TRY(qwen38_gpu_session_create(f->device,f->config,f->capacity,&f->session,e));
    TRY(residency(f,f->session,e));
    TRY(q38_packed_gpu_create(f->device,library,&f->b,e)); TRY(q38c_gpu_create(f->device,library,&f->ops,e));
    TRY(q38d_gpu_create(f->device,library,&f->d,e)); TRY(q38e_gpu_create(f->device,library,&f->e,e));
    TRY(q38f_routed_create(f->device,library,&f->moe,e));
    id<MTLFunction> fn=[(__bridge id<MTLLibrary>)library newFunctionWithName:@"q38g_copy"];
    id<MTLComputePipelineState> p=fn ? [(__bridge id<MTLDevice>)f->device newComputePipelineStateWithFunction:fn error:NULL] : nil;
    if (!p) return fail(e,Q38_GPU_ERROR,"graph copy pipeline"); f->copy=(__bridge_retained void *)p;
    f->gpu_argmax=!q38_ref_kernels("argmax");  /* oracle mode builds no argmax pipeline (as 360acf0) */
    if (f->gpu_argmax && !(f->argmax=argmax_pipeline(f->device,library))) return fail(e,Q38_GPU_ERROR,"graph argmax pipeline");
    TRY(bind_weight(f,QWEN38_GLOBAL_LAYER,Q38_TOKEN_EMBD,&f->embed,e));
    TRY(bind_weight(f,QWEN38_GLOBAL_LAYER,Q38_LM_HEAD,&f->head,e));
    if (f->embed.layout.k_logical!=f->c.hidden_size || f->embed.layout.n_out!=f->c.vocab_size ||
        f->head.layout.k_logical!=f->c.hidden_size || f->head.layout.n_out!=f->c.vocab_size)
        return fail(e,Q38_SHAPE,"embedding/head config");
    TRY(hc_bind(f,QWEN38_GLOBAL_LAYER,final_roles,&f->final,e));
    f->layer=calloc(f->c.blocks,sizeof *f->layer); f->traces=calloc(2+9*f->c.trunk_blocks,sizeof *f->traces);
    if (!f->layer || !f->traces) return fail(e,Q38_OOM,"graph metadata");
    TRY(scratch(f,e));
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) {
        G2Layer *l=&f->layer[i]; const Qwen38Layer *kind; TRY(qwen38_config_layer(f->config,i,&kind,e)); l->kind=*kind;
        if (kind->domain!=Q38_TRUNK) return fail(e,Q38_UNSUPPORTED,"nontrunk layer");
        if (kind->kind==Q38_LAYER_GDN) TRY(create_gdn(f,i,e)); else TRY(create_qsa(f,i,e));
        TRY(hc_bind(f,i,ffn_roles,&l->ffn,e));
        const Qwen38Role roles[]={Q38_MOE_ROUTER,Q38_MOE_GATE,Q38_MOE_UP,Q38_MOE_DOWN,Q38_SHARED_GATE,Q38_SHARED_UP,Q38_SHARED_DOWN,Q38_SHARED_ROUTER};
        for (size_t j=0;j<8;j++) { G2Weight w; TRY(bind_weight(f,i,roles[j],&w,e)); l->fb[j]=w.binding; l->fl[j]=w.layout; }
        if (kind->has_ple) TRY(create_ple(f,i,e));
    }
    f->main_histories=[(__bridge NSArray *)f->histories count];
    return Q38_OK;
}
Qwen38Status qwen38_gpu_forward_create(const Gguf *g,const Qwen38Config *config,const Qwen38Weights *weights,
    const GpuWeightStore *store,void *device,void *library,uint32_t max_tokens,uint32_t capacity,
    uint32_t max_logits_rows,Qwen38GpuForward **out,Qwen38Error *e) {
    if (out) *out=NULL;
    if (!g || !config || !weights || !store || !device || !library || !out) return fail(e,Q38_INVALID_ARGUMENT,"graph arguments");
    const Qwen38ConfigView *c=qwen38_config_view(config);
    if (!c || !max_tokens || max_tokens>1024 || max_tokens>capacity || capacity>32768 || capacity>c->max_positions || max_logits_rows>max_tokens)
        return fail(e,Q38_CAPACITY,"graph chunk/context/logits capacities");
    Qwen38GpuForward *f=calloc(1,sizeof *f); if (!f) return fail(e,Q38_OOM,"graph allocation");
    f->gguf=g; f->config=config; f->weights=weights; f->store=store; f->c=*c; f->max_tokens=max_tokens;
    f->capacity=capacity; f->max_logits_rows=max_logits_rows; f->valid=true;
    f->device=(__bridge_retained void *)(__bridge id<MTLDevice>)device;
    f->library=(__bridge_retained void *)(__bridge id<MTLLibrary>)library;
    f->buffers=(__bridge_retained void *)[NSMutableArray new]; f->histories=(__bridge_retained void *)[NSMutableArray new];
    Qwen38Status rc=create(f,library,e); if (rc) { qwen38_gpu_forward_destroy(f); return rc; }
    *out=f; return fail(e,Q38_OK,NULL);
}
size_t qwen38_gpu_forward_bytes(const Qwen38GpuForward *f) { return f ? f->bytes : 0; }
Qwen38GpuF32 q38g_gpu_state(Qwen38GpuForward *f,uint32_t layer,uint32_t slot) {
    if (!f || f->busy || layer>=f->c.trunk_blocks+(f->mtp_enabled ? 1u : 0u) || slot>=Q38GS_COUNT) return (Qwen38GpuF32){0};
    G2Layer *l=&f->layer[layer];
    Qwen38GpuF32 v[]={l->cw.history,l->cw.state,l->dw.key_cache,l->dw.value_cache,l->dw.index_cache,l->dw.block_cache,l->ew.history};
    return v[slot];
}
Qwen38StateInfo q38g_gpu_ngram_info(const Qwen38GpuForward *f,uint32_t layer) {
    if (!f || f->busy || layer>=f->c.trunk_blocks) return (Qwen38StateInfo){0};
    return qwen38_ngram_state_info(f->layer[layer].estate);
}
Qwen38StateInfo qwen38_gpu_forward_info(const Qwen38GpuForward *f) {
    if (!f) return (Qwen38StateInfo){0};
    Qwen38StateInfo info=qwen38_gpu_session_info(f->session); info.valid &= f->valid; info.busy |= f->busy;
    info.next_pos=f->next_pos; return info;
}
Qwen38GpuForwardStats qwen38_gpu_forward_stats(const Qwen38GpuForward *f) {
    return f ? f->stats : (Qwen38GpuForwardStats){0};
}
void qwen38_gpu_forward_destroy(Qwen38GpuForward *f) {
    if (!f || f->busy) return;
    for (uint32_t i=0;f->layer && i<f->c.trunk_blocks;i++) {
        G2Layer *l=&f->layer[i];
        if (l->job) { qwen38_ngram_abort(l->job,NULL); qwen38_ngram_job_release(l->job,NULL); }
        if (l->estate) qwen38_ngram_state_destroy(l->estate,NULL);
        if (l->es) qwen38_ngram_close(l->es,NULL);
        free(l->c_constants);
    }
    q38_packed_gpu_free(f->b); q38c_gpu_destroy(f->ops); q38d_gpu_destroy(f->d); q38e_gpu_destroy(f->e); q38f_routed_destroy(f->moe);
    if (f->session) qwen38_gpu_session_destroy(f->session,NULL);
    if (f->mtp_session) qwen38_gpu_session_destroy(f->mtp_session,NULL);
    if (f->mtp_concat) CFRelease(f->mtp_concat);
    if (f->mtp_sum) CFRelease(f->mtp_sum);
    if (f->library) CFRelease(f->library);
    if (f->copy) CFRelease(f->copy);
    if (f->argmax) CFRelease(f->argmax);
    free(f->host_row);
    if (f->buffers) CFRelease(f->buffers);
    if (f->histories) CFRelease(f->histories);
    if (f->device) CFRelease(f->device);
    free(f->snapshot); free(f->layer); free(f->traces); free(f);
}

static void barrier(Qwen38GpuCommand cmd) { [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
static void copy(Qwen38GpuForward *f,Qwen38GpuCommand cmd,Qwen38GpuF32 src,Qwen38GpuF32 dst,bool repeat) {
    uint32_t a[4]={(uint32_t)(dst.span.bytes/4),f->c.hidden_size,(uint32_t)f->c.hc.streams,repeat};
    if (!a[0]) return;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)f->copy;
    [enc setComputePipelineState:p];
    [enc setBuffer:(__bridge id<MTLBuffer>)src.span.buffer offset:src.span.offset_bytes atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)dst.span.buffer offset:dst.span.offset_bytes atIndex:1];
    [enc setBytes:a length:sizeof a atIndex:2];
    [enc dispatchThreads:MTLSizeMake(a[0],1,1) threadsPerThreadgroup:MTLSizeMake(MIN(a[0],p.maxTotalThreadsPerThreadgroup),1,1)];
    barrier(cmd);
}
static void emit(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t layer,Qwen38GraphStage stage,Qwen38GpuF32 src) {
    if (!f->trace_span.buffer) return;
    Qwen38GpuSpan s=f->trace_span; s.offset_bytes=f->trace_offset;
    Qwen38GpuF32 dst=shape(s,src.rank,src.dim);
    copy(f,cmd,src,dst,false);
    f->traces[f->trace_count++]=(G2Trace){layer,stage,dst}; f->trace_offset+=dst.span.bytes;
}
Qwen38Status q38g_forward_set_profile(Qwen38GpuForward *f,Q38Profile *p,Qwen38Error *e) {
    if (!f) return fail(e,Q38_INVALID_ARGUMENT,"null graph");
    if (f->busy) return fail(e,Q38_BUSY,"graph busy");
    f->prof=p; return fail(e,Q38_OK,NULL);
}
void q38g_profile_mark(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,uint32_t layer,const char *op,uint32_t type,uint64_t bytes) {
    if (!f->prof_ticket) return;
    void *pass=q38_profile_group(f->prof,layer,op,type,bytes);
    /* R0.3: the session checks the encoder by identity, so the split sees the
     * real one; the graph then encodes through the counting proxy. */
    cmd->compute_encoder=q38_profile_unwrap(f->prof,cmd->compute_encoder);
    if (!pass || q38g_session_split(f->session,f->prof_ticket,pass,cmd,NULL)) f->prof_failed=true;
    else cmd->compute_encoder=q38_profile_wrap(f->prof,cmd->compute_encoder);
}
/* Arguments are evaluated only in profile mode. */
#define MARK(layer,op,type,bytes) do { if (f->prof_ticket) q38g_profile_mark(f,cmd,layer,op,type,bytes); } while(0)
static Qwen38Status linear(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,uint32_t layer,const char *op,G2Weight w,Qwen38GpuF32 x,Qwen38GpuF32 y,Qwen38Error *e) {
    MARK(layer,op,w.binding->tensor.type,w.binding->tensor.bytes);   /* nominal: the whole matrix once */
    TRY(q38_packed_encode(f->b,cmd,f->store,w.binding,&w.layout,0,flat(x),flat(y),e)); barrier(*cmd); return Q38_OK;
}
/* R1.4: n projections of the same x in one multi call (one dispatch per
 * format), one profile group and one barrier for all of them. */
static Qwen38Status linears(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,uint32_t layer,const char *op,const G2Weight *w,size_t n,
        Qwen38GpuF32 x,const Qwen38GpuF32 *y,Qwen38Error *e) {
    const GgufBinding *b[Q38_MULTI_MAX]; Q38PackedLayout l[Q38_MULTI_MAX]; Qwen38GpuF32 out[Q38_MULTI_MAX];
    uint32_t type=w[0].binding->tensor.type; uint64_t bytes=0;
    if (n>Q38_MULTI_MAX) return fail(e,Q38_INVALID_ARGUMENT,"too many projections");
    for (size_t j=0;j<n;j++) {
        b[j]=w[j].binding; l[j]=w[j].layout; out[j]=flat(y[j]); bytes+=b[j]->tensor.bytes;
        if (b[j]->tensor.type!=type) type=Q38_PROFILE_MIXED;
    }
    MARK(layer,op,type,bytes);
    TRY(q38_packed_encode_multi(f->b,cmd,f->store,b,l,n,flat(x),out,e)); barrier(*cmd); return Q38_OK;
}
/* Profile group names of one HC read, in encode order; C ops keep their Q38C_ name. */
enum { HC_ATTN,HC_FFN,HC_FINAL };
static const char *const hc_ops[3][7]={
    {"hc_attn.rms","hc_attn.down","hc_attn.hc_low","hc_attn.up","hc_attn.hc_read","hc_attn.inject","hc_attn.hc_inject"},
    {"hc_ffn.rms","hc_ffn.down","hc_ffn.hc_low","hc_ffn.up","hc_ffn.hc_read","hc_ffn.inject","hc_ffn.hc_inject"},
    {"final.rms","final.down","final.hc_low","final.up","final.hc_read","final.inject","final.hc_inject"}};
static Qwen38Status hc_read(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,uint32_t layer,const G2Hc *hc,int kind,size_t t,Qwen38Error *e) {
    Q38CGpuSpec spec={.hc=f->c.hc,.eps=f->c.hc.rms_eps}; const char *const *n=hc_ops[kind];
#define V(x) time_view(f->x,t)
#define OP(k,op,...) do { MARK(layer,n[k],Q38_PROFILE_NONE,0); Qwen38GpuF32 vv[]={__VA_ARGS__}; \
    TRY(q38c_gpu_encode(f->ops,*cmd,Q38C_##op,&spec,vv,sizeof vv/sizeof *vv,e)); } while(0)
    OP(0,RMS,V(r),hc->gain,V(normed));
    /* R2.5: down and inject read the same normed rows: one multi call (bit-identical
     * to two single calls), profiled as the down group; the lone inject matvec
     * (4 rows) ran at ~10 GB/s. */
    if (hc->inject.binding)
        TRY(linears(f,cmd,layer,n[1],(G2Weight[]){hc->down,hc->inject},2,V(normed),(Qwen38GpuF32[]){V(low),V(il)},e));
    else TRY(linear(f,cmd,layer,n[1],hc->down,V(normed),V(low),e));
    { /* R2.6: low, up, read and inject in one dispatch, bit-identical; the sequence when unsupported */
        MARK(layer,n[3],hc->up.binding->tensor.type,hc->up.binding->tensor.bytes);
        bool inj=hc->inject.binding!=NULL; const Qwen38GpuF32 none={0};
        Qwen38Status status=q38_packed_encode_hc_tail(f->b,cmd,f->store,hc->up.binding,&hc->up.layout,f->c.hc.streams,
            flat(V(low)),flat(V(normed)),inj ? flat(V(il)) : none,flat(V(act)),flat(V(gate)),flat(V(mixed)),
            inj ? flat(V(inject)) : none,e);
        if (!status) { barrier(*cmd); return Q38_OK; }
        if (status!=Q38_UNSUPPORTED) return status;
    }
    OP(2,HC_LOW,V(low),V(act));
    TRY(linear(f,cmd,layer,n[3],hc->up,V(act),V(gate),e));
    OP(4,HC_READ,V(normed),V(gate),V(mixed));
    if (hc->inject.binding) OP(6,HC_INJECT,V(il),V(inject));
#undef OP
#undef V
    return Q38_OK;
}
static Qwen38Status hc_add(Qwen38GpuForward *f,Qwen38GpuCommand cmd,size_t t,Qwen38Error *e) {
    Q38CGpuSpec spec={.hc=f->c.hc};
    Qwen38GpuF32 v[]={time_view(f->r,t),time_view(f->block,t),time_view(f->inject,t)};
    return q38c_gpu_encode(f->ops,cmd,Q38C_HC_ADD,&spec,v,3,e);
}
static Qwen38Status ple(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; Q38EGpuWork w=l->ew; w.rows=time_view(w.rows,t); w.residual=time_view(f->r,t);
#define W(x) w.x=time_view(f->ew.x,t)
    W(key); W(query); W(value); W(gate); W(u); W(z); W(conv); W(delta);
#undef W
    TRY(q38e_ple_encode(f->e,f->b,f->ops,cmd,l->nc,f->store,l->eb,l->el,&w,e));
    emit(f,cmd,i,Q38G_PLE_DELTA,w.delta);
    return q38e_add_encode(f->e,cmd,w.residual,w.delta,w.residual,e);
}
typedef struct { Qwen38GpuForward *f; uint32_t layer; } GdnMark;
static void gdn_mark(void *ctx,Qwen38GpuCommand *cmd,const char *op,uint32_t type,uint64_t bytes) {
    GdnMark *m=ctx; q38g_profile_mark(m->f,cmd,m->layer,op,type,bytes);
}
static Qwen38Status gdn(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; Q38CGpuWork w=l->cw; w.residual=time_view(f->r,t);
    GdnMark mark={f,i}; if (f->prof_ticket) { w.mark=gdn_mark; w.mark_ctx=&mark; }   /* R1.8: sub-groups, profile only */
    for (size_t j=0;j<Q38CT_COUNT;j++) w.trace[j]=time_view(w.trace[j],t);
    TRY(q38c_block_encode(f->ops,f->b,cmd,&l->gdn,f->store,l->cb,&w,e));
    emit(f,cmd,i,Q38G_ATTN_MIXED,w.trace[Q38CT_MIX]); emit(f,cmd,i,Q38G_ATTN_INJECT,w.trace[Q38CT_INJECT]);
    emit(f,cmd,i,Q38G_ATTN_CONTRIBUTION,w.trace[Q38CT_OUT]); return Q38_OK;
}
static Qwen38Status qsa(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,uint32_t i,size_t t,uint32_t pos,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; Q38DGpuLayer w=l->dw;
#define W(x) w.x=time_view(f->dw.x,t)
    W(qg); W(k); W(v); W(iq); W(ik); W(qn); W(qr); W(gate); W(kn); W(iqn); W(iqr); W(out);
#undef W
    w.pooled=f->dw.pooled; w.pooled_n=f->dw.pooled_n;
    w.scores=time_view(f->dw.scores,f->capacity/l->ac.ratio); w.keep=f->dw.keep; w.ids=f->dw.ids; w.part=f->dw.part;
    TRY(hc_read(f,cmd,i,&l->attn,HC_ATTN,t,e));
    Qwen38GpuF32 outputs[]={w.qg,w.k,w.v,w.iq,w.ik};
    TRY(linears(f,cmd,i,"qsa.proj",l->qsa,5,time_view(f->mixed,t),outputs,e));
    MARK(i,"qsa.core",Q38_PROFILE_MIXED,0); TRY(q38d_layer_encode(f->d,f->ops,*cmd,&l->ac,pos,(uint32_t)t,&w,e));
    TRY(linear(f,cmd,i,"qsa.out",l->qsa[5],w.out,time_view(f->block,t),e));
    emit(f,*cmd,i,Q38G_ATTN_MIXED,time_view(f->mixed,t)); emit(f,*cmd,i,Q38G_ATTN_INJECT,time_view(f->inject,t));
    emit(f,*cmd,i,Q38G_ATTN_CONTRIBUTION,time_view(f->block,t)); MARK(i,"hc_add",Q38_PROFILE_NONE,0); return hc_add(f,*cmd,t,e);
}
typedef struct { Qwen38GpuForward *f; uint32_t layer; } MoeMark;
static void moe_mark(void *ctx,Qwen38GpuCommand *cmd,const char *op,uint32_t type,uint64_t bytes) {
    MoeMark *m=ctx; q38g_profile_mark(m->f,cmd,m->layer,op,type,bytes);
}
static Qwen38Status moe(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,Qwen38Error *e) {
    G2Layer *l=&f->layer[i];
    Qwen38MoeConfig mc=f->c.moe;
    if (i>=f->c.trunk_blocks) mc.down_physical=mc.intermediate;
    MoeMark mark={f,i};
    for (size_t start=0;start<t;start+=32) {
        size_t n=MIN(t-start,32),p=n*f->c.moe.top_k; Q38FRoutedWork w=f->fw;
#define W(x,rows) w.x=time_view(w.x,rows)
        W(logits,n); W(probs,n); W(weights,n); W(gate,p); W(up,p); W(act,p); W(down,p);
        W(sgate,n); W(sup,n); W(sact,n); W(shared,n); W(slogit,n); W(ssig,n); W(routed,n);
#undef W
        if (f->prof_ticket) { w.mark=moe_mark; w.mark_ctx=&mark; }   /* R1.9a: sub-groups, profile only */
        TRY(q38f_routed_encode(f->moe,f->b,f->ops,cmd,&mc,f->store,l->fb,l->fl,
            slice(f->mixed,start,n),slice(f->block,start,n),&w,e));
        if (!cmd.concurrent) barrier(cmd);   /* R1.6b: routed_encode already ends with one */
    }
    return Q38_OK;
}
/* R0.3 weight bytes of the composite groups: the tensors their kernels read in
 * this append. gdn is split in sub-groups inside q38c_block_encode (R1.8); ple: key/value plus the staged F32 n-gram rows; moe is split in sub-groups
 * inside q38f_routed_encode (R1.9a): router and shared weights once per
 * 32-token microbatch plus top_k experts of gate/up/down per token, exact for
 * decode and an upper bound for t>1 (tokens may share experts). qsa.core reads KV/index caches, no weights: it stays 0. */
static uint64_t ple_bytes(const G2Layer *l,size_t t) {
    return l->eb[0]->tensor.bytes+l->eb[1]->tensor.bytes+t*l->ew.rows.stride_bytes[0];
}
/* Profile groups: embed reads t rows, linears their whole matrix once. */
static Qwen38Status run(Qwen38GpuForward *f,Qwen38GpuCommand *cmd,size_t t,uint32_t layers,size_t rows,uint32_t pos,Qwen38Error *e) {
    MARK(QWEN38_GLOBAL_LAYER,"embed",f->embed.binding->tensor.type,t*(f->embed.binding->tensor.bytes/f->c.vocab_size));
    TRY(q38_packed_lookup(f->b,cmd,f->store,f->embed.binding,&f->embed.layout,f->tokens.span,time_view(f->mixed,t),e));
    barrier(*cmd); emit(f,*cmd,QWEN38_GLOBAL_LAYER,Q38G_EMBEDDING,time_view(f->mixed,t));
    copy(f,*cmd,time_view(f->mixed,t),time_view(f->r,t),true);
    for (uint32_t i=0;i<layers;i++) {
        G2Layer *l=&f->layer[i];
        if (l->es) { MARK(i,"ple",Q38_PROFILE_MIXED,ple_bytes(l,t)); TRY(ple(f,*cmd,i,t,e)); }
        if (l->kind.kind==Q38_LAYER_GDN) { TRY(gdn(f,*cmd,i,t,e)); }
        else TRY(qsa(f,cmd,i,t,pos,e));
        emit(f,*cmd,i,Q38G_ATTN_RESIDUAL,time_view(f->r,t));
        TRY(hc_read(f,cmd,i,&l->ffn,HC_FFN,t,e));
        emit(f,*cmd,i,Q38G_FFN_MIXED,time_view(f->mixed,t)); emit(f,*cmd,i,Q38G_FFN_INJECT,time_view(f->inject,t));
        TRY(moe(f,*cmd,i,t,e)); emit(f,*cmd,i,Q38G_MOE,time_view(f->block,t));
        MARK(i,"hc_add",Q38_PROFILE_NONE,0); TRY(hc_add(f,*cmd,t,e)); emit(f,*cmd,i,Q38G_OUTPUT,time_view(f->r,t));
    }
    TRY(hc_read(f,cmd,QWEN38_GLOBAL_LAYER,&f->final,HC_FINAL,t,e)); emit(f,*cmd,QWEN38_GLOBAL_LAYER,Q38G_FINAL,time_view(f->mixed,t));
    if (rows) TRY(linear(f,cmd,QWEN38_GLOBAL_LAYER,"lm_head",f->head,slice(f->mixed,t-rows,rows),time_view(f->logits,rows),e));
    if (rows && f->want_id) {   /* the concurrent encoder orders only explicit barriers */
        barrier(*cmd); argmax_encode(f->argmax,*cmd,f->logits.span,f->argmax_id.span,f->c.vocab_size);
    }
    return Q38_OK;
}
static Qwen38Status jobs(Qwen38GpuForward *f,bool commit,Qwen38Status rc,Qwen38Error *e) {
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) {
        G2Layer *l=&f->layer[i]; if (!l->job) continue;
        Qwen38Error local;
        Qwen38Status s=commit ? q38e_ngram_gpu_commit(l->job,&local) : qwen38_ngram_abort(l->job,&local);
        if (!rc && s) { rc=s; if (e) *e=local; }
        s=qwen38_ngram_job_release(l->job,&local); l->job=NULL;
        if (!rc && s) { rc=s; if (e) *e=local; }
    }
    return rc;
}
static Qwen38Status prepare(Qwen38GpuForward *f,const int32_t *tokens,size_t t,uint32_t layers,uint64_t pos,Qwen38Error *e) {
    memcpy(contents(f->tokens),tokens,t*4);
    for (uint32_t i=0;i<layers;i++) {
        G2Layer *l=&f->layer[i]; if (!l->es) continue;
        TRY(qwen38_ngram_prepare(l->es,l->estate,tokens,t,pos,&l->job,e));
        TRY(qwen38_ngram_read(l->job,e)); Qwen38F32 rows; TRY(qwen38_ngram_rows(l->job,&rows,e));
        Q38eReadStats stats=q38e_reader_stats(l->es->reader);
        f->stats.ngram_calls+=stats.calls; f->stats.ngram_bytes+=stats.bytes_read;
        f->stats.ngram_rows+=stats.unique_rows;
        memcpy(contents(l->ew.rows),rows.data,rows.capacity_bytes);
    }
    return Q38_OK;
}
static Qwen38Status claim(Qwen38GpuForward *f,Qwen38GpuSession *session,const Qwen38GpuTicket *ticket,Qwen38Error *e) {
    for (id<MTLBuffer> b in (__bridge NSArray *)f->buffers)
        TRY(qwen38_gpu_session_claim(session,ticket,(Qwen38GpuSpan){(__bridge void *)b,0,b.length},Q38_GPU_USE_WRITE,e));
    if (f->trace_span.buffer) TRY(qwen38_gpu_session_claim(session,ticket,f->trace_span,Q38_GPU_USE_WRITE,e));
    const GgufPlan *plan=gpu_weights_plan(f->store);
    /* R1.7a: weights are pinned once per session, all or nothing: a failed pass clears
     * its partial pins, only a finished one is sealed. `claim` oracle: every command. */
    bool pin=qwen38_gpu_session_pins_enabled(session);
    if (pin && qwen38_gpu_session_pins_sealed(session)) return Q38_OK;
    for (size_t i=0;i<plan->weight_count;i++) {
        const GgufBinding *w=&plan->weights[i];
        Qwen38GpuSpan span={gpu_weights_buffer(f->store,w),w->offset,w->tensor.bytes};
        Qwen38Status rc=pin ? qwen38_gpu_session_pin_read(session,ticket,span,e) : qwen38_gpu_session_claim(session,ticket,span,Q38_GPU_USE_READ,e);
        if (rc) { if (pin) qwen38_gpu_session_pins_clear(session); return rc; }
    }
    if (pin) qwen38_gpu_session_pins_seal(session);
    return Q38_OK;
}
/* want!=NULL: GPU argmax of the last row's logits, only the id is read back (logits ignored). */
static Qwen38Status forward(Qwen38GpuForward *f,const int32_t *tokens,size_t count,uint32_t layers,
    Qwen38F32Mut logits,int32_t *want,Qwen38GraphTrace trace,void *user,Qwen38Error *e) {
    if (!f || (count && !tokens)) return fail(e,Q38_INVALID_ARGUMENT,"append arguments");
    Qwen38StateInfo before=qwen38_gpu_forward_info(f);
    if (before.busy) return fail(e,Q38_BUSY,"graph busy");
    if (!before.valid) return fail(e,Q38_BAD_STATE,"graph requires reset");
    if (layers>f->c.trunk_blocks) return fail(e,Q38_RANGE,"trunk depth");
    if (layers<f->c.trunk_blocks && before.next_pos) return fail(e,Q38_BAD_STATE,"prefix requires fresh graph");
    if (count>f->max_tokens || count>f->capacity-before.next_pos) return fail(e,Q38_CAPACITY,"append capacity");
    if (want) { logits=(Qwen38F32Mut){.rank=2,.dim={1,f->c.vocab_size}}; if (!count) return fail(e,Q38_INVALID_ARGUMENT,"argmax needs a token"); }
    else {
    TRY(q38c_view(q38c_ro(logits),e));
    if (logits.rank!=2 || logits.dim[0]>count || logits.dim[0]>f->max_logits_rows || logits.dim[1]!=f->c.vocab_size ||
        logits.stride_bytes[0]!=(size_t)f->c.vocab_size*4) return fail(e,Q38_SHAPE,"dense bounded logits required");
    }
    for (size_t i=0;i<count;i++) if (tokens[i]<0 || (uint32_t)tokens[i]>=f->c.vocab_size) return fail(e,Q38_RANGE,"token outside vocabulary");
    if (!count) return fail(e,Q38_OK,NULL);
    size_t output_bytes=want ? 0 : q38c_span(q38c_ro(logits));
    if (output_bytes) {
        for (id<MTLBuffer> b in (__bridge NSArray *)f->buffers)
            if (q38c_overlap(logits.data,output_bytes,b.contents,b.length)) return fail(e,Q38_INVALID_ARGUMENT,"logits alias graph storage");
        const GgufPlan *plan=gpu_weights_plan(f->store);
        for (size_t i=0;i<plan->weight_count;i++) {
            Qwen38CpuSpan weight; TRY(qwen38_weights_cpu_span(f->store,&plan->weights[i],&weight,e));
            if (q38c_overlap(logits.data,output_bytes,weight.data,weight.bytes)) return fail(e,Q38_INVALID_ARGUMENT,"logits alias weights");
        }
    }
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> trace_buffer=nil;
    if (trace) {
        size_t per=2*f->c.hidden_size,h=f->c.hc.streams,d=f->c.hidden_size;
        for (uint32_t i=0;i<layers;i++) per+=4*d+2*h+2*h*d+(f->layer[i].es ? h*d : 0);
        size_t bytes;
        if (!q38c_mul(per,count*4,&bytes) || bytes>(256u<<20)) return fail(e,Q38_CAPACITY,"trace exceeds 256 MiB");
        trace_buffer=[(__bridge id<MTLDevice>)f->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!trace_buffer) return fail(e,Q38_OOM,"trace buffer");
        f->trace_span=(Qwen38GpuSpan){(__bridge void *)trace_buffer,0,bytes};
    }
    f->trace_count=0; f->trace_offset=0; f->busy=true; f->want_id=want!=NULL;
    f->stats=(Qwen38GpuForwardStats){0}; double prepare_start=clock_s();
    Qwen38Status rc=prepare(f,tokens,count,layers,before.next_pos,e);
    double encode_start=clock_s(); f->stats.prepare_s=encode_start-prepare_start;
    Qwen38GpuTicket ticket; Qwen38GpuCommand cmd; bool begun=false,submitted=false;
    if (!rc) { rc=qwen38_gpu_session_begin(f->session,f->config,before.next_pos,count,&ticket,&cmd,e); begun=!rc; }
    if (!rc) rc=claim(f,f->session,&ticket,e);
    if (!rc) { f->prof_ticket=f->prof ? &ticket : NULL; f->prof_failed=false; }
    if (!rc) rc=run(f,&cmd,count,layers,logits.dim[0],(uint32_t)before.next_pos,e);
    if (!rc && f->prof_failed) rc=fail(e,Q38_GPU_ERROR,"profile encoder split");
    f->prof_ticket=NULL; f->want_id=false;
    if (!rc) { rc=qwen38_gpu_session_submit(f->session,&ticket,e); submitted=!rc; }
    double wait_start=clock_s(); f->stats.encode_s=wait_start-encode_start;
    if (submitted) rc=qwen38_gpu_session_finish(f->session,&ticket,e);
    else if (begun) qwen38_gpu_session_abort(f->session,&ticket,false,NULL);
    f->stats.wait_s=clock_s()-wait_start;
    if (f->prof && !q38_profile_end_append(f->prof,!rc && submitted) && !rc) rc=fail(e,Q38_GPU_ERROR,"profile samples");
    if (submitted) f->stats.gpu_s=qwen38_gpu_session_seconds(f->session);
    rc=jobs(f,!rc && layers==f->c.trunk_blocks,rc,e);
    if (rc || layers<f->c.trunk_blocks) f->valid=false;
    else { f->last_start=f->next_pos; f->last_count=count; f->next_pos+=count; }
    if (!rc) {
        if (want) *want=*(int32_t *)contents(f->argmax_id);
        else if (logits.dim[0]) memcpy(logits.data,contents(f->logits),logits.dim[0]*f->c.vocab_size*4);
        if (trace) for (size_t j=0;j<f->trace_count;j++) {
            G2Trace *r=&f->traces[j]; Qwen38F32 v={.data=contents(r->view),.capacity_bytes=r->view.span.bytes,.rank=r->view.rank};
            memcpy(v.dim,r->view.dim,sizeof v.dim); memcpy(v.stride_bytes,r->view.stride_bytes,sizeof v.stride_bytes);
            trace(user,r->layer,r->stage,v);
        }
    }
    f->trace_span=(Qwen38GpuSpan){0}; f->busy=false;
    return rc ? rc : fail(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gpu_forward(Qwen38GpuForward *f,const int32_t *tokens,size_t count,uint32_t layers,
    Qwen38F32Mut logits,Qwen38GraphTrace trace,void *user,Qwen38Error *e) {
    return forward(f,tokens,count,layers,logits,NULL,trace,user,e);
}
Qwen38Status qwen38_gpu_forward_argmax(Qwen38GpuForward *f,const int32_t *tokens,size_t count,uint32_t layers,
    int32_t *id,Qwen38Error *e) {
    if (!f || !id || !tokens || !count) return fail(e,Q38_INVALID_ARGUMENT,"argmax arguments");
    if (f->gpu_argmax) return forward(f,tokens,count,layers,(Qwen38F32Mut){0},id,NULL,NULL,e);
    /* `argmax` oracle: today's path, the last row read back and scanned on the CPU. */
    uint32_t n=f->c.vocab_size;
    if (!f->host_row && !(f->host_row=malloc((size_t)n*4))) return fail(e,Q38_OOM,"argmax row");
    Qwen38F32Mut y={.data=f->host_row,.capacity_bytes=(size_t)n*4,.rank=2,.dim={1,n},.stride_bytes={(size_t)n*4,4}};
    TRY(forward(f,tokens,count,layers,y,NULL,NULL,NULL,e));
    int32_t best=0;
    for (uint32_t j=0;j<n;j++) {
        if (!isfinite(f->host_row[j])) return fail(e,Q38_GPU_ERROR,"nonfinite logit");
        if (f->host_row[j]>f->host_row[best]) best=(int32_t)j;
    }
    *id=best; return fail(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gpu_forward_reset(Qwen38GpuForward *f,Qwen38Error *e) {
    if (!f) return fail(e,Q38_INVALID_ARGUMENT,"null graph");
    if (f->busy) return fail(e,Q38_BUSY,"graph busy");
    if (f->mtp_failed) return fail(e,Q38_BAD_STATE,"MTP allocation failed; destroy graph");
    f->valid=false; TRY(qwen38_gpu_session_reset(f->session,f->config,e));
    if (f->mtp_session) TRY(qwen38_gpu_session_reset(f->mtp_session,f->config,e));
    uint64_t epoch=qwen38_gpu_session_info(f->session).key.epoch;
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) if (f->layer[i].estate) TRY(qwen38_ngram_state_reset(f->layer[i].estate,epoch,e));
    for (id<MTLBuffer> b in (__bridge NSArray *)f->histories) memset(b.contents,0,b.length);
    f->next_pos=0; f->last_count=0; f->snapshot_serial++; f->valid=true; return fail(e,Q38_OK,NULL);
}

static Qwen38Status mtp_create(Qwen38GpuForward *f,Qwen38Error *e) {
    uint32_t i=f->c.trunk_blocks; size_t t=f->max_tokens,h=f->c.hc.streams,d=f->c.hidden_size;
    const Qwen38Layer *kind; TRY(qwen38_config_layer(f->config,i,&kind,e));
    f->layer[i].kind=*kind;
    if (kind->kind!=Q38_LAYER_QSA || kind->has_ple) return fail(e,Q38_UNSUPPORTED,"MTP requires QSA without PLE");
    TRY(qwen38_gpu_session_create(f->device,f->config,f->capacity,&f->mtp_session,e));
    TRY(residency(f,f->mtp_session,e));
    TRY(create_qsa(f,i,e)); TRY(hc_bind(f,i,ffn_roles,&f->layer[i].ffn,e));
    const Qwen38Role roles[]={Q38_MOE_ROUTER,Q38_MOE_GATE,Q38_MOE_UP,Q38_MOE_DOWN,Q38_SHARED_GATE,Q38_SHARED_UP,Q38_SHARED_DOWN,Q38_SHARED_ROUTER};
    for (size_t j=0;j<8;j++) { G2Weight w; TRY(bind_weight(f,i,roles[j],&w,e)); f->layer[i].fb[j]=w.binding; f->layer[i].fl[j]=w.layout; }
    const Qwen38Role head_roles[]={Q38_MTP_HEAD_NORM,Q38_MTP_HEAD_DOWN,Q38_MTP_HEAD_UP,Q38_ROLE_COUNT};
    TRY(hc_bind(f,i,head_roles,&f->mtp_head,e)); TRY(bind_weight(f,i,Q38_MTP_EH,&f->mtp_eh,e));
    TRY(constant(f,i,Q38_MTP_E_NORM,&f->mtp_egain,1,(size_t[]){d},e));
    TRY(constant(f,i,Q38_MTP_H_NORM,&f->mtp_hgain,1,(size_t[]){h*d},e));
    ALLOC(f->mtp_en,false,t,1,d); ALLOC(f->mtp_hn,false,t,1,h*d);
    ALLOC(f->mtp_cat,false,t*(h+1),2*d); ALLOC(f->mtp_proj,false,t*(h+1),d);
    ALLOC(f->mtp_r,false,t,h,d);
    NSString *names[]={@"q38g_mtp_cat",@"q38g_mtp_sum"}; void **dst[]={&f->mtp_concat,&f->mtp_sum};
    for (size_t j=0;j<2;j++) {
        id<MTLFunction> fn=[(__bridge id<MTLLibrary>)f->library newFunctionWithName:names[j]];
        id<MTLComputePipelineState> p=fn ? [(__bridge id<MTLDevice>)f->device newComputePipelineStateWithFunction:fn error:NULL] : nil;
        if (!p) return fail(e,Q38_GPU_ERROR,"MTP wiring pipeline"); *dst[j]=(__bridge_retained void *)p;
    }
    return Q38_OK;
}
Qwen38Status qwen38_gpu_mtp_enable(Qwen38GpuForward *f,Qwen38Error *e) {
    if (!f) return fail(e,Q38_INVALID_ARGUMENT,"null graph");
    Qwen38StateInfo info=qwen38_gpu_forward_info(f);
    if (info.busy) return fail(e,Q38_BUSY,"graph busy");
    if (!info.valid || info.next_pos || f->mtp_enabled || f->mtp_failed) return fail(e,Q38_BAD_STATE,"MTP requires fresh graph");
    if (f->c.mtp_blocks!=1) return fail(e,Q38_UNSUPPORTED,"exactly one MTP block required");
    Qwen38Status rc=mtp_create(f,e);
    if (rc) { f->mtp_failed=true; f->valid=false; } else f->mtp_enabled=true;
    return rc ? rc : fail(e,Q38_OK,NULL);
}
Qwen38StateInfo qwen38_gpu_mtp_info(const Qwen38GpuForward *f) {
    if (!f || !f->mtp_enabled) return (Qwen38StateInfo){0};
    Qwen38StateInfo info=qwen38_gpu_session_info(f->mtp_session); info.valid &= f->valid; info.busy |= f->busy; return info;
}
static void mtp_wire(Qwen38GpuForward *f,Qwen38GpuCommand cmd,size_t t,bool sum) {
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)(sum ? f->mtp_sum : f->mtp_concat);
    uint32_t h=(uint32_t)f->c.hc.streams,d=f->c.hidden_size;
    uint32_t a[4]={(uint32_t)t*(sum ? h*d : (h+1)*2*d),d,h,0};
    [enc setComputePipelineState:p];
    Qwen38GpuSpan spans[4]={sum ? f->mtp_proj.span : f->mtp_en.span,sum ? f->mtp_r.span : f->mtp_hn.span,f->mtp_cat.span,cmd.error_word};
    for (size_t j=0;j<(sum ? 2u : 3u);j++) [enc setBuffer:(__bridge id<MTLBuffer>)spans[j].buffer offset:spans[j].offset_bytes atIndex:j];
    if (sum) [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:3];
    [enc setBytes:a length:sizeof a atIndex:sum ? 2 : 3];
    [enc dispatchThreads:MTLSizeMake(a[0],1,1) threadsPerThreadgroup:MTLSizeMake(MIN(a[0],p.maxTotalThreadsPerThreadgroup),1,1)]; barrier(cmd);
}
static Qwen38Status mtp_run(Qwen38GpuForward *f,Qwen38GpuCommand cmd,size_t offset,size_t t,size_t rows,uint32_t pos,Qwen38Error *e) {
    size_t d=f->c.hidden_size,h=f->c.hc.streams; Q38CGpuSpec spec={.eps=f->c.hc.rms_eps};
    TRY(q38_packed_lookup(f->b,&cmd,f->store,f->embed.binding,&f->embed.layout,f->tokens.span,time_view(f->mixed,t),e)); barrier(cmd);
    Qwen38GpuF32 ev[]={shape(f->mixed.span,3,(size_t[]){t,1,d}),f->mtp_egain,time_view(f->mtp_en,t)};
    TRY(q38c_gpu_encode(f->ops,cmd,Q38C_RMS,&spec,ev,3,e));
    Qwen38GpuF32 hv[]={shape(slice(f->r,offset,t).span,3,(size_t[]){t,1,h*d}),f->mtp_hgain,time_view(f->mtp_hn,t)};
    TRY(q38c_gpu_encode(f->ops,cmd,Q38C_RMS,&spec,hv,3,e));
    mtp_wire(f,cmd,t,false);
    TRY(linear(f,&cmd,f->c.trunk_blocks,"mtp.eh",f->mtp_eh,time_view(f->mtp_cat,t*(h+1)),time_view(f->mtp_proj,t*(h+1)),e));
    mtp_wire(f,cmd,t,true);
    /* The shared operators use r. Restore this borrowed view on EVERY exit. */
    Qwen38GpuF32 trunk=f->r; f->r=f->mtp_r; uint32_t i=f->c.trunk_blocks;
    Qwen38Status rc=qsa(f,&cmd,i,t,pos,e);
    if (!rc) rc=hc_read(f,&cmd,i,&f->layer[i].ffn,HC_FFN,t,e);
    if (!rc) rc=moe(f,cmd,i,t,e);
    if (!rc) rc=hc_add(f,cmd,t,e);
    if (!rc && rows) rc=hc_read(f,&cmd,QWEN38_GLOBAL_LAYER,&f->mtp_head,HC_FINAL,t,e);
    if (!rc && rows) rc=linear(f,&cmd,QWEN38_GLOBAL_LAYER,"lm_head",f->head,slice(f->mixed,t-rows,rows),time_view(f->logits,rows),e);
    f->r=trunk; return rc;
}
Qwen38Status qwen38_gpu_mtp(Qwen38GpuForward *f,size_t offset,const int32_t *tokens,size_t count,Qwen38F32Mut logits,Qwen38Error *e) {
    if (!f || (count && !tokens)) return fail(e,Q38_INVALID_ARGUMENT,"MTP arguments");
    if (f->busy) return fail(e,Q38_BUSY,"graph busy");
    Qwen38StateInfo before=qwen38_gpu_mtp_info(f);
    if (!before.valid) return fail(e,Q38_BAD_STATE,"MTP disabled or requires reset");
    if (offset>f->last_count || count>f->last_count-offset || before.next_pos!=f->last_start+offset)
        return fail(e,Q38_BAD_STATE,"MTP requires consecutive last-trunk residual pairs");
    if (count>f->capacity-before.next_pos) return fail(e,Q38_CAPACITY,"MTP capacity");
    TRY(q38c_view(q38c_ro(logits),e));
    if (logits.rank!=2 || logits.dim[0]>count || logits.dim[0]>f->max_logits_rows || logits.dim[1]!=f->c.vocab_size ||
        logits.stride_bytes[0]!=(size_t)f->c.vocab_size*4) return fail(e,Q38_SHAPE,"dense bounded MTP logits required");
    for (size_t j=0;j<count;j++) if (tokens[j]<0 || (uint32_t)tokens[j]>=f->c.vocab_size) return fail(e,Q38_RANGE,"MTP token outside vocabulary");
    size_t bytes=q38c_span(q38c_ro(logits));
    if (bytes) {
        for (id<MTLBuffer> b in (__bridge NSArray *)f->buffers)
            if (q38c_overlap(logits.data,bytes,b.contents,b.length)) return fail(e,Q38_INVALID_ARGUMENT,"MTP logits alias storage");
        const GgufPlan *plan=gpu_weights_plan(f->store);
        for (size_t j=0;j<plan->weight_count;j++) {
            Qwen38CpuSpan w; TRY(qwen38_weights_cpu_span(f->store,&plan->weights[j],&w,e));
            if (q38c_overlap(logits.data,bytes,w.data,w.bytes)) return fail(e,Q38_INVALID_ARGUMENT,"MTP logits alias weights");
        }
    }
    if (!count) return fail(e,Q38_OK,NULL);
    f->busy=true; f->stats=(Qwen38GpuForwardStats){0}; double start=clock_s(); memcpy(contents(f->tokens),tokens,count*4);
    Qwen38GpuTicket ticket; Qwen38GpuCommand cmd; bool begun=false,submitted=false;
    Qwen38Status rc=qwen38_gpu_session_begin(f->mtp_session,f->config,before.next_pos,count,&ticket,&cmd,e); begun=!rc;
    if (!rc) rc=claim(f,f->mtp_session,&ticket,e);
    if (!rc) rc=mtp_run(f,cmd,offset,count,logits.dim[0],(uint32_t)before.next_pos,e);
    if (!rc) { rc=qwen38_gpu_session_submit(f->mtp_session,&ticket,e); submitted=!rc; }
    double wait=clock_s(); f->stats.encode_s=wait-start;
    if (submitted) rc=qwen38_gpu_session_finish(f->mtp_session,&ticket,e);
    else if (begun) qwen38_gpu_session_abort(f->mtp_session,&ticket,false,NULL);
    f->stats.wait_s=clock_s()-wait;
    if (submitted) f->stats.gpu_s=qwen38_gpu_session_seconds(f->mtp_session);
    if (rc) f->valid=false;
    else if (bytes) memcpy(logits.data,contents(f->logits),bytes);
    f->busy=false; return rc ? rc : fail(e,Q38_OK,NULL);
}
size_t qwen38_gpu_checkpoint_bytes(const Qwen38GpuForward *f) { return f ? f->snapshot_bytes : 0; }
static size_t checkpoint_transfer(Qwen38GpuForward *f,void *buffer,bool restore) {
    size_t n=0; NSArray *histories=(__bridge NSArray *)f->histories;
#define TRANSFER(ptr,len) do { size_t z_=(len); if (buffer) { if (restore) memcpy((ptr),(char *)buffer+n,z_); else memcpy((char *)buffer+n,(ptr),z_); } n+=z_; } while(0)
    for (size_t j=0;j<f->main_histories;j++) { id<MTLBuffer> b=histories[j]; TRANSFER(b.contents,b.length); }
    TRANSFER(contents(f->r),f->r.span.bytes);
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) if (f->layer[i].estate) {
        Qwen38NgramState *s=f->layer[i].estate; TRANSFER(s->previous,(s->store->config.order-1)*sizeof *s->previous);
    }
#undef TRANSFER
    return n;
}
Qwen38Status qwen38_gpu_checkpoint_save(Qwen38GpuForward *f,Qwen38GpuCheckpoint *out,Qwen38Error *e) {
    if (out) *out=(Qwen38GpuCheckpoint){0};
    if (!f || !out) return fail(e,Q38_INVALID_ARGUMENT,"checkpoint arguments");
    Qwen38StateInfo info=qwen38_gpu_forward_info(f);
    if (info.busy) return fail(e,Q38_BUSY,"graph busy");
    if (!info.valid) return fail(e,Q38_BAD_STATE,"checkpoint requires valid graph");
    if (f->snapshot_serial==UINT64_MAX) return fail(e,Q38_OVERFLOW,"checkpoint serial");
    if (!f->snapshot) {
        size_t n=checkpoint_transfer(f,NULL,false); f->snapshot=malloc(n ? n : 1);
        if (!f->snapshot) return fail(e,Q38_OOM,"host checkpoint"); f->snapshot_bytes=n;
    }
    checkpoint_transfer(f,f->snapshot,false); f->saved_start=f->last_start; f->saved_count=f->last_count;
    *out=f->saved=(Qwen38GpuCheckpoint){f,info.key.epoch,++f->snapshot_serial,info.next_pos,qwen38_gpu_mtp_info(f).next_pos};
    return fail(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gpu_checkpoint_restore(Qwen38GpuForward *f,Qwen38GpuCheckpoint s,Qwen38Error *e) {
    if (!f) return fail(e,Q38_INVALID_ARGUMENT,"null graph");
    if (s.owner!=f) return fail(e,Q38_OWNER,"foreign checkpoint");
    Qwen38StateInfo info=qwen38_gpu_forward_info(f);
    if (info.busy) return fail(e,Q38_BUSY,"graph busy");
    if (!f->snapshot || s.epoch!=info.key.epoch || s.serial!=f->snapshot_serial ||
        s.position!=f->saved.position || s.mtp_position!=f->saved.mtp_position) return fail(e,Q38_STALE,"stale checkpoint");
    if (!info.valid || s.position>info.next_pos || info.next_pos-s.position>2 || s.mtp_position!=qwen38_gpu_mtp_info(f).next_pos)
        return fail(e,Q38_BAD_STATE,"checkpoint window or predictor changed");
    TRY(q38g_session_rewind(f->session,f->config,s.epoch,s.position,e));
    f->valid=false; uint64_t epoch=qwen38_gpu_session_info(f->session).key.epoch;
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) if (f->layer[i].estate) {
        TRY(qwen38_ngram_state_reset(f->layer[i].estate,epoch,e)); f->layer[i].estate->info.next_pos=s.position;
    }
    checkpoint_transfer(f,f->snapshot,true); f->next_pos=s.position;
    f->last_start=f->saved_start; f->last_count=f->saved_count; f->snapshot_serial++;
    f->valid=true; return fail(e,Q38_OK,NULL);
}
