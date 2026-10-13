/* Ordinary G2 graph. Operators stay in B/C/D/E/F; this file owns resources,
 * graph order, bounded observations and the one G0 transaction per append. */
#import <Metal/Metal.h>
#include "qwen38_forward_gpu.h"
#include "qwen38_g_internal.h"
#include "qwen38_e_internal.h"
#include "qwen38_d_internal.h"
#include "qwen38_f_internal.h"
#include <stdlib.h>

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
    void *device,*buffers,*histories,*copy; size_t bytes;
    bool valid,busy; Qwen38GpuSession *session;
    Q38PackedGpu *b; Q38CGpu *ops; Q38DGpu *d; Q38EGpu *e; Q38FRouted *moe;
    G2Weight embed,head; G2Hc final; G2Layer *layer;
    Qwen38GpuF32 r,normed,low,act,gate,il,inject,mixed,block,logits,tokens,c_arena;
    Q38DGpuLayer dw; Q38EGpuWork ew; Q38FRoutedWork fw;
    G2Trace *traces; size_t trace_count,trace_offset; Qwen38GpuSpan trace_span;
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
    ALLOC(f->tokens,false,t); ALLOC(f->logits,false,f->max_logits_rows,f->c.vocab_size);
    const Qwen38AttentionConfig *a=&f->c.attention;
    ALLOC(f->dw.qg,false,t,a->heads,2,a->head_dim);
    ALLOC(f->dw.k,false,t,a->kv_heads,a->head_dim); ALLOC(f->dw.v,false,t,a->kv_heads,a->head_dim);
    ALLOC(f->dw.iq,false,t,a->index_heads,a->index_dim); ALLOC(f->dw.ik,false,t,a->index_dim);
    ALLOC(f->dw.qn,false,t,a->heads,a->head_dim); ALLOC(f->dw.qr,false,t,a->heads,a->head_dim);
    ALLOC(f->dw.gate,false,t,a->heads,a->head_dim); ALLOC(f->dw.kn,false,t,a->kv_heads,a->head_dim);
    ALLOC(f->dw.iqn,false,t,a->index_heads,a->index_dim); ALLOC(f->dw.iqr,false,t,a->index_heads,a->index_dim);
    ALLOC(f->dw.pooled,false,1,1,a->index_dim); ALLOC(f->dw.pooled_n,false,1,1,a->index_dim);
    ALLOC(f->dw.scores,false,f->capacity); ALLOC(f->dw.out,false,t,a->heads,a->head_dim);
    Qwen38GpuF32 ids,keep; ALLOC(ids,false,(size_t)a->token_budget+f->capacity); ALLOC(keep,false,f->capacity);
    f->dw.ids=ids.span; f->dw.keep=keep.span;
    const Qwen38MoeConfig *m=&f->c.moe; size_t mt=MIN(t,32),p=mt*m->top_k;
    ALLOC(f->fw.logits,false,mt,m->experts); ALLOC(f->fw.probs,false,mt,m->experts); ALLOC(f->fw.weights,false,mt,m->top_k);
    ALLOC(f->fw.gate,false,p,m->intermediate); ALLOC(f->fw.up,false,p,m->intermediate); ALLOC(f->fw.act,false,p,m->intermediate);
    ALLOC(f->fw.down,false,p,d); ALLOC(f->fw.sgate,false,mt,m->shared_intermediate);
    ALLOC(f->fw.sup,false,mt,m->shared_intermediate); ALLOC(f->fw.sact,false,mt,m->shared_intermediate);
    ALLOC(f->fw.shared,false,mt,d); ALLOC(f->fw.slogit,false,mt,1); ALLOC(f->fw.ssig,false,mt,1); ALLOC(f->fw.routed,false,mt,d);
    ALLOC(ids,false,p); ALLOC(keep,false,p); f->fw.ids=ids.span; f->fw.identity=keep.span;
    return Q38_OK;
}
static Qwen38Status create(Qwen38GpuForward *f,void *library,Qwen38Error *e) {
    TRY(qwen38_gpu_session_create(f->device,f->config,f->capacity,&f->session,e));
    TRY(q38_packed_gpu_create(f->device,library,&f->b,e)); TRY(q38c_gpu_create(f->device,library,&f->ops,e));
    TRY(q38d_gpu_create(f->device,library,&f->d,e)); TRY(q38e_gpu_create(f->device,library,&f->e,e));
    TRY(q38f_routed_create(f->device,library,&f->moe,e));
    id<MTLFunction> fn=[(__bridge id<MTLLibrary>)library newFunctionWithName:@"q38g_copy"];
    id<MTLComputePipelineState> p=fn ? [(__bridge id<MTLDevice>)f->device newComputePipelineStateWithFunction:fn error:NULL] : nil;
    if (!p) return fail(e,Q38_GPU_ERROR,"graph copy pipeline"); f->copy=(__bridge_retained void *)p;
    TRY(bind_weight(f,QWEN38_GLOBAL_LAYER,Q38_TOKEN_EMBD,&f->embed,e));
    TRY(bind_weight(f,QWEN38_GLOBAL_LAYER,Q38_LM_HEAD,&f->head,e));
    if (f->embed.layout.k_logical!=f->c.hidden_size || f->embed.layout.n_out!=f->c.vocab_size ||
        f->head.layout.k_logical!=f->c.hidden_size || f->head.layout.n_out!=f->c.vocab_size)
        return fail(e,Q38_SHAPE,"embedding/head config");
    TRY(hc_bind(f,QWEN38_GLOBAL_LAYER,final_roles,&f->final,e));
    f->layer=calloc(f->c.trunk_blocks,sizeof *f->layer); f->traces=calloc(2+9*f->c.trunk_blocks,sizeof *f->traces);
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
    f->buffers=(__bridge_retained void *)[NSMutableArray new]; f->histories=(__bridge_retained void *)[NSMutableArray new];
    Qwen38Status rc=create(f,library,e); if (rc) { qwen38_gpu_forward_destroy(f); return rc; }
    *out=f; return fail(e,Q38_OK,NULL);
}
size_t qwen38_gpu_forward_bytes(const Qwen38GpuForward *f) { return f ? f->bytes : 0; }
Qwen38GpuF32 q38g_gpu_state(Qwen38GpuForward *f,uint32_t layer,uint32_t slot) {
    if (!f || f->busy || layer>=f->c.trunk_blocks || slot>=Q38GS_COUNT) return (Qwen38GpuF32){0};
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
    if (f->copy) CFRelease(f->copy);
    if (f->buffers) CFRelease(f->buffers);
    if (f->histories) CFRelease(f->histories);
    if (f->device) CFRelease(f->device);
    free(f->layer); free(f->traces); free(f);
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
static Qwen38Status linear(Qwen38GpuForward *f,Qwen38GpuCommand cmd,G2Weight w,Qwen38GpuF32 x,Qwen38GpuF32 y,Qwen38Error *e) {
    TRY(q38_packed_encode(f->b,&cmd,f->store,w.binding,&w.layout,0,flat(x),flat(y),e)); barrier(cmd); return Q38_OK;
}
static Qwen38Status hc_read(Qwen38GpuForward *f,Qwen38GpuCommand cmd,const G2Hc *hc,size_t t,Qwen38Error *e) {
    Q38CGpuSpec spec={.hc=f->c.hc,.eps=f->c.hc.rms_eps};
#define V(x) time_view(f->x,t)
#define OP(op,...) do { Qwen38GpuF32 vv[]={__VA_ARGS__}; TRY(q38c_gpu_encode(f->ops,cmd,Q38C_##op,&spec,vv,sizeof vv/sizeof *vv,e)); } while(0)
    OP(RMS,V(r),hc->gain,V(normed));
    TRY(linear(f,cmd,hc->down,V(normed),V(low),e));
    OP(HC_LOW,V(low),V(act));
    TRY(linear(f,cmd,hc->up,V(act),V(gate),e));
    OP(HC_READ,V(normed),V(gate),V(mixed));
    if (hc->inject.binding) {
        TRY(linear(f,cmd,hc->inject,V(normed),V(il),e)); OP(HC_INJECT,V(il),V(inject));
    }
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
static Qwen38Status gdn(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; Q38CGpuWork w=l->cw; w.residual=time_view(f->r,t);
    for (size_t j=0;j<Q38CT_COUNT;j++) w.trace[j]=time_view(w.trace[j],t);
    TRY(q38c_block_encode(f->ops,f->b,cmd,&l->gdn,f->store,l->cb,&w,e));
    emit(f,cmd,i,Q38G_ATTN_MIXED,w.trace[Q38CT_MIX]); emit(f,cmd,i,Q38G_ATTN_INJECT,w.trace[Q38CT_INJECT]);
    emit(f,cmd,i,Q38G_ATTN_CONTRIBUTION,w.trace[Q38CT_OUT]); return Q38_OK;
}
static Qwen38Status qsa(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,uint32_t pos,Qwen38Error *e) {
    G2Layer *l=&f->layer[i]; Q38DGpuLayer w=l->dw;
#define W(x) w.x=time_view(f->dw.x,t)
    W(qg); W(k); W(v); W(iq); W(ik); W(qn); W(qr); W(gate); W(kn); W(iqn); W(iqr); W(out);
#undef W
    w.pooled=f->dw.pooled; w.pooled_n=f->dw.pooled_n;
    w.scores=time_view(f->dw.scores,f->capacity/l->ac.ratio); w.keep=f->dw.keep; w.ids=f->dw.ids;
    TRY(hc_read(f,cmd,&l->attn,t,e));
    Qwen38GpuF32 outputs[]={w.qg,w.k,w.v,w.iq,w.ik};
    for (size_t j=0;j<5;j++) TRY(linear(f,cmd,l->qsa[j],time_view(f->mixed,t),outputs[j],e));
    TRY(q38d_layer_encode(f->d,f->ops,cmd,&l->ac,pos,(uint32_t)t,&w,e));
    TRY(linear(f,cmd,l->qsa[5],w.out,time_view(f->block,t),e));
    emit(f,cmd,i,Q38G_ATTN_MIXED,time_view(f->mixed,t)); emit(f,cmd,i,Q38G_ATTN_INJECT,time_view(f->inject,t));
    emit(f,cmd,i,Q38G_ATTN_CONTRIBUTION,time_view(f->block,t)); return hc_add(f,cmd,t,e);
}
static Qwen38Status moe(Qwen38GpuForward *f,Qwen38GpuCommand cmd,uint32_t i,size_t t,Qwen38Error *e) {
    G2Layer *l=&f->layer[i];
    for (size_t start=0;start<t;start+=32) {
        size_t n=MIN(t-start,32),p=n*f->c.moe.top_k; Q38FRoutedWork w=f->fw;
#define W(x,rows) w.x=time_view(w.x,rows)
        W(logits,n); W(probs,n); W(weights,n); W(gate,p); W(up,p); W(act,p); W(down,p);
        W(sgate,n); W(sup,n); W(sact,n); W(shared,n); W(slogit,n); W(ssig,n); W(routed,n);
#undef W
        TRY(q38f_routed_encode(f->moe,f->b,f->ops,cmd,&f->c.moe,f->store,l->fb,l->fl,
            slice(f->mixed,start,n),slice(f->block,start,n),&w,e)); barrier(cmd);
    }
    return Q38_OK;
}
static Qwen38Status run(Qwen38GpuForward *f,Qwen38GpuCommand cmd,size_t t,uint32_t layers,size_t rows,uint32_t pos,Qwen38Error *e) {
    TRY(q38_packed_lookup(f->b,&cmd,f->store,f->embed.binding,&f->embed.layout,f->tokens.span,time_view(f->mixed,t),e));
    barrier(cmd); emit(f,cmd,QWEN38_GLOBAL_LAYER,Q38G_EMBEDDING,time_view(f->mixed,t));
    copy(f,cmd,time_view(f->mixed,t),time_view(f->r,t),true);
    for (uint32_t i=0;i<layers;i++) {
        G2Layer *l=&f->layer[i];
        if (l->es) TRY(ple(f,cmd,i,t,e));
        if (l->kind.kind==Q38_LAYER_GDN) TRY(gdn(f,cmd,i,t,e)); else TRY(qsa(f,cmd,i,t,pos,e));
        emit(f,cmd,i,Q38G_ATTN_RESIDUAL,time_view(f->r,t));
        TRY(hc_read(f,cmd,&l->ffn,t,e));
        emit(f,cmd,i,Q38G_FFN_MIXED,time_view(f->mixed,t)); emit(f,cmd,i,Q38G_FFN_INJECT,time_view(f->inject,t));
        TRY(moe(f,cmd,i,t,e)); emit(f,cmd,i,Q38G_MOE,time_view(f->block,t));
        TRY(hc_add(f,cmd,t,e)); emit(f,cmd,i,Q38G_OUTPUT,time_view(f->r,t));
    }
    TRY(hc_read(f,cmd,&f->final,t,e)); emit(f,cmd,QWEN38_GLOBAL_LAYER,Q38G_FINAL,time_view(f->mixed,t));
    if (rows) TRY(linear(f,cmd,f->head,slice(f->mixed,t-rows,rows),time_view(f->logits,rows),e));
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
        memcpy(contents(l->ew.rows),rows.data,rows.capacity_bytes);
    }
    return Q38_OK;
}
static Qwen38Status claim(Qwen38GpuForward *f,const Qwen38GpuTicket *ticket,Qwen38Error *e) {
    for (id<MTLBuffer> b in (__bridge NSArray *)f->buffers)
        TRY(qwen38_gpu_session_claim(f->session,ticket,(Qwen38GpuSpan){(__bridge void *)b,0,b.length},Q38_GPU_USE_WRITE,e));
    if (f->trace_span.buffer) TRY(qwen38_gpu_session_claim(f->session,ticket,f->trace_span,Q38_GPU_USE_WRITE,e));
    const GgufPlan *plan=gpu_weights_plan(f->store);
    for (size_t i=0;i<plan->weight_count;i++) {
        const GgufBinding *w=&plan->weights[i];
        TRY(qwen38_gpu_session_claim(f->session,ticket,(Qwen38GpuSpan){gpu_weights_buffer(f->store,w),w->offset,w->tensor.bytes},Q38_GPU_USE_READ,e));
    }
    return Q38_OK;
}
Qwen38Status qwen38_gpu_forward(Qwen38GpuForward *f,const int32_t *tokens,size_t count,uint32_t layers,
    Qwen38F32Mut logits,Qwen38GraphTrace trace,void *user,Qwen38Error *e) {
    if (!f || (count && !tokens)) return fail(e,Q38_INVALID_ARGUMENT,"append arguments");
    Qwen38StateInfo before=qwen38_gpu_forward_info(f);
    if (before.busy) return fail(e,Q38_BUSY,"graph busy");
    if (!before.valid) return fail(e,Q38_BAD_STATE,"graph requires reset");
    if (layers>f->c.trunk_blocks) return fail(e,Q38_RANGE,"trunk depth");
    if (layers<f->c.trunk_blocks && before.next_pos) return fail(e,Q38_BAD_STATE,"prefix requires fresh graph");
    if (count>f->max_tokens || count>f->capacity-before.next_pos) return fail(e,Q38_CAPACITY,"append capacity");
    TRY(q38c_view(q38c_ro(logits),e));
    if (logits.rank!=2 || logits.dim[0]>count || logits.dim[0]>f->max_logits_rows || logits.dim[1]!=f->c.vocab_size ||
        logits.stride_bytes[0]!=(size_t)f->c.vocab_size*4) return fail(e,Q38_SHAPE,"dense bounded logits required");
    for (size_t i=0;i<count;i++) if (tokens[i]<0 || (uint32_t)tokens[i]>=f->c.vocab_size) return fail(e,Q38_RANGE,"token outside vocabulary");
    if (!count) return fail(e,Q38_OK,NULL);
    size_t output_bytes=q38c_span(q38c_ro(logits));
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
    f->trace_count=0; f->trace_offset=0; f->busy=true;
    Qwen38Status rc=prepare(f,tokens,count,layers,before.next_pos,e);
    Qwen38GpuTicket ticket; Qwen38GpuCommand cmd; bool begun=false,submitted=false;
    if (!rc) { rc=qwen38_gpu_session_begin(f->session,f->config,before.next_pos,count,&ticket,&cmd,e); begun=!rc; }
    if (!rc) rc=claim(f,&ticket,e);
    if (!rc) rc=run(f,cmd,count,layers,logits.dim[0],(uint32_t)before.next_pos,e);
    if (!rc) { rc=qwen38_gpu_session_submit(f->session,&ticket,e); submitted=!rc; }
    if (submitted) rc=qwen38_gpu_session_finish(f->session,&ticket,e);
    else if (begun) qwen38_gpu_session_abort(f->session,&ticket,false,NULL);
    rc=jobs(f,!rc && layers==f->c.trunk_blocks,rc,e);
    if (rc || layers<f->c.trunk_blocks) f->valid=false;
    else f->next_pos+=count;
    if (!rc) {
        if (logits.dim[0]) memcpy(logits.data,contents(f->logits),logits.dim[0]*f->c.vocab_size*4);
        if (trace) for (size_t j=0;j<f->trace_count;j++) {
            G2Trace *r=&f->traces[j]; Qwen38F32 v={.data=contents(r->view),.capacity_bytes=r->view.span.bytes,.rank=r->view.rank};
            memcpy(v.dim,r->view.dim,sizeof v.dim); memcpy(v.stride_bytes,r->view.stride_bytes,sizeof v.stride_bytes);
            trace(user,r->layer,r->stage,v);
        }
    }
    f->trace_span=(Qwen38GpuSpan){0}; f->busy=false;
    return rc ? rc : fail(e,Q38_OK,NULL);
}
Qwen38Status qwen38_gpu_forward_reset(Qwen38GpuForward *f,Qwen38Error *e) {
    if (!f) return fail(e,Q38_INVALID_ARGUMENT,"null graph");
    if (f->busy) return fail(e,Q38_BUSY,"graph busy");
    f->valid=false; TRY(qwen38_gpu_session_reset(f->session,f->config,e));
    uint64_t epoch=qwen38_gpu_session_info(f->session).key.epoch;
    for (uint32_t i=0;i<f->c.trunk_blocks;i++) if (f->layer[i].estate) TRY(qwen38_ngram_state_reset(f->layer[i].estate,epoch,e));
    for (id<MTLBuffer> b in (__bridge NSArray *)f->histories) memset(b.contents,0,b.length);
    f->next_pos=0; f->valid=true; return fail(e,Q38_OK,NULL);
}
