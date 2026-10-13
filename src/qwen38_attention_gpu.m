/* Private D Metal stage glue; caller retains buffers and owns submission. */
#import <Metal/Metal.h>
#include "qwen38_d_internal.h"
#include "qwen38_c_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
struct Q38DGpu { void *device, *pipeline[Q38D_ALL_OPS]; };
static Qwen38Status result(Qwen38Error *e,Qwen38Status code) {
    if (e) { memset(e,0,sizeof(*e)); e->code=code; } return code;
}
void q38d_gpu_destroy(Q38DGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    for (size_t i=0;i<Q38D_ALL_OPS;++i) if (g->pipeline[i]) CFRelease(g->pipeline[i]);
    free(g);
}
Qwen38Status q38d_gpu_create(void *device,void *library,Q38DGpu **out,Qwen38Error *e) {
    if (!out) return result(e,Q38_INVALID_ARGUMENT);
    *out=NULL;
    if (!device || !library) return result(e,Q38_INVALID_ARGUMENT);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device!=dev) return result(e,Q38_OWNER);
    Q38DGpu *g=calloc(1,sizeof(*g)); if (!g) return result(e,Q38_OOM);
    g->device=(__bridge_retained void *)dev;
    NSString *names[]={@"q38d_rope_gpu",@"q38d_pool_gpu",@"q38d_scores_gpu",@"q38d_attend_gpu",
        @"q38d_rank_gpu",@"q38d_compact_gpu",@"q38d_tail_gpu",@"q38d_copy_gpu"};
    for (size_t i=0;i<Q38D_ALL_OPS;++i) {
        id<MTLFunction> f=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=f ? [dev newComputePipelineStateWithFunction:f error:NULL] : nil;
        if (!p) { q38d_gpu_destroy(g); return result(e,Q38_GPU_ERROR); }
        g->pipeline[i]=(__bridge_retained void *)p;
    }
    *out=g; return result(e,Q38_OK);
}
static Qwen38Status span(Qwen38GpuSpan s,id<MTLDevice> dev,size_t need) {
    if (!s.buffer) return Q38_INVALID_ARGUMENT;
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)s.buffer;
    if (b.device!=dev) return Q38_OWNER;
    if (s.offset_bytes%4) return Q38_STRIDE;
    if (s.offset_bytes>b.length || s.bytes>b.length-s.offset_bytes || need>s.bytes) return Q38_CAPACITY;
    if (b.hazardTrackingMode==MTLHazardTrackingModeUntracked) return Q38_UNSUPPORTED;
    return Q38_OK;
}
static bool overlaps(Qwen38GpuSpan a,size_t na,Qwen38GpuSpan b,size_t nb) {
    if (!na || !nb) return false;
    if (a.buffer==b.buffer) return a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared) {
        uintptr_t p=(uintptr_t)x.contents+a.offset_bytes,q=(uintptr_t)y.contents+b.offset_bytes;
        return p<=q ? q-p<na : p-q<nb;
    }
    return false;
}
Qwen38Status q38d_gpu_encode(Q38DGpu *g,Qwen38GpuCommand cmd,uint32_t op,
    const Q38DGpuArgs *a,const Qwen38GpuSpan *v,size_t count,Qwen38Error *e) {
    if (!g || !a || !v || !cmd.device || !cmd.compute_encoder) return result(e,Q38_INVALID_ARGUMENT);
    if (op>=Q38D_GPU_OPS) return result(e,Q38_UNSUPPORTED);
    size_t slots=op==Q38D_ATTEND ? 6 : (op==Q38D_SCORES ? 3 : 2);
    if (count!=slots) return result(e,Q38_SHAPE);
    /* Explicit conservative cap; all products below bounded within uint32. */
    if (!a->h || a->h>128 || !a->d || a->d>256 || a->n>32768) return result(e,Q38_RANGE);
    size_t hd=(size_t)a->h*a->d,need[6]={0},work=a->h;
    if (op==Q38D_ROPE) {
        if (!a->r || a->r%2 || a->r>a->d || !isfinite(a->theta) || a->theta<=0) return result(e,Q38_RANGE);
        need[0]=need[1]=hd*4;
    } else if (op==Q38D_POOL) {
        if (!a->n) return result(e,Q38_RANGE);
        need[0]=(size_t)a->n*a->d*4; need[1]=a->d*4; work=a->d;
    } else if (op==Q38D_SCORES) {
        need[0]=hd*4; need[1]=(size_t)a->n*a->d*4; need[2]=a->n*4; work=a->n;
    } else {
        if (!a->j || a->j>a->h || a->h%a->j || !a->m || a->m>a->n) return result(e,Q38_SHAPE);
        need[0]=need[1]=need[5]=hd*4;
        need[2]=need[3]=(size_t)a->n*a->j*a->d*4; need[4]=a->m*4;
    }
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return result(e,Q38_OWNER);
    Qwen38Status code=span(cmd.error_word,dev,4);
    if (code!=Q38_OK) return result(e,code);
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared)
        return result(e,Q38_CAPACITY);
    for (size_t i=0;i<count;++i) {
        code=span(v[i],dev,need[i]); if (code!=Q38_OK) return result(e,code);
        if (overlaps(v[i],need[i],cmd.error_word,4)) return result(e,Q38_INVALID_ARGUMENT);
        for (size_t j=0;j<i;++j)
            if (overlaps(v[i],need[i],v[j],need[j])) return result(e,Q38_INVALID_ARGUMENT);
    }
    if (!work) return result(e,Q38_OK);
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)g->pipeline[op];
    [enc setComputePipelineState:p];
    for (size_t i=0;i<count;++i)
        [enc setBuffer:(__bridge id<MTLBuffer>)v[i].buffer offset:v[i].offset_bytes atIndex:i];
    [enc setBytes:a length:sizeof(*a) atIndex:6];
    if (op==Q38D_ROPE) {
        float frequencies[128];
        for (uint32_t i=0;i<a->r/2;++i) frequencies[i]=q38d_rope_frequency(a->theta,i,a->r);
        [enc setBytes:frequencies length:a->r/2*sizeof(float) atIndex:8];
    }
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:7];
    [enc dispatchThreads:MTLSizeMake(work,1,1)
        threadsPerThreadgroup:MTLSizeMake(MIN(work,p.maxTotalThreadsPerThreadgroup),1,1)];
    return result(e,Q38_OK);
}

/* ---- G2 layer composition ---- */
static Qwen38GpuSpan row(Qwen38GpuF32 v, size_t first, size_t rows) {
    size_t pitch=v.stride_bytes[0];
    return (Qwen38GpuSpan){v.span.buffer, v.span.offset_bytes+first*pitch, rows*pitch};
}
static Qwen38GpuSpan words(Qwen38GpuSpan s, size_t first, size_t count) {
    return (Qwen38GpuSpan){s.buffer, s.offset_bytes+first*4, count*4};
}
/* Private dispatch of the selection/copy helpers; spans checked by caller. */
static void helper(Q38DGpu *g, Qwen38GpuCommand cmd, uint32_t op, const Q38DGpuArgs *a,
        const Qwen38GpuSpan *v, size_t count, size_t work) {
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)g->pipeline[op];
    [enc setComputePipelineState:p];
    for (size_t i=0;i<count;++i)
        [enc setBuffer:(__bridge id<MTLBuffer>)v[i].buffer offset:v[i].offset_bytes atIndex:i];
    [enc setBytes:a length:sizeof(*a) atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:7];
    [enc dispatchThreads:MTLSizeMake(work,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(work,p.maxTotalThreadsPerThreadgroup),1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static Qwen38Status stage(Q38DGpu *g, Qwen38GpuCommand cmd, uint32_t op, Q38DGpuArgs a,
        const Qwen38GpuSpan *v, size_t count, Qwen38Error *e) {
    Qwen38Status code=q38d_gpu_encode(g,cmd,op,&a,v,count,e);
    if (code==Q38_OK) [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return code;
}
Qwen38Status q38d_select_encode(Q38DGpu *g, Qwen38GpuCommand cmd,
        uint32_t visible, uint32_t ratio, uint32_t budget, Qwen38GpuSpan scores,
        Qwen38GpuSpan keep, Qwen38GpuSpan ids, Qwen38Error *e) {
    if (!g || !cmd.device || !cmd.compute_encoder) return result(e,Q38_INVALID_ARGUMENT);
    if (!ratio || !budget || budget%ratio || visible>32768) return result(e,Q38_RANGE);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return result(e,Q38_OWNER);
    uint32_t n=visible/ratio, sel=MIN(n,budget/ratio), tail=visible%ratio;
    Qwen38GpuSpan v[]={scores,keep,ids,cmd.error_word};
    size_t need[]={(size_t)n*4,(size_t)n*4,((size_t)sel*ratio+tail)*4,4};
    for (size_t i=0;i<4;++i) {
        Qwen38Status code=span(v[i],dev,need[i]);
        if (code) return result(e,code);
        for (size_t j=0;j<i;++j)
            if (overlaps(v[i],need[i],v[j],need[j])) return result(e,Q38_INVALID_ARGUMENT);
    }
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared)
        return result(e,Q38_CAPACITY);
    if (n) {
        Q38DGpuArgs a={.n=n,.m=sel,.r=ratio};
        helper(g,cmd,Q38D_RANK,&a,(Qwen38GpuSpan[]){scores,keep},2,n);
        helper(g,cmd,Q38D_COMPACT,&a,(Qwen38GpuSpan[]){keep,ids},2,n);
    }
    if (tail) {
        Q38DGpuArgs a={.m=sel*ratio,.n=n*ratio,.pos=visible-1};
        helper(g,cmd,Q38D_TAIL,&a,&ids,1,tail);
    }
    return result(e,Q38_OK);
}
/* Admit the complete composition before recording anything. Stage checks
 * alone cannot catch a later write aliasing an earlier stage's input/cache. */
static Qwen38Status layer_check(Q38DGpu *g, Qwen38GpuCommand cmd,
        const Qwen38AttentionConfig *c, uint32_t pos, uint32_t t, const Q38DGpuLayer *l) {
    size_t cap=l->key_cache.dim[0];
    if (!cap || cap>32768 || t>cap || pos>cap-t) return Q38_CAPACITY;
    Qwen38AttentionPlan plan;
    Qwen38Status code=qwen38_attention_plan(c,(uint32_t)cap,t ? t : 1,&plan,NULL);
    if (code) return code;
    if (c->heads>128 || c->index_heads>128 || c->head_dim>256 || c->index_dim>256 ||
        c->ratio>32768 || c->token_budget>32768) return Q38_RANGE;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return Q38_OWNER;
    if ((code=span(cmd.error_word,dev,4))) return code;
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared) return Q38_CAPACITY;
    size_t h=c->heads,j=c->kv_heads,d=c->head_dim,hi=c->index_heads,di=c->index_dim,nb=cap/c->ratio;
    Qwen38GpuF32 v[]={l->qg,l->k,l->v,l->iq,l->ik,l->gamma_q,l->gamma_k,l->gamma_iq,l->gamma_ik,
        l->key_cache,l->value_cache,l->index_cache,l->block_cache,l->qn,l->qr,l->gate,l->kn,l->iqn,l->iqr,
        l->pooled,l->pooled_n,l->scores,l->out};
    const size_t ranks[]={4,3,3,3,2,1,1,1,1,3,3,2,2,3,3,3,3,3,3,3,3,1,3};
    const size_t dims[][4]={{t,h,2,d},{t,j,d},{t,j,d},{t,hi,di},{t,di},{d},{d},{di},{di},
        {cap,j,d},{cap,j,d},{cap,di},{nb,di},{t,h,d},{t,h,d},{t,h,d},{t,j,d},{t,hi,di},{t,hi,di},
        {1,1,di},{1,1,di},{nb},{t,h,d}};
    Qwen38GpuSpan spans[25]; size_t need[25];
    for (size_t i=0;i<23;++i) {
        if (v[i].rank!=ranks[i]) return Q38_SHAPE;
        size_t bytes=4;
        for (size_t a=ranks[i];a-- >0;) {
            if (v[i].dim[a]!=dims[i][a]) return Q38_SHAPE;
            if (v[i].stride_bytes[a]!=bytes) return Q38_STRIDE;
            bytes*=dims[i][a]; /* bounded by the configuration caps above */
        }
        spans[i]=v[i].span; need[i]=bytes;
    }
    spans[23]=l->keep; need[23]=nb*4;
    spans[24]=l->ids; need[24]=((size_t)c->token_budget+c->ratio-1)*4;
    for (size_t i=0;i<25;++i) {
        if ((code=span(spans[i],dev,need[i]))) return code;
        if (overlaps(spans[i],need[i],cmd.error_word,4)) return Q38_INVALID_ARGUMENT;
        for (size_t j0=0;j0<i;++j0)
            if (i>=9 && overlaps(spans[i],need[i],spans[j0],need[j0])) return Q38_INVALID_ARGUMENT;
    }
    return Q38_OK;
}
Qwen38Status q38d_layer_encode(Q38DGpu *g, Q38CGpu *ops, Qwen38GpuCommand cmd,
        const Qwen38AttentionConfig *c, uint32_t pos0, uint32_t t, const Q38DGpuLayer *l, Qwen38Error *e) {
    if (!g || !ops || !c || !l || !cmd.device || !cmd.compute_encoder) return result(e,Q38_INVALID_ARGUMENT);
    Qwen38Status code=layer_check(g,cmd,c,pos0,t,l);
    if (code) return result(e,code);
    uint32_t h=c->heads, j=c->kv_heads, d=c->head_dim, hi=c->index_heads, di=c->index_dim, r=c->ratio;
    uint32_t kb=c->token_budget/r;
    if (!t) return result(e,Q38_OK);
    Q38CGpuSpec spec={.eps=c->rms_eps,.activation=Q38_SIGMOID};
    /* Batch: C norms per head and the sigmoid gate, over the whole append. */
    Qwen38GpuF32 q=l->qg, gate_in=l->qg;
    q.rank=3; q.dim[1]=h; q.dim[2]=d; q.stride_bytes[1]=2*(size_t)d*4; q.stride_bytes[2]=4;
    gate_in=q; gate_in.span.offset_bytes+=(size_t)d*4; gate_in.span.bytes-=(size_t)d*4;
#define C_OP(op,...) do { Qwen38GpuF32 vv[]={__VA_ARGS__}; \
    code=q38c_gpu_encode(ops,cmd,Q38C_##op,&spec,vv,sizeof vv/sizeof *vv,e); if (code) return code; } while (0)
    C_OP(RMS,q,l->gamma_q,l->qn);
    C_OP(ACT,gate_in,l->gate);
    C_OP(RMS,l->k,l->gamma_k,l->kn);
    C_OP(RMS,l->iq,l->gamma_iq,l->iqn);
    /* Raw V and raw index keys enter the cache for the whole append. */
    Q38DGpuArgs copy={.n=t*j*d};
    helper(g,cmd,Q38D_COPY,&copy,(Qwen38GpuSpan[]){l->v.span,row(l->value_cache,pos0,t)},2,copy.n);
    copy.n=t*di;
    helper(g,cmd,Q38D_COPY,&copy,(Qwen38GpuSpan[]){l->ik.span,row(l->index_cache,pos0,t)},2,copy.n);
    for (uint32_t i=0;i<t;++i) {
        uint32_t p=pos0+i, n=(p+1)/r, sel=n<kb ? n : kb;
        Q38DGpuArgs a={.h=h,.d=d,.r=c->rotary_dim,.pos=p,.theta=c->rope_theta};
        if ((code=stage(g,cmd,Q38D_ROPE,a,(Qwen38GpuSpan[]){row(l->qn,i,1),row(l->qr,i,1)},2,e))) return code;
        a.h=j;
        if ((code=stage(g,cmd,Q38D_ROPE,a,(Qwen38GpuSpan[]){row(l->kn,i,1),row(l->key_cache,p,1)},2,e))) return code;
        a.h=hi; a.d=di;
        if ((code=stage(g,cmd,Q38D_ROPE,a,(Qwen38GpuSpan[]){row(l->iqn,i,1),row(l->iqr,i,1)},2,e))) return code;
        if ((p+1)%r==0) {                         /* block b completes at this token */
            uint32_t b=p/r;
            Q38DGpuArgs pool={.h=1,.d=di,.n=r};
            if ((code=stage(g,cmd,Q38D_POOL,pool,(Qwen38GpuSpan[]){row(l->index_cache,(size_t)b*r,r),l->pooled.span},2,e)))
                return code;
            C_OP(RMS,l->pooled,l->gamma_ik,l->pooled_n);
            Q38DGpuArgs rope={.h=1,.d=di,.r=c->rotary_dim,.pos=b*r,.theta=c->rope_theta};
            if ((code=stage(g,cmd,Q38D_ROPE,rope,(Qwen38GpuSpan[]){l->pooled_n.span,row(l->block_cache,b,1)},2,e)))
                return code;
        }
        if (n) {
            Q38DGpuArgs sc={.h=hi,.d=di,.n=n,.m=sel,.r=r};
            if ((code=stage(g,cmd,Q38D_SCORES,sc,(Qwen38GpuSpan[]){row(l->iqr,i,1),row(l->block_cache,0,n),
                words(l->scores.span,0,n)},3,e))) return code;
        }
        uint32_t tail=p+1-n*r, m=sel*r+tail;
        if ((code=q38d_select_encode(g,cmd,p+1,r,c->token_budget,l->scores.span,l->keep,l->ids,e))) return code;
        Q38DGpuArgs at={.h=h,.j=j,.d=d,.n=p+1,.m=m};
        if ((code=stage(g,cmd,Q38D_ATTEND,at,(Qwen38GpuSpan[]){row(l->qr,i,1),row(l->gate,i,1),
            row(l->key_cache,0,p+1),row(l->value_cache,0,p+1),words(l->ids,0,m),row(l->out,i,1)},6,e))) return code;
    }
#undef C_OP
    return result(e,Q38_OK);
}
