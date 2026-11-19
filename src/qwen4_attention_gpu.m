/* Private D Metal stage glue; caller retains buffers and owns submission. */
#import <Metal/Metal.h>
#include "qwen4_attention_internal.h"
#include "qwen4_cpu_internal.h"
#include "qwen4_ref.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
struct QW4DGpu { void *device, *pipeline[QW4D_ALL_OPS], *mat[3]; bool scalar_attention, reference, mma; };
/* P5.1: the MMA split kernel for this layout and this many keys per split (else the
 * R2.1 split kernel; QW4_QSA_SPLIT set at creation keeps the R2.1 kernel always). */
static bool use_mma(const QW4DGpu *g, uint32_t h, uint32_t j, uint32_t d, uint32_t keys) {
    return g->mma && d==QW4D_MMA_D && j && h%j==0 && h/j>=8 && h/j<=16 && keys<=QW4D_MMA_KEYS;
}
static Qwen4Status result(Qwen4Error *e,Qwen4Status code) {
    if (e) { memset(e,0,sizeof(*e)); e->code=code; } return code;
}
void qw4d_gpu_destroy(QW4DGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    for (size_t i=0;i<QW4D_ALL_OPS;++i) if (g->pipeline[i]) CFRelease(g->pipeline[i]);
    for (size_t i=0;i<3;++i) if (g->mat[i]) CFRelease(g->mat[i]);
    free(g);
}
Qwen4Status qw4d_gpu_create(void *device,void *library,QW4DGpu **out,Qwen4Error *e) {
    if (!out) return result(e,QW4_INVALID_ARGUMENT);
    *out=NULL;
    if (!device || !library) return result(e,QW4_INVALID_ARGUMENT);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device!=dev) return result(e,QW4_OWNER);
    QW4DGpu *g=calloc(1,sizeof(*g)); if (!g) return result(e,QW4_OOM);
    g->device=(__bridge_retained void *)dev;
    NSString *names[]={@"qw4d_rope_gpu",@"qw4d_pool_gpu",@"qw4d_scores_gpu",@"qw4d_attend_gpu",
        @"qw4d_rank_gpu",@"qw4d_compact_gpu",@"qw4d_tail_gpu",@"qw4d_copy_gpu",
        @"qw4d_attend_split_gpu",@"qw4d_attend_merge_gpu",
        @"qw4d_scores_fast_gpu",@"qw4d_select_gpu",@"qw4d_rope_fast_gpu",
        @"qw4d_rope_batch_gpu",@"qw4d_pool_batch_gpu",@"qw4d_scores_batch_gpu",@"qw4d_select_batch_gpu",
        @"qw4d_tail_batch_gpu",@"qw4d_split_batch_gpu",@"qw4d_merge_batch_gpu",
        @"qw4d_attend_split_mma_gpu",@"qw4d_split_mma_batch_gpu",@"qw4d_merge_mma_batch_gpu"};
    g->scalar_attention=getenv("QW4_QSA_SCALAR")!=NULL;
    g->reference=qw4_ref_kernels("attention");
    g->mma=getenv("QW4_QSA_SPLIT")==NULL;
    if (g->scalar_attention) names[QW4D_ATTEND]=@"qw4d_attend_scalar_gpu";
    for (size_t i=0;i<QW4D_ALL_OPS;++i) {
        id<MTLFunction> f=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=f ? [dev newComputePipelineStateWithFunction:f error:NULL] : nil;
        if (!p) { qw4d_gpu_destroy(g); return result(e,QW4_GPU_ERROR); }
        if (i==QW4D_ATTEND && !g->scalar_attention &&
            (p.maxTotalThreadsPerThreadgroup<256 || p.staticThreadgroupMemoryLength>dev.maxThreadgroupMemoryLength)) {
            qw4d_gpu_destroy(g); return result(e,QW4_UNSUPPORTED);
        }
        if (i>=QW4D_SPLIT_MMA) {   /* 4 simdgroups and 13 KB of threadgroup memory (merge: d threads), or the R2.1 kernels */
            if (p.maxTotalThreadsPerThreadgroup<(i==QW4D_MERGE_MMA_BATCH ? 256u : 128u) || p.threadExecutionWidth!=32 ||
                p.staticThreadgroupMemoryLength>dev.maxThreadgroupMemoryLength) g->mma=false;
            g->pipeline[i]=(__bridge_retained void *)p;
            continue;
        }
        if ((i==QW4D_SPLIT && p.maxTotalThreadsPerThreadgroup<128) || (i>=QW4D_MERGE && p.maxTotalThreadsPerThreadgroup<256))
            g->reference=true;  /* the fast path needs 4 simdgroups and one thread per feature */
        if (i>=QW4D_SPLIT && p.threadExecutionWidth!=32)
            g->reference=true;  /* the fast kernels' simd_sum, simd_any and 8-lane groups assume 32 lanes */
        g->pipeline[i]=(__bridge_retained void *)p;
    }
    /* Prefill-gap: the materialized attention for prefill batches (8K FP32: qsa.core 4.91 ->
     * 2.57 s). QW4_QSA_MAT=0 or QW4_REF_KERNELS=exact keep the MMA split (and its decode bits);
     * the attention oracle family keeps the reference path anyway. */
    const char *mv=getenv("QW4_QSA_MAT");
    if (!(mv && *mv=='0') && !qw4_ref_kernels("exact")) {
        NSString *const mat[3]={@"qw4d_mat_scores_gpu",@"qw4d_mat_softmax_gpu",@"qw4d_mat_pv_gpu"};
        bool ok=true; id<MTLComputePipelineState> ps[3];
        for (size_t i=0;i<3;i++) {
            id<MTLFunction> f=[lib newFunctionWithName:mat[i]];
            ps[i]=f ? [dev newComputePipelineStateWithFunction:f error:NULL] : nil;
            ok=ok && ps[i] && ps[i].maxTotalThreadsPerThreadgroup>=(i==1 ? 256u : 128u) && ps[i].threadExecutionWidth==32;
        }
        if (ok) for (size_t i=0;i<3;i++) g->mat[i]=(__bridge_retained void *)ps[i];
    }
    *out=g; return result(e,QW4_OK);
}
static Qwen4Status span(Qwen4GpuSpan s,id<MTLDevice> dev,size_t need) {
    if (!s.buffer) return QW4_INVALID_ARGUMENT;
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)s.buffer;
    if (b.device!=dev) return QW4_OWNER;
    if (s.offset_bytes%4) return QW4_STRIDE;
    if (s.offset_bytes>b.length || s.bytes>b.length-s.offset_bytes || need>s.bytes) return QW4_CAPACITY;
    if (b.hazardTrackingMode==MTLHazardTrackingModeUntracked) return QW4_UNSUPPORTED;
    return QW4_OK;
}
static bool overlaps(Qwen4GpuSpan a,size_t na,Qwen4GpuSpan b,size_t nb) {
    if (!na || !nb) return false;
    if (a.buffer==b.buffer) return a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
    id<MTLBuffer> x=(__bridge id<MTLBuffer>)a.buffer,y=(__bridge id<MTLBuffer>)b.buffer;
    if (x.storageMode==MTLStorageModeShared && y.storageMode==MTLStorageModeShared) {
        uintptr_t p=(uintptr_t)x.contents+a.offset_bytes,q=(uintptr_t)y.contents+b.offset_bytes;
        return p<=q ? q-p<na : p-q<nb;
    }
    return false;
}
/* Admission of one public op: shapes, caps, spans, aliasing. *work = threads. */
static Qwen4Status admit(QW4DGpu *g,Qwen4GpuCommand cmd,uint32_t op,
    const QW4DGpuArgs *a,const Qwen4GpuSpan *v,size_t count,size_t *work_out) {
    if (!g || !a || !v || !cmd.device || !cmd.compute_encoder) return QW4_INVALID_ARGUMENT;
    if (op>=QW4D_GPU_OPS) return QW4_UNSUPPORTED;
    size_t slots=op==QW4D_ATTEND ? 6 : (op==QW4D_SCORES ? 3 : 2);
    if (count!=slots) return QW4_SHAPE;
    /* Explicit conservative cap; all products below bounded within uint32. */
    if (!a->h || a->h>128 || !a->d || a->d>256 || a->n>QWEN4_MAX_CONTEXT) return QW4_RANGE;
    size_t hd=(size_t)a->h*a->d,need[6]={0},work=a->h;
    if (op==QW4D_ROPE) {
        if (!a->r || a->r%2 || a->r>a->d || !isfinite(a->theta) || a->theta<=0) return QW4_RANGE;
        need[0]=need[1]=hd*4;
    } else if (op==QW4D_POOL) {
        if (!a->n) return QW4_RANGE;
        need[0]=(size_t)a->n*a->d*4; need[1]=a->d*4; work=a->d;
    } else if (op==QW4D_SCORES) {
        need[0]=hd*4; need[1]=(size_t)a->n*a->d*4; need[2]=a->n*4; work=a->n;
    } else {
        if (!a->j || a->j>a->h || a->h%a->j || !a->m || a->m>a->n) return QW4_SHAPE;
        need[0]=need[1]=need[5]=hd*4;
        need[2]=need[3]=(size_t)a->n*a->j*a->d*4; need[4]=a->m*4;
    }
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return QW4_OWNER;
    Qwen4Status code=span(cmd.error_word,dev,4);
    if (code!=QW4_OK) return code;
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared)
        return QW4_CAPACITY;
    for (size_t i=0;i<count;++i) {
        code=span(v[i],dev,need[i]); if (code!=QW4_OK) return code;
        if (overlaps(v[i],need[i],cmd.error_word,4)) return QW4_INVALID_ARGUMENT;
        for (size_t j=0;j<i;++j)
            if (overlaps(v[i],need[i],v[j],need[j])) return QW4_INVALID_ARGUMENT;
    }
    *work_out=work; return QW4_OK;
}
static void bind(QW4DGpu *g,Qwen4GpuCommand cmd,uint32_t op,const QW4DGpuArgs *a,const Qwen4GpuSpan *v,size_t count) {
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->pipeline[op]];
    for (size_t i=0;i<count;++i)
        [enc setBuffer:(__bridge id<MTLBuffer>)v[i].buffer offset:v[i].offset_bytes atIndex:i];
    [enc setBytes:a length:sizeof(*a) atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:7];
}
Qwen4Status qw4d_gpu_encode(QW4DGpu *g,Qwen4GpuCommand cmd,uint32_t op,
    const QW4DGpuArgs *a,const Qwen4GpuSpan *v,size_t count,Qwen4Error *e) {
    size_t work=0;
    Qwen4Status code=admit(g,cmd,op,a,v,count,&work);
    if (code || !work) return result(e,code);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    /* R2.2/R2.8: the exact fast scores and RoPE (same bits) unless the oracle is selected:
     * scores QW4D_SCORE_LANES threads per block (one per head), RoPE one per element. */
    if (op==QW4D_SCORES && !g->reference && a->h<=QW4D_SCORE_LANES) { op=QW4D_SCORES_FAST; work=(size_t)a->n*QW4D_SCORE_LANES; }
    if (op==QW4D_ROPE && !g->reference) { op=QW4D_ROPE_FAST; work=(size_t)a->h*a->d; }
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)g->pipeline[op];
    bind(g,cmd,op,a,v,count);
    if (op==QW4D_ROPE || op==QW4D_ROPE_FAST) {
        float frequencies[128];
        for (uint32_t i=0;i<a->r/2;++i) frequencies[i]=qw4d_rope_frequency(a->theta,i,a->r);
        [enc setBytes:frequencies length:a->r/2*sizeof(float) atIndex:8];
    }
    if (op==QW4D_ATTEND && !g->scalar_attention)
        [enc dispatchThreadgroups:MTLSizeMake(a->h,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    else if (op==QW4D_SCORES_FAST || op==QW4D_ROPE_FAST) /* whole 8-lane groups per threadgroup */
        [enc dispatchThreads:MTLSizeMake(work,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(work,(size_t)256),1,1)];
    else [enc dispatchThreads:MTLSizeMake(work,1,1)
        threadsPerThreadgroup:MTLSizeMake(MIN(work,p.maxTotalThreadsPerThreadgroup),1,1)];
    return result(e,QW4_OK);
}
void qw4d_gpu_set_reference(QW4DGpu *g,bool reference) { if (g) g->reference=reference; }
Qwen4Status qw4d_attend_encode(QW4DGpu *g,Qwen4GpuCommand cmd,const QW4DGpuArgs *a,
    const Qwen4GpuSpan v[6],Qwen4GpuSpan part,Qwen4Error *e) {
    size_t work=0;
    Qwen4Status code=admit(g,cmd,QW4D_ATTEND,a,v,6,&work);
    if (code) return result(e,code);
    uint32_t splits,keys; qw4d_attend_splits(a->m,&splits,&keys);
    size_t need=(size_t)splits*a->h*(a->d+2)*4;
    if (g->reference || g->scalar_attention || a->d%32 || a->h/a->j>16 || !part.buffer || part.bytes<need)
        return qw4d_gpu_encode(g,cmd,QW4D_ATTEND,a,v,6,e);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    if ((code=span(part,dev,need))) return result(e,code);
    if (overlaps(part,need,cmd.error_word,4)) return result(e,QW4_INVALID_ARGUMENT);
    for (size_t i=0;i<6;++i)  /* the views' extents as admitted by qw4d_gpu_encode */
        if (overlaps(part,need,v[i],v[i].bytes)) return result(e,QW4_INVALID_ARGUMENT);
    QW4DGpuArgs s=*a; s.r=keys; s.pos=splits;  /* split kernels: r = keys per split, pos = splits */
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    const bool mma=use_mma(g,a->h,a->j,a->d,keys);
    /* P5.2: an MMA threadgroup covers QW4D_MMA_GROUP splits; the merge sees its partials. */
    const uint32_t parts=mma ? (splits+QW4D_MMA_GROUP-1)/QW4D_MMA_GROUP : splits;
    bind(g,cmd,mma ? QW4D_SPLIT_MMA : QW4D_SPLIT,&s,v,6);
    [enc setBuffer:(__bridge id<MTLBuffer>)part.buffer offset:part.offset_bytes atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(parts,a->j,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    if (parts==1) return result(e,QW4_OK);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    s.pos=parts;
    bind(g,cmd,QW4D_MERGE,&s,v,6);
    [enc setBuffer:(__bridge id<MTLBuffer>)part.buffer offset:part.offset_bytes atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(a->h,1,1) threadsPerThreadgroup:MTLSizeMake(a->d,1,1)];
    return result(e,QW4_OK);
}

/* ---- G2 layer composition ---- */
static Qwen4GpuSpan row(Qwen4GpuF32 v, size_t first, size_t rows) {
    size_t pitch=v.stride_bytes[0];
    return (Qwen4GpuSpan){v.span.buffer, v.span.offset_bytes+first*pitch, rows*pitch};
}
static Qwen4GpuSpan words(Qwen4GpuSpan s, size_t first, size_t count) {
    return (Qwen4GpuSpan){s.buffer, s.offset_bytes+first*4, count*4};
}
/* Private dispatch of the selection/copy helpers; spans checked by caller. */
static void helper(QW4DGpu *g, Qwen4GpuCommand cmd, uint32_t op, const QW4DGpuArgs *a,
        const Qwen4GpuSpan *v, size_t count, size_t work) {
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
static Qwen4Status stage(QW4DGpu *g, Qwen4GpuCommand cmd, uint32_t op, QW4DGpuArgs a,
        const Qwen4GpuSpan *v, size_t count, Qwen4Error *e) {
    Qwen4Status code=qw4d_gpu_encode(g,cmd,op,&a,v,count,e);
    if (code==QW4_OK) [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return code;
}
/* Threads of a selection threadgroup: up to 1024, a whole number of simdgroups. */
static NSUInteger select_threads(QW4DGpu *g,uint32_t op) {
    NSUInteger most=((__bridge id<MTLComputePipelineState>)g->pipeline[op]).maxTotalThreadsPerThreadgroup;
    return MIN(most,(NSUInteger)1024)/32*32;
}
Qwen4Status qw4d_select_encode(QW4DGpu *g, Qwen4GpuCommand cmd,
        uint32_t visible, uint32_t ratio, uint32_t budget, Qwen4GpuSpan scores,
        Qwen4GpuSpan keep, Qwen4GpuSpan ids, Qwen4Error *e) {
    if (!g || !cmd.device || !cmd.compute_encoder) return result(e,QW4_INVALID_ARGUMENT);
    if (!ratio || !budget || budget%ratio || visible>QWEN4_MAX_CONTEXT) return result(e,QW4_RANGE);
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return result(e,QW4_OWNER);
    uint32_t n=visible/ratio, sel=MIN(n,budget/ratio), tail=visible%ratio;
    Qwen4GpuSpan v[]={scores,keep,ids,cmd.error_word};
    size_t need[]={(size_t)n*4,(size_t)n*4,((size_t)sel*ratio+tail)*4,4};
    for (size_t i=0;i<4;++i) {
        Qwen4Status code=span(v[i],dev,need[i]);
        if (code) return result(e,code);
        for (size_t j=0;j<i;++j)
            if (overlaps(v[i],need[i],v[j],need[j])) return result(e,QW4_INVALID_ARGUMENT);
    }
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared)
        return result(e,QW4_CAPACITY);
    if (n && !g->reference) {  /* long context: O(n) radix selection, same keep flags and IDs */
        QW4DGpuArgs a={.n=n,.m=sel,.r=ratio};
        bind(g,cmd,QW4D_SELECT_FAST,&a,(Qwen4GpuSpan[]){scores,keep,ids},3);
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(select_threads(g,QW4D_SELECT_FAST),1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else if (n) {
        QW4DGpuArgs a={.n=n,.m=sel,.r=ratio};
        helper(g,cmd,QW4D_RANK,&a,(Qwen4GpuSpan[]){scores,keep},2,n);
        helper(g,cmd,QW4D_COMPACT,&a,(Qwen4GpuSpan[]){keep,ids},2,n);
    }
    if (tail) {
        QW4DGpuArgs a={.m=sel*ratio,.n=n*ratio,.pos=visible-1};
        helper(g,cmd,QW4D_TAIL,&a,&ids,1,tail);
    }
    return result(e,QW4_OK);
}
/* Admit the complete composition before recording anything. Stage checks
 * alone cannot catch a later write aliasing an earlier stage's input/cache. */
static Qwen4Status layer_check(QW4DGpu *g, Qwen4GpuCommand cmd,
        const Qwen4AttentionConfig *c, uint32_t pos, uint32_t t, const QW4DGpuLayer *l) {
    size_t cap=l->key_cache.dim[0];
    if (!cap || cap>QWEN4_MAX_CONTEXT || t>cap || pos>cap-t) return QW4_CAPACITY;
    Qwen4AttentionPlan plan;
    Qwen4Status code=qwen4_attention_plan(c,(uint32_t)cap,t ? t : 1,&plan,NULL);
    if (code) return code;
    if (c->heads>128 || c->index_heads>128 || c->head_dim>256 || c->index_dim>256 ||
        c->ratio>32768 || c->token_budget>32768) return QW4_RANGE;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (cmd.device!=g->device || enc.device!=dev) return QW4_OWNER;
    if ((code=span(cmd.error_word,dev,4))) return code;
    if ([(__bridge id<MTLBuffer>)cmd.error_word.buffer storageMode]!=MTLStorageModeShared) return QW4_CAPACITY;
    size_t h=c->heads,j=c->kv_heads,d=c->head_dim,hi=c->index_heads,di=c->index_dim,nb=cap/c->ratio;
    Qwen4GpuF32 v[]={l->qg,l->k,l->v,l->iq,l->ik,l->gamma_q,l->gamma_k,l->gamma_iq,l->gamma_ik,
        l->key_cache,l->value_cache,l->index_cache,l->block_cache,l->qn,l->qr,l->gate,l->kn,l->iqn,l->iqr,
        l->pooled,l->pooled_n,l->scores,l->out};
    const size_t ranks[]={4,3,3,3,2,1,1,1,1,3,3,2,2,3,3,3,3,3,3,3,3,1,3};
    const size_t dims[][4]={{t,h,2,d},{t,j,d},{t,j,d},{t,hi,di},{t,di},{d},{d},{di},{di},
        {cap,j,d},{cap,j,d},{cap,di},{nb,di},{t,h,d},{t,h,d},{t,h,d},{t,j,d},{t,hi,di},{t,hi,di},
        {1,1,di},{1,1,di},{nb},{t,h,d}};
    Qwen4GpuSpan spans[25]; size_t need[25];
    for (size_t i=0;i<23;++i) {
        if (v[i].rank!=ranks[i]) return QW4_SHAPE;
        size_t bytes=4;
        for (size_t a=ranks[i];a-- >0;) {
            if (v[i].dim[a]!=dims[i][a]) return QW4_SHAPE;
            if (v[i].stride_bytes[a]!=bytes) return QW4_STRIDE;
            bytes*=dims[i][a]; /* bounded by the configuration caps above */
        }
        spans[i]=v[i].span; need[i]=bytes;
    }
    spans[23]=l->keep; need[23]=nb*4;
    spans[24]=l->ids; need[24]=((size_t)c->token_budget+c->ratio-1)*4;
    for (size_t i=0;i<25;++i) {
        if ((code=span(spans[i],dev,need[i]))) return code;
        if (overlaps(spans[i],need[i],cmd.error_word,4)) return QW4_INVALID_ARGUMENT;
        for (size_t j0=0;j0<i;++j0)
            if (i>=9 && overlaps(spans[i],need[i],spans[j0],need[j0])) return QW4_INVALID_ARGUMENT;
    }
    if (l->part.buffer) {  /* optional split partials: written, so disjoint from every view */
        if ((code=span(l->part,dev,l->part.bytes))) return code;
        if (overlaps(l->part,l->part.bytes,cmd.error_word,4)) return QW4_INVALID_ARGUMENT;
        for (size_t i=0;i<25;++i) if (overlaps(l->part,l->part.bytes,spans[i],need[i])) return QW4_INVALID_ARGUMENT;
    }
    return QW4_OK;
}
/* R3.3: one batch dispatch, grid (x, count); z rows of the batch kernel. */
static void batch(QW4DGpu *g, Qwen4GpuCommand cmd, uint32_t op, const QW4DGpuArgs *a, const QW4DBatchArgs *b,
        const Qwen4GpuSpan *v, size_t count, MTLSize grid, MTLSize threads, bool groups) {
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    bind(g,cmd,op,a,v,count);
    if (count>6) [enc setBuffer:(__bridge id<MTLBuffer>)v[6].buffer offset:v[6].offset_bytes atIndex:8]; /* split partials */
    [enc setBytes:b length:sizeof(*b) atIndex:9];
    if (op==QW4D_ROPE_BATCH) {
        float frequencies[128];
        for (uint32_t i=0;i<a->r/2;++i) frequencies[i]=qw4d_rope_frequency(a->theta,i,a->r);
        [enc setBytes:frequencies length:a->r/2*sizeof(float) atIndex:8];
    }
    if (groups) [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    else [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}
static void wait_all(Qwen4GpuCommand cmd) {
    [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static Qwen4GpuF32 rows3(Qwen4GpuF32 v, size_t n) { v.dim[0]=n; v.span.bytes=n*v.stride_bytes[0]; return v; }
/* The per-token loop of qw4d_layer_encode as whole-append stages (see
 * qwen4_attention.metal, R3.3): keys, index keys and queries rotated for all
 * tokens, then the blocks completed in the append pooled, normed and rotated,
 * then scores, rank, compact, tail and split attention per token in parallel. */
static Qwen4Status layer_batch(QW4DGpu *g, QW4CGpu *ops, Qwen4GpuCommand cmd,
        const Qwen4AttentionConfig *c, uint32_t pos0, uint32_t t, const QW4DGpuLayer *l, Qwen4Error *e) {
    uint32_t h=c->heads, j=c->kv_heads, d=c->head_dim, hi=c->index_heads, di=c->index_dim, r=c->ratio;
    uint32_t cap=(uint32_t)l->key_cache.dim[0], stride=cap/r, ids_stride=c->token_budget+r-1;
    uint32_t n0=pos0/r, n1=(pos0+t)/r, nb=n1-n0;
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device; Qwen4Status code;
    const Qwen4GpuSpan need_v[]={l->bscores,l->bkeep,l->bids,l->bpart,l->bpooled.span,l->bpooled_n.span};
    const size_t need[]={(size_t)t*stride*4,(size_t)t*stride*4,(size_t)t*ids_stride*4,
        (size_t)QW4D_BATCH_SPLIT_TOKENS*QW4D_MAX_SPLITS*h*(d+2)*4,((size_t)nb+1)*di*4,((size_t)nb+1)*di*4};
    for (size_t i=0;i<6;i++) {
        if ((code=span(need_v[i],dev,need[i]))) return result(e,code);
        if (overlaps(need_v[i],need[i],cmd.error_word,4)) return result(e,QW4_INVALID_ARGUMENT);
    }
    QW4DGpuArgs a={.h=h,.d=d,.r=c->rotary_dim,.theta=c->rope_theta};
    QW4DBatchArgs b={.pos0=pos0,.step=1,.x_stride=h*d,.y_stride=h*d,.ratio=r,.budget=c->token_budget,
        .scores_stride=stride,.ids_stride=ids_stride,.count=t};
    batch(g,cmd,QW4D_ROPE_BATCH,&a,&b,(Qwen4GpuSpan[]){row(l->qn,0,t),row(l->qr,0,t)},2,
          MTLSizeMake(h*d,t,1),MTLSizeMake(256,1,1),false);
    a.h=j; b.x_stride=b.y_stride=j*d;
    batch(g,cmd,QW4D_ROPE_BATCH,&a,&b,(Qwen4GpuSpan[]){row(l->kn,0,t),row(l->key_cache,pos0,t)},2,
          MTLSizeMake(j*d,t,1),MTLSizeMake(256,1,1),false);
    a.h=hi; a.d=di; b.x_stride=b.y_stride=hi*di;
    batch(g,cmd,QW4D_ROPE_BATCH,&a,&b,(Qwen4GpuSpan[]){row(l->iqn,0,t),row(l->iqr,0,t)},2,
          MTLSizeMake(hi*di,t,1),MTLSizeMake(256,1,1),false);
    wait_all(cmd);
    if (nb) { /* blocks n0 .. n1-1 complete in this append */
        QW4DGpuArgs p={.h=1,.d=di,.n=r};
        QW4DBatchArgs pb={.x_stride=r*di,.y_stride=di,.ratio=r,.count=nb};
        batch(g,cmd,QW4D_POOL_BATCH,&p,&pb,(Qwen4GpuSpan[]){row(l->index_cache,(size_t)n0*r,(size_t)nb*r),l->bpooled.span},2,
              MTLSizeMake(di,nb,1),MTLSizeMake(MIN(di,256u),1,1),false);
        wait_all(cmd);
        QW4CGpuSpec spec={.eps=c->rms_eps};
        Qwen4GpuF32 vv[]={rows3(l->bpooled,nb),l->gamma_ik,rows3(l->bpooled_n,nb)};
        if ((code=qw4c_gpu_encode(ops,cmd,QW4C_RMS,&spec,vv,3,e))) return code;
        wait_all(cmd);
        QW4DGpuArgs rope={.h=1,.d=di,.r=c->rotary_dim,.theta=c->rope_theta};
        QW4DBatchArgs rb={.pos0=n0*r,.step=r,.x_stride=di,.y_stride=di,.ratio=r,.count=nb};
        batch(g,cmd,QW4D_ROPE_BATCH,&rope,&rb,(Qwen4GpuSpan[]){l->bpooled_n.span,row(l->block_cache,n0,nb)},2,
              MTLSizeMake(di,nb,1),MTLSizeMake(MIN(di,256u),1,1),false);
        wait_all(cmd);
    }
    if (n1) { /* token z sees blocks 0 .. (pos0+z+1)/r - 1 */
        QW4DGpuArgs sa={.h=hi,.d=di}; b.x_stride=hi*di;
        batch(g,cmd,QW4D_SCORES_BATCH,&sa,&b,(Qwen4GpuSpan[]){row(l->iqr,0,t),row(l->block_cache,0,n1),l->bscores},3,
              MTLSizeMake((n1+31)/32,(t+7)/8,1),MTLSizeMake(256,1,1),true); /* P5.3: 32 blocks x 8 tokens */
        wait_all(cmd);
        batch(g,cmd,QW4D_SELECT_BATCH,&sa,&b,(Qwen4GpuSpan[]){l->bscores,l->bkeep,l->bids},3,
              MTLSizeMake(1,t,1),MTLSizeMake(select_threads(g,QW4D_SELECT_BATCH),1,1),true);
    }
    batch(g,cmd,QW4D_TAIL_BATCH,&a,&b,&l->bids,1,MTLSizeMake(r,t,1),MTLSizeMake(MIN(r,256u),1,1),false);
    wait_all(cmd);
    QW4DGpuArgs at={.h=h,.j=j,.d=d};
    /* Prefill-gap: materialized attention, S [tokens][j][16][sp] then the row sums, in the
     * split-partials buffer (need checked above: 128 tokens x 2 x 16 x 2112 floats fit). */
    const uint32_t sp=(ids_stride+63u)/64u*64u, mat_step=128u;
    if (g->mat[0] && d%64u==0 && d<=256u && j && h%j==0 && h/j<=16u &&
        ((size_t)mat_step*j*16u*sp+(size_t)mat_step*j*16u)*4u<=l->bpart.bytes) {
        id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
        for (uint32_t first=0;first<t;first+=mat_step) {
            b.first=first; b.count=MIN(mat_step,t-first);
            Qwen4GpuSpan v[]={row(l->qr,0,t),row(l->gate,0,t),row(l->key_cache,0,pos0+t),row(l->value_cache,0,pos0+t),
                l->bids,row(l->out,0,t)};
            /* the batch's last token sees the most keys: at most min(id stride, visible) */
            const uint32_t most=MIN(ids_stride,pos0+first+b.count), group=h/j;
            const MTLSize grids[3]={MTLSizeMake((most+63u)/64u,j,b.count),MTLSizeMake(group,j,b.count),MTLSizeMake(d/64u,j,b.count)};
            for (size_t k=0;k<3;k++) {
                [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->mat[k]];
                for (size_t i=0;i<6;++i) [enc setBuffer:(__bridge id<MTLBuffer>)v[i].buffer offset:v[i].offset_bytes atIndex:i];
                [enc setBytes:&at length:sizeof at atIndex:6];
                [enc setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:7];
                [enc setBuffer:(__bridge id<MTLBuffer>)l->bpart.buffer offset:l->bpart.offset_bytes atIndex:8];
                [enc setBytes:&b length:sizeof b atIndex:9];
                [enc dispatchThreadgroups:grids[k] threadsPerThreadgroup:MTLSizeMake(k==1 ? 256 : 128,1,1)];
                wait_all(cmd);
            }
        }
        return result(e,QW4_OK);
    }
    uint32_t most_keys; { uint32_t splits; qw4d_attend_splits(ids_stride,&splits,&most_keys); } /* m <= budget+r-1 */
    const uint32_t split_op=use_mma(g,h,j,d,most_keys) ? QW4D_SPLIT_MMA_BATCH : QW4D_SPLIT_BATCH;
    /* P5.2: the MMA path writes MAX_SPLITS/GROUP partials per token, so the same buffer holds GROUP x the tokens. */
    const uint32_t step=split_op==QW4D_SPLIT_MMA_BATCH ? QW4D_BATCH_SPLIT_TOKENS*QW4D_MMA_GROUP : QW4D_BATCH_SPLIT_TOKENS;
    for (uint32_t first=0;first<t;first+=step) {
        b.first=first; b.count=MIN(step,t-first);
        Qwen4GpuSpan v[]={row(l->qr,0,t),row(l->gate,0,t),row(l->key_cache,0,pos0+t),row(l->value_cache,0,pos0+t),
            l->bids,row(l->out,0,t),l->bpart};
        batch(g,cmd,split_op,&at,&b,v,7,MTLSizeMake(split_op==QW4D_SPLIT_MMA_BATCH ? QW4D_MAX_SPLITS/QW4D_MMA_GROUP : QW4D_MAX_SPLITS,
              j,b.count),MTLSizeMake(128,1,1),true);
        wait_all(cmd);
        batch(g,cmd,split_op==QW4D_SPLIT_MMA_BATCH ? QW4D_MERGE_MMA_BATCH : QW4D_MERGE_BATCH,&at,&b,v,7,MTLSizeMake(h,b.count,1),MTLSizeMake(d,1,1),true);
        wait_all(cmd);
    }
    return result(e,QW4_OK);
}
Qwen4Status qw4d_layer_encode(QW4DGpu *g, QW4CGpu *ops, Qwen4GpuCommand cmd,
        const Qwen4AttentionConfig *c, uint32_t pos0, uint32_t t, const QW4DGpuLayer *l, Qwen4Error *e) {
    if (!g || !ops || !c || !l || !cmd.device || !cmd.compute_encoder) return result(e,QW4_INVALID_ARGUMENT);
    Qwen4Status code=layer_check(g,cmd,c,pos0,t,l);
    if (code) return result(e,code);
    uint32_t h=c->heads, j=c->kv_heads, d=c->head_dim, hi=c->index_heads, di=c->index_dim, r=c->ratio;
    uint32_t kb=c->token_budget/r;
    if (!t) return result(e,QW4_OK);
    QW4CGpuSpec spec={.eps=c->rms_eps,.activation=QW4_SIGMOID};
    /* Batch: C norms per head and the sigmoid gate, over the whole append. */
    Qwen4GpuF32 q=l->qg, gate_in=l->qg;
    q.rank=3; q.dim[1]=h; q.dim[2]=d; q.stride_bytes[1]=2*(size_t)d*4; q.stride_bytes[2]=4;
    gate_in=q; gate_in.span.offset_bytes+=(size_t)d*4; gate_in.span.bytes-=(size_t)d*4;
#define C_OP(op,...) do { Qwen4GpuF32 vv[]={__VA_ARGS__}; \
    code=qw4c_gpu_encode(ops,cmd,QW4C_##op,&spec,vv,sizeof vv/sizeof *vv,e); if (code) return code; } while (0)
    C_OP(RMS,q,l->gamma_q,l->qn);
    C_OP(ACT,gate_in,l->gate);
    C_OP(RMS,l->k,l->gamma_k,l->kn);
    C_OP(RMS,l->iq,l->gamma_iq,l->iqn);
    /* Raw V and raw index keys enter the cache for the whole append. */
    QW4DGpuArgs copy={.n=t*j*d};
    helper(g,cmd,QW4D_COPY,&copy,(Qwen4GpuSpan[]){l->v.span,row(l->value_cache,pos0,t)},2,copy.n);
    copy.n=t*di;
    helper(g,cmd,QW4D_COPY,&copy,(Qwen4GpuSpan[]){l->ik.span,row(l->index_cache,pos0,t)},2,copy.n);
    /* R3.3: a prefill append runs the same stages batched over its tokens, same bits. */
    if (t>=QW4D_BATCH_MIN_TOKENS && !g->reference && !g->scalar_attention && hi<=QW4D_SCORE_LANES &&
        l->bscores.buffer && l->part.buffer)
        return layer_batch(g,ops,cmd,c,pos0,t,l,e);
    for (uint32_t i=0;i<t;++i) {
        uint32_t p=pos0+i, n=(p+1)/r, sel=n<kb ? n : kb;
        QW4DGpuArgs a={.h=h,.d=d,.r=c->rotary_dim,.pos=p,.theta=c->rope_theta};
        if ((code=stage(g,cmd,QW4D_ROPE,a,(Qwen4GpuSpan[]){row(l->qn,i,1),row(l->qr,i,1)},2,e))) return code;
        a.h=j;
        if ((code=stage(g,cmd,QW4D_ROPE,a,(Qwen4GpuSpan[]){row(l->kn,i,1),row(l->key_cache,p,1)},2,e))) return code;
        a.h=hi; a.d=di;
        if ((code=stage(g,cmd,QW4D_ROPE,a,(Qwen4GpuSpan[]){row(l->iqn,i,1),row(l->iqr,i,1)},2,e))) return code;
        if ((p+1)%r==0) {                         /* block b completes at this token */
            uint32_t b=p/r;
            QW4DGpuArgs pool={.h=1,.d=di,.n=r};
            if ((code=stage(g,cmd,QW4D_POOL,pool,(Qwen4GpuSpan[]){row(l->index_cache,(size_t)b*r,r),l->pooled.span},2,e)))
                return code;
            C_OP(RMS,l->pooled,l->gamma_ik,l->pooled_n);
            QW4DGpuArgs rope={.h=1,.d=di,.r=c->rotary_dim,.pos=b*r,.theta=c->rope_theta};
            if ((code=stage(g,cmd,QW4D_ROPE,rope,(Qwen4GpuSpan[]){l->pooled_n.span,row(l->block_cache,b,1)},2,e)))
                return code;
        }
        if (n) {
            QW4DGpuArgs sc={.h=hi,.d=di,.n=n,.m=sel,.r=r};
            if ((code=stage(g,cmd,QW4D_SCORES,sc,(Qwen4GpuSpan[]){row(l->iqr,i,1),row(l->block_cache,0,n),
                words(l->scores.span,0,n)},3,e))) return code;
        }
        uint32_t tail=p+1-n*r, m=sel*r+tail;
        if ((code=qw4d_select_encode(g,cmd,p+1,r,c->token_budget,l->scores.span,l->keep,l->ids,e))) return code;
        QW4DGpuArgs at={.h=h,.j=j,.d=d,.n=p+1,.m=m};
        if ((code=qw4d_attend_encode(g,cmd,&at,(Qwen4GpuSpan[]){row(l->qr,i,1),row(l->gate,i,1),
            row(l->key_cache,0,p+1),row(l->value_cache,0,p+1),words(l->ids,0,m),row(l->out,i,1)},l->part,e))) return code;
        [(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
#undef C_OP
    return result(e,QW4_OK);
}
