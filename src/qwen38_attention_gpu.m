/* Private D Metal stage glue; caller retains buffers and owns submission. */
#import <Metal/Metal.h>
#include "qwen38_d_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
struct Q38DGpu { void *device, *pipeline[Q38D_GPU_OPS]; };
static Qwen38Status result(Qwen38Error *e,Qwen38Status code) {
    if (e) { memset(e,0,sizeof(*e)); e->code=code; } return code;
}
void q38d_gpu_destroy(Q38DGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    for (size_t i=0;i<Q38D_GPU_OPS;++i) if (g->pipeline[i]) CFRelease(g->pipeline[i]);
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
    NSString *names[]={@"q38d_rope_gpu",@"q38d_pool_gpu",@"q38d_scores_gpu",@"q38d_attend_gpu"};
    for (size_t i=0;i<Q38D_GPU_OPS;++i) {
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
