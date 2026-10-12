/* Private per-stage harness glue. No submission, waits or numerical CPU path. */
#import <Metal/Metal.h>
#include "qwen38_c_internal.h"
#include <stdlib.h>
struct Q38CGpu { void *device, *check, *pipeline[Q38C_GPU_OPS]; };
void q38c_gpu_destroy(Q38CGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    if (g->check) CFRelease(g->check);
    for (size_t i=0;i<Q38C_GPU_OPS;++i) if (g->pipeline[i]) CFRelease(g->pipeline[i]);
    free(g);
}
Qwen38Status q38c_gpu_create(void *device, void *library, Q38CGpu **out, Qwen38Error *e) {
    if (!out) return q38c_error(e,Q38_INVALID_ARGUMENT,"null GPU output");
    *out=NULL;
    if (!device || !library) return q38c_error(e,Q38_INVALID_ARGUMENT,"missing Metal device/library");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device!=dev) return q38c_error(e,Q38_OWNER,"foreign Metal library");
    Q38CGpu *g=calloc(1,sizeof(*g));
    if (!g) return q38c_error(e,Q38_OOM,"GPU context allocation failed");
    g->device=(__bridge_retained void *)dev;
    NSString *names[]={@"q38c_activate",@"q38c_check"};
    for (size_t i=0;i<Q38C_GPU_OPS+1;++i) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=fn ? [dev newComputePipelineStateWithFunction:fn error:NULL] : nil;
        if (!p) { q38c_gpu_destroy(g); return q38c_error(e,Q38_GPU_ERROR,"missing or invalid C pipeline"); }
        if (i==Q38C_GPU_OPS) g->check=(__bridge_retained void *)p;
        else g->pipeline[i]=(__bridge_retained void *)p;
    }
    *out=g; return q38c_error(e,Q38_OK,NULL);
}
static Qwen38Status gpu_span(Qwen38GpuSpan s, id<MTLDevice> dev, Qwen38Error *e) {
    if (!s.buffer) return q38c_error(e,Q38_INVALID_ARGUMENT,"null Metal buffer");
    id<MTLBuffer> b=(__bridge id<MTLBuffer>)s.buffer;
    if (b.device!=dev) return q38c_error(e,Q38_OWNER,"foreign Metal buffer");
    if (s.offset_bytes%4) return q38c_error(e,Q38_STRIDE,"unaligned GPU offset");
    if (s.offset_bytes>b.length || s.bytes>b.length-s.offset_bytes)
        return q38c_error(e,Q38_CAPACITY,"GPU span exceeds buffer");
    if (b.hazardTrackingMode==MTLHazardTrackingModeUntracked)
        return q38c_error(e,Q38_UNSUPPORTED,"untracked GPU hazards");
    return Q38_OK;
}
static Qwen38Status gpu_view(Qwen38GpuF32 v, id<MTLDevice> dev, Q38CGpuView *packed,
        size_t *extent, Qwen38Error *e) {
    Qwen38F32 layout={.rank=v.rank};
    memcpy(layout.dim,v.dim,sizeof(v.dim)); memcpy(layout.stride_bytes,v.stride_bytes,sizeof(v.stride_bytes));
    Q38C_TRY(q38c_layout(layout,extent,e)); Q38C_TRY(gpu_span(v.span,dev,e));
    if (*extent>v.span.bytes) return q38c_error(e,Q38_CAPACITY,"GPU view exceeds span");
    if (*extent/4>UINT32_MAX) return q38c_error(e,Q38_RANGE,"GPU view exceeds uint32 indexing");
    packed->rank=(uint32_t)v.rank; packed->count=(uint32_t)q38c_count(layout);
    for (size_t i=0;i<v.rank;++i) {
        if (v.stride_bytes[i]/4>UINT32_MAX) return q38c_error(e,Q38_RANGE,"GPU stride exceeds uint32");
        packed->dim[i]=(uint32_t)v.dim[i]; packed->stride[i]=(uint32_t)(v.stride_bytes[i]/4);
    }
    return Q38_OK;
}
static bool gpu_overlap(Qwen38GpuSpan a, size_t na, Qwen38GpuSpan b, size_t nb) {
    return a.buffer==b.buffer && na && nb && a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
}
static void dispatch(id<MTLComputeCommandEncoder> enc, void *pipeline, size_t count) {
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)pipeline;
    [enc setComputePipelineState:p];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(count,p.maxTotalThreadsPerThreadgroup),1,1)];
}
static bool gpu_same(Q38CGpuView a, Q38CGpuView b) {
    return a.rank==b.rank && !memcmp(a.dim,b.dim,a.rank*sizeof(uint32_t));
}
Qwen38Status q38c_gpu_encode(Q38CGpu *g, Qwen38GpuCommand cmd, Q38CGpuOp op,
        const Q38CGpuSpec *spec, const Qwen38GpuF32 *v, size_t count, Qwen38Error *e) {
    if (!g || !spec || !v || !cmd.compute_encoder || !cmd.device)
        return q38c_error(e,Q38_INVALID_ARGUMENT,"missing GPU dispatch argument");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (g->device!=cmd.device || enc.device!=dev) return q38c_error(e,Q38_OWNER,"foreign GPU command");
    Q38C_TRY(gpu_span(cmd.error_word,dev,e));
    id<MTLBuffer> error=(__bridge id<MTLBuffer>)cmd.error_word.buffer;
    if (error.storageMode!=MTLStorageModeShared || cmd.error_word.bytes<4)
        return q38c_error(e,Q38_CAPACITY,"error word needs four Shared bytes");
    if (op!=Q38C_ACT) return q38c_error(e,Q38_UNSUPPORTED,"unknown C kernel");
    if (count!=2) return q38c_error(e,Q38_SHAPE,"activation needs input/output");
    Q38CGpuArgs a={0}; size_t span[8]={0};
    for (size_t j=0;j<count;++j) {
        Q38C_TRY(gpu_view(v[j],dev,&a.v[j],&span[j],e));
        if (gpu_overlap(v[j].span,span[j],cmd.error_word,4))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"view aliases GPU error word");
    }
    if (!gpu_same(a.v[0],a.v[1])) return q38c_error(e,Q38_SHAPE,"activation shape mismatch");
    if (gpu_overlap(v[0].span,span[0],v[1].span,span[1]))
        return q38c_error(e,Q38_INVALID_ARGUMENT,"forbidden GPU alias");
    if (spec->activation<Q38_SIGMOID || spec->activation>Q38_SOFTPLUS)
        return q38c_error(e,Q38_UNSUPPORTED,"unknown activation");
    a.p[0]=(uint32_t)spec->activation; a.read_mask=1;
    if (!a.v[0].count) return q38c_error(e,Q38_OK,NULL);
    /* Bind unused slots to a real buffer: no nil/unbound arguments in MSL. */
    for (size_t j=0;j<8;++j) {
        Qwen38GpuSpan s=v[j<count ? j : 0].span;
        [enc setBuffer:(__bridge id<MTLBuffer>)s.buffer offset:s.offset_bytes atIndex:j];
    }
    [enc setBytes:&a length:sizeof(a) atIndex:8];
    [enc setBuffer:error offset:cmd.error_word.offset_bytes atIndex:9];
    dispatch(enc,g->check,a.v[0].count);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    dispatch(enc,g->pipeline[op],a.v[0].count);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return q38c_error(e,Q38_OK,NULL);
}
