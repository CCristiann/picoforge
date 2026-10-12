/* Metal interface only: admission, retained pipelines, and command encoding. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include <stdlib.h>

struct Q38PackedGpu { void *device, *linear; };

Qwen38Status q38_packed_gpu_create(void *device, void *library,
                                  Q38PackedGpu **out, Qwen38Error *err) {
    if (out) *out=NULL;
    if (!out || !device || !library) return q38_b_result(err,Q38_INVALID_ARGUMENT,"missing Metal context");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device != dev) return q38_b_result(err,Q38_OWNER,"library device differs");
    id<MTLFunction> function=[lib newFunctionWithName:@"q38_packed_linear"];
    if (!function) return q38_b_result(err,Q38_MISSING,"packed kernel absent from library");
    NSError *failure=nil;
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:function error:&failure];
    if (!pipeline) return q38_b_result(err,Q38_GPU_ERROR,"cannot create packed pipeline");
    Q38PackedGpu *ctx=calloc(1,sizeof *ctx);
    if (!ctx) return q38_b_result(err,Q38_OOM,"cannot allocate packed context");
    ctx->device=(__bridge_retained void *)dev;
    ctx->linear=(__bridge_retained void *)pipeline;
    *out=ctx;
    return q38_b_result(err,Q38_OK,"");
}
void q38_packed_gpu_free(Q38PackedGpu *ctx) {
    if (!ctx) return;
    CFRelease(ctx->linear); CFRelease(ctx->device); free(ctx);
}

static Qwen38Status gpu_span(id<MTLDevice> dev, Qwen38GpuSpan span, size_t needed,
                             id<MTLBuffer> __strong *out) {
    *out=(__bridge id<MTLBuffer>)span.buffer;
    if (!*out) return !needed && !span.bytes && !span.offset_bytes ? Q38_OK : Q38_INVALID_ARGUMENT;
    if ((*out).device != dev) return Q38_OWNER;
    if (span.offset_bytes%4) return Q38_STRIDE;
    if (span.offset_bytes > (*out).length || span.bytes > (*out).length-span.offset_bytes)
        return Q38_RANGE;
    return needed > span.bytes ? Q38_CAPACITY : Q38_OK;
}
static bool gpu_alias(id<MTLBuffer> a, size_t oa, size_t na,
                      id<MTLBuffer> b, size_t ob, size_t nb) {
    if (!na || !nb) return false;
    if (a==b) return oa < ob+nb && ob < oa+na;
    /* Also catch overlapping Shared no-copy allocations represented by different objects. */
    if (a.storageMode==MTLStorageModeShared && b.storageMode==MTLStorageModeShared) {
        const unsigned char *pa=a.contents, *pb=b.contents;
        return q38_b_overlap(pa+oa,na,pb+ob,nb);
    }
    return false;
}

Qwen38Status q38_packed_encode(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    if (!ctx || !command || !store || !binding || !layout || !command->compute_encoder)
        return q38_b_result(err,Q38_INVALID_ARGUMENT,"missing encode argument");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)ctx->device;
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    if (command->device != ctx->device || encoder.device != dev)
        return q38_b_result(err,Q38_OWNER,"command device differs");
    Qwen38CpuSpan resident;
    Qwen38Status status=qwen38_weights_cpu_span(store,binding,&resident,err);
    if (status) return status; /* Includes exact live-catalog identity; never calls fatal storage on a forgery. */
    Q38PackedLayout expected;
    status=q38_packed_layout_init(&binding->tensor,layout->k_logical,&expected,err);
    if (status) return status;
    if (!q38_b_same_layout(layout,&expected)) return q38_b_result(err,Q38_FORMAT,"binding/layout mismatch");
    if (expert>=layout->n_experts) return q38_b_result(err,Q38_RANGE,"expert outside tensor");
    size_t xb,yb;
    status=q38_b_matrix(x.rank,x.dim,x.stride_bytes,x.span.bytes,&xb);
    if (status) return q38_b_result(err,status,"invalid GPU input view");
    status=q38_b_matrix(y.rank,y.dim,y.stride_bytes,y.span.bytes,&yb);
    if (status) return q38_b_result(err,status,"invalid GPU output view");
    if (x.dim[0]!=y.dim[0] || x.dim[1]!=layout->k_logical || y.dim[1]!=layout->n_out)
        return q38_b_result(err,Q38_SHAPE,"GPU matrix dimensions differ");
    if (layout->k_storage>UINT32_MAX || layout->n_out>UINT32_MAX || x.dim[0]>UINT32_MAX)
        return q38_b_result(err,Q38_RANGE,"kernel dimensions exceed uint32");
    id<MTLBuffer> input=nil,output=nil,word=nil;
    status=gpu_span(dev,x.span,xb,&input);
    if (status) return q38_b_result(err,status,"invalid GPU input span");
    status=gpu_span(dev,y.span,yb,&output);
    if (status) return q38_b_result(err,status,"invalid GPU output span");
    status=gpu_span(dev,command->error_word,4,&word);
    if (status) return q38_b_result(err,status,"invalid GPU error word");
    if (word.storageMode!=MTLStorageModeShared)
        return q38_b_result(err,Q38_TYPE,"error word must use Shared storage");
    id<MTLBuffer> weights=(__bridge id<MTLBuffer>)gpu_weights_buffer(store,binding);
    if (weights.device!=dev) return q38_b_result(err,Q38_OWNER,"weight device differs");
    size_t wo=binding->offset, wn=layout->tensor_bytes, xo=x.span.offset_bytes, yo=y.span.offset_bytes;
    size_t eo=command->error_word.offset_bytes;
    if (gpu_alias(input,xo,xb,output,yo,yb) || gpu_alias(weights,wo,wn,input,xo,xb) ||
        gpu_alias(weights,wo,wn,output,yo,yb) || gpu_alias(word,eo,4,input,xo,xb) ||
        gpu_alias(word,eo,4,output,yo,yb) || gpu_alias(word,eo,4,weights,wo,wn))
        return q38_b_result(err,Q38_INVALID_ARGUMENT,"aliased GPU resources");
    if (!x.dim[0]) return q38_b_result(err,Q38_OK,"");
    size_t offset=wo+expert*layout->expert_bytes;
    Q38PackedArgs args={.type=layout->gguf_type,.k_storage=(uint32_t)layout->k_storage,
        .k_logical=(uint32_t)layout->k_logical,.n_out=(uint32_t)layout->n_out,.rows=(uint32_t)x.dim[0],
        .row_bytes=layout->row_bytes,.weight_offset=offset,.weight_end=offset+layout->expert_bytes,
        .x_stride=x.stride_bytes[0],.y_stride=y.stride_bytes[0],.x_bytes=xb,.y_bytes=yb};
    id<MTLComputePipelineState> pipeline=(__bridge id<MTLComputePipelineState>)ctx->linear;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:xo atIndex:1];
    [encoder setBuffer:output offset:yo atIndex:2];
    [encoder setBuffer:word offset:eo atIndex:3];
    [encoder setBytes:&args length:sizeof args atIndex:4];
    NSUInteger threads=MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(layout->n_out,x.dim[0],1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    return q38_b_result(err,Q38_OK,"");
}
