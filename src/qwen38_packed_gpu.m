/* Metal interface only: admission, retained pipelines, and command encoding. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include <stdlib.h>

struct Q38PackedGpu { void *device, *linear, *indexed, *lookup; };

Qwen38Status q38_packed_gpu_create(void *device, void *library,
                                  Q38PackedGpu **out, Qwen38Error *err) {
    if (out) *out=NULL;
    if (!out || !device || !library) return q38_b_result(err,Q38_INVALID_ARGUMENT,"missing Metal context");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib=(__bridge id<MTLLibrary>)library;
    if (lib.device != dev) return q38_b_result(err,Q38_OWNER,"library device differs");
    id<MTLFunction> function=[lib newFunctionWithName:@"q38_packed_linear"];
    id<MTLFunction> gather=[lib newFunctionWithName:@"q38_packed_linear_indexed"];
    id<MTLFunction> lookup=[lib newFunctionWithName:@"q38_packed_lookup"];
    if (!function || !gather || !lookup) return q38_b_result(err,Q38_MISSING,"packed kernel absent from library");
    NSError *failure=nil;
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:function error:&failure];
    id<MTLComputePipelineState> indexed=[dev newComputePipelineStateWithFunction:gather error:&failure];
    id<MTLComputePipelineState> embedding=[dev newComputePipelineStateWithFunction:lookup error:&failure];
    if (!pipeline || !indexed || !embedding) return q38_b_result(err,Q38_GPU_ERROR,"cannot create packed pipeline");
    Q38PackedGpu *ctx=calloc(1,sizeof *ctx);
    if (!ctx) return q38_b_result(err,Q38_OOM,"cannot allocate packed context");
    ctx->device=(__bridge_retained void *)dev;
    ctx->linear=(__bridge_retained void *)pipeline;
    ctx->indexed=(__bridge_retained void *)indexed;
    ctx->lookup=(__bridge_retained void *)embedding;
    *out=ctx;
    return q38_b_result(err,Q38_OK,"");
}
void q38_packed_gpu_free(Q38PackedGpu *ctx) {
    if (!ctx) return;
    CFRelease(ctx->linear); CFRelease(ctx->indexed); CFRelease(ctx->lookup); CFRelease(ctx->device); free(ctx);
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

/* Shared admission of both entry points; rows_x is the reachable input rows. */
typedef struct { id<MTLBuffer> input, output, word, weights; size_t xb, yb; } Admitted;
static Qwen38Status admit(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        Qwen38GpuF32 x, Qwen38GpuF32 y, size_t rows_x, Admitted *a, Qwen38Error *err) {
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
    status=q38_b_matrix(x.rank,x.dim,x.stride_bytes,x.span.bytes,&a->xb);
    if (status) return q38_b_result(err,status,"invalid GPU input view");
    status=q38_b_matrix(y.rank,y.dim,y.stride_bytes,y.span.bytes,&a->yb);
    if (status) return q38_b_result(err,status,"invalid GPU output view");
    if (x.dim[0]!=rows_x || x.dim[1]!=layout->k_logical || y.dim[1]!=layout->n_out)
        return q38_b_result(err,Q38_SHAPE,"GPU matrix dimensions differ");
    if (layout->k_storage>UINT32_MAX || layout->n_out>UINT32_MAX || y.dim[0]>UINT32_MAX ||
        layout->n_experts>UINT32_MAX)
        return q38_b_result(err,Q38_RANGE,"kernel dimensions exceed uint32");
    status=gpu_span(dev,x.span,a->xb,&a->input);
    if (status) return q38_b_result(err,status,"invalid GPU input span");
    status=gpu_span(dev,y.span,a->yb,&a->output);
    if (status) return q38_b_result(err,status,"invalid GPU output span");
    status=gpu_span(dev,command->error_word,4,&a->word);
    if (status) return q38_b_result(err,status,"invalid GPU error word");
    if (a->word.storageMode!=MTLStorageModeShared)
        return q38_b_result(err,Q38_TYPE,"error word must use Shared storage");
    a->weights=(__bridge id<MTLBuffer>)gpu_weights_buffer(store,binding);
    if (a->weights.device!=dev) return q38_b_result(err,Q38_OWNER,"weight device differs");
    size_t wo=binding->offset, wn=layout->tensor_bytes, xo=x.span.offset_bytes, yo=y.span.offset_bytes;
    size_t eo=command->error_word.offset_bytes, xb=a->xb, yb=a->yb;
    if (gpu_alias(a->input,xo,xb,a->output,yo,yb) || gpu_alias(a->weights,wo,wn,a->input,xo,xb) ||
        gpu_alias(a->weights,wo,wn,a->output,yo,yb) || gpu_alias(a->word,eo,4,a->input,xo,xb) ||
        gpu_alias(a->word,eo,4,a->output,yo,yb) || gpu_alias(a->word,eo,4,a->weights,wo,wn))
        return q38_b_result(err,Q38_INVALID_ARGUMENT,"aliased GPU resources");
    return Q38_OK;
}
static Q38PackedArgs wire(const Q38PackedLayout *layout, size_t offset, size_t end, size_t rows,
        Qwen38GpuF32 x, Qwen38GpuF32 y, const Admitted *a) {
    return (Q38PackedArgs){.type=layout->gguf_type,.k_storage=(uint32_t)layout->k_storage,
        .k_logical=(uint32_t)layout->k_logical,.n_out=(uint32_t)layout->n_out,.rows=(uint32_t)rows,
        .row_bytes=layout->row_bytes,.weight_offset=offset,.weight_end=end,
        .x_stride=x.stride_bytes[0],.y_stride=y.stride_bytes[0],.x_bytes=a->xb,.y_bytes=a->yb};
}
static void bind(id<MTLComputeCommandEncoder> encoder, const Admitted *a, Qwen38GpuF32 x,
        Qwen38GpuF32 y, const Qwen38GpuCommand *command) {
    [encoder setBuffer:a->weights offset:0 atIndex:0];
    [encoder setBuffer:a->input offset:x.span.offset_bytes atIndex:1];
    [encoder setBuffer:a->output offset:y.span.offset_bytes atIndex:2];
    [encoder setBuffer:a->word offset:command->error_word.offset_bytes atIndex:3];
}

Qwen38Status q38_packed_encode(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    Admitted a;
    Qwen38Status status=admit(ctx,command,store,binding,layout,x,y,y.dim[0],&a,err);
    if (status) return status;
    if (expert>=layout->n_experts) return q38_b_result(err,Q38_RANGE,"expert outside tensor");
    if (!x.dim[0]) return q38_b_result(err,Q38_OK,"");
    size_t offset=binding->offset+expert*layout->expert_bytes;
    Q38PackedArgs args=wire(layout,offset,offset+layout->expert_bytes,x.dim[0],x,y,&a);
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    id<MTLComputePipelineState> pipeline=(__bridge id<MTLComputePipelineState>)ctx->linear;
    [encoder setComputePipelineState:pipeline];
    bind(encoder,&a,x,y,command);
    [encoder setBytes:&args length:sizeof args atIndex:4];
    NSUInteger threads=MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(layout->n_out,x.dim[0],1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    return q38_b_result(err,Q38_OK,"");
}

Qwen38Status q38_packed_encode_indexed(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        Qwen38GpuSpan ids, size_t divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    if (!divisor || divisor>UINT32_MAX || y.dim[0]%divisor)
        return q38_b_result(err,Q38_SHAPE,"output rows must be a multiple of the input divisor");
    Admitted a;
    Qwen38Status status=admit(ctx,command,store,binding,layout,x,y,y.dim[0]/divisor,&a,err);
    if (status) return status;
    id<MTLBuffer> route=nil;
    status=gpu_span((__bridge id<MTLDevice>)ctx->device,ids,y.dim[0]*4,&route);
    if (status) return q38_b_result(err,status,"invalid GPU expert-ID span");
    size_t wo=binding->offset, io=ids.offset_bytes, n=y.dim[0]*4;
    if (gpu_alias(route,io,n,a.output,y.span.offset_bytes,a.yb) ||
        gpu_alias(route,io,n,a.word,command->error_word.offset_bytes,4) ||
        gpu_alias(route,io,n,a.weights,wo,layout->tensor_bytes))
        return q38_b_result(err,Q38_INVALID_ARGUMENT,"expert IDs alias an output or weights");
    if (!y.dim[0]) return q38_b_result(err,Q38_OK,"");
    Q38PackedIndexedArgs args={.base=wire(layout,wo,wo+layout->tensor_bytes,y.dim[0],x,y,&a),
        .expert_bytes=layout->expert_bytes,.experts=(uint32_t)layout->n_experts,.x_divisor=(uint32_t)divisor};
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    id<MTLComputePipelineState> pipeline=(__bridge id<MTLComputePipelineState>)ctx->indexed;
    [encoder setComputePipelineState:pipeline];
    bind(encoder,&a,x,y,command);
    [encoder setBytes:&args length:sizeof args atIndex:4];
    [encoder setBuffer:route offset:io atIndex:5];
    NSUInteger threads=MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(layout->n_out,y.dim[0],1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    return q38_b_result(err,Q38_OK,"");
}

Qwen38Status q38_packed_lookup(Q38PackedGpu *ctx,const Qwen38GpuCommand *cmd,
    const GpuWeightStore *store,const GgufBinding *binding,const Q38PackedLayout *l,
    Qwen38GpuSpan ids,Qwen38GpuF32 y,Qwen38Error *e) {
    if (!ctx || !cmd || !cmd->compute_encoder || !store || !binding || !l)
        return q38_b_result(e,Q38_INVALID_ARGUMENT,"missing lookup argument");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)ctx->device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd->compute_encoder;
    if (cmd->device!=ctx->device || enc.device!=dev) return q38_b_result(e,Q38_OWNER,"lookup device");
    Qwen38CpuSpan resident; Q38PackedLayout expected; size_t bytes;
    Qwen38Status code=qwen38_weights_cpu_span(store,binding,&resident,e); if (code) return code;
    code=q38_packed_layout_init(&binding->tensor,l->k_logical,&expected,e); if (code) return code;
    if (!q38_b_same_layout(l,&expected)) return q38_b_result(e,Q38_FORMAT,"lookup layout");
    if (l->n_experts!=1 || y.rank!=2 || y.dim[1]!=l->k_logical) return q38_b_result(e,Q38_SHAPE,"lookup shape");
    if (l->n_out>UINT32_MAX || l->k_storage>UINT32_MAX || y.dim[0]>UINT32_MAX)
        return q38_b_result(e,Q38_RANGE,"lookup dimensions");
    code=q38_b_matrix(y.rank,y.dim,y.stride_bytes,y.span.bytes,&bytes); if (code) return q38_b_result(e,code,"lookup output");
    id<MTLBuffer> buffers[4];
    Qwen38GpuSpan spans[]={ids,y.span,cmd->error_word,{gpu_weights_buffer(store,binding),binding->offset,l->tensor_bytes}};
    size_t needs[]={y.dim[0]*4,bytes,4,l->tensor_bytes};
    for (size_t i=0;i<4;i++) {
        code=gpu_span(dev,spans[i],needs[i],&buffers[i]); if (code) return q38_b_result(e,code,"lookup span");
        if (buffers[i].hazardTrackingMode==MTLHazardTrackingModeUntracked) return q38_b_result(e,Q38_UNSUPPORTED,"lookup hazards");
        for (size_t j=0;j<i;j++) if (gpu_alias(buffers[i],spans[i].offset_bytes,needs[i],buffers[j],spans[j].offset_bytes,needs[j]))
            return q38_b_result(e,Q38_INVALID_ARGUMENT,"lookup alias");
    }
    if (buffers[2].storageMode!=MTLStorageModeShared) return q38_b_result(e,Q38_TYPE,"lookup error storage");
    if (!y.dim[0]) return q38_b_result(e,Q38_OK,"");
    Q38PackedArgs args={.type=l->gguf_type,.k_storage=(uint32_t)l->k_storage,.k_logical=(uint32_t)l->k_logical,
        .n_out=(uint32_t)l->n_out,.rows=(uint32_t)y.dim[0],.row_bytes=l->row_bytes,.weight_offset=binding->offset,
        .weight_end=binding->offset+l->tensor_bytes,.y_stride=y.stride_bytes[0],.y_bytes=bytes};
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)ctx->lookup;
    [enc setComputePipelineState:p];
    [enc setBuffer:buffers[3] offset:0 atIndex:0];
    [enc setBuffer:buffers[0] offset:ids.offset_bytes atIndex:1];
    [enc setBuffer:buffers[1] offset:y.span.offset_bytes atIndex:2];
    [enc setBuffer:buffers[2] offset:cmd->error_word.offset_bytes atIndex:3];
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreads:MTLSizeMake(l->k_logical,y.dim[0],1) threadsPerThreadgroup:MTLSizeMake(MIN(l->k_logical,p.maxTotalThreadsPerThreadgroup),1,1)];
    return q38_b_result(e,Q38_OK,"");
}
