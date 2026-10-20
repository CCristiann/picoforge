/* Metal interface only: admission, retained pipelines, and command encoding. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include "qwen38_ref.h"
#include <stdlib.h>

/* ref_*: run the oracle kernel for that family (qwen38_ref.h), fixed at creation.
 * q8_0, f16, f32: the cooperative matvecs of qwen38_matvec.metal, one per
 * format (q8_0 is the default linear path for Q8_0); multi: their multi-GEMV
 * entry points (Q8_0, F16, F32); indexed_fast: the indexed expert matvecs
 * (IQ2_XXS, Q2_K); each is NULL on a GPU whose simdgroups are not 32 wide. */
enum { Q38_INDEXED_FAST = 2 };
struct Q38PackedGpu { void *device, *linear, *indexed, *lookup, *q8_0, *f16, *f32, *multi[3],
                      *indexed_fast[Q38_INDEXED_FAST]; bool ref_linear, ref_indexed; };
/* The fast kernels' tile (Q8_T, DENSE_T in the kernel): a threadgroup computes
 * one output row for that many rows of x, with one 32-lane simdgroup (Q8_0)
 * or dense_split(K) of them (F16, F32; at most Q38_DENSE_S). */
enum { Q38_Q8_T = 4, Q38_DENSE_T = 4, Q38_DENSE_S = 8, Q38_SIMD = 32 };
/* Indexed matvecs: IDX_SG in the kernel, simdgroups (one output each) per threadgroup. */
enum { Q38_IDX_SG = 4 };
static const uint32_t q38_indexed_types[Q38_INDEXED_FAST]={16,10};
/* Mirror of dense_split in qwen38_matvec.metal (the kernel checks the match):
 * simdgroups per output row from K alone, doubling while each lane keeps at
 * least 5 chunks of 8 columns. */
static size_t dense_split(size_t k) {
    size_t s=1;
    while (s<Q38_DENSE_S && k>=2*s*Q38_SIMD*8*5) s*=2;
    return s;
}

/* The fast kernels' simd_sum and lane-to-column map assume 32-wide
 * simdgroups; elsewhere the slot stays NULL, the fast path answers
 * UNSUPPORTED and q38_packed_encode runs the oracle. */
static void *fast_pipeline(id<MTLComputePipelineState> pipeline, size_t simdgroups) {
    if (pipeline.threadExecutionWidth!=Q38_SIMD || pipeline.maxTotalThreadsPerThreadgroup<Q38_SIMD*simdgroups)
        return NULL;
    return (__bridge_retained void *)pipeline;
}

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
    id<MTLFunction> q8=[lib newFunctionWithName:@"q38_matvec_q8_0"];
    id<MTLFunction> f16=[lib newFunctionWithName:@"q38_matvec_f16"];
    id<MTLFunction> f32=[lib newFunctionWithName:@"q38_matvec_f32"];
    static NSString *const multi_names[3]={@"q38_matvec_multi_q8_0",@"q38_matvec_multi_f16",@"q38_matvec_multi_f32"};
    id<MTLFunction> multi[3];
    for (size_t i=0;i<3;i++) multi[i]=[lib newFunctionWithName:multi_names[i]];
    static NSString *const indexed_names[Q38_INDEXED_FAST]={@"q38_matvec_indexed_iq2_xxs",@"q38_matvec_indexed_q2_k"};
    id<MTLFunction> indexed_fn[Q38_INDEXED_FAST];
    bool present=true;
    for (size_t i=0;i<Q38_INDEXED_FAST;i++) present=(indexed_fn[i]=[lib newFunctionWithName:indexed_names[i]])!=nil && present;
    if (!function || !gather || !lookup || !q8 || !f16 || !f32 || !multi[0] || !multi[1] || !multi[2] || !present)
        return q38_b_result(err,Q38_MISSING,"packed kernel absent from library");
    NSError *failure=nil;
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:function error:&failure];
    id<MTLComputePipelineState> indexed=[dev newComputePipelineStateWithFunction:gather error:&failure];
    id<MTLComputePipelineState> embedding=[dev newComputePipelineStateWithFunction:lookup error:&failure];
    id<MTLComputePipelineState> matvec_q8=[dev newComputePipelineStateWithFunction:q8 error:&failure];
    id<MTLComputePipelineState> matvec_f16=[dev newComputePipelineStateWithFunction:f16 error:&failure];
    id<MTLComputePipelineState> matvec_f32=[dev newComputePipelineStateWithFunction:f32 error:&failure];
    id<MTLComputePipelineState> matvec_multi[3];
    for (size_t i=0;i<3;i++) matvec_multi[i]=[dev newComputePipelineStateWithFunction:multi[i] error:&failure];
    id<MTLComputePipelineState> matvec_indexed[Q38_INDEXED_FAST];
    for (size_t i=0;i<Q38_INDEXED_FAST;i++)
        present=(matvec_indexed[i]=[dev newComputePipelineStateWithFunction:indexed_fn[i] error:&failure])!=nil && present;
    if (!pipeline || !indexed || !embedding || !matvec_q8 || !matvec_f16 || !matvec_f32 ||
        !matvec_multi[0] || !matvec_multi[1] || !matvec_multi[2] || !present)
        return q38_b_result(err,Q38_GPU_ERROR,"cannot create packed pipeline");
    Q38PackedGpu *ctx=calloc(1,sizeof *ctx);
    if (!ctx) return q38_b_result(err,Q38_OOM,"cannot allocate packed context");
    ctx->device=(__bridge_retained void *)dev;
    ctx->linear=(__bridge_retained void *)pipeline;
    ctx->indexed=(__bridge_retained void *)indexed;
    ctx->lookup=(__bridge_retained void *)embedding;
    ctx->q8_0=fast_pipeline(matvec_q8,1);
    ctx->f16=fast_pipeline(matvec_f16,Q38_DENSE_S);
    ctx->f32=fast_pipeline(matvec_f32,Q38_DENSE_S);
    for (size_t i=0;i<3;i++) ctx->multi[i]=fast_pipeline(matvec_multi[i],i ? Q38_DENSE_S : 1);
    for (size_t i=0;i<Q38_INDEXED_FAST;i++) ctx->indexed_fast[i]=fast_pipeline(matvec_indexed[i],Q38_IDX_SG);
    ctx->ref_linear=q38_ref_kernels("linear");
    ctx->ref_indexed=q38_ref_kernels("indexed");
    *out=ctx;
    return q38_b_result(err,Q38_OK,"");
}
void q38_packed_gpu_free(Q38PackedGpu *ctx) {
    if (!ctx) return;
    CFRelease(ctx->linear); CFRelease(ctx->indexed); CFRelease(ctx->lookup);
    if (ctx->q8_0) CFRelease(ctx->q8_0);
    if (ctx->f16) CFRelease(ctx->f16);
    if (ctx->f32) CFRelease(ctx->f32);
    for (size_t i=0;i<3;i++) if (ctx->multi[i]) CFRelease(ctx->multi[i]);
    for (size_t i=0;i<Q38_INDEXED_FAST;i++) if (ctx->indexed_fast[i]) CFRelease(ctx->indexed_fast[i]);
    CFRelease(ctx->device); free(ctx);
}
/* Tests force the choice regardless of the environment read at creation. */
void q38_packed_gpu_set_reference(Q38PackedGpu *ctx, bool linear, bool indexed) {
    if (ctx) { ctx->ref_linear=linear; ctx->ref_indexed=indexed; }
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

/* The fast kernel of a format and its rows of x per threadgroup; NULL: none. */
static void *fast_kernel(const Q38PackedGpu *ctx, uint32_t type, size_t *tile) {
    *tile=type==8 ? Q38_Q8_T : Q38_DENSE_T;
    return type==8 ? ctx->q8_0 : type==1 ? ctx->f16 : type==0 ? ctx->f32 : NULL;
}

/* One linear encode, shared by both entry points so validation is identical:
 * admit() and the expert check come FIRST, a bad call fails the same way on
 * either path. Fast today: Q8_0 (R1.2), F16 and F32 (R1.3), for any number of
 * rows (each measured faster than the oracle at 1..1024 rows). Not
 * bit-identical to the oracle: Q38_REF_KERNELS=linear (or 1) restores the
 * oracle, bit for bit as 360acf0. A format without a fast kernel, or a GPU
 * without 32-wide simdgroups (NULL pipeline), runs the oracle.
 * force_fast: q38_packed_encode_fast, which never falls back to the oracle and
 * reports UNSUPPORTED for a format with no fast kernel. */
static Qwen38Status encode_linear(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, bool force_fast, Qwen38Error *err) {
    Admitted a;
    Qwen38Status status=admit(ctx,command,store,binding,layout,x,y,y.dim[0],&a,err);
    if (status) return status;
    if (expert>=layout->n_experts) return q38_b_result(err,Q38_RANGE,"expert outside tensor");
    size_t tile;
    void *kernel=fast_kernel(ctx,layout->gguf_type,&tile);
    bool fast=kernel && (force_fast || !ctx->ref_linear);
    if (force_fast && !fast) return q38_b_result(err,Q38_UNSUPPORTED,"no fast kernel for this format");
    if (!x.dim[0]) return q38_b_result(err,Q38_OK,"");
    size_t offset=binding->offset+expert*layout->expert_bytes;
    Q38PackedArgs args=wire(layout,offset,offset+layout->expert_bytes,x.dim[0],x,y,&a);
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    id<MTLComputePipelineState> pipeline=(__bridge id<MTLComputePipelineState>)(fast ? kernel : ctx->linear);
    [encoder setComputePipelineState:pipeline];
    bind(encoder,&a,x,y,command);
    [encoder setBytes:&args length:sizeof args atIndex:4];
    if (fast) {
        size_t simdgroups=layout->gguf_type==8 ? 1 : dense_split(layout->k_logical);
        MTLSize groups=MTLSizeMake(layout->n_out,(x.dim[0]+tile-1)/tile,1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(Q38_SIMD*simdgroups,1,1)];
    } else {
        NSUInteger threads=MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(layout->n_out,x.dim[0],1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    }
    return q38_b_result(err,Q38_OK,"");
}
Qwen38Status q38_packed_encode(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    return encode_linear(ctx,command,store,binding,layout,expert,x,y,false,err);
}
/* Always the fast kernel, never the oracle: the harness measures it against
 * the oracle selected in q38_packed_encode. */
Qwen38Status q38_packed_encode_fast(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        size_t expert, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    return encode_linear(ctx,command,store,binding,layout,expert,x,y,true,err);
}

/* The indexed fast kernel of a format (R1.5); NULL: none. */
static void *indexed_kernel(const Q38PackedGpu *ctx, uint32_t type) {
    for (size_t i=0;i<Q38_INDEXED_FAST;i++) if (q38_indexed_types[i]==type) return ctx->indexed_fast[i];
    return NULL;
}
/* One indexed encode for both entry points, validation first as in
 * encode_linear. Fast by default for the formats with an indexed kernel
 * (R1.5: IQ2_XXS, Q2_K), at any number of rows (measured faster than the
 * oracle at 1..128 tokens of 10 slots). Not bit-identical to the oracle (the
 * order inside a row differs): Q38_REF_KERNELS=indexed (or 1) restores the
 * oracle bit for bit. force_fast: q38_packed_encode_indexed_fast, UNSUPPORTED
 * for a format without an indexed fast kernel. */
static Qwen38Status encode_indexed(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        Qwen38GpuSpan ids, size_t divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, bool force_fast, Qwen38Error *err) {
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
    void *kernel=indexed_kernel(ctx,layout->gguf_type);
    bool fast=kernel && (force_fast || !ctx->ref_indexed);
    if (force_fast && !fast) return q38_b_result(err,Q38_UNSUPPORTED,"no indexed fast kernel for this format");
    if (!y.dim[0]) return q38_b_result(err,Q38_OK,"");
    Q38PackedIndexedArgs args={.base=wire(layout,wo,wo+layout->tensor_bytes,y.dim[0],x,y,&a),
        .expert_bytes=layout->expert_bytes,.experts=(uint32_t)layout->n_experts,.x_divisor=(uint32_t)divisor};
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    id<MTLComputePipelineState> pipeline=(__bridge id<MTLComputePipelineState>)(fast ? kernel : ctx->indexed);
    [encoder setComputePipelineState:pipeline];
    bind(encoder,&a,x,y,command);
    [encoder setBytes:&args length:sizeof args atIndex:4];
    [encoder setBuffer:route offset:io atIndex:5];
    if (fast) {
        MTLSize groups=MTLSizeMake((layout->n_out+Q38_IDX_SG-1)/Q38_IDX_SG,y.dim[0],1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(Q38_SIMD*Q38_IDX_SG,1,1)];
    } else {
        NSUInteger threads=MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(layout->n_out,y.dim[0],1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    }
    return q38_b_result(err,Q38_OK,"");
}
Qwen38Status q38_packed_encode_indexed(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        Qwen38GpuSpan ids, size_t divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    return encode_indexed(ctx,command,store,binding,layout,ids,divisor,x,y,false,err);
}
Qwen38Status q38_packed_encode_indexed_fast(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *binding, const Q38PackedLayout *layout,
        Qwen38GpuSpan ids, size_t divisor, Qwen38GpuF32 x, Qwen38GpuF32 y, Qwen38Error *err) {
    return encode_indexed(ctx,command,store,binding,layout,ids,divisor,x,y,true,err);
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

/* R1.4 multi-GEMV: the projections of one x, one dispatch per fast format.
 * Mirror of MultiArgs in qwen38_matvec.metal: first[j] is projection j's
 * first threadgroup column (sum of the earlier n_out of its dispatch). */
typedef struct { uint32_t count, first[Q38_MULTI_MAX]; } Q38MultiArgs;
_Static_assert(sizeof(Q38MultiArgs)==36,"multi Metal argument layout");
Qwen38Status q38_packed_encode_multi(Q38PackedGpu *ctx, const Qwen38GpuCommand *command,
        const GpuWeightStore *store, const GgufBinding *const *bindings, const Q38PackedLayout *layouts,
        size_t count, Qwen38GpuF32 x, const Qwen38GpuF32 *y, Qwen38Error *err) {
    if (!bindings || !layouts || !y || !count || count>Q38_MULTI_MAX)
        return q38_b_result(err,Q38_INVALID_ARGUMENT,"multi-GEMV needs 1..8 projections");
    Admitted a[Q38_MULTI_MAX];
    for (size_t i=0;i<count;i++) { /* each admitted as alone; no output overlaps another output or weights */
        Qwen38Status status=admit(ctx,command,store,bindings[i],&layouts[i],x,y[i],y[i].dim[0],&a[i],err);
        if (status) return status;
        for (size_t j=0;j<i;j++)
            if (gpu_alias(a[i].output,y[i].span.offset_bytes,a[i].yb,a[j].output,y[j].span.offset_bytes,a[j].yb) ||
                gpu_alias(a[i].output,y[i].span.offset_bytes,a[i].yb,a[j].weights,bindings[j]->offset,layouts[j].tensor_bytes) ||
                gpu_alias(a[j].output,y[j].span.offset_bytes,a[j].yb,a[i].weights,bindings[i]->offset,layouts[i].tensor_bytes))
                return q38_b_result(err,Q38_INVALID_ARGUMENT,"aliased GPU outputs");
    }
    /* group[f]: projections of fast format f (0 Q8_0, 1 F16, 2 F32) in call
     * order; the rest (no fast kernel, Q38_REF_KERNELS=linear, no 32-wide
     * simdgroups) run q38_packed_encode one by one: the oracle, bit for bit. */
    size_t group[3][Q38_MULTI_MAX], size[3]={0};
    for (size_t i=0;i<count;i++) {
        uint32_t type=layouts[i].gguf_type;
        size_t f=type==8 ? 0 : type==1 ? 1 : type==0 ? 2 : 3;
        if (f<3 && ctx->multi[f] && !ctx->ref_linear) group[f][size[f]++]=i;
        else {
            Qwen38Status status=encode_linear(ctx,command,store,bindings[i],&layouts[i],0,x,y[i],false,err);
            if (status) return status;
        }
    }
    if (!x.dim[0]) return q38_b_result(err,Q38_OK,"");
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)command->compute_encoder;
    for (size_t f=0;f<3;f++) {
        if (!size[f]) continue;
        Q38PackedArgs args[Q38_MULTI_MAX]; Q38MultiArgs multi={(uint32_t)size[f],{0}};
        size_t columns=0;
        [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)ctx->multi[f]];
        for (size_t j=0;j<Q38_MULTI_MAX;j++) { /* unused slots: the error word, never read */
            [encoder setBuffer:a[0].word offset:command->error_word.offset_bytes atIndex:j];
            [encoder setBuffer:a[0].word offset:command->error_word.offset_bytes atIndex:8+j];
        }
        for (size_t j=0;j<size[f];j++) {
            size_t i=group[f][j], offset=bindings[i]->offset;
            args[j]=wire(&layouts[i],offset,offset+layouts[i].expert_bytes,x.dim[0],x,y[i],&a[i]);
            multi.first[j]=(uint32_t)columns;
            columns+=layouts[i].n_out;
            if (columns>UINT32_MAX) return q38_b_result(err,Q38_RANGE,"multi-GEMV rows exceed uint32");
            [encoder setBuffer:a[i].weights offset:0 atIndex:j];
            [encoder setBuffer:a[i].output offset:y[i].span.offset_bytes atIndex:8+j];
        }
        [encoder setBuffer:a[0].input offset:x.span.offset_bytes atIndex:16];
        [encoder setBuffer:a[0].word offset:command->error_word.offset_bytes atIndex:17];
        [encoder setBytes:args length:size[f]*sizeof *args atIndex:18];
        [encoder setBytes:&multi length:sizeof multi atIndex:19];
        size_t tile=f ? Q38_DENSE_T : Q38_Q8_T, simdgroups=f ? dense_split(x.dim[1]) : 1;
        [encoder dispatchThreadgroups:MTLSizeMake(columns,(x.dim[0]+tile-1)/tile,1)
                threadsPerThreadgroup:MTLSizeMake(Q38_SIMD*simdgroups,1,1)];
    }
    return q38_b_result(err,Q38_OK,"");
}
