/* Private per-stage harness glue. No submission, waits or numerical CPU path. */
#import <Metal/Metal.h>
#include "qwen38_c_internal.h"
#include "qwen38_b_internal.h"
#include "qwen38_ref.h"
#include <stdlib.h>
/* rms_fast: q38c_rms_fast or NULL (no 32-wide simdgroup / 256 threads: the
 * oracle stays). ref_fuse: RMS and GDN_NORM run check + q38c_rms (R1.6a). */
struct Q38CGpu { void *device, *check, *history, *pipeline[Q38C_GPU_OPS], *rms_fast; bool ref_fuse; };
void q38c_gpu_set_reference(Q38CGpu *g, bool fuse) { if (g) g->ref_fuse=fuse; }
void q38c_gpu_destroy(Q38CGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    if (g->check) CFRelease(g->check);
    if (g->history) CFRelease(g->history);
    if (g->rms_fast) CFRelease(g->rms_fast);
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
    NSString *names[]={@"q38c_activate",@"q38c_rms",@"q38c_activate",@"q38c_activate",
        @"q38c_hc_read",@"q38c_hc_add",@"q38c_conv",@"q38c_gdn_prepare",@"q38c_rms",
        @"q38c_gdn_scan",@"q38c_check",@"q38c_history"};
    for (size_t i=0;i<Q38C_GPU_OPS+2;++i) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        id<MTLComputePipelineState> p=fn ? [dev newComputePipelineStateWithFunction:fn error:NULL] : nil;
        if (!p) { q38c_gpu_destroy(g); return q38c_error(e,Q38_GPU_ERROR,"missing or invalid C pipeline"); }
        if (i==Q38C_GPU_OPS+1) g->history=(__bridge_retained void *)p;
        else if (i==Q38C_GPU_OPS) g->check=(__bridge_retained void *)p;
        else g->pipeline[i]=(__bridge_retained void *)p;
    }
    id<MTLFunction> fn=[lib newFunctionWithName:@"q38c_rms_fast"];
    id<MTLComputePipelineState> p=fn ? [dev newComputePipelineStateWithFunction:fn error:NULL] : nil;
    if (p && p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=Q38C_RMS_THREADS)
        g->rms_fast=(__bridge_retained void *)p;
    g->ref_fuse=q38_ref_kernels("fuse"); /* once per context, never per token */
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
    if (!na || !nb) return false;
    if (a.buffer==b.buffer)
        return a.offset_bytes<b.offset_bytes+nb && b.offset_bytes<a.offset_bytes+na;
    /* Separate no-copy Metal objects may still wrap the same Shared memory. */
    id<MTLBuffer> ba=(__bridge id<MTLBuffer>)a.buffer, bb=(__bridge id<MTLBuffer>)b.buffer;
    if (ba.storageMode==MTLStorageModeShared && bb.storageMode==MTLStorageModeShared) {
        uintptr_t pa=(uintptr_t)ba.contents+a.offset_bytes, pb=(uintptr_t)bb.contents+b.offset_bytes;
        return pa<=pb ? pb-pa<na : pa-pb<nb;
    }
    return false;
}
static void dispatch(id<MTLComputeCommandEncoder> enc, void *pipeline, size_t count) {
    id<MTLComputePipelineState> p=(__bridge id<MTLComputePipelineState>)pipeline;
    [enc setComputePipelineState:p];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(count,p.maxTotalThreadsPerThreadgroup),1,1)];
}
static bool gpu_same(Q38CGpuView a, Q38CGpuView b) {
    return a.rank==b.rank && !memcmp(a.dim,b.dim,a.rank*sizeof(uint32_t));
}
static bool gpu_dims(Q38CGpuView a, size_t rank, const size_t *dims) {
    if (a.rank!=rank) return false;
    for (size_t i=0;i<rank;++i) if (a.dim[i]!=dims[i]) return false;
    return true;
}
static Qwen38Status stage(Q38CGpuOp op, const Q38CGpuSpec *s, Q38CGpuArgs *a,
        size_t *work, uint32_t *writes, uint64_t *inplace, Qwen38Error *e) {
    Q38CGpuView *v=a->v; size_t t=v[0].dim[0];
    a->p[5]=(uint32_t)op; *work=v[0].count; *inplace=0;
    if (op>=Q38C_HC_LOW && op<=Q38C_HC_ADD) {
        if (!s->hc.streams || !s->hc.hidden_size || !s->hc.low_rank ||
            s->hc.streams>UINT32_MAX || s->hc.hidden_size>UINT32_MAX || s->hc.low_rank>UINT32_MAX)
            return q38c_error(e,Q38_SHAPE,"invalid GPU HC shape");
        if (!isfinite(s->hc.rms_eps)) return q38c_error(e,Q38_NONFINITE,"nonfinite HC epsilon");
        if (s->hc.rms_eps<=0) return q38c_error(e,Q38_RANGE,"nonpositive HC epsilon");
        a->p[1]=(uint32_t)s->hc.streams;
    }
    if (op==Q38C_ACT || op==Q38C_HC_LOW || op==Q38C_HC_INJECT) {
        if (!gpu_same(v[0],v[1])) goto shape;
        if (op==Q38C_ACT) {
            if (s->activation<Q38_SIGMOID || s->activation>Q38_SOFTPLUS)
                return q38c_error(e,Q38_UNSUPPORTED,"unknown activation");
            a->p[0]=(uint32_t)s->activation; a->p[1]=a->p[2]=1;
        } else {
            size_t width=op==Q38C_HC_LOW ? s->hc.low_rank : s->hc.streams;
            if (!gpu_dims(v[0],2,(size_t[]){t,width})) goto shape;
            a->p[0]=op==Q38C_HC_LOW ? Q38_SILU : Q38_SIGMOID;
            a->p[2]=op==Q38C_HC_LOW ? 1 : 2; *inplace=UINT64_C(1)<<8;
        }
        a->read_mask=1; *writes=2;
    } else if (op==Q38C_RMS || op==Q38C_GDN_NORM) {
        if (v[0].rank!=3 || !gpu_same(v[0],v[2])) goto shape;
        size_t groups=v[0].dim[1], width=v[0].dim[2];
        if (!(gpu_dims(v[1],1,(size_t[]){width}) || gpu_dims(v[1],2,(size_t[]){groups,width}))) goto shape;
        float eps=s->eps; a->read_mask=3;
        if (op==Q38C_GDN_NORM) {
            Qwen38StateSize size; Q38C_TRY(qwen38_gdn_state_bytes(&s->gdn,&size,e));
            if (!gpu_dims(v[0],3,(size_t[]){t,s->gdn.value_heads,s->gdn.value_dim}) ||
                !gpu_same(v[0],v[3]) || v[1].rank!=1) goto shape;
            eps=s->gdn.rms_eps; a->p[0]=1; a->read_mask=11;
        }
        if (!isfinite(eps)) return q38c_error(e,Q38_NONFINITE,"nonfinite RMS epsilon");
        if (eps<=0) return q38c_error(e,Q38_RANGE,"nonpositive RMS epsilon");
        a->eps=eps; *writes=4; *inplace=UINT64_C(1)<<16; *work=t*groups;
    } else if (op==Q38C_GDN_PREP || op==Q38C_GDN_SCAN) {
        Qwen38StateSize size; Q38C_TRY(qwen38_gdn_state_bytes(&s->gdn,&size,e));
        size_t hk=s->gdn.key_heads, hv=s->gdn.value_heads, dk=s->gdn.key_dim, dv=s->gdn.value_dim;
        if (!gpu_dims(v[0],2,(size_t[]){t,2*hk*dk+hv*dv}) || !gpu_dims(v[1],2,(size_t[]){t,hv}) ||
            !gpu_same(v[1],v[2])) goto shape;
        a->p[0]=(uint32_t)hk; a->p[1]=(uint32_t)hv; a->p[2]=(uint32_t)dk; a->p[3]=(uint32_t)dv;
        if (op==Q38C_GDN_PREP) {
            if (!gpu_dims(v[3],1,(size_t[]){hv}) || !gpu_same(v[3],v[4]) ||
                !gpu_same(v[0],v[5]) || !gpu_same(v[1],v[6]) || !gpu_same(v[2],v[7])) goto shape;
            a->p[4]=1; a->eps=s->gdn.l2_eps; a->read_mask=31; *writes=224; *work=t;
            *inplace=(UINT64_C(1)<<40)|(UINT64_C(1)<<49)|(UINT64_C(1)<<58);
        } else {
            if (!gpu_dims(v[3],3,(size_t[]){hv,dv,dk}) || !gpu_dims(v[4],3,(size_t[]){t,hv,dv})) goto shape;
            a->p[4]=2; a->read_mask=15; *writes=24; *work=t ? hv*dv : 0;
        }
    } else if (op==Q38C_CONV) {
        Qwen38StateSize size; Q38C_TRY(qwen38_conv_state_bytes(&s->conv,&size,e));
        size_t c=s->conv.channels, k=s->conv.kernel, d=s->conv.dilation, l=(k-1)*d;
        if (!gpu_dims(v[0],2,(size_t[]){t,c}) || !gpu_dims(v[1],2,(size_t[]){c,k}) ||
            !gpu_same(v[0],v[2]) || !gpu_dims(v[3],2,(size_t[]){l,c})) goto shape;
        a->p[0]=(uint32_t)k; a->p[1]=(uint32_t)d; a->p[2]=s->conv.apply_silu;
        a->read_mask=11; *writes=12;
    } else {
        size_t h=s->hc.streams, width=s->hc.hidden_size;
        if (!gpu_dims(v[0],3,(size_t[]){t,h,width})) goto shape;
        if (op==Q38C_HC_READ) {
            if (!gpu_same(v[0],v[1]) || !gpu_dims(v[2],2,(size_t[]){t,width})) goto shape;
            a->read_mask=3; *writes=4; *work=t*width;
        } else {
            if (!gpu_dims(v[1],2,(size_t[]){t,width}) || !gpu_dims(v[2],2,(size_t[]){t,h})) goto shape;
            a->read_mask=7; *writes=1;
        }
    }
    return Q38_OK;
shape: return q38c_error(e,Q38_SHAPE,"GPU stage shape mismatch");
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
    if ((unsigned)op>=Q38C_GPU_OPS) return q38c_error(e,Q38_UNSUPPORTED,"unknown C kernel");
    size_t expected_count=(op==Q38C_ACT || op==Q38C_HC_LOW || op==Q38C_HC_INJECT) ? 2 : 3;
    if (op==Q38C_CONV) expected_count=4;
    if (op==Q38C_GDN_PREP) expected_count=8;
    if (op==Q38C_GDN_NORM) expected_count=4;
    if (op==Q38C_GDN_SCAN) expected_count=5;
    if (count!=expected_count) return q38c_error(e,Q38_SHAPE,"wrong GPU view count");
    Q38CGpuArgs a={0}; size_t span[8]={0};
    for (size_t j=0;j<count;++j) {
        Q38C_TRY(gpu_view(v[j],dev,&a.v[j],&span[j],e));
        if (gpu_overlap(v[j].span,span[j],cmd.error_word,4))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"view aliases GPU error word");
    }
    size_t work, check_count=0; uint32_t writes; uint64_t inplace;
    Q38C_TRY(stage(op,spec,&a,&work,&writes,&inplace,e));
    for (size_t i=0;i<count;++i) {
        if (a.read_mask&(1u<<i)) check_count=MAX(check_count,a.v[i].count);
        for (size_t j=0;j<count;++j) if (i!=j && (writes&(1u<<i)) && gpu_overlap(v[i].span,span[i],v[j].span,span[j])) {
            if ((inplace&(UINT64_C(1)<<(i*8+j))) && v[i].span.buffer==v[j].span.buffer &&
                v[i].span.offset_bytes==v[j].span.offset_bytes &&
                gpu_same(a.v[i],a.v[j]) && !memcmp(a.v[i].stride,a.v[j].stride,sizeof(a.v[i].stride))) continue;
            return q38c_error(e,Q38_INVALID_ARGUMENT,"forbidden GPU alias");
        }
    }
    if (!work) return q38c_error(e,Q38_OK,NULL);
    /* Bind unused slots to a real buffer: no nil/unbound arguments in MSL. */
    for (size_t j=0;j<8;++j) {
        Qwen38GpuSpan s=v[j<count ? j : 0].span;
        [enc setBuffer:(__bridge id<MTLBuffer>)s.buffer offset:s.offset_bytes atIndex:j];
    }
    [enc setBytes:&a length:sizeof(a) atIndex:8];
    [enc setBuffer:error offset:cmd.error_word.offset_bytes atIndex:9];
    if ((op==Q38C_RMS || op==Q38C_GDN_NORM) && g->rms_fast && !g->ref_fuse) {
        /* R1.6a: one threadgroup per row, q38c_check folded into the kernel. */
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g->rms_fast];
        [enc dispatchThreadgroups:MTLSizeMake(work,1,1) threadsPerThreadgroup:MTLSizeMake(Q38C_RMS_THREADS,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return q38c_error(e,Q38_OK,NULL);
    }
    dispatch(enc,g->check,check_count);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    dispatch(enc,g->pipeline[op],work);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (op==Q38C_CONV && a.v[3].count) {
        dispatch(enc,g->history,a.v[0].dim[1]);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    return q38c_error(e,Q38_OK,NULL);
}

static Qwen38GpuF32 group_view(Qwen38GpuF32 v, size_t groups, size_t width) {
    v.rank=3; v.dim[1]=groups; v.dim[2]=width; v.stride_bytes[1]=width*4; v.stride_bytes[2]=4; return v;
}
Qwen38Status q38c_block_encode(Q38CGpu *g, Q38PackedGpu *b, Qwen38GpuCommand cmd,
        const Q38CLayer *l, const GpuWeightStore *store, const GgufBinding *const bindings[Q38CW_COUNT],
        const Q38CGpuWork *w, Qwen38Error *e) {
    if (!g || !b || !l || !l->bound || !store || !bindings || !w)
        return q38c_error(e,Q38_INVALID_ARGUMENT,"missing composed GPU inputs");
    if (!cmd.compute_encoder || !cmd.device || cmd.device!=g->device)
        return q38c_error(e,Q38_OWNER,"foreign composed GPU command");
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    if (enc.device!=dev) return q38c_error(e,Q38_OWNER,"foreign composed GPU encoder");
    Q38C_TRY(gpu_span(cmd.error_word,dev,e));
    size_t t=w->residual.dim[0], bytes; Q38C_TRY(q38c_work_size(l,t,&bytes,e));
    Qwen38GpuF32 all[Q38CT_COUNT+8]; size_t extent[Q38CT_COUNT+8], count=0;
    all[count++]=w->residual; all[count++]=w->history; all[count++]=w->state;
    for (size_t i=0;i<Q38CT_COUNT;++i) all[count++]=w->trace[i];
    const size_t constants[]={Q38CW_HN,Q38CW_A,Q38CW_DT,Q38CW_CONV,Q38CW_GN};
    for (size_t i=0;i<5;++i) all[count++]=w->constant[constants[i]];
    size_t physical_bytes=0;
    for (size_t i=0;i<count;++i) {
        Q38CGpuView v={0}; Q38C_TRY(gpu_view(all[i],dev,&v,&extent[i],e));
        if (!q38c_add(physical_bytes,extent[i],&physical_bytes) || physical_bytes>512u*1024u*1024u)
            return q38c_error(e,Q38_CAPACITY,"composed GPU physical views exceed 512 MiB");
        if (gpu_overlap(all[i].span,extent[i],cmd.error_word,4))
            return q38c_error(e,Q38_INVALID_ARGUMENT,"composed view aliases error word");
        for (size_t j=0;j<i;++j)
            if (gpu_overlap(all[i].span,extent[i],all[j].span,extent[j]))
                return q38c_error(e,Q38_INVALID_ARGUMENT,"composed GPU views overlap");
    }
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        Qwen38CpuSpan span; Q38C_TRY(qwen38_weights_cpu_span(store,bindings[i],&span,e));
        const GgufTensor *got=&bindings[i]->tensor,*want=&l->tensor[i].storage;
        if (strcmp(got->name,want->name) || got->type!=want->type || got->rank!=want->rank ||
            memcmp(got->shape,want->shape,got->rank*sizeof(size_t)) || got->bytes!=want->bytes)
            return q38c_error(e,Q38_FORMAT,"C catalog and GPU weight descriptor differ");
        Qwen38GpuSpan weight={gpu_weights_buffer(store,bindings[i]),bindings[i]->offset,got->bytes};
        for (size_t j=0;j<count;++j)
            if (gpu_overlap(weight,weight.bytes,all[j].span,extent[j]))
                return q38c_error(e,Q38_INVALID_ARGUMENT,"composed GPU view aliases weights");
    }
    Q38CGpuSpec spec={.hc=l->config.hc,.gdn=l->config.gdn,.conv=l->conv,.eps=l->config.hc.rms_eps};
    size_t h=l->config.hc.streams, width=l->config.hidden_size, hv=l->config.gdn.value_heads, dv=l->config.gdn.value_dim;
#define V(name) w->trace[Q38CT_##name]
#define K(name) w->constant[Q38CW_##name]
#define OP(op,...) do { Qwen38GpuF32 views[]={__VA_ARGS__}; \
    Q38C_TRY(q38c_gpu_encode(g,cmd,Q38C_##op,&spec,views,sizeof(views)/sizeof(*views),e)); } while (0)
#define LINEAR(slot,input,output) do { Q38C_TRY(q38_packed_encode(b,&cmd,store,bindings[Q38CW_##slot], \
    &l->weight[Q38CW_##slot].layout,0,V(input),V(output),e)); \
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while (0)
    OP(RMS,w->residual,K(HN),group_view(V(N),h,width));
    LINEAR(HD,N,LOW); OP(HC_LOW,V(LOW),V(ACT)); LINEAR(HU,ACT,HG);
    OP(HC_READ,group_view(V(N),h,width),group_view(V(HG),h,width),V(MIX));
    LINEAR(HI,N,IL); OP(HC_INJECT,V(IL),V(INJECT));
    { /* R1.4: the four projections of MIX in one multi call, one barrier */
        const Q38CWeight slot[]={Q38CW_QKV,Q38CW_Z,Q38CW_AP,Q38CW_BP};
        const GgufBinding *mb[4]; Q38PackedLayout ml[4];
        const Qwen38GpuF32 out[]={V(QKV),V(Z),V(ALPHA),V(BR)};
        for (size_t i=0;i<4;++i) { mb[i]=bindings[slot[i]]; ml[i]=l->weight[slot[i]].layout; }
        Q38C_TRY(q38_packed_encode_multi(b,&cmd,store,mb,ml,4,V(MIX),out,e));
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    OP(CONV,V(QKV),K(CONV),V(CONV),w->history);
    OP(GDN_PREP,V(CONV),V(ALPHA),V(BR),K(A),K(DT),V(PREP),V(LOGG),V(BETA));
    OP(GDN_SCAN,V(PREP),V(LOGG),V(BETA),w->state,group_view(V(SCAN),hv,dv));
    OP(GDN_NORM,group_view(V(SCAN),hv,dv),K(GN),group_view(V(NORM),hv,dv),group_view(V(Z),hv,dv));
    LINEAR(OUT,NORM,OUT); OP(HC_ADD,w->residual,V(OUT),V(INJECT));
#undef LINEAR
#undef OP
#undef V
#undef K
    return q38c_error(e,Q38_OK,NULL);
}
