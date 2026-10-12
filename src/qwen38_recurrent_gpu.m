/* Private per-stage harness glue. No submission, waits or numerical CPU path. */
#import <Metal/Metal.h>
#include "qwen38_c_internal.h"
#include <stdlib.h>
struct Q38CGpu { void *device, *check, *history, *pipeline[Q38C_GPU_OPS]; };
void q38c_gpu_destroy(Q38CGpu *g) {
    if (!g) return;
    if (g->device) CFRelease(g->device);
    if (g->check) CFRelease(g->check);
    if (g->history) CFRelease(g->history);
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
