/* Isolated per-stage Metal harness; caller submits, waits, checks and retains. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_c_internal.h"
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLCommandBuffer> command;
static id<MTLComputeCommandEncoder> encoder;
static id<MTLBuffer> error_buffer;
static Q38CGpu *gpu;
static double maximum_error, case_error;
static size_t peak_metal_bytes;
typedef struct { Qwen38GpuF32 gpu; Qwen38F32Mut cpu; size_t slots; } View;
static View view(size_t rank, const size_t *dims, size_t pad) {
    View v={.gpu={.rank=rank},.cpu={.rank=rank}}; size_t extent=4;
    for (size_t j=rank;j-- >0;) {
        v.gpu.dim[j]=v.cpu.dim[j]=dims[j];
        v.gpu.stride_bytes[j]=v.cpu.stride_bytes[j]=(j==rank-1 ? 4 : extent+4*pad);
        extent+=(dims[j] ? dims[j]-1 : 0)*v.cpu.stride_bytes[j];
    }
    if (!dims[0]) extent=0;
    id<MTLBuffer> b=[device newBufferWithLength:extent+32 options:MTLResourceStorageModeShared]; assert(b);
    peak_metal_bytes=MAX(peak_metal_bytes,device.currentAllocatedSize);
    v.gpu.span=(Qwen38GpuSpan){(__bridge_retained void *)b,16,extent};
    v.cpu.data=(float *)b.contents+4; v.cpu.capacity_bytes=extent; v.slots=extent/4+8;
    for (size_t j=0;j<v.slots;++j) ((float *)b.contents)[j]=123456.0f;
    return v;
}
static void release_view(View v) { CFRelease(v.gpu.span.buffer); }
static void pattern(View v, float scale) {
    Qwen38F32 r=q38c_ro(v.cpu);
    for (size_t i=0;i<q38c_count(r);++i) v.cpu.data[q38c_offset(r,i)]=scale*sinf((float)(i%997)*0.71f);
}
static void guards(View v) {
    bool *used=calloc(v.slots,sizeof(bool)); assert(used); Qwen38F32 r=q38c_ro(v.cpu);
    for (size_t i=0;i<q38c_count(r);++i) used[q38c_offset(r,i)+4]=true;
    float *all=v.cpu.data-4;
    for (size_t i=0;i<v.slots;++i) if (!used[i]) assert(all[i]==123456.0f);
    free(used);
}
static float *expected(View v) {
    float *p=malloc(v.cpu.capacity_bytes ? v.cpu.capacity_bytes : 4); assert(p);
    memcpy(p,v.cpu.data,v.cpu.capacity_bytes); return p;
}
static void compare(View v, const float *ref, double tolerance) {
    Qwen38F32 r=q38c_ro(v.cpu); size_t width=r.dim[r.rank-1], count=q38c_count(r);
    for (size_t row=0;row<count/width;++row) {
        double scale=1, error=0;
        for (size_t j=0;j<width;++j) { size_t at=q38c_offset(r,row*width+j);
            assert(isfinite(v.cpu.data[at]) && isfinite(ref[at]));
            scale=fmax(scale,fabs(ref[at])); error=fmax(error,fabs((double)v.cpu.data[at]-ref[at])); }
        maximum_error=fmax(maximum_error,error/scale); case_error=fmax(case_error,error/scale);
        assert(error/scale<=tolerance);
    }
    guards(v);
}
static Qwen38GpuCommand begin(uint32_t initial_error) {
    *(uint32_t *)error_buffer.contents=initial_error;
    command=[queue commandBuffer]; assert(command.retainedReferences);
    encoder=[command computeCommandEncoder];
    return (Qwen38GpuCommand){(__bridge void *)device,(__bridge void *)encoder,
        {(__bridge void *)error_buffer,0,4},false};
}
static void finish(uint32_t expected_error) {
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.error) fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);
    assert(command.status==MTLCommandBufferStatusCompleted);
    assert(*(uint32_t *)error_buffer.contents==expected_error); encoder=nil; command=nil;
}
static void activations(void) {
    for (size_t rank=1;rank<=4;++rank) for (int kind=0;kind<3;++kind) {
        size_t dims[]={3,2,3,257}; View x=view(rank,dims+4-rank,3), y=view(rank,dims+4-rank,7);
        pattern(x,90); float *ref=expected(y); Qwen38F32Mut out=y.cpu; out.data=ref;
        assert(qwen38_activate((Qwen38Activation)kind,q38c_ro(x.cpu),out,NULL)==Q38_OK);
        Q38CGpuSpec spec={.activation=(Qwen38Activation)kind}; Qwen38GpuF32 v[]={x.gpu,y.gpu};
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_ACT,&spec,v,2,NULL)==Q38_OK); finish(0);
        Qwen38F32 yr=q38c_ro(y.cpu);
        for (size_t i=0;i<q38c_count(yr);++i) { size_t at=q38c_offset(yr,i);
            assert(fabs((double)y.cpu.data[at]-ref[at])<=2e-6+2e-5*fabs(ref[at])); }
        compare(y,ref,1e-5); guards(x); free(ref); release_view(x); release_view(y);
    }
    View x=view(1,(size_t[]){3},0), y=view(1,(size_t[]){3},0); pattern(x,1); x.cpu.data[2]=NAN;
    Q38CGpuSpec s={.activation=Q38_SILU}; Qwen38GpuF32 v[]={x.gpu,y.gpu};
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_ACT,&s,v,2,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
    for (size_t i=0;i<3;++i) assert(y.cpu.data[i]==123456);
    x.cpu.data[2]=1;
    assert(q38c_gpu_encode(gpu,begin(Q38_GPU_STATE),Q38C_ACT,&s,v,2,NULL)==Q38_OK); finish(Q38_GPU_STATE);
    for (size_t i=0;i<3;++i) assert(y.cpu.data[i]==123456);
    Qwen38GpuCommand cmd=begin(0); v[1]=x.gpu;
    assert(q38c_gpu_encode(gpu,cmd,Q38C_ACT,&s,v,2,NULL)==Q38_INVALID_ARGUMENT);
    v[1]=y.gpu; v[1].span.bytes=4;
    assert(q38c_gpu_encode(gpu,cmd,Q38C_ACT,&s,v,2,NULL)==Q38_CAPACITY); finish(0);
    release_view(x); release_view(y);
}
/* R1.6b: on a concurrent command (8-byte word) the input check ORs into word 1,
 * unless the op writes what the check reads; then word 0 and the barrier stay. */
static void concurrent_words(void) {
    uint32_t *w=(uint32_t *)error_buffer.contents;
    View x=view(1,(size_t[]){3},0), y=view(1,(size_t[]){3},0); pattern(x,1); x.cpu.data[2]=-INFINITY;
    Q38CGpuSpec s={.activation=Q38_SIGMOID}; Qwen38GpuF32 v[]={x.gpu,y.gpu};
    Qwen38GpuCommand cmd=begin(0); cmd.error_word.bytes=8; cmd.concurrent=true; w[1]=0;
    assert(q38c_gpu_encode(gpu,cmd,Q38C_ACT,&s,v,2,NULL)==Q38_OK); finish(0);
    assert(w[1]==Q38_GPU_NONFINITE && y.cpu.data[2]==0);   /* sigmoid(-inf)=0: the op is not stopped */
    release_view(x); release_view(y);
    Q38CGpuSpec hs={.hc={1,33,17,1e-6f}};
    View r=view(3,(size_t[]){1,1,33},0), m=view(2,(size_t[]){1,33},0), j=view(2,(size_t[]){1,1},0);
    pattern(r,2); pattern(m,2); pattern(j,2); r.cpu.data[0]=NAN; float kept=r.cpu.data[1];
    Qwen38GpuF32 rv[]={r.gpu,m.gpu,j.gpu}; cmd=begin(0); cmd.error_word.bytes=8; cmd.concurrent=true; w[1]=0;
    assert(q38c_gpu_encode(gpu,cmd,Q38C_HC_ADD,&hs,rv,3,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
    assert(w[1]==0 && r.cpu.data[1]==kept);                /* in place: word 0, the op skipped */
    release_view(r); release_view(m); release_view(j);
}
static void norms(void) {
    size_t widths[]={1,3,128,2560};
    for (size_t k=0;k<4;++k) for (size_t grouped=0;grouped<2;++grouped) for (size_t inplace=0;inplace<2;++inplace) {
        size_t w=widths[k], gain_dims[]={3,w};
        View x=view(3,(size_t[]){5,3,w},3), gain=view(1+grouped,gain_dims+1-grouped,5);
        View y=view(3,(size_t[]){5,3,w},7); pattern(x,0.7f); pattern(gain,1.4f); gain.cpu.data[0]=1e-10f;
        if (inplace) { release_view(y); y=x; }
        float *ref=expected(y); Qwen38F32Mut out=y.cpu; out.data=ref;
        assert(qwen38_rmsnorm_groups(q38c_ro(x.cpu),q38c_ro(gain.cpu),out,1e-6f,NULL)==Q38_OK);
        Q38CGpuSpec spec={.eps=1e-6f}; Qwen38GpuF32 v[]={x.gpu,gain.gpu,y.gpu};
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_RMS,&spec,v,3,NULL)==Q38_OK); finish(0);
        compare(y,ref,1e-5); guards(x); guards(gain); free(ref);
        if (!inplace) release_view(y);
        release_view(x); release_view(gain);
    }
    View x=view(3,(size_t[]){1,1,3},0), gain=view(1,(size_t[]){3},0), y=view(3,(size_t[]){1,1,3},0);
    pattern(gain,1); x.cpu.data[0]=FLT_MAX;
    Q38CGpuSpec s={.eps=1e-6f}; Qwen38GpuF32 v[]={x.gpu,gain.gpu,y.gpu};
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_RMS,&s,v,3,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
    release_view(x); release_view(y);
    x=view(3,(size_t[]){0,1,3},0); y=view(3,(size_t[]){0,1,3},0); gain.cpu.data[0]=NAN;
    v[0]=x.gpu; v[2]=y.gpu;
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_RMS,&s,v,3,NULL)==Q38_OK); finish(0);
    guards(x); guards(y); release_view(x); release_view(y); release_view(gain);
}
static void hc(void) {
    for (size_t h=1;h<=7;h+=3) {
        Q38CGpuSpec s={.hc={h,33,17,1e-6f}};
        View low=view(2,(size_t[]){5,17},3), inj=view(2,(size_t[]){5,h},3);
        View norm=view(3,(size_t[]){5,h,33},3), gate=view(3,(size_t[]){5,h,33},7);
        View mixed=view(2,(size_t[]){5,33},9), residual=view(3,(size_t[]){5,h,33},5);
        pattern(low,3); pattern(inj,3); pattern(norm,2); pattern(gate,3); pattern(residual,2);
        float *lr=expected(low), *ir=expected(inj), *mr=expected(mixed), *rr=expected(residual);
        Qwen38F32Mut lo=low.cpu, io=inj.cpu, mo=mixed.cpu, ro=residual.cpu;
        lo.data=lr; io.data=ir; mo.data=mr; ro.data=rr;
        assert(qwen38_hc_lowrank_activate(&s.hc,q38c_ro(low.cpu),lo,NULL)==Q38_OK);
        assert(qwen38_hc_inject_weights(&s.hc,q38c_ro(inj.cpu),io,NULL)==Q38_OK);
        assert(qwen38_hc_read(&s.hc,q38c_ro(norm.cpu),q38c_ro(gate.cpu),mo,NULL)==Q38_OK);
        assert(qwen38_hc_add(&s.hc,ro,q38c_ro(mo),q38c_ro(io),NULL)==Q38_OK);
        Qwen38GpuCommand cmd=begin(0);
        Qwen38GpuF32 lv[]={low.gpu,low.gpu}, iv[]={inj.gpu,inj.gpu};
        Qwen38GpuF32 mv[]={norm.gpu,gate.gpu,mixed.gpu}, rv[]={residual.gpu,mixed.gpu,inj.gpu};
        assert(q38c_gpu_encode(gpu,cmd,Q38C_HC_LOW,&s,lv,2,NULL)==Q38_OK);
        assert(q38c_gpu_encode(gpu,cmd,Q38C_HC_INJECT,&s,iv,2,NULL)==Q38_OK);
        assert(q38c_gpu_encode(gpu,cmd,Q38C_HC_READ,&s,mv,3,NULL)==Q38_OK);
        assert(q38c_gpu_encode(gpu,cmd,Q38C_HC_ADD,&s,rv,3,NULL)==Q38_OK); finish(0);
        compare(low,lr,1e-5); compare(inj,ir,1e-5); compare(mixed,mr,2e-5); compare(residual,rr,2e-5);
        guards(norm); guards(gate); free(lr); free(ir); free(mr); free(rr);
        release_view(low); release_view(inj); release_view(norm); release_view(gate); release_view(mixed); release_view(residual);
    }
}
static Qwen38GpuF32 slice(View v, size_t start, size_t tokens) {
    Qwen38GpuF32 s=v.gpu; s.dim[0]=tokens;
    if (tokens) { size_t offset=start*s.stride_bytes[0]; s.span.offset_bytes+=offset; s.span.bytes-=offset; }
    return s;
}
static void *backing(size_t bytes) {
    void *p=NULL; assert(posix_memalign(&p,64,bytes ? bytes : 64)==0); return p;
}
static void convolutions(void) {
    size_t channels[]={1,31,33}, kernels[]={1,2,4};
    for (size_t ci=0;ci<3;++ci) for (size_t ki=0;ki<3;++ki) for (size_t d=1;d<=3;d+=2) for (size_t silu=0;silu<2;++silu) {
        size_t c=channels[ci], k=kernels[ki], l=(k-1)*d;
        Q38CGpuSpec spec={.conv={c,k,d,silu!=0}}; Qwen38StateSize bytes;
        assert(qwen38_conv_state_bytes(&spec.conv,&bytes,NULL)==Q38_OK);
        void *metadata=backing(bytes.metadata_bytes), *payload=backing(bytes.payload_bytes); Qwen38ConvState *state=NULL;
        assert(qwen38_conv_state_init(&spec.conv,(Qwen38StateKey){&spec,0,1},
            (Qwen38Memory){metadata,bytes.metadata_bytes},(Qwen38Memory){payload,bytes.payload_bytes},&state,NULL)==Q38_OK);
        View x=view(2,(size_t[]){13,c},5), taps=view(2,(size_t[]){c,k},3);
        View history=view(2,(size_t[]){l,c},7), split_history=view(2,(size_t[]){l,c},7);
        View y=view(2,(size_t[]){13,c},9), split_y=view(2,(size_t[]){13,c},9);
        pattern(x,0.7f); pattern(taps,0.2f); pattern(history,0.6f); pattern(split_history,0.6f);
        float *warm=expected(history); Qwen38F32Mut warm_out=history.cpu; warm_out.data=warm;
        assert(qwen38_conv_causal(&spec.conv,q38c_ro(history.cpu),q38c_ro(taps.cpu),warm_out,state,0,NULL)==Q38_OK);
        float *ref=expected(y), *href=expected(history); Qwen38F32Mut out=y.cpu; out.data=ref;
        assert(qwen38_conv_causal(&spec.conv,q38c_ro(x.cpu),q38c_ro(taps.cpu),out,state,l,NULL)==Q38_OK);
        Qwen38F32Mut ho=history.cpu; ho.data=href;
        assert(qwen38_conv_state_export(state,ho,NULL)==Q38_OK);
        Qwen38GpuF32 v[]={x.gpu,taps.gpu,y.gpu,history.gpu}; Qwen38GpuCommand cmd=begin(0);
        assert(q38c_gpu_encode(gpu,cmd,Q38C_CONV,&spec,v,4,NULL)==Q38_OK);
        size_t chunks[]={0,1,3,0,2,7}, start=0;
        for (size_t j=0;j<6;++j) {
            Qwen38GpuF32 sv[]={slice(x,start,chunks[j]),taps.gpu,slice(split_y,start,chunks[j]),split_history.gpu};
            assert(q38c_gpu_encode(gpu,cmd,Q38C_CONV,&spec,sv,4,NULL)==Q38_OK); start+=chunks[j];
        }
        finish(0); compare(y,ref,1e-5); compare(split_y,ref,1e-5);
        compare(history,href,0); compare(split_history,href,0); compare(split_y,y.cpu.data,0);
        guards(x); guards(taps); free(warm); free(ref); free(href); free(metadata); free(payload);
        release_view(x); release_view(taps); release_view(history); release_view(split_history); release_view(y); release_view(split_y);
    }
}
static void gdn_preparation(void) {
    Qwen38GdnShape shapes[]={{2,6,3,5,1e-6f,1e-6f},{2,2,4,7,1e-6f,1e-6f},{16,48,128,128,1e-6f,1e-6f}};
    for (size_t si=0;si<3;++si) for (size_t inplace=0;inplace<2;++inplace) {
        Q38CGpuSpec spec={.gdn=shapes[si]}; size_t hk=spec.gdn.key_heads, hv=spec.gdn.value_heads;
        size_t dk=spec.gdn.key_dim, dv=spec.gdn.value_dim, c=2*hk*dk+hv*dv, t=si==2 ? 3 : 129;
        View all[]={view(2,(size_t[]){t,c},3),view(2,(size_t[]){t,hv},3),view(2,(size_t[]){t,hv},5),
            view(1,(size_t[]){hv},0),view(1,(size_t[]){hv},0),view(2,(size_t[]){t,c},7),
            view(2,(size_t[]){t,hv},7),view(2,(size_t[]){t,hv},9)};
        for (size_t j=0;j<5;++j) pattern(all[j],0.7f);
        for (size_t h=0;h<hv;++h) all[3].cpu.data[h]=-fabsf(all[3].cpu.data[h]);
        /* Include exactly zero and sub-epsilon Q/K, and A==0. */
        for (size_t j=0;j<2*hk*dk;++j) all[0].cpu.data[j]=j<dk ? 0 : 1e-10f;
        View dest[]={all[inplace ? 0 : 5],all[inplace ? 1 : 6],all[inplace ? 2 : 7]};
        float *refs[]={expected(dest[0]),expected(dest[1]),expected(dest[2])};
        Qwen38F32Mut out[3]; for (size_t j=0;j<3;++j) { out[j]=dest[j].cpu; out[j].data=refs[j]; }
        assert(qwen38_gdn_prepare(&spec.gdn,q38c_ro(all[0].cpu),q38c_ro(all[1].cpu),q38c_ro(all[2].cpu),
            q38c_ro(all[3].cpu),q38c_ro(all[4].cpu),out[0],out[1],out[2],NULL)==Q38_OK);
        Qwen38GpuF32 v[8]; for (size_t j=0;j<8;++j) v[j]=all[j].gpu;
        for (size_t j=0;j<3;++j) v[j+5]=dest[j].gpu;
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_PREP,&spec,v,8,NULL)==Q38_OK); finish(0);
        for (size_t j=0;j<3;++j) { compare(dest[j],refs[j],1e-5); free(refs[j]); }
        for (size_t j=0;j<8;++j) { guards(all[j]); release_view(all[j]); }
        View o=view(3,(size_t[]){t,hv,dv},5), z=view(3,(size_t[]){t,hv,dv},3);
        View gain=view(1,(size_t[]){dv},0), y=view(3,(size_t[]){t,hv,dv},7);
        pattern(o,0.5f); pattern(z,2); pattern(gain,1.5f); gain.cpu.data[0]=1e-10f;
        View target=inplace ? o : y; float *ref=expected(target); Qwen38F32Mut norm=target.cpu; norm.data=ref;
        assert(qwen38_gdn_norm_gate(&spec.gdn,q38c_ro(o.cpu),q38c_ro(z.cpu),q38c_ro(gain.cpu),norm,NULL)==Q38_OK);
        Qwen38GpuF32 nv[]={o.gpu,gain.gpu,target.gpu,z.gpu};
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_NORM,&spec,nv,4,NULL)==Q38_OK); finish(0);
        compare(target,ref,1e-5); free(ref); guards(o); guards(z); guards(gain);
        release_view(o); release_view(z); release_view(gain); release_view(y);
    }
}
/* Install a non-symmetric initial state using ONLY public basis-key writes.
 * The GPU receives the exported snapshot, never the opaque CPU payload. */
static void seed_gdn(Qwen38GdnShape s, Qwen38GdnState *state, View snapshot) {
    size_t hk=s.key_heads, hv=s.value_heads, dk=s.key_dim, dv=s.value_dim, c=2*hk*dk+hv*dv;
    View q=view(2,(size_t[]){dk,c},0), g=view(2,(size_t[]){dk,hv},0), b=view(2,(size_t[]){dk,hv},0);
    View y=view(3,(size_t[]){dk,hv,dv},0); pattern(q,0); pattern(g,0);
    for (size_t t=0;t<dk;++t) {
        for (size_t h=0;h<hk;++h) q.cpu.data[t*c+hk*dk+h*dk+t]=1;
        for (size_t h=0;h<hv;++h) {
            b.cpu.data[t*hv+h]=1;
            for (size_t v=0;v<dv;++v) q.cpu.data[t*c+2*hk*dk+h*dv+v]=
                (float)(h+1)*0.01f+(float)(v+1)*0.001f+(float)(t+1)*0.0001f;
        }
    }
    assert(qwen38_gdn_scan(&s,q38c_ro(q.cpu),q38c_ro(g.cpu),q38c_ro(b.cpu),y.cpu,state,0,NULL)==Q38_OK);
    assert(qwen38_gdn_state_export(state,snapshot.cpu,NULL)==Q38_OK);
    release_view(q); release_view(g); release_view(b); release_view(y);
}
static void gdn_scans(void) {
    size_t lengths[]={0,1,3,4,5,63,64,65,127,128,129,1023,1024,1025,8192,16384,32768};
    for (size_t test=0;test<19;++test) {
        case_error=0;
        bool real=test>=17; size_t t=real ? (test==17 ? 1 : 65) : lengths[test];
        Q38CGpuSpec spec={.gdn=real ? (Qwen38GdnShape){16,48,128,128,1e-6f,1e-6f} :
            (Qwen38GdnShape){2,6,3,5,1e-6f,1e-6f}};
        size_t hk=spec.gdn.key_heads,hv=spec.gdn.value_heads,dk=spec.gdn.key_dim,dv=spec.gdn.value_dim,c=2*hk*dk+hv*dv;
        Qwen38StateSize bytes; assert(qwen38_gdn_state_bytes(&spec.gdn,&bytes,NULL)==Q38_OK);
        void *metadata=backing(bytes.metadata_bytes), *payload=backing(bytes.payload_bytes); Qwen38GdnState *state=NULL;
        assert(qwen38_gdn_state_init(&spec.gdn,(Qwen38StateKey){&spec,0,1},
            (Qwen38Memory){metadata,bytes.metadata_bytes},(Qwen38Memory){payload,bytes.payload_bytes},&state,NULL)==Q38_OK);
        View q=view(2,(size_t[]){t,c},3), g=view(2,(size_t[]){t,hv},5), b=view(2,(size_t[]){t,hv},7);
        View a=view(1,(size_t[]){hv},0), dt=view(1,(size_t[]){hv},0);
        View s=view(3,(size_t[]){hv,dv,dk},3), ss=view(3,(size_t[]){hv,dv,dk},3);
        View y=view(3,(size_t[]){t,hv,dv},5), sy=view(3,(size_t[]){t,hv,dv},5);
        seed_gdn(spec.gdn,state,s); memcpy(ss.cpu.data,s.cpu.data,s.cpu.capacity_bytes);
        pattern(q,0.7f); pattern(g,0.2f); pattern(b,0.3f); pattern(dt,0);
        for (size_t h=0;h<hv;++h) a.cpu.data[h]=h ? -0.0001f : 0;
        assert(qwen38_gdn_prepare(&spec.gdn,q38c_ro(q.cpu),q38c_ro(g.cpu),q38c_ro(b.cpu),q38c_ro(a.cpu),
            q38c_ro(dt.cpu),q.cpu,g.cpu,b.cpu,NULL)==Q38_OK);
        float *ref=expected(y), *sref=expected(s); Qwen38F32Mut out=y.cpu, sout=s.cpu; out.data=ref; sout.data=sref;
        assert(qwen38_gdn_scan(&spec.gdn,q38c_ro(q.cpu),q38c_ro(g.cpu),q38c_ro(b.cpu),out,state,dk,NULL)==Q38_OK);
        assert(qwen38_gdn_state_export(state,sout,NULL)==Q38_OK);
        Qwen38GpuF32 v[]={q.gpu,g.gpu,b.gpu,s.gpu,y.gpu}; Qwen38GpuCommand cmd=begin(0);
        assert(q38c_gpu_encode(gpu,cmd,Q38C_GDN_SCAN,&spec,v,5,NULL)==Q38_OK);
        size_t chunks[]={0,1,3,0,7,63,1024}, start=0, part=0;
        do {
            size_t n=MIN(t-start,chunks[MIN(part,(size_t)6)]);
            Qwen38GpuF32 sv[]={slice(q,start,n),slice(g,start,n),slice(b,start,n),ss.gpu,slice(sy,start,n)};
            assert(q38c_gpu_encode(gpu,cmd,Q38C_GDN_SCAN,&spec,sv,5,NULL)==Q38_OK); start+=n; ++part;
        } while (start<t);
        finish(0); double tol=t<=1025 ? 5e-5 : 2e-4;
        compare(y,ref,tol); compare(sy,ref,tol); compare(s,sref,tol); compare(ss,sref,tol);
        compare(sy,y.cpu.data,0); compare(ss,s.cpu.data,0);
        /* Drain occurred in finish; reset/replay now uses zero S, same inputs. */
        if (test==10) {
            assert(qwen38_gdn_state_reset(state,2,NULL)==Q38_OK);
            assert(qwen38_gdn_state_export(state,s.cpu,NULL)==Q38_OK);
            assert(qwen38_gdn_scan(&spec.gdn,q38c_ro(q.cpu),q38c_ro(g.cpu),q38c_ro(b.cpu),out,state,0,NULL)==Q38_OK);
            assert(qwen38_gdn_state_export(state,sout,NULL)==Q38_OK);
            assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_SCAN,&spec,v,5,NULL)==Q38_OK); finish(0);
            compare(y,ref,tol); compare(s,sref,tol);
        }
        printf("GDN T=%zu Hk=%zu Hv=%zu Dk=%zu Dv=%zu scaled=%.9g bound=%.1g\n",t,hk,hv,dk,dv,case_error,tol);
        guards(q); guards(g); guards(b); free(ref); free(sref); free(metadata); free(payload);
        release_view(q); release_view(g); release_view(b); release_view(a); release_view(dt);
        release_view(s); release_view(ss); release_view(y); release_view(sy);
    }
}
static void admission(void) {
    View x=view(2,(size_t[]){3,5},2), y=view(2,(size_t[]){3,5},3); pattern(x,1);
    Qwen38GpuCommand cmd=begin(0); Q38CGpuSpec s={.activation=Q38_SILU};
    Qwen38GpuF32 good[]={x.gpu,y.gpu}, v[2]; Qwen38Error error;
    for (size_t test=0;test<8;++test) {
        memcpy(v,good,sizeof(v)); Qwen38Status want;
        switch (test) {
        case 0: v[0].stride_bytes[1]=8; want=Q38_STRIDE; break;
        case 1: v[1].span.offset_bytes+=2; want=Q38_STRIDE; break;
        case 2: v[0].span.bytes=SIZE_MAX; want=Q38_CAPACITY; break;
        case 3: v[1].dim[0]=2; want=Q38_SHAPE; break;
        case 4: v[1]=x.gpu; want=Q38_INVALID_ARGUMENT; break;
        case 5: v[0].rank=0; want=Q38_SHAPE; break;
        case 6: v[0].span.buffer=NULL; want=Q38_INVALID_ARGUMENT; break;
        default: v[0].dim[0]=(size_t)UINT32_MAX+1; want=Q38_RANGE; break;
        }
        assert(q38c_gpu_encode(gpu,cmd,Q38C_ACT,&s,v,2,&error)==want && error.code==want);
    }
    Qwen38GpuCommand bad=cmd; bad.error_word=x.gpu.span;
    assert(q38c_gpu_encode(gpu,bad,Q38C_ACT,&s,good,2,NULL)==Q38_INVALID_ARGUMENT);
    finish(0); Qwen38F32 yr=q38c_ro(y.cpu);
    for (size_t i=0;i<q38c_count(yr);++i) assert(q38c_get(yr,i)==123456);
    guards(x); guards(y); release_view(x); release_view(y);
}
static void state_errors(void) {
    Q38CGpuSpec c={.conv={1,2,1,true}};
    View x=view(2,(size_t[]){1,1},0), taps=view(2,(size_t[]){1,2},0);
    View y=view(2,(size_t[]){1,1},0), h=view(2,(size_t[]){1,1},0);
    taps.cpu.data[0]=0; taps.cpu.data[1]=2; h.cpu.data[0]=1;
    Qwen38GpuF32 cv[]={x.gpu,taps.gpu,y.gpu,h.gpu};
    for (size_t test=0;test<2;++test) {
        x.cpu.data[0]=test ? FLT_MAX : NAN;
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_CONV,&c,cv,4,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
        assert(h.cpu.data[0]==1 && y.cpu.data[0]==123456);
    }
    release_view(x); release_view(taps); release_view(y); release_view(h);
    Q38CGpuSpec s={.gdn={1,1,1,1,1e-6f,1e-6f}};
    View q=view(2,(size_t[]){1,3},0), g=view(2,(size_t[]){1,1},0), b=view(2,(size_t[]){1,1},0);
    View state=view(3,(size_t[]){1,1,1},0), out=view(3,(size_t[]){1,1,1},0);
    q.cpu.data[0]=q.cpu.data[1]=1; q.cpu.data[2]=0.25f; state.cpu.data[0]=0;
    Qwen38GpuF32 v[]={q.gpu,g.gpu,b.gpu,state.gpu,out.gpu};
    for (size_t test=0;test<3;++test) {
        g.cpu.data[0]=test==0 ? 0.1f : 0; b.cpu.data[0]=test==1 ? 2 : test==2 ? NAN : 1;
        assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_SCAN,&s,v,5,NULL)==Q38_OK);
        finish(test==2 ? Q38_GPU_NONFINITE : Q38_GPU_STATE);
        assert(state.cpu.data[0]==0 && out.cpu.data[0]==123456);
    }
    b.cpu.data[0]=1; q.cpu.data[2]=FLT_MAX; state.cpu.data[0]=-FLT_MAX;
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_SCAN,&s,v,5,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
    /* The failed state is discarded, even if this particular row was unchanged. */
    state.cpu.data[0]=0; q.cpu.data[2]=0.25f;
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_GDN_SCAN,&s,v,5,NULL)==Q38_OK); finish(0);
    assert(state.cpu.data[0]==0.25f && out.cpu.data[0]==0.25f);
    release_view(q); release_view(g); release_view(b); release_view(state); release_view(out);
}
static void shared_aliases(void) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE); void *memory=NULL;
    assert(posix_memalign(&memory,page,page)==0); memset(memory,0,page);
    id<MTLBuffer> a=[device newBufferWithBytesNoCopy:memory length:page
        options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> b=[device newBufferWithBytesNoCopy:memory length:page
        options:MTLResourceStorageModeShared deallocator:nil];
    assert(a && b && a!=b);
    Qwen38GpuF32 x={.span={(__bridge void *)a,0,8},.rank=1,.dim={2},.stride_bytes={4}};
    Qwen38GpuF32 y=x; y.span=(Qwen38GpuSpan){(__bridge void *)b,4,8};
    Q38CGpuSpec spec={.activation=Q38_SIGMOID}; Qwen38GpuF32 views[]={x,y};
    assert(q38c_gpu_encode(gpu,begin(0),Q38C_ACT,&spec,views,2,NULL)==Q38_INVALID_ARGUMENT);
    finish(0);
    /* Distinct wrappers also must not hide an alias of the error word. */
    y.span.offset_bytes=16; views[1]=y;
    Qwen38GpuCommand cmd=begin(0); cmd.error_word=(Qwen38GpuSpan){(__bridge void *)b,0,4};
    assert(q38c_gpu_encode(gpu,cmd,Q38C_ACT,&spec,views,2,NULL)==Q38_INVALID_ARGUMENT);
    finish(0); a=nil; b=nil; free(memory);
}
static FILE *compose_file(const char *dir, const char *name, const char *mode) {
    char path[4096]; int n=snprintf(path,sizeof(path),"%s/%s",dir,name);
    assert(n>0 && (size_t)n<sizeof(path)); FILE *f=fopen(path,mode); assert(f); return f;
}
static void compose_read(const char *dir, const char *name, View v) {
    FILE *f=compose_file(dir,name,"rb"); size_t bytes=q38c_count(q38c_ro(v.cpu))*4;
    assert(fread(v.cpu.data,1,bytes,f)==bytes && fgetc(f)==EOF && !ferror(f)); assert(!fclose(f));
}
static void compose_write(const char *dir, const char *name, View v) {
    guards(v); FILE *f=compose_file(dir,name,"wb"); size_t bytes=q38c_count(q38c_ro(v.cpu))*4;
    assert(fwrite(v.cpu.data,1,bytes,f)==bytes); assert(!fclose(f));
}
/* Real bounded store, as in B's harness: only the selected tensor payloads.
 * Preserve A's names/types/shapes; relocate offsets in this fixture transport.
 * Never pass the original checkpoint to gpu_weights_open. */
static GpuWeightStore *compose_store(const Q38CLayer *l, const char *dir, Gguf *g, GgufTensor tensors[Q38CW_COUNT]) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE), offset=page;
    char path[]="build/qwen38-c/weights-XXXXXX"; g->fd=mkstemp(path); assert(g->fd>=0 && !unlink(path));
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        tensors[i]=l->tensor[i].storage; tensors[i].file_offset=offset;
        offset=(offset+tensors[i].bytes+page-1)&~(page-1);
    }
    assert(offset<=96*1024*1024); g->file_size=offset; g->tensor_count=Q38CW_COUNT; g->tensors=tensors;
    assert(!ftruncate(g->fd,(off_t)offset));
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        char name[64]; snprintf(name,sizeof(name),"weight-%zu.bin",i); FILE *f=compose_file(dir,name,"rb");
        unsigned char bytes[65536]; size_t done=0;
        while (done<tensors[i].bytes) {
            size_t n=MIN(sizeof(bytes),tensors[i].bytes-done); assert(fread(bytes,1,n,f)==n);
            size_t written=0;
            while (written<n) {
                ssize_t rc=pwrite(g->fd,bytes+written,n-written,(off_t)(tensors[i].file_offset+done+written));
                if (rc<0 && errno==EINTR) continue;
                assert(rc>0); written+=(size_t)rc;
            }
            done+=n;
        }
        assert(fgetc(f)==EOF && !ferror(f)); assert(!fclose(f));
    }
    return gpu_weights_open((__bridge void *)device,g,32*1024*1024);
}
static Qwen38GpuF32 subview(Qwen38GpuF32 v, size_t pos, size_t count) {
    size_t skip=pos*v.stride_bytes[0]; assert(skip<=v.span.bytes);
    v.span.offset_bytes+=skip; v.span.bytes-=skip; v.dim[0]=count; return v;
}
static void composed(id<MTLLibrary> lib, const char *source, uint32_t index, const char *dir,
        const char *output, size_t tokens, size_t chunk) {
    assert(tokens && tokens<=129 && chunk<=129); size_t initial=gpu_weights_live_mapped_bytes();
    Gguf original; gguf_open(source,&original); Qwen38Config *config=NULL; Qwen38Weights *catalog=NULL;
    assert(qwen38_config_load(&original,&config,NULL)==Q38_OK);
    assert(qwen38_weights_describe(&original,config,&catalog,NULL)==Q38_OK);
    Q38CLayer l; assert(q38c_layer_plan(config,catalog,index,&l,NULL)==Q38_OK);
    Gguf transport={0}; GgufTensor tensors[Q38CW_COUNT]; GpuWeightStore *store=compose_store(&l,dir,&transport,tensors);
    const GgufBinding *binding[Q38CW_COUNT]; Qwen38CpuSpan spans[Q38CW_COUNT];
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        binding[i]=gpu_weights_find(store,l.tensor[i].storage.name); assert(binding[i]);
        assert(qwen38_weights_cpu_span(store,binding[i],&spans[i],NULL)==Q38_OK);
    }
    size_t cb=(l.constant_bytes+63)&~(size_t)63; void *constants=aligned_alloc(64,cb); assert(constants);
    assert(q38c_layer_bind(&l,spans,(Qwen38Memory){constants,cb},NULL)==Q38_OK);
    size_t wb; assert(q38c_work_size(&l,tokens,&wb,NULL)==Q38_OK); void *wm=aligned_alloc(64,wb); assert(wm);
    Q38CWork cpu; assert(q38c_work_init(&l,tokens,(Qwen38Memory){wm,wb},&cpu,NULL)==Q38_OK);
    View trace[Q38CT_COUNT], fixed[Q38CW_COUNT]={0}; Q38CGpuWork w={0};
    for (size_t i=0;i<Q38CT_COUNT;++i) { trace[i]=view(2,cpu.trace[i].dim,0); w.trace[i]=trace[i].gpu; }
    const size_t slots[]={Q38CW_HN,Q38CW_A,Q38CW_DT,Q38CW_CONV,Q38CW_GN};
    for (size_t i=0;i<5;++i) {
        size_t slot=slots[i]; Qwen38F32 k=l.constant[slot]; fixed[slot]=view(k.rank,k.dim,0);
        memcpy(fixed[slot].cpu.data,k.data,q38c_count(k)*4); w.constant[slot]=fixed[slot].gpu;
    }
    w.constant[Q38CW_HN].rank=2; w.constant[Q38CW_HN].dim[0]=l.config.hc.streams;
    w.constant[Q38CW_HN].dim[1]=l.config.hidden_size; w.constant[Q38CW_HN].stride_bytes[0]=l.config.hidden_size*4;
    w.constant[Q38CW_HN].stride_bytes[1]=4;
    View residual=view(3,(size_t[]){tokens,l.config.hc.streams,l.config.hidden_size},0);
    View history=view(2,(size_t[]){l.conv.kernel-1,l.conv.channels},0);
    View state=view(3,(size_t[]){l.config.gdn.value_heads,l.config.gdn.value_dim,l.config.gdn.key_dim},0);
    w.residual=residual.gpu; w.history=history.gpu; w.state=state.gpu;
    Q38PackedGpu *packed=NULL; assert(q38_packed_gpu_create((__bridge void *)device,(__bridge void *)lib,&packed,NULL)==Q38_OK);
    compose_read(dir,"input.f32",residual); memset(history.cpu.data,0,history.cpu.capacity_bytes); memset(state.cpu.data,0,state.cpu.capacity_bytes);
    Q38CGpuWork bad=w; bad.trace[Q38CT_LOW].span=w.trace[Q38CT_N].span;
    assert(q38c_block_encode(gpu,packed,begin(0),&l,store,binding,&bad,NULL)==Q38_INVALID_ARGUMENT);
    [encoder endEncoding]; encoder=nil; command=nil; /* Discard failed encode. */
    bad=w; bad.trace[Q38CT_OUT].dim[1]--;
    assert(q38c_block_encode(gpu,packed,begin(0),&l,store,binding,&bad,NULL)==Q38_SHAPE);
    [encoder endEncoding]; encoder=nil; command=nil; /* Earlier stages were encoded, never submitted. */
    for (size_t i=0;i<q38c_count(q38c_ro(state.cpu));++i) assert(state.cpu.data[i]==0);
    Qwen38GpuCommand rejected=begin(Q38_GPU_STATE);
    assert(q38c_block_encode(gpu,packed,rejected,&l,store,binding,&w,NULL)==Q38_OK); finish(Q38_GPU_STATE);
    for (size_t i=0;i<q38c_count(q38c_ro(state.cpu));++i) assert(state.cpu.data[i]==0);
    /* Numerical failure after histories can advance; reset both raw buffers. */
    float old_gain=fixed[Q38CW_GN].cpu.data[0]; fixed[Q38CW_GN].cpu.data[0]=NAN;
    bad=w; bad.residual=subview(w.residual,0,1);
    for (size_t i=0;i<Q38CT_COUNT;++i) bad.trace[i]=subview(w.trace[i],0,1);
    assert(q38c_block_encode(gpu,packed,begin(0),&l,store,binding,&bad,NULL)==Q38_OK); finish(Q38_GPU_NONFINITE);
    fixed[Q38CW_GN].cpu.data[0]=old_gain;
    float *snapshots[Q38CT_COUNT+3]={0}; View outputs[Q38CT_COUNT+3];
    for (size_t i=0;i<Q38CT_COUNT;++i) outputs[i]=trace[i];
    outputs[Q38CT_COUNT]=residual; outputs[Q38CT_COUNT+1]=history; outputs[Q38CT_COUNT+2]=state;
    for (size_t replay=0;replay<2;++replay) {
        compose_read(dir,"input.f32",residual); memset(history.cpu.data,0,history.cpu.capacity_bytes); memset(state.cpu.data,0,state.cpu.capacity_bytes);
        Qwen38GpuCommand cmd=begin(0);
        for (size_t pos=0;pos<tokens;) {
            size_t n=chunk ? 1+pos%chunk : tokens; if (n>tokens-pos) n=tokens-pos;
            Q38CGpuWork part=w; part.residual=subview(w.residual,pos,n);
            for (size_t i=0;i<Q38CT_COUNT;++i) part.trace[i]=subview(w.trace[i],pos,n);
            Qwen38Error error; Qwen38Status rc=q38c_block_encode(gpu,packed,cmd,&l,store,binding,&part,&error);
            if (rc) fprintf(stderr,"C4 GPU: %s\n",error.detail); assert(rc==Q38_OK);
            pos+=n;
            part.residual=subview(w.residual,pos,0);
            for (size_t i=0;i<Q38CT_COUNT;++i) part.trace[i]=subview(w.trace[i],pos,0);
            assert(q38c_block_encode(gpu,packed,cmd,&l,store,binding,&part,NULL)==Q38_OK);
        }
        finish(0); /* One command for ALL chunks/stages; no per-operator wait. */
        for (size_t i=0;i<Q38CT_COUNT+3;++i) {
            guards(outputs[i]);
            if (!replay) snapshots[i]=expected(outputs[i]);
            else assert(!memcmp(snapshots[i],outputs[i].cpu.data,outputs[i].cpu.capacity_bytes));
        }
    }
    for (size_t i=0;i<Q38CT_COUNT;++i) { char name[64]; snprintf(name,sizeof(name),"%s.f32",q38c_stage_names[i]); compose_write(output,name,trace[i]); }
    compose_write(output,"residual.f32",residual); compose_write(output,"history.f32",history); compose_write(output,"state.f32",state);
    printf("{\"tokens\":%zu,\"metal_peak_bytes\":%zu,\"mapped_bytes\":%zu,\"reset_replay_bitwise\":true,\"commands_per_run\":1}\n",tokens,peak_metal_bytes,gpu_weights_live_mapped_bytes()-initial);
    for (size_t i=0;i<Q38CT_COUNT+3;++i) { free(snapshots[i]); release_view(outputs[i]); }
    for (size_t i=0;i<5;++i) release_view(fixed[slots[i]]);
    q38_packed_gpu_free(packed);gpu_weights_close(store);assert(!close(transport.fd));
    free(wm);free(constants);qwen38_weights_free(catalog);qwen38_config_free(config);gguf_close(&original);
}
int main(int argc, char **argv) {
    assert(argc==2 || (argc==9 && !strcmp(argv[2],"--compose")));
    @autoreleasepool {
        device=MTLCreateSystemDefaultDevice(); assert(device); queue=[device newCommandQueue];
        NSError *error=nil;
        id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 1; }
        assert(q38c_gpu_create((__bridge void *)device,(__bridge void *)lib,&gpu,NULL)==Q38_OK);
        error_buffer=[device newBufferWithLength:8 options:MTLResourceStorageModeShared];
        if (argc==9) composed(lib,argv[3],(uint32_t)strtoul(argv[4],NULL,10),argv[5],argv[6],
            strtoul(argv[7],NULL,10),strtoul(argv[8],NULL,10));
        else {
        shared_aliases(); activations(); concurrent_words(); norms(); hc(); convolutions(); gdn_preparation(); gdn_scans(); admission(); state_errors();
        printf("PASS: Metal C stages; max scaled error %.9g; observed Metal allocation peak %zu bytes\n",maximum_error,peak_metal_bytes);
        }
        q38c_gpu_destroy(gpu); error_buffer=nil; queue=nil; device=nil;
    }
    assert(gpu_weights_live_mapped_bytes()==0);
    return 0;
}
