/* Isolated per-stage Metal harness; caller submits, waits, checks and retains. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_c_internal.h"
#include <assert.h>
#include <stdlib.h>
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLCommandBuffer> command;
static id<MTLComputeCommandEncoder> encoder;
static id<MTLBuffer> error_buffer;
static Q38CGpu *gpu;
static double maximum_error;
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
        maximum_error=fmax(maximum_error,error/scale); assert(error/scale<=tolerance);
    }
    guards(v);
}
static Qwen38GpuCommand begin(uint32_t initial_error) {
    *(uint32_t *)error_buffer.contents=initial_error;
    command=[queue commandBuffer]; assert(command.retainedReferences);
    encoder=[command computeCommandEncoder];
    return (Qwen38GpuCommand){(__bridge void *)device,(__bridge void *)encoder,
        {(__bridge void *)error_buffer,0,4}};
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
int main(int argc, char **argv) {
    assert(argc==2);
    @autoreleasepool {
        device=MTLCreateSystemDefaultDevice(); assert(device); queue=[device newCommandQueue];
        NSError *error=nil;
        id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 1; }
        assert(q38c_gpu_create((__bridge void *)device,(__bridge void *)lib,&gpu,NULL)==Q38_OK);
        error_buffer=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        activations(); norms(); hc(); convolutions(); gdn_preparation(); q38c_gpu_destroy(gpu);
        printf("PASS: Metal C stages; max scaled error %.9g\n",maximum_error);
        error_buffer=nil; queue=nil; device=nil;
    }
    return 0;
}
