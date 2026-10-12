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
int main(int argc, char **argv) {
    assert(argc==2);
    @autoreleasepool {
        device=MTLCreateSystemDefaultDevice(); assert(device); queue=[device newCommandQueue];
        NSError *error=nil;
        id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); return 1; }
        assert(q38c_gpu_create((__bridge void *)device,(__bridge void *)lib,&gpu,NULL)==Q38_OK);
        error_buffer=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        activations(); q38c_gpu_destroy(gpu);
        printf("PASS: Metal C stages; max scaled error %.9g\n",maximum_error);
        error_buffer=nil; queue=nil; device=nil;
    }
    return 0;
}
