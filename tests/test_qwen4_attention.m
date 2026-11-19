/* Binary, bounded stage harness. Submission/lifetime live here, never in D. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen4_attention_internal.h"
#include "qwen4_cpu_internal.h"
#include <assert.h>
#include <unistd.h>
static id<MTLDevice> device;
static NSMutableArray<id<MTLBuffer>> *buffers;
static Qwen4GpuSpan buffer(size_t bytes,bool read_input) {
    assert(bytes<=32u*1024u*1024u);
    id<MTLBuffer> b=[device newBufferWithLength:bytes+32 options:MTLResourceStorageModeShared]; assert(b);
    [buffers addObject:b]; memset(b.contents,0x5a,b.length);
    if (read_input) assert(fread((char *)b.contents+16,1,bytes,stdin)==bytes);
    return (Qwen4GpuSpan){(__bridge void *)b,16,bytes};
}
static void *data(Qwen4GpuSpan s) { return (char *)[(__bridge id<MTLBuffer>)s.buffer contents]+s.offset_bytes; }
static void guards(void) {
    for (id<MTLBuffer> b in buffers) for (size_t i=0;i<16;++i) {
        unsigned char *p=b.contents; assert(p[i]==0x5a && p[b.length-16+i]==0x5a);
    }
}
static void read_value(void *p,size_t bytes) { assert(fread(p,1,bytes,stdin)==bytes); }
static Qwen4GpuF32 view(Qwen4GpuSpan s,uint32_t h,uint32_t d,bool gain) {
    Qwen4GpuF32 v={s,3,{1,h,d,0},{(size_t)h*d*4,(size_t)d*4,4,0}};
    if (gain) { v.rank=1; v.dim[0]=d; v.stride_bytes[0]=4; } return v;
}
static void admission(QW4DGpu *g,Qwen4GpuCommand cmd,uint32_t op,
                      QW4DGpuArgs a,Qwen4GpuSpan *v,size_t count) {
    Qwen4GpuSpan bad[6]; memcpy(bad,v,count*sizeof(*v));
    bad[count-1].bytes=0;
    if (v[count-1].bytes) assert(qw4d_gpu_encode(g,cmd,op,&a,bad,count,NULL)==QW4_CAPACITY);
    bad[0]=v[0]; bad[0].offset_bytes++;
    assert(qw4d_gpu_encode(g,cmd,op,&a,bad,count,NULL)==QW4_STRIDE);
    assert(qw4d_gpu_encode(g,cmd,op,&a,v,count+1,NULL)==QW4_SHAPE);
    QW4DGpuArgs wrong=a; wrong.d=257;
    assert(qw4d_gpu_encode(g,cmd,op,&wrong,v,count,NULL)==QW4_RANGE);
    Qwen4GpuCommand badcmd=cmd; badcmd.error_word=v[0];
    assert(qw4d_gpu_encode(g,badcmd,op,&a,v,count,NULL)==QW4_INVALID_ARGUMENT);
}
/* Separate Shared no-copy objects must not bypass alias admission. */
static void nocopy(QW4DGpu *g,Qwen4GpuCommand cmd) {
    size_t page=(size_t)getpagesize(); void *memory=NULL;
    assert(posix_memalign(&memory,page,page)==0);
    id<MTLBuffer> b=[device newBufferWithBytesNoCopy:memory length:page
        options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> c=[device newBufferWithBytesNoCopy:memory length:page
        options:MTLResourceStorageModeShared deallocator:nil]; assert(b && c);
    Qwen4GpuSpan v[]={{(__bridge void *)b,0,32},{(__bridge void *)c,0,32}};
    QW4DGpuArgs a={.h=1,.d=8,.r=4,.theta=10000};
    assert(qw4d_gpu_encode(g,cmd,QW4D_ROPE,&a,v,2,NULL)==QW4_INVALID_ARGUMENT);
    b=nil; c=nil; free(memory);
}
int main(int argc,char **argv) { @autoreleasepool {
    assert(argc==3); device=MTLCreateSystemDefaultDevice(); assert(device);
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); assert(library);
    QW4DGpu *gpu=NULL; assert(qw4d_gpu_create((__bridge void *)device,(__bridge void *)library,&gpu,NULL)==QW4_OK);
    buffers=[NSMutableArray array];
    bool prep=!strcmp(argv[2],"prep");
    uint32_t initial=prep ? 0 : QW4_GPU_STATE;
    Qwen4GpuSpan err=buffer(4,false); *(uint32_t *)data(err)=initial;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer]; assert(command.retainedReferences);
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    Qwen4GpuCommand cmd={(__bridge void *)device,(__bridge void *)encoder,err,false};
    nocopy(gpu,cmd);
    uint32_t dims[4]; read_value(dims,sizeof(dims));
    for (size_t i=0;i<4;++i) assert(dims[i]<=QWEN4_MAX_CONTEXT);
    QW4DGpuArgs a={0}; Qwen4GpuSpan v[6]={0},output={0}; size_t count=0; uint32_t op=0;
    QW4CGpu *c_gpu=NULL;
    bool selection=!strcmp(argv[2],"select_tokens");
    if (selection) {
        uint32_t visible=dims[0],r=dims[1],budget=dims[2]; assert(r && budget && budget%r==0);
        uint32_t n=visible/r,m=MIN(n,budget/r)*r+visible%r;
        v[0]=buffer(n*4,true); v[1]=buffer(n*4,false); v[2]=buffer(m*4,false); count=3;
        Qwen4GpuSpan bad=v[2]; bad.bytes=0;
        if (m) assert(qw4d_select_encode(gpu,cmd,visible,r,budget,v[0],v[1],bad,NULL)==QW4_CAPACITY);
        bad=v[0]; bad.offset_bytes++;
        assert(qw4d_select_encode(gpu,cmd,visible,r,budget,bad,v[1],v[2],NULL)==QW4_STRIDE);
        if (n) assert(qw4d_select_encode(gpu,cmd,visible,r,budget,v[0],v[0],v[2],NULL)==QW4_INVALID_ARGUMENT);
        assert(qw4d_select_encode(gpu,cmd,QWEN4_MAX_CONTEXT+1,r,budget,v[0],v[1],v[2],NULL)==QW4_RANGE);
        assert(qw4d_select_encode(gpu,cmd,visible,0,budget,v[0],v[1],v[2],NULL)==QW4_RANGE);
    } else if (!strcmp(argv[2],"rope") || prep) {
        op=QW4D_ROPE; a.h=dims[0]; a.d=dims[1]; a.r=dims[2]; a.pos=dims[3];
        read_value(&a.theta,4); float eps=0; if (prep) read_value(&eps,4);
        v[0]=buffer((size_t)a.h*a.d*4,true); v[1]=buffer(v[0].bytes,false); count=2;
        if (prep) {
            Qwen4GpuSpan gamma=buffer(a.d*4,true),gate=buffer(v[0].bytes,true);
            Qwen4GpuSpan normalized=buffer(v[0].bytes,false); output=buffer(v[0].bytes*2,false);
            assert(qw4c_gpu_create((__bridge void *)device,(__bridge void *)library,&c_gpu,NULL)==QW4_OK);
            QW4CGpuSpec spec={.eps=eps,.activation=QW4_SIGMOID};
            Qwen4GpuF32 norms[]={view(v[0],a.h,a.d,false),view(gamma,1,a.d,true),view(normalized,a.h,a.d,false)};
            assert(qw4c_gpu_encode(c_gpu,cmd,QW4C_RMS,&spec,norms,3,NULL)==QW4_OK);
            v[0]=normalized; v[1]=output; v[1].bytes/=2;
            Qwen4GpuSpan activated=output; activated.bytes/=2; activated.offset_bytes+=activated.bytes;
            Qwen4GpuF32 gates[]={view(gate,a.h,a.d,false),view(activated,a.h,a.d,false)};
            assert(qw4c_gpu_encode(c_gpu,cmd,QW4C_ACT,&spec,gates,2,NULL)==QW4_OK);
        }
    } else if (!strcmp(argv[2],"pool")) {
        op=QW4D_POOL; a.h=1; a.n=dims[0]; a.d=dims[1]; count=2;
        v[0]=buffer((size_t)a.n*a.d*4,true); v[1]=buffer(a.d*4,false);
    } else if (!strcmp(argv[2],"scores")) {
        op=QW4D_SCORES; a.h=dims[0]; a.d=dims[1]; a.n=dims[2]; count=3;
        v[0]=buffer((size_t)a.h*a.d*4,true); v[1]=buffer((size_t)a.n*a.d*4,true); v[2]=buffer(a.n*4,false);
    } else {
        bool sparse=!strcmp(argv[2],"sparse_attend");
        assert(sparse || !strcmp(argv[2],"attend")); op=QW4D_ATTEND;
        a.h=dims[0]; a.j=dims[1]; a.d=dims[2]; a.n=dims[3]; count=6;
        v[0]=buffer((size_t)a.h*a.d*4,true); v[1]=buffer(v[0].bytes,true);
        v[2]=buffer((size_t)a.n*a.j*a.d*4,true); v[3]=buffer(v[2].bytes,true);
        if (sparse) {
            uint32_t r,budget; read_value(&r,4); read_value(&budget,4);
            assert(r && budget && budget%r==0);
            uint32_t blocks=a.n/r;
            a.m=MIN(blocks,budget/r)*r+a.n%r;
            Qwen4GpuSpan scores=buffer(blocks*4,true),keep=buffer(blocks*4,false);
            v[4]=buffer(a.m*4,false);
            assert(qw4d_select_encode(gpu,cmd,a.n,r,budget,scores,keep,v[4],NULL)==QW4_OK);
        } else {
            read_value(&a.m,4); assert(a.m<=a.n);
            v[4]=buffer(a.m*4,true);
        }
        v[5]=buffer(v[0].bytes,false);
    }
    if (!prep) output=v[count-1];
    if (!selection) admission(gpu,cmd,op,a,v,count);
    Qwen4Status status=selection ? qw4d_select_encode(gpu,cmd,dims[0],dims[1],dims[2],v[0],v[1],v[2],NULL)
        : qw4d_gpu_encode(gpu,cmd,op,&a,v,count,NULL);
    assert(status==QW4_OK);
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.error) fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);
    assert(command.status==MTLCommandBufferStatusCompleted);
    uint32_t bits=*(uint32_t *)data(err); assert((bits&QW4_GPU_STATE)==initial);
    status=(bits&QW4_GPU_NONFINITE) ? QW4_NONFINITE : ((bits&QW4_GPU_BOUNDS) ? QW4_RANGE : QW4_OK);
    guards(); assert(fwrite(&status,sizeof(status),1,stdout)==1);
    assert(fwrite(data(output),1,output.bytes,stdout)==output.bytes); assert(getchar()==EOF);
    fprintf(stderr,"QW4_D_METAL_BYTES=%llu\n",(unsigned long long)device.currentAllocatedSize);
    qw4c_gpu_destroy(c_gpu); qw4d_gpu_destroy(gpu); return 0;
} }
