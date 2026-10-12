/* Bounded Metal transport and ABI tests. Run only through qwen38_gpu_lock.py. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    uint32_t type;
    size_t physical, logical, n, experts, expert, m, bytes;
    bool decode;
} Request;
static size_t argument(const char *s) {
    assert(*s && strspn(s,"0123456789")==strlen(s));
    errno=0;
    unsigned long long n=strtoull(s,NULL,10);
    assert(!errno && n<=SIZE_MAX);
    return (size_t)n;
}
static uint32_t format_argument(const char *s) {
    size_t n=argument(s); assert(n<=UINT32_MAX); return (uint32_t)n;
}
static Request request(int argc,char **argv) {
    Request r={0};
    r.decode=argc==4;
    if (r.decode) {
        r.type=format_argument(argv[1]); r.physical=argument(argv[2]); r.n=argument(argv[3]);
        r.logical=r.physical; r.experts=1; r.m=1;
    } else {
        assert(argc==9 && !strcmp(argv[1],"matmul"));
        r.type=format_argument(argv[2]); r.physical=argument(argv[3]); r.logical=argument(argv[4]);
        r.n=argument(argv[5]); r.experts=argument(argv[6]); r.expert=argument(argv[7]); r.m=argument(argv[8]);
    }
    const size_t limit=32u*1024u*1024u;
    const GgufFormat *format=NULL;
    assert(q38_b_format(r.type,&format)==Q38_OK);
    assert(r.physical && r.logical && r.logical<=r.physical && r.n && r.experts && r.expert<r.experts);
    assert(r.physical<=limit/4 && r.n<=limit/4/r.physical && r.experts<=limit/4/r.physical/r.n);
    assert(r.physical%format->block==0 && r.m<=128);
    r.bytes=r.physical/format->block*format->bytes*r.n*r.experts;
    assert(r.bytes<=limit);
    return r;
}
static GpuWeightStore *make_store(id<MTLDevice> dev,Request r,const unsigned char *raw,
                                   Gguf *g,GgufTensor tensors[2]) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE), extent=(r.bytes+32+page-1)/page*page;
    char path[]="build/qwen38-b/weights-XXXXXX";
    g->fd=mkstemp(path);
    assert(g->fd>=0 && unlink(path)==0);
    g->file_size=2*extent+3*page;
    assert(ftruncate(g->fd,(off_t)g->file_size)==0);
    for (size_t i=0;i<2;i++) {
        tensors[i]=(GgufTensor){.type=r.type,.rank=3,.shape={r.physical,r.n,r.experts},
            .file_offset=page+i*(extent+page)+32,.bytes=r.bytes};
        snprintf(tensors[i].name,sizeof tensors[i].name,"fixture%zu",i);
        size_t done=0;
        while (done<r.bytes) {
            ssize_t wrote=pwrite(g->fd,raw+done,r.bytes-done,(off_t)(tensors[i].file_offset+done));
            if (wrote<0 && errno==EINTR) continue;
            assert(wrote>0); done+=(size_t)wrote;
        }
    }
    g->tensors=tensors; g->tensor_count=2;
    GpuWeightStore *store=gpu_weights_open((__bridge void *)dev,g,extent);
    assert(gpu_weights_plan(store)->view_count==2);
    return store;
}
static id<MTLBuffer> shared(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> buffer=[dev newBufferWithLength:MAX(bytes,(size_t)4) options:MTLResourceStorageModeShared];
    assert(buffer); return buffer;
}
static id<MTLLibrary> load_library(id<MTLDevice> dev) {
    const char *path=getenv("Q38_PACKED_LIBRARY");
    if (!path) path="build/qwen38-b/packed.metallib";
    NSError *error=nil;
    id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
    if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); abort(); }
    return lib;
}
static void decode_rows(id<MTLDevice> dev,id<MTLLibrary> lib,id<MTLComputeCommandEncoder> enc,
        const GpuWeightStore *store,const GgufBinding *binding,Request r,id<MTLBuffer> out,id<MTLBuffer> word) {
    NSError *error=nil;
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q38_packed_decode"] error:&error];
    assert(pipeline);
    Q38PackedLayout layout;
    assert(q38_packed_layout_init(&binding->tensor,r.logical,&layout,NULL)==Q38_OK);
    Q38PackedArgs args={.type=r.type,.k_storage=(uint32_t)r.physical,.n_out=(uint32_t)r.n,
        .row_bytes=layout.row_bytes,.weight_offset=binding->offset,
        .weight_end=binding->offset+layout.expert_bytes,.y_bytes=r.n*r.physical*4};
    if (getenv("Q38_PACKED_BOUNDS")) args.weight_end=binding->offset+1;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)gpu_weights_buffer(store,binding) offset:0 atIndex:0];
    [enc setBuffer:out offset:16 atIndex:1];
    [enc setBuffer:word offset:4 atIndex:2];
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreads:MTLSizeMake(r.physical,r.n,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
}

static void admission(Q38PackedGpu *ctx,Qwen38GpuCommand cmd,const GpuWeightStore *store,
        const GgufBinding *binding,Q38PackedLayout layout,size_t expert,Qwen38GpuF32 x,Qwen38GpuF32 y) {
    Qwen38Error err;
    GgufBinding copy=*binding;
    assert(q38_packed_encode(ctx,&cmd,store,&copy,&layout,expert,x,y,&err)==Q38_OWNER && err.code==Q38_OWNER);
    Q38PackedLayout bad=layout; bad.row_bytes++;
    assert(q38_packed_encode(ctx,&cmd,store,binding,&bad,expert,x,y,&err)==Q38_FORMAT);
    assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,layout.n_experts,x,y,&err)==Q38_RANGE);
    Qwen38GpuF32 changed=y; changed.dim[1]++;
    assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,x,changed,&err)==Q38_SHAPE);
    changed=x; changed.stride_bytes[1]=8;
    assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,changed,y,&err)==Q38_STRIDE);
    changed=x; changed.span.offset_bytes=((__bridge id<MTLBuffer>)x.span.buffer).length+4;
    assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,changed,y,&err)==Q38_RANGE);
    Qwen38GpuCommand badcmd=cmd; badcmd.device=NULL;
    assert(q38_packed_encode(ctx,&badcmd,store,binding,&layout,expert,x,y,&err)==Q38_OWNER);
    badcmd=cmd; badcmd.error_word.offset_bytes++;
    assert(q38_packed_encode(ctx,&badcmd,store,binding,&layout,expert,x,y,&err)==Q38_STRIDE);
    if (x.dim[0]) {
        changed=x; changed.span.bytes=(x.dim[0]-1)*x.stride_bytes[0]+x.dim[1]*4-1;
        assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,changed,y,&err)==Q38_CAPACITY);
        badcmd=cmd; badcmd.error_word=y.span;
        assert(q38_packed_encode(ctx,&badcmd,store,binding,&layout,expert,x,y,&err)==Q38_INVALID_ARGUMENT);
        size_t ybytes=(y.dim[0]-1)*y.stride_bytes[0]+y.dim[1]*4;
        if (x.span.bytes>=ybytes) {
            changed=y; changed.span=x.span;
            assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,x,changed,&err)==Q38_INVALID_ARGUMENT);
        }
    }
    /* A well-shaped empty batch must encode no work and may have nil spans. */
    changed=x; Qwen38GpuF32 empty_y=y;
    changed.dim[0]=empty_y.dim[0]=0;
    changed.span=(Qwen38GpuSpan){0}; empty_y.span=(Qwen38GpuSpan){0};
    assert(q38_packed_encode(ctx,&cmd,store,binding,&layout,expert,changed,empty_y,&err)==Q38_OK);
}
static void linear_rows(Q38PackedGpu *ctx,Qwen38GpuCommand cmd,const GpuWeightStore *store,
        const GgufBinding *binding,Request r,id<MTLBuffer> input,id<MTLBuffer> output) {
    Q38PackedLayout layout;
    assert(q38_packed_layout_init(&binding->tensor,r.logical,&layout,NULL)==Q38_OK);
    Qwen38GpuF32 x={{(__bridge void *)input,16,r.m*(r.logical+3)*4},2,
        {r.m,r.logical,1,1},{(r.logical+3)*4,4,0,0}};
    Qwen38GpuF32 y={{(__bridge void *)output,16,r.m*(r.n+3)*4},2,
        {r.m,r.n,1,1},{(r.n+3)*4,4,0,0}};
    admission(ctx,cmd,store,binding,layout,r.expert,x,y);
    Qwen38Error err;
    Qwen38Status status=q38_packed_encode(ctx,&cmd,store,binding,&layout,r.expert,x,y,&err);
    if (status) { fprintf(stderr,"encode: %d %s\n",status,err.detail); abort(); }
}

int main(int argc,char **argv) {
    const char *lock=getenv("PICOFORGE_GPU_LOCK_FD");
    if (!lock || fcntl((int)argument(lock),F_GETFD)<0) {
        fprintf(stderr,"run through tools/qwen38_gpu_lock.py; inherited lock fd required\n"); return 2;
    }
    Request r=request(argc,argv);
    unsigned char *raw=malloc(r.bytes);
    assert(raw && fread(raw,1,r.bytes,stdin)==r.bytes);
    size_t initial=gpu_weights_live_mapped_bytes();
    int status=0;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        id<MTLLibrary> lib=load_library(dev);
        Q38PackedGpu *ctx=(Q38PackedGpu *)(uintptr_t)1;
        assert(q38_packed_gpu_create((__bridge void *)dev,NULL,&ctx,NULL)==Q38_INVALID_ARGUMENT && !ctx);
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        Gguf g={0}; GgufTensor tensors[2];
        GpuWeightStore *store=make_store(dev,r,raw,&g,tensors);
        const GgufBinding *binding=&gpu_weights_plan(store)->weights[r.expert%2];
        size_t input_bytes=r.decode ? 0 : r.m*(r.logical+3)*4;
        size_t output_bytes=r.decode ? r.n*r.physical*4 : r.m*(r.n+3)*4;
        assert(input_bytes<=32u*1024u*1024u && output_bytes<=32u*1024u*1024u);
        id<MTLBuffer> input=shared(dev,input_bytes+32),output=shared(dev,output_bytes+32),word=shared(dev,12);
        memset(output.contents,0xa5,output_bytes+32);
        float *x=(float *)((unsigned char *)input.contents+16);
        float *y=(float *)((unsigned char *)output.contents+16);
        for(size_t i=0;i<output_bytes/4;i++) y[i]=NAN;
        if (!r.decode) for(size_t m=0;m<r.m;m++) {
            assert(fread(x+m*(r.logical+3),4,r.logical,stdin)==r.logical);
            for(size_t k=r.logical;k<r.logical+3;k++) x[m*(r.logical+3)+k]=NAN;
            for(size_t n=r.n;n<r.n+3;n++) y[m*(r.n+3)+n]=123;
        }
        assert(fgetc(stdin)==EOF && !ferror(stdin));
        uint32_t *flags=word.contents;
        uint32_t seed=getenv("Q38_PACKED_STICKY") ? Q38_GPU_STATE : 0;
        flags[0]=flags[2]=0x12345678; flags[1]=seed;
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        assert(command.retainedReferences);
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,4,4}};
        if (r.decode) decode_rows(dev,lib,encoder,store,binding,r,output,word);
        else linear_rows(ctx,cmd,store,binding,r,input,output);
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
        assert(flags[0]==0x12345678 && flags[2]==0x12345678 && (flags[1]&seed)==seed);
        unsigned char *guard=output.contents;
        for(size_t i=0;i<16;i++) assert(guard[i]==0xa5 && guard[16+output_bytes+i]==0xa5);
        if(flags[1]&~seed) {
            status=(flags[1]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
            fprintf(stderr,"qwen38 Metal: %d error bits %u\n",status,flags[1]);
        } else if(r.decode) assert(fwrite(y,4,r.n*r.physical,stdout)==r.n*r.physical);
        else for(size_t m=0;m<r.m;m++) {
            for(size_t n=r.n;n<r.n+3;n++) assert(y[m*(r.n+3)+n]==123);
            assert(fwrite(y+m*(r.n+3),4,r.n,stdout)==r.n);
        }
        q38_packed_gpu_free(ctx); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw);
    return status;
}
