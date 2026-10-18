/* R1 fast-kernel harness, native side. `selector` checks Q38_REF_KERNELS
 * parsing (no GPU). GPU modes run only through qwen38_gpu_lock.py:
 *   <oracle|encode|fast> type physical logical n m
 * oracle = q38_packed_encode with the oracle forced, encode = q38_packed_encode
 * as configured by the environment, fast = q38_packed_encode_fast.
 * stdin: n packed rows, then x [m,logical] FP32. stdout: y [m,n] FP32, then
 * the n rows decoded on the CPU (q38_packed_decode_row) [n,physical] FP32,
 * the source of the FP64 reference. Exit 8 (Q38_UNSUPPORTED) with no output:
 * no fast kernel; 16/20 (NONFINITE/GPU_ERROR): error word set. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include "qwen38_ref.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

/* test_selector_parsing: pure parser cases, then once through the environment. */
static int selector_cases(void) {
    static const struct { const char *value, *family; bool want; } cases[] = {
        {NULL,"linear",false}, {"","linear",false}, {"0","argmax",false},
        {"1","linear",true}, {"1","argmax",true}, {"all","claim",true},
        {"linear,moe","linear",true}, {"linear,moe","moe",true}, {"linear,moe","indexed",false},
        {"linear","linear",true}, {"linear","barrier",false},
        /* R18: any unknown token (typo, number, empty) selects every oracle. */
        {"lineer","moe",true}, {"2","linear",true}, {"linear,,moe","indexed",true},
        {"linear,","fuse",true}, {",moe","argmax",true}, {"Linear","linear",true},
    };
    for (size_t i=0;i<sizeof cases/sizeof *cases;i++)
        if (q38_ref_parse(cases[i].value,cases[i].family)!=cases[i].want) {
            fprintf(stderr,"selector case %zu (\"%s\", %s) expected %d\n",i,
                    cases[i].value ? cases[i].value : "(unset)",cases[i].family,cases[i].want);
            return 1;
        }
    assert(setenv("Q38_REF_KERNELS","indexed",1)==0);
    assert(q38_ref_kernels("indexed") && !q38_ref_kernels("linear"));
    assert(unsetenv("Q38_REF_KERNELS")==0 && !q38_ref_kernels("indexed"));
    return 0;
}

typedef struct { const char *mode; uint32_t type; size_t physical, logical, n, m, bytes; } Request;
static size_t number(const char *s) {
    assert(*s && strspn(s,"0123456789")==strlen(s));
    errno=0;
    unsigned long long v=strtoull(s,NULL,10);
    assert(!errno && v<=UINT32_MAX);
    return (size_t)v;
}
static Request request(int argc,char **argv) {
    assert(argc==7);
    Request r={argv[1],(uint32_t)number(argv[2]),number(argv[3]),number(argv[4]),number(argv[5]),number(argv[6]),0};
    assert(!strcmp(r.mode,"oracle") || !strcmp(r.mode,"encode") || !strcmp(r.mode,"fast"));
    const GgufFormat *format=NULL;
    assert(q38_b_format(r.type,&format)==Q38_OK && r.physical%format->block==0);
    assert(r.logical && r.logical<=r.physical && r.n && r.m);
    const size_t limit=32u*1024u*1024u; /* the B harness's transport budget */
    assert(r.physical<=limit/4/r.n && r.m<=limit/4/MAX(r.logical,r.n));
    r.bytes=r.physical/format->block*format->bytes*r.n;
    return r;
}
/* One rank-2 tensor in an unlinked file, 32 bytes past a page boundary so the
 * binding offset is not page aligned (as in test_qwen38_packed_gpu.m). */
static GpuWeightStore *make_store(id<MTLDevice> dev,Request r,const unsigned char *raw,
                                  Gguf *g,GgufTensor *t) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE), extent=(r.bytes+32+page-1)/page*page;
    char path[]="build/qwen38-r1/weights-XXXXXX";
    g->fd=mkstemp(path);
    assert(g->fd>=0 && unlink(path)==0);
    g->file_size=extent+2*page;
    assert(ftruncate(g->fd,(off_t)g->file_size)==0);
    *t=(GgufTensor){.name="fixture",.type=r.type,.rank=2,.shape={r.physical,r.n},
                    .file_offset=page+32,.bytes=r.bytes};
    for (size_t done=0;done<r.bytes;) {
        ssize_t wrote=pwrite(g->fd,raw+done,r.bytes-done,(off_t)(t->file_offset+done));
        if (wrote<0 && errno==EINTR) continue;
        assert(wrote>0); done+=(size_t)wrote;
    }
    g->tensors=t; g->tensor_count=1;
    GpuWeightStore *store=gpu_weights_open((__bridge void *)dev,g,extent);
    assert(store && gpu_weights_plan(store)->view_count==1);
    return store;
}
static id<MTLBuffer> shared(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> buffer=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    assert(buffer); return buffer;
}
/* Contiguous [rows,cols] view 16 bytes into the buffer (guards on both sides). */
static Qwen38GpuF32 matrix(id<MTLBuffer> b,size_t rows,size_t cols) {
    return (Qwen38GpuF32){{(__bridge void *)b,16,rows*cols*4},2,{rows,cols,1,1},{cols*4,4,0,0}};
}
static Qwen38Status encode(Q38PackedGpu *ctx,Request r,Qwen38GpuCommand cmd,const GpuWeightStore *store,
        const GgufBinding *binding,Qwen38GpuF32 x,Qwen38GpuF32 y) {
    Q38PackedLayout layout; Qwen38Error err;
    assert(q38_packed_layout_init(&binding->tensor,r.logical,&layout,NULL)==Q38_OK);
    if (!strcmp(r.mode,"oracle")) q38_packed_gpu_set_reference(ctx,true,true);
    Qwen38Status status=!strcmp(r.mode,"fast") ?
        q38_packed_encode_fast(ctx,&cmd,store,binding,&layout,0,x,y,&err) :
        q38_packed_encode(ctx,&cmd,store,binding,&layout,0,x,y,&err);
    if (status && status!=Q38_UNSUPPORTED) { fprintf(stderr,"encode: %d %s\n",status,err.detail); abort(); }
    return status;
}
static void write_decoded(const GpuWeightStore *store,const GgufBinding *binding,Request r) {
    Q38PackedCpuView w; Qwen38Error err;
    assert(qwen38_weights_cpu_span(store,binding,&w.span,&err)==Q38_OK);
    assert(q38_packed_layout_init(&binding->tensor,r.logical,&w.layout,NULL)==Q38_OK);
    float *row=malloc(r.physical*4); assert(row);
    for (size_t i=0;i<r.n;i++) {
        assert(q38_packed_decode_row(&w,0,i,row,r.physical,&err)==Q38_OK);
        assert(fwrite(row,4,r.physical,stdout)==r.physical);
    }
    free(row);
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"selector")) return selector_cases();
    const char *lock=getenv("PICOFORGE_GPU_LOCK_FD"), *library=getenv("Q38_PACKED_LIBRARY");
    if (!lock || fcntl((int)number(lock),F_GETFD)<0 || !library) {
        fprintf(stderr,"run through tools/qwen38_gpu_lock.py with Q38_PACKED_LIBRARY set\n"); return 2;
    }
    Request r=request(argc,argv);
    unsigned char *raw=malloc(r.bytes);
    assert(raw && fread(raw,1,r.bytes,stdin)==r.bytes);
    size_t initial=gpu_weights_live_mapped_bytes(), xb=r.m*r.logical*4, yb=r.m*r.n*4;
    int status=0;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); abort(); }
        Q38PackedGpu *ctx=NULL;
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        Gguf g={0}; GgufTensor tensor;
        GpuWeightStore *store=make_store(dev,r,raw,&g,&tensor);
        const GgufBinding *binding=&gpu_weights_plan(store)->weights[0];
        id<MTLBuffer> input=shared(dev,xb+32),output=shared(dev,yb+32),word=shared(dev,4);
        assert(fread((unsigned char *)input.contents+16,1,xb,stdin)==xb && fgetc(stdin)==EOF);
        memset(output.contents,0xa5,yb+32);
        float *y=(float *)((unsigned char *)output.contents+16);
        for (size_t i=0;i<r.m*r.n;i++) y[i]=NAN; /* an unwritten output cannot pass */
        uint32_t *flags=word.contents; flags[0]=0;
        id<MTLCommandBuffer> command=[[dev newCommandQueue] commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4}};
        status=(int)encode(ctx,r,cmd,store,binding,matrix(input,r.m,r.logical),matrix(output,r.m,r.n));
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
        const unsigned char *guard=output.contents; /* no write outside y */
        for (size_t i=0;i<16;i++) assert(guard[i]==0xa5 && guard[16+yb+i]==0xa5);
        if (!status && flags[0]) {
            status=(flags[0]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
            fprintf(stderr,"qwen38 Metal: error bits %u\n",flags[0]);
        } else if (!status) {
            assert(fwrite(y,4,r.m*r.n,stdout)==r.m*r.n);
            write_decoded(store,binding,r);
        }
        q38_packed_gpu_free(ctx); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw);
    return status;
}
