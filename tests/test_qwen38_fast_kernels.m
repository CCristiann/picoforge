/* R1 fast-kernel harness, native side. `selector` checks Q38_REF_KERNELS
 * parsing (no GPU). GPU modes run only through qwen38_gpu_lock.py:
 *   <oracle|encode|fast> type physical logical n m [xpitch ypitch offset]
 *   bench <oracle|fast> n K m
 * oracle = q38_packed_encode with the oracle forced, encode = q38_packed_encode
 * as configured by the environment, fast = q38_packed_encode_fast.
 * stdin: n packed rows, then x [m,logical] FP32. stdout: y [m,n] FP32, then
 * the n rows decoded on the CPU (q38_packed_decode_row) [n,physical] FP32,
 * the source of the FP64 reference. Exit 8 (Q38_UNSUPPORTED) with no output:
 * no fast kernel; 16/20 (NONFINITE/GPU_ERROR): error word set.
 * Views: row pitches in floats (default logical and n), both views start
 * `offset` floats past a 16-byte guard. Every x float outside the view is NaN,
 * so a read past k_logical or between rows poisons y; every y byte outside the
 * view is 0xa5 and must stay so (no write outside y).
 * bench: Q8_0 [n,K] and x from a fixed seed (no stdin), 3 warm-ups + 20
 * measured command buffers, GPU time from GPUStartTime/GPUEndTime. */
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

typedef struct { const char *mode; uint32_t type; size_t physical, logical, n, m, bytes, xpitch, ypitch, offset; } Request;
static size_t number(const char *s) {
    assert(*s && strspn(s,"0123456789")==strlen(s));
    errno=0;
    unsigned long long v=strtoull(s,NULL,10);
    assert(!errno && v<=UINT32_MAX);
    return (size_t)v;
}
static Request request(int argc,char **argv) {
    assert(argc>1);
    bool bench=!strcmp(argv[1],"bench");
    assert(bench ? argc==6 : argc==7 || argc==10);
    Request r={argv[1+bench],8,0,0,0,0,0,0,0,0};
    if (bench) { r.physical=r.logical=number(argv[4]); r.n=number(argv[3]); r.m=number(argv[5]); }
    else { r.type=(uint32_t)number(argv[2]); r.physical=number(argv[3]); r.logical=number(argv[4]);
           r.n=number(argv[5]); r.m=number(argv[6]); }
    r.xpitch=argc==10 ? number(argv[7]) : r.logical;
    r.ypitch=argc==10 ? number(argv[8]) : r.n;
    r.offset=argc==10 ? number(argv[9]) : 0;
    assert(!strcmp(r.mode,"oracle") || !strcmp(r.mode,"fast") || (!bench && !strcmp(r.mode,"encode")));
    const GgufFormat *format=NULL;
    assert(q38_b_format(r.type,&format)==Q38_OK && r.physical%format->block==0);
    assert(r.logical && r.logical<=r.physical && r.n && r.m && r.xpitch>=r.logical && r.ypitch>=r.n);
    /* the B harness's transport budget; bench data never crosses stdin */
    const size_t limit=bench ? 1024u*1024u*1024u : 32u*1024u*1024u;
    assert(r.physical<=limit/4/r.n && r.m<=limit/4/MAX(r.xpitch,r.ypitch) && r.offset<=limit/4);
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
/* [rows,cols] view, row pitch `pitch` floats, `offset` floats past the 16-byte
 * leading guard; the buffer ends with a 16-byte guard after the last row. */
static size_t floats(Request r,size_t cols,size_t pitch) { return r.offset+(r.m-1)*pitch+cols; }
static Qwen38GpuF32 matrix(id<MTLBuffer> b,Request r,size_t cols,size_t pitch) {
    size_t bytes=(floats(r,cols,pitch)-r.offset)*4;
    return (Qwen38GpuF32){{(__bridge void *)b,16+r.offset*4,bytes},2,{r.m,cols,1,1},{pitch*4,4,0,0}};
}
static bool in_view(Request r,size_t cols,size_t pitch,size_t i) { /* i: float after the guard */
    return i>=r.offset && (i-r.offset)/pitch<r.m && (i-r.offset)%pitch<cols;
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

/* bench data: xorshift64 bytes; Q8_0 scales finite in [2^-7, 2^-6) so y stays finite. */
static uint64_t mix(uint64_t *s) { *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static void synthesize(unsigned char *raw,size_t bytes,uint64_t *seed) {
    assert(raw && bytes%34==0);
    for (size_t b=0;b<bytes;b+=34) {
        for (size_t i=2;i<34;i+=8) { uint64_t v=mix(seed); memcpy(raw+b+i,&v,MIN((size_t)8,34-i)); }
        raw[b]=(unsigned char)mix(seed); raw[b+1]=(unsigned char)(0x20u|(mix(seed)&3u));
    }
}
static int ascending(const void *a,const void *b) {
    double u=*(const double *)a, v=*(const double *)b; return (u>v)-(u<v);
}
static double quantile(const double *v,size_t n,double q) { /* numpy's linear rule */
    double at=q*(double)(n-1); size_t lo=(size_t)at; double hi=lo+1<n ? v[lo+1] : v[lo];
    return v[lo]+(at-(double)lo)*(hi-v[lo]);
}
/* Median and p10/p90 of the 20 measured runs; bytes = weights + x + y. */
static void report(Request r,double *seconds,const float *y) {
    for (size_t i=0;i<r.m*r.n;i++) assert(isfinite(y[i]));
    double *t=seconds+3; qsort(t,20,sizeof *t,ascending);
    double bytes=(double)(r.bytes+4*r.m*(r.logical+r.n)), median=quantile(t,20,0.5);
    printf("{\"mode\": \"%s\", \"n\": %zu, \"k\": %zu, \"m\": %zu, \"bytes\": %.0f, "
           "\"median_us\": %.1f, \"p10_us\": %.1f, \"p90_us\": %.1f, \"gbps\": %.1f}\n",
           r.mode,r.n,r.logical,r.m,bytes,median*1e6,quantile(t,20,0.1)*1e6,quantile(t,20,0.9)*1e6,
           bytes/median/1e9);
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"selector")) return selector_cases();
    const char *lock=getenv("PICOFORGE_GPU_LOCK_FD"), *library=getenv("Q38_PACKED_LIBRARY");
    if (!lock || fcntl((int)number(lock),F_GETFD)<0 || !library) {
        fprintf(stderr,"run through tools/qwen38_gpu_lock.py with Q38_PACKED_LIBRARY set\n"); return 2;
    }
    Request r=request(argc,argv);
    bool timing=!strcmp(argv[1],"bench");
    unsigned char *raw=malloc(r.bytes);
    uint64_t seed=3813;
    if (timing) synthesize(raw,r.bytes,&seed); else assert(raw && fread(raw,1,r.bytes,stdin)==r.bytes);
    size_t initial=gpu_weights_live_mapped_bytes();
    size_t xw=floats(r,r.logical,r.xpitch)+8, yw=floats(r,r.n,r.ypitch)+8; /* words incl. guards */
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
        id<MTLBuffer> input=shared(dev,xw*4),output=shared(dev,yw*4),word=shared(dev,4);
        float *x=(float *)input.contents+4, *y=(float *)output.contents+4;
        for (size_t i=0;i<xw;i++) ((float *)input.contents)[i]=NAN; /* guards and gaps */
        for (size_t i=0;i<r.m;i++) {
            float *row=x+r.offset+i*r.xpitch;
            if (!timing) assert(fread(row,4,r.logical,stdin)==r.logical);
            else for (size_t k=0;k<r.logical;k++) row[k]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
        }
        assert(timing || fgetc(stdin)==EOF);
        memset(output.contents,0xa5,yw*4);
        for (size_t i=0;i<r.m;i++)
            for (size_t j=0;j<r.n;j++) y[r.offset+i*r.ypitch+j]=NAN; /* an unwritten output cannot pass */
        Qwen38GpuF32 xv=matrix(input,r,r.logical,r.xpitch), yv=matrix(output,r,r.n,r.ypitch);
        uint32_t *flags=word.contents;
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        double seconds[23];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            flags[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4}};
            status=(int)encode(ctx,r,cmd,store,binding,xv,yv);
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || (!status && !flags[0]));
        }
        const uint32_t *words=output.contents; /* no write outside y */
        for (size_t i=0;i<yw;i++)
            assert((i>=4 && i<yw-4 && in_view(r,r.n,r.ypitch,i-4)) || words[i]==0xa5a5a5a5u);
        if (timing) report(r,seconds,y);
        else if (!status && flags[0]) {
            status=(flags[0]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
            fprintf(stderr,"qwen38 Metal: error bits %u\n",flags[0]);
        } else if (!status) {
            for (size_t i=0;i<r.m;i++) assert(fwrite(y+r.offset+i*r.ypitch,4,r.n,stdout)==r.n);
            write_decoded(store,binding,r);
        }
        q38_packed_gpu_free(ctx); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw);
    return status;
}
