/* R1 fast-kernel harness, native side. `selector` checks Q38_REF_KERNELS
 * parsing (no GPU). GPU modes run only through qwen38_gpu_lock.py:
 *   <oracle|encode|fast> type physical logical n m [xpitch ypitch offset]
 *   bench <oracle|fast> n K m [reps [type]]
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
 * bench: a [n,K] tensor of `type` (default 8, Q8_0; 1 F16, 0 F32) and x from a
 * fixed seed (no stdin), 3 warm-ups + 20
 * measured command buffers, GPU time from GPUStartTime/GPUEndTime. reps > 1
 * encodes the kernel reps times back to back in each command buffer (as in
 * the decode graph) and reports the time per encode. */
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

typedef struct { const char *mode; uint32_t type; size_t physical, logical, n, m, bytes, xpitch, ypitch, offset, reps; } Request;
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
    assert(bench ? argc>=6 && argc<=8 : argc==7 || argc==10);
    Request r={argv[1+bench],8,0,0,0,0,0,0,0,0,1};
    if (bench) { r.physical=r.logical=number(argv[4]); r.n=number(argv[3]); r.m=number(argv[5]);
                 r.reps=argc>=7 ? number(argv[6]) : 1; assert(r.reps && r.reps<=1000);
                 r.type=argc==8 ? (uint32_t)number(argv[7]) : 8; assert(r.type==8 || r.type<=1); }
    else { r.type=(uint32_t)number(argv[2]); r.physical=number(argv[3]); r.logical=number(argv[4]);
           r.n=number(argv[5]); r.m=number(argv[6]); }
    r.xpitch=argc==10 ? number(argv[7]) : r.logical;
    r.ypitch=argc==10 ? number(argv[8]) : r.n;
    r.offset=argc==10 ? number(argv[9]) : 0;
    assert(!strcmp(r.mode,"oracle") || !strcmp(r.mode,"fast") || (!bench && !strcmp(r.mode,"encode")));
    const GgufFormat *format=NULL;
    assert(q38_b_format(r.type,&format)==Q38_OK && r.physical%format->block==0);
    assert(r.logical && r.logical<=r.physical && r.n && r.m && r.xpitch>=r.logical && r.ypitch>=r.n);
    /* The B harness's transport budget (decoded rows cross stdout); bench data
     * never crosses a pipe and is bounded by its packed size instead. */
    const size_t limit=bench ? 1024u*1024u*1024u : 32u*1024u*1024u;
    assert(r.physical<=(bench ? limit/r.n : limit/4/r.n) && r.m<=limit/4/MAX(r.xpitch,r.ypitch) && r.offset<=limit/4);
    r.bytes=r.physical/format->block*format->bytes*r.n;
    assert(r.bytes<=limit);
    return r;
}
/* count rank-2 tensors (raw back to back) in an unlinked file, each 32 bytes
 * past its own page boundary so no binding offset is page aligned (as in
 * test_qwen38_packed_gpu.m); the view cap (the largest tensor) splits them
 * over several Metal buffers, so a multi call binds different weight buffers. */
static GpuWeightStore *make_store(id<MTLDevice> dev,size_t count,const Request *r,const unsigned char *raw,
                                  Gguf *g,GgufTensor *t) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE), extent=0, at=page;
    for (size_t i=0;i<count;i++) extent=MAX(extent,(r[i].bytes+32+page-1)/page*page);
    char path[]="build/qwen38-r1/weights-XXXXXX";
    g->fd=mkstemp(path);
    assert(g->fd>=0 && unlink(path)==0);
    for (size_t i=0;i<count;at+=(r[i].bytes+32+page-1)/page*page,raw+=r[i++].bytes) {
        t[i]=(GgufTensor){.type=r[i].type,.rank=2,.shape={r[i].physical,r[i].n},.file_offset=at+32,.bytes=r[i].bytes};
        snprintf(t[i].name,sizeof t[i].name,"fixture%zu",i);
        for (size_t done=0;done<r[i].bytes;) {
            ssize_t wrote=pwrite(g->fd,raw+done,r[i].bytes-done,(off_t)(t[i].file_offset+done));
            if (wrote<0 && errno==EINTR) continue;
            assert(wrote>0); done+=(size_t)wrote;
        }
    }
    g->file_size=at+page;
    assert(ftruncate(g->fd,(off_t)g->file_size)==0);
    g->tensors=t; g->tensor_count=count;
    GpuWeightStore *store=gpu_weights_open((__bridge void *)dev,g,extent);
    assert(store && gpu_weights_plan(store)->weight_count==count && (count>1 || gpu_weights_plan(store)->view_count==1));
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
    for (size_t i=0;i<r.n;i++) { /* a row it rejects (nonfinite padding) is sent as NaN */
        Qwen38Status status=q38_packed_decode_row(&w,0,i,row,r.physical,&err);
        assert(status==Q38_OK || status==Q38_NONFINITE);
        for (size_t k=0;status && k<r.physical;k++) row[k]=NAN;
        assert(fwrite(row,4,r.physical,stdout)==r.physical);
    }
    free(row);
}

/* bench data: xorshift64 bytes; Q8_0 scales finite in [2^-7, 2^-6), F16/F32
 * elements finite with magnitude below 1, so y stays finite. */
static uint64_t mix(uint64_t *s) { *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static void synthesize(unsigned char *raw,size_t bytes,uint64_t *seed,uint32_t type) {
    if (type==1) { /* sign, exponent 1..14 (2^-14..2^-1), random fraction */
        for (size_t i=0;i+1<bytes;i+=2) {
            uint64_t v=mix(seed); uint16_t h=(uint16_t)((v&0x83ffu)|((1u+(v>>16)%14u)<<10));
            memcpy(raw+i,&h,2);
        }
        return;
    }
    if (type==0) { /* sign, exponent 2^-20..2^-1, random fraction */
        for (size_t i=0;i+3<bytes;i+=4) {
            uint64_t v=mix(seed); uint32_t f=(uint32_t)((v&0x807fffffu)|((107u+(v>>32)%20u)<<23));
            memcpy(raw+i,&f,4);
        }
        return;
    }
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
static void report(Request r,double *seconds,const float *y,const char *specs) {
    for (size_t i=0;y && i<r.m*r.n;i++) assert(isfinite(y[i]));
    double *t=seconds+3; qsort(t,20,sizeof *t,ascending);
    for (size_t i=0;i<20;i++) t[i]/=(double)r.reps;
    double bytes=(double)(r.bytes+4*r.m*(r.logical+r.n)), median=quantile(t,20,0.5);
    printf("{\"mode\": \"%s\", \"type\": %u, \"n\": %zu, \"k\": %zu, \"m\": %zu, \"reps\": %zu, \"bytes\": %.0f, "
           "\"median_us\": %.1f, \"p10_us\": %.1f, \"p90_us\": %.1f, \"gbps\": %.1f%s%s%s}\n",
           r.mode,r.type,r.n,r.logical,r.m,r.reps,bytes,median*1e6,quantile(t,20,0.1)*1e6,quantile(t,20,0.9)*1e6,
           bytes/median/1e9,specs ? ", \"specs\": \"" : "",specs ? specs : "",specs ? "\"" : "");
}

/* Counts the dispatches a multi call encodes (forwarding as qwen38_profile.m). */
@interface Q38DispatchCounter : NSProxy { @public id<MTLComputeCommandEncoder> target; uint32_t dispatches; }
@end
@implementation Q38DispatchCounter
- (id)forwardingTargetForSelector:(SEL)s { (void)s; return target; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)s { return [(NSObject *)target methodSignatureForSelector:s]; }
- (void)forwardInvocation:(NSInvocation *)i { [i invokeWithTarget:target]; }
- (void)dispatchThreads:(MTLSize)n threadsPerThreadgroup:(MTLSize)g { dispatches++; [target dispatchThreads:n threadsPerThreadgroup:g]; }
- (void)dispatchThreadgroups:(MTLSize)n threadsPerThreadgroup:(MTLSize)g { dispatches++; [target dispatchThreadgroups:n threadsPerThreadgroup:g]; }
@end

/* R1.4 multi-GEMV, one store of count tensors sharing x [m,K]:
 *   multi m xpitch ypad offset type:physical:logical:n ...
 * encodes q38_packed_encode_multi as the environment selects; stdin: the
 * packed tensors back to back, then x; stdout: the number of dispatches it
 * encoded (uint32), then y_0 [m,n_0], y_1, ... Views
 * as above, y_i pitch n_i+ypad, each y_i in its own guarded buffer.
 *   mbench <multi|single> m reps type:physical:logical:n ...
 * times one multi call, or the fast single calls back to back (the graph
 * before R1.4), each followed by one barrier, on synthetic data. */
static int multi_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"mbench");
    size_t first=timing ? 5 : 6, count=(size_t)argc-first;
    assert((size_t)argc>first && count<=Q38_MULTI_MAX);
    size_t xpitch=0, ypad=timing ? 0 : number(argv[4]), total=0, outputs=0, bytes=0;
    Request p[Q38_MULTI_MAX], r={.mode=timing ? argv[2] : "multi",.m=number(argv[2+timing]),
        .offset=timing ? 0 : number(argv[5]),.reps=timing ? number(argv[4]) : 1};
    assert(!strcmp(r.mode,"multi") || !strcmp(r.mode,"single"));
    char specs[512]="";
    for (size_t i=0;i<count;i++) {
        unsigned type; size_t physical,logical,n; int used=0;
        assert(sscanf(argv[first+i],"%u:%zu:%zu:%zu%n",&type,&physical,&logical,&n,&used)==4 && !argv[first+i][(size_t)used]);
        const GgufFormat *format=NULL;
        assert(q38_b_format(type,&format)==Q38_OK && physical%format->block==0 && logical<=physical && n);
        assert(!i || logical==r.logical); /* one x: every projection has the same K */
        p[i]=(Request){.type=type,.physical=physical,.logical=logical,.n=n,.bytes=physical/format->block*format->bytes*n};
        r.logical=logical; total+=p[i].bytes; outputs+=n; assert(total<=(1u<<30) && outputs<=(1u<<24));
        snprintf(specs+strlen(specs),sizeof specs-strlen(specs),"%s%s",i ? " " : "",argv[first+i]);
    }
    xpitch=timing ? r.logical : number(argv[3]);
    assert(r.m && r.m<=4096 && r.reps && r.reps<=1000 && xpitch>=r.logical && r.offset<=1u<<20 && ypad<=1u<<20);
    unsigned char *raw=malloc(total); assert(raw);
    uint64_t seed=3813;
    for (size_t i=0;i<count;bytes+=p[i++].bytes)
        if (timing) synthesize(raw+bytes,p[i].bytes,&seed,p[i].type); else assert(fread(raw+bytes,1,p[i].bytes,stdin)==p[i].bytes);
    size_t initial=gpu_weights_live_mapped_bytes(), xw=floats(r,r.logical,xpitch)+8;
    int status=0; uint32_t dispatches=0;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); abort(); }
        Q38PackedGpu *ctx=NULL;
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        Gguf g={0}; GgufTensor tensors[Q38_MULTI_MAX];
        GpuWeightStore *store=make_store(dev,count,p,raw,&g,tensors);
        const GgufBinding *bindings[Q38_MULTI_MAX]; Q38PackedLayout layouts[Q38_MULTI_MAX];
        id<MTLBuffer> input=shared(dev,xw*4), output[Q38_MULTI_MAX], word=shared(dev,4);
        Qwen38GpuF32 yv[Q38_MULTI_MAX];
        float *x=(float *)input.contents+4;
        for (size_t i=0;i<xw;i++) ((float *)input.contents)[i]=NAN; /* guards and gaps */
        for (size_t i=0;i<r.m;i++) {
            float *row=x+r.offset+i*xpitch;
            if (!timing) assert(fread(row,4,r.logical,stdin)==r.logical);
            else for (size_t k=0;k<r.logical;k++) row[k]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
        }
        assert(timing || fgetc(stdin)==EOF);
        for (size_t i=0;i<count;i++) {
            bindings[i]=&gpu_weights_plan(store)->weights[i];
            assert(q38_packed_layout_init(&bindings[i]->tensor,r.logical,&layouts[i],NULL)==Q38_OK);
            size_t pitch=p[i].n+ypad, yw=floats(r,p[i].n,pitch)+8;
            output[i]=shared(dev,yw*4); memset(output[i].contents,0xa5,yw*4);
            for (size_t m=0;m<r.m;m++)
                for (size_t j=0;j<p[i].n;j++) ((float *)output[i].contents)[4+r.offset+m*pitch+j]=NAN;
            yv[i]=matrix(output[i],r,p[i].n,pitch);
        }
        Qwen38GpuF32 xv=matrix(input,r,r.logical,xpitch);
        uint32_t *flags=word.contents;
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        double seconds[23];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            flags[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
            counter->target=encoder;
            if (!timing) encoder=(id<MTLComputeCommandEncoder>)counter;
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4}};
            for (size_t rep=0;rep<r.reps && !status;rep++) {
                Qwen38Error err={0};
                if (!strcmp(r.mode,"multi")) {
                    status=(int)q38_packed_encode_multi(ctx,&cmd,store,bindings,layouts,count,xv,yv,&err);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                } else for (size_t i=0;i<count && !status;i++) {
                    status=(int)q38_packed_encode_fast(ctx,&cmd,store,bindings[i],&layouts[i],0,xv,yv[i],&err);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                if (status && status!=Q38_UNSUPPORTED) { fprintf(stderr,"encode: %d %s\n",status,err.detail); abort(); }
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || (!status && !flags[0]));
            dispatches=counter->dispatches;
        }
        for (size_t i=0;i<count;i++) { /* no write outside any y */
            const uint32_t *words=output[i].contents; size_t pitch=p[i].n+ypad, yw=floats(r,p[i].n,pitch)+8;
            for (size_t w=0;w<yw;w++)
                assert((w>=4 && w<yw-4 && in_view(r,p[i].n,pitch,w-4)) || words[w]==0xa5a5a5a5u);
            for (size_t m=0;timing && m<r.m;m++)
                for (size_t j=0;j<p[i].n;j++) assert(isfinite(((float *)output[i].contents)[4+r.offset+m*pitch+j]));
        }
        r.n=outputs; r.bytes=total; r.type=p[0].type;
        if (timing) report(r,seconds,NULL,specs);
        else if (!status && flags[0]) {
            status=(flags[0]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
            fprintf(stderr,"qwen38 Metal: error bits %u\n",flags[0]);
        } else if (!status) assert(fwrite(&dispatches,4,1,stdout)==1);
        for (size_t i=0;!timing && !status && i<count;i++)
            for (size_t m=0;m<r.m;m++)
                assert(fwrite((float *)output[i].contents+4+r.offset+m*(p[i].n+ypad),4,p[i].n,stdout)==p[i].n);
        q38_packed_gpu_free(ctx); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw);
    return status;
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"selector")) return selector_cases();
    const char *lock=getenv("PICOFORGE_GPU_LOCK_FD"), *library=getenv("Q38_PACKED_LIBRARY");
    if (!lock || fcntl((int)number(lock),F_GETFD)<0 || !library) {
        fprintf(stderr,"run through tools/qwen38_gpu_lock.py with Q38_PACKED_LIBRARY set\n"); return 2;
    }
    if (argc>1 && (!strcmp(argv[1],"multi") || !strcmp(argv[1],"mbench"))) return multi_main(argc,argv,library);
    Request r=request(argc,argv);
    bool timing=!strcmp(argv[1],"bench");
    unsigned char *raw=malloc(r.bytes);
    uint64_t seed=3813;
    if (timing) synthesize(raw,r.bytes,&seed,r.type); else assert(raw && fread(raw,1,r.bytes,stdin)==r.bytes);
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
        GpuWeightStore *store=make_store(dev,1,&r,raw,&g,&tensor);
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
            for (size_t rep=0;rep<r.reps && !status;rep++) status=(int)encode(ctx,r,cmd,store,binding,xv,yv);
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || (!status && !flags[0]));
        }
        const uint32_t *words=output.contents; /* no write outside y */
        for (size_t i=0;i<yw;i++)
            assert((i>=4 && i<yw-4 && in_view(r,r.n,r.ypitch,i-4)) || words[i]==0xa5a5a5a5u);
        if (timing) report(r,seconds,y,NULL);
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
