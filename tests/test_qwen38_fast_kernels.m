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
#include "qwen38_c_internal.h"
#include "qwen38_d_internal.h"
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
        {"attention","attention",true}, {"attention","linear",false}, {"linear","attention",false},
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

/* experts: 0 or 1 = a rank-2 tensor; more = rank 3 [physical, n, experts] (indexed modes). */
typedef struct { const char *mode; uint32_t type; size_t physical, logical, n, m, bytes, xpitch, ypitch, offset, reps, experts; } Request;
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
    Request r={argv[1+bench],8,0,0,0,0,0,0,0,0,1,0};
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
        t[i]=(GgufTensor){.type=r[i].type,.rank=r[i].experts>1 ? 3 : 2,.shape={r[i].physical,r[i].n,MAX(r[i].experts,(size_t)1)},
                          .file_offset=at+32,.bytes=r[i].bytes};
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
    for (size_t i=0;i<r.n*MAX(r.experts,(size_t)1);i++) { /* a row it rejects (nonfinite padding) is sent as NaN */
        Qwen38Status status=q38_packed_decode_row(&w,i/r.n,i%r.n,row,r.physical,&err);
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
    if (type!=8) { /* R1.5 block formats: random codes, small finite scales, so y stays finite */
        for (size_t i=0;i<bytes;i+=8) { uint64_t v=mix(seed); memcpy(raw+i,&v,MIN((size_t)8,bytes-i)); }
        const GgufFormat *format=gguf_format(type);
        for (size_t b=0;b<bytes;b+=format->bytes) {
            uint16_t h[2];
            for (size_t j=0;j<2;j++) { uint64_t v=mix(seed); h[j]=(uint16_t)((v&0x83ffu)|((5u+(v>>16)%5u)<<10)); }
            if (type==39) raw[b]=(unsigned char)(118u+mix(seed)%7u); /* scale 2^-9..2^-3 */
            else memcpy(raw+b+(type==10 ? 80 : 0),h,type==16 ? 2 : 4); /* d (and dmin) 2^-10..2^-5 */
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
@interface Q38DispatchCounter : NSProxy { @public id<MTLComputeCommandEncoder> target; uint32_t dispatches, barriers; }
@end
@implementation Q38DispatchCounter
- (id)forwardingTargetForSelector:(SEL)s { (void)s; return target; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)s { return [(NSObject *)target methodSignatureForSelector:s]; }
- (void)forwardInvocation:(NSInvocation *)i { [i invokeWithTarget:target]; }
- (void)memoryBarrierWithScope:(MTLBarrierScope)s { barriers++; [target memoryBarrierWithScope:s]; }
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
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4},false};
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

/* R1.5 indexed expert matvecs (q38_packed_encode_indexed's contract):
 *   indexed <oracle|encode|fast> type physical logical n experts rows divisor [xpitch ypitch offset]
 * stdin: the packed [experts, n, physical] tensor, ids uint32 [rows], then x
 * [rows/divisor, logical]. stdout: y [rows, n] (even when the error word is
 * set: an element that is not written stays the NaN it was filled with),
 * then, with a clean error word, the CPU-decoded rows [experts*n, physical].
 * Views as above (x has rows/divisor rows, y has rows).
 *   ibench <oracle|fast> type n physical logical experts rows divisor reps
 * synthetic tensor, ids distinct within each run of 10 rows (a token's
 * slots), 3 warm-ups + 20 measured command buffers of reps encodes each.
 * R1.5b fused gate+up+SwiGLU, same arguments with type = gate[/up] type:
 *   gateup <fused|sequence|oracle> ...   gbench <fused|sequence> ...
 * fused = q38_packed_encode_gate_up (exit 8, no output: UNSUPPORTED);
 * sequence = q38f_routed_encode's unfused sequence with the indexed fast
 * kernels (gate, up, q38c ACT SILU, q38f_product, a barrier after each);
 * oracle = the same with the indexed oracle. stdin: the gate then the up
 * tensor, ids, x; stdout: act [rows, n], then the decoded gate and up rows. */
static Qwen38Status sequence(Q38PackedGpu *ctx,Q38CGpu *ops,id<MTLComputePipelineState> product,Qwen38GpuCommand cmd,
        const GpuWeightStore *store,const GgufBinding *const *b,const Q38PackedLayout *l,Qwen38GpuSpan ids,size_t divisor,
        Qwen38GpuF32 x,Qwen38GpuF32 gate,Qwen38GpuF32 up,Qwen38GpuF32 act,bool fast,Qwen38Error *err) {
    id<MTLComputeCommandEncoder> encoder=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    Qwen38GpuF32 out[2]={gate,up};
    for (size_t i=0;i<2;i++) {
        Qwen38Status status=fast ? q38_packed_encode_indexed_fast(ctx,&cmd,store,b[i],&l[i],ids,divisor,x,out[i],err) :
            q38_packed_encode_indexed(ctx,&cmd,store,b[i],&l[i],ids,divisor,x,out[i],err);
        if (status) return status;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    Q38CGpuSpec spec={.activation=Q38_SILU}; Qwen38GpuF32 views[]={gate,act};
    Qwen38Status status=q38c_gpu_encode(ops,cmd,Q38C_ACT,&spec,views,2,err);
    if (status) return status;
    [encoder setComputePipelineState:product];
    [encoder setBuffer:(__bridge id<MTLBuffer>)act.span.buffer offset:act.span.offset_bytes atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)up.span.buffer offset:up.span.offset_bytes atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)cmd.error_word.buffer offset:cmd.error_word.offset_bytes atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(act.dim[0]*act.dim[1],1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    return Q38_OK;
}
/* Weight bytes of one row the indexed kernels read for k logical columns:
 * whole blocks, and of a partial Q2_K block (the 640-of-768 down) only its
 * scale bytes, the 32-byte code halves it touches and d/dmin; another format's
 * partial block counts whole. */
static size_t read_bytes(uint32_t type,size_t k) {
    const GgufFormat *f=gguf_format(type);
    size_t whole=k/f->block*f->bytes, c=k%f->block;
    if (!c) return whole;
    return whole+(type==10 ? (c+15)/16+32*((c+127)/128)+4 : f->bytes);
}
static int indexed_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"ibench") || !strcmp(argv[1],"gbench"), pair=argv[1][0]=='g';
    assert(timing ? argc==11 : argc==10 || argc==13);
    unsigned types[2]={0,0}; int parsed=0;
    int fields=sscanf(argv[3],"%u%n/%u%n",&types[0],&parsed,&types[1],&parsed);
    assert((fields==1 || (pair && fields==2)) && !argv[3][(size_t)parsed]);
    if (fields==1) types[1]=types[0];
    Request r={.mode=argv[2],.type=types[0]};
    size_t at=4;
    if (timing) { r.n=number(argv[at++]); r.physical=number(argv[at++]); }
    else { r.physical=number(argv[at++]); r.logical=number(argv[at++]); r.n=number(argv[at++]); }
    if (timing) r.logical=number(argv[at++]);
    r.experts=number(argv[at++]);
    size_t rows=number(argv[at++]), divisor=number(argv[at++]);
    r.reps=timing ? number(argv[at++]) : 1;
    assert(pair ? !strcmp(r.mode,"fused") || !strcmp(r.mode,"sequence") || (!timing && !strcmp(r.mode,"oracle")) :
           !strcmp(r.mode,"oracle") || !strcmp(r.mode,"fast") || (!timing && !strcmp(r.mode,"encode")));
    assert(r.n && r.experts && rows && divisor && rows%divisor==0 && r.reps && r.reps<=1000);
    r.m=rows/divisor;
    r.xpitch=argc==13 ? number(argv[10]) : r.logical;
    r.ypitch=argc==13 ? number(argv[11]) : r.n;
    r.offset=argc==13 ? number(argv[12]) : 0;
    const size_t limit=timing ? 1024u*1024u*1024u : 32u*1024u*1024u;
    assert(r.xpitch>=r.logical && r.ypitch>=r.n && r.offset<=limit/4 && r.m<=limit/4/r.xpitch && rows<=limit/4/r.ypitch);
    assert(r.physical<=(timing ? limit/r.n/r.experts : limit/4/r.n/r.experts/(1+pair)));
    assert(!pair || !strcmp(r.mode,"fused") || r.ypitch==r.n); /* q38f_product needs a dense act */
    Request t[2]={r,r};
    size_t total=0;
    for (size_t i=0;i<1+pair;i++) {
        const GgufFormat *format=NULL;
        t[i].type=types[i];
        assert(q38_b_format(t[i].type,&format)==Q38_OK && r.physical%format->block==0 && r.logical && r.logical<=r.physical);
        t[i].bytes=r.physical/format->block*format->bytes*r.n*r.experts;
        total+=t[i].bytes;
    }
    assert(total<=limit);
    r.bytes=t[0].bytes;
    Request ry=r; ry.m=rows; /* the y view's rows */
    unsigned char *raw=malloc(total); assert(raw);
    uint32_t *ids=malloc(rows*4); assert(ids);
    uint64_t seed=3813;
    if (timing) {
        for (size_t i=0;i<1+pair;i++) synthesize(raw+(i ? t[0].bytes : 0),t[i].bytes,&seed,t[i].type);
        for (size_t p=0;p<rows;p++) /* distinct within a token's 10 slots */
            for (bool again=true;again;) {
                ids[p]=(uint32_t)(mix(&seed)%r.experts); again=false;
                for (size_t q=p/10*10;q<p && r.experts>=10;q++) again=again || ids[q]==ids[p];
            }
    } else assert(fread(raw,1,total,stdin)==total && fread(ids,4,rows,stdin)==rows);
    size_t initial=gpu_weights_live_mapped_bytes();
    size_t xw=floats(r,r.logical,r.xpitch)+8, yw=floats(ry,r.n,r.ypitch)+8;
    int status=0;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); abort(); }
        Q38PackedGpu *ctx=NULL; Q38CGpu *ops=NULL;
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        assert(q38c_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ops,NULL)==Q38_OK);
        id<MTLComputePipelineState> product=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q38f_product"] error:&error];
        assert(product);
        Gguf g={0}; GgufTensor tensors[2];
        GpuWeightStore *store=make_store(dev,1+pair,t,raw,&g,tensors);
        const GgufBinding *bindings[2]={&gpu_weights_plan(store)->weights[0],&gpu_weights_plan(store)->weights[pair]};
        Q38PackedLayout layouts[2];
        for (size_t i=0;i<2;i++) assert(q38_packed_layout_init(&bindings[i]->tensor,r.logical,&layouts[i],NULL)==Q38_OK);
        id<MTLBuffer> input=shared(dev,xw*4),output=shared(dev,yw*4),word=shared(dev,4),route=shared(dev,rows*4);
        id<MTLBuffer> scratch[2]={shared(dev,rows*r.n*4),shared(dev,rows*r.n*4)}; /* the sequence's gate and up */
        memcpy(route.contents,ids,rows*4);
        float *x=(float *)input.contents+4, *y=(float *)output.contents+4;
        for (size_t i=0;i<xw;i++) ((float *)input.contents)[i]=NAN; /* guards and gaps */
        for (size_t i=0;i<r.m;i++) {
            float *row=x+r.offset+i*r.xpitch;
            if (!timing) assert(fread(row,4,r.logical,stdin)==r.logical);
            else for (size_t k=0;k<r.logical;k++) row[k]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
        }
        assert(timing || fgetc(stdin)==EOF);
        memset(output.contents,0xa5,yw*4);
        for (size_t i=0;i<rows;i++)
            for (size_t j=0;j<r.n;j++) y[r.offset+i*r.ypitch+j]=NAN; /* an unwritten output cannot pass */
        Qwen38GpuF32 xv=matrix(input,r,r.logical,r.xpitch), yv=matrix(output,ry,r.n,r.ypitch), sv[2];
        for (size_t i=0;i<2;i++) sv[i]=(Qwen38GpuF32){{(__bridge void *)scratch[i],0,rows*r.n*4},2,{rows,r.n,1,1},{r.n*4,4,0,0}};
        Qwen38GpuSpan span={(__bridge void *)route,0,rows*4};
        uint32_t *flags=word.contents;
        if (!strcmp(r.mode,"oracle")) q38_packed_gpu_set_reference(ctx,true,true);
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        double seconds[23];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            flags[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4},false};
            for (size_t rep=0;rep<r.reps && !status;rep++) {
                Qwen38Error err={0};
                if (!pair) status=(int)(!strcmp(r.mode,"fast") ?
                    q38_packed_encode_indexed_fast(ctx,&cmd,store,bindings[0],&layouts[0],span,divisor,xv,yv,&err) :
                    q38_packed_encode_indexed(ctx,&cmd,store,bindings[0],&layouts[0],span,divisor,xv,yv,&err));
                else if (!strcmp(r.mode,"fused"))
                    status=(int)q38_packed_encode_gate_up(ctx,&cmd,store,bindings[0],&layouts[0],bindings[1],&layouts[1],
                                                         span,divisor,xv,yv,&err);
                else status=(int)sequence(ctx,ops,product,cmd,store,bindings,layouts,span,divisor,xv,sv[0],sv[1],yv,
                                          !strcmp(r.mode,"sequence"),&err);
                if (status && status!=Q38_UNSUPPORTED) { fprintf(stderr,"encode: %d %s\n",status,err.detail); abort(); }
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || (!status && !flags[0]));
        }
        const uint32_t *words=output.contents; /* no write outside y */
        for (size_t i=0;i<yw;i++)
            assert((i>=4 && i<yw-4 && in_view(ry,r.n,r.ypitch,i-4)) || words[i]==0xa5a5a5a5u);
        if (timing) {
            size_t distinct=0; /* bytes actually read: each distinct expert once (per tensor), x and y */
            for (size_t e=0;e<r.experts;e++) { bool used=false; for (size_t p=0;p<rows && !used;p++) used=ids[p]==e; distinct+=used; }
            Request tr=ry; /* report() adds 4*rows*(logical+n): keep x at its rows/divisor rows (modular size_t) */
            tr.bytes=distinct*r.n*(read_bytes(t[0].type,r.logical)+(pair ? read_bytes(t[1].type,r.logical) : 0))+4*r.m*r.logical-4*rows*r.logical;
            report(tr,seconds,NULL,NULL);
        } else if (!status) {
            for (size_t i=0;i<rows;i++) assert(fwrite(y+r.offset+i*r.ypitch,4,r.n,stdout)==r.n);
            if (flags[0]) {
                status=(flags[0]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
                fprintf(stderr,"qwen38 Metal: error bits %u\n",flags[0]);
            } else for (size_t i=0;i<1+pair;i++) write_decoded(store,bindings[i],t[i]);
        }
        q38_packed_gpu_free(ctx); q38c_gpu_destroy(ops); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw); free(ids);
    return status;
}

/* R1.5c fused routed down + weighted reduce (q38_packed_encode_down_reduce):
 *   downreduce <fused|sequence|oracle> type physical logical n experts tokens slots [xpitch ypitch offset]
 * stdin: the packed [experts, n, physical] tensor, ids uint32 [tokens*slots],
 * act [tokens*slots, logical], scores [tokens, slots], shared [tokens, n],
 * sigmoid [tokens]. stdout: y [tokens, n] and routed [tokens, n] (also with
 * the error word set: unwritten elements stay NaN), then, with a clean word,
 * the CPU-decoded rows. fused = q38_packed_encode_down_reduce (exit 8, no
 * output: UNSUPPORTED); sequence = q38f_routed_encode's unfused tail: indexed
 * fast down into a [pairs, n] scratch, barrier, q38f_reduce with the identity
 * map; oracle = the same with the indexed oracle. Views: act pitch xpitch, y
 * pitch ypitch, both `offset` floats past the guard; the rest dense.
 *   dbench <fused|sequence> type n physical logical experts tokens slots reps
 * synthetic data as ibench; bytes = the fused pass's: distinct experts' rows
 * read, act, shared, y and routed (the same count for both modes).
 *   rbench experts slots tokens reps
 * times q38f_route alone (one thread per token, as the graph dispatches it). */
static int down_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"dbench"), route=!strcmp(argv[1],"rbench");
    assert(route ? argc==6 : timing ? argc==11 : argc==10 || argc==13);
    Request r={.mode=route ? "route" : argv[2]};
    size_t at=route ? 2 : 3, tokens, slots;
    if (!route) r.type=(uint32_t)number(argv[at++]);
    if (timing) { r.n=number(argv[at++]); r.physical=number(argv[at++]); r.logical=number(argv[at++]); }
    else if (!route) { r.physical=number(argv[at++]); r.logical=number(argv[at++]); r.n=number(argv[at++]); }
    r.experts=number(argv[at++]);
    if (route) { slots=number(argv[at++]); tokens=number(argv[at++]); }
    else { tokens=number(argv[at++]); slots=number(argv[at++]); }
    r.reps=timing || route ? number(argv[at++]) : 1;
    assert(tokens && slots && r.experts && r.reps<=1000 && r.reps && tokens<=4096 && slots<=64);
    size_t pairs=tokens*slots;
    uint64_t seed=3813;
    if (route) {
        assert(slots<=r.experts && r.experts<=65536);
        r.n=slots; r.logical=r.experts; r.m=tokens; r.bytes=4*tokens*r.experts; /* + logits and ids in report() */
        double seconds[23];
        @autoreleasepool {
            id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
            NSError *error=nil;
            id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q38f_route"] error:&error];
            assert(p);
            id<MTLBuffer> logits=shared(dev,4*tokens*r.experts),probs=shared(dev,4*tokens*r.experts),ids=shared(dev,4*pairs),
                weights=shared(dev,4*pairs),word=shared(dev,4);
            for (size_t i=0;i<tokens*r.experts;i++) ((float *)logits.contents)[i]=(float)(mix(&seed)>>40)*0x1p-21f-4.0f;
            struct { uint32_t t,e,k,norm,pitch; } ra={(uint32_t)tokens,(uint32_t)r.experts,(uint32_t)slots,1,(uint32_t)r.experts};
            id<MTLCommandQueue> queue=[dev newCommandQueue];
            for (int run=0;run<23;run++) {
                ((uint32_t *)word.contents)[0]=0;
                id<MTLCommandBuffer> command=[queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                for (size_t rep=0;rep<r.reps;rep++) { /* as q38f_routed_encode's run(): 64-thread groups, a barrier */
                    [encoder setComputePipelineState:p];
                    [encoder setBuffer:logits offset:0 atIndex:0]; [encoder setBuffer:probs offset:0 atIndex:1];
                    [encoder setBuffer:ids offset:0 atIndex:2]; [encoder setBuffer:weights offset:0 atIndex:3];
                    [encoder setBytes:&ra length:sizeof ra atIndex:4]; [encoder setBuffer:word offset:0 atIndex:5];
                    [encoder dispatchThreads:MTLSizeMake(tokens,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)64,p.maxTotalThreadsPerThreadgroup),1,1)];
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
                assert(command.status==MTLCommandBufferStatusCompleted && !command.error && !((uint32_t *)word.contents)[0]);
                seconds[run]=command.GPUEndTime-command.GPUStartTime;
            }
        }
        report(r,seconds,NULL,NULL);
        return 0;
    }
    assert(timing ? !strcmp(r.mode,"fused") || !strcmp(r.mode,"sequence") :
           !strcmp(r.mode,"fused") || !strcmp(r.mode,"sequence") || !strcmp(r.mode,"oracle"));
    r.m=pairs;
    r.xpitch=argc==13 ? number(argv[10]) : r.logical;
    r.ypitch=argc==13 ? number(argv[11]) : r.n;
    r.offset=argc==13 ? number(argv[12]) : 0;
    const size_t limit=timing ? 1024u*1024u*1024u : 32u*1024u*1024u;
    const GgufFormat *format=NULL;
    assert(q38_b_format(r.type,&format)==Q38_OK && r.physical%format->block==0 && r.logical && r.logical<=r.physical);
    assert(r.xpitch>=r.logical && r.ypitch>=r.n && r.offset<=limit/4 && pairs<=limit/4/r.xpitch && tokens<=limit/4/r.ypitch);
    assert(r.n<=limit/4/pairs && r.physical<=(timing ? limit/r.n/r.experts : limit/4/r.n/r.experts));
    r.bytes=r.physical/format->block*format->bytes*r.n*r.experts;
    Request ry=r; ry.m=tokens;
    unsigned char *raw=malloc(r.bytes); assert(raw);
    uint32_t *ids=malloc(pairs*4); float *aux=malloc(4*tokens*(slots+r.n+1)); assert(ids && aux);
    if (timing) {
        synthesize(raw,r.bytes,&seed,r.type);
        for (size_t p=0;p<pairs;p++) /* distinct within a token's slots */
            for (bool again=true;again;) {
                ids[p]=(uint32_t)(mix(&seed)%r.experts); again=false;
                for (size_t q=p/slots*slots;q<p && r.experts>=slots;q++) again=again || ids[q]==ids[p];
            }
        for (size_t i=0;i<tokens*(slots+r.n+1);i++) aux[i]=(float)(mix(&seed)>>40)*0x1p-24f;
    } else assert(fread(raw,1,r.bytes,stdin)==r.bytes && fread(ids,4,pairs,stdin)==pairs);
    size_t initial=gpu_weights_live_mapped_bytes();
    size_t xw=floats(r,r.logical,r.xpitch)+8, yw=floats(ry,r.n,r.ypitch)+8;
    int status=0;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); abort(); }
        Q38PackedGpu *ctx=NULL;
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        id<MTLComputePipelineState> reduce=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q38f_reduce"] error:&error];
        assert(reduce);
        Gguf g={0}; GgufTensor tensor;
        GpuWeightStore *store=make_store(dev,1,&r,raw,&g,&tensor);
        const GgufBinding *binding=&gpu_weights_plan(store)->weights[0];
        Q38PackedLayout layout;
        assert(q38_packed_layout_init(&binding->tensor,r.logical,&layout,NULL)==Q38_OK);
        id<MTLBuffer> input=shared(dev,xw*4),output=shared(dev,yw*4),word=shared(dev,4),route_ids=shared(dev,pairs*4),
            scores=shared(dev,tokens*slots*4),sh=shared(dev,tokens*r.n*4),sig=shared(dev,tokens*4),
            routed=shared(dev,tokens*r.n*4),down=shared(dev,pairs*r.n*4),identity=shared(dev,pairs*4);
        memcpy(route_ids.contents,ids,pairs*4);
        for (size_t p=0;p<pairs;p++) ((uint32_t *)identity.contents)[p]=(uint32_t)p;
        float *x=(float *)input.contents+4, *y=(float *)output.contents+4;
        for (size_t i=0;i<xw;i++) ((float *)input.contents)[i]=NAN; /* guards and gaps */
        for (size_t i=0;i<pairs;i++) {
            float *row=x+r.offset+i*r.xpitch;
            if (!timing) assert(fread(row,4,r.logical,stdin)==r.logical);
            else for (size_t k=0;k<r.logical;k++) row[k]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
        }
        if (!timing) assert(fread(aux,4,tokens*(slots+r.n+1),stdin)==tokens*(slots+r.n+1) && fgetc(stdin)==EOF);
        memcpy(scores.contents,aux,tokens*slots*4);
        memcpy(sh.contents,aux+tokens*slots,tokens*r.n*4);
        memcpy(sig.contents,aux+tokens*(slots+r.n),tokens*4);
        memset(output.contents,0xa5,yw*4);
        for (size_t i=0;i<tokens;i++)
            for (size_t j=0;j<r.n;j++) y[r.offset+i*r.ypitch+j]=NAN; /* an unwritten output cannot pass */
        for (size_t i=0;i<tokens*r.n;i++) ((float *)routed.contents)[i]=NAN;
        Qwen38GpuF32 xv=matrix(input,r,r.logical,r.xpitch), yv=matrix(output,ry,r.n,r.ypitch);
        Qwen38GpuF32 dv={{(__bridge void *)down,0,pairs*r.n*4},2,{pairs,r.n,1,1},{r.n*4,4,0,0}};
        Q38DownReduceViews views={{{(__bridge void *)scores,0,tokens*slots*4},2,{tokens,slots,1,1},{slots*4,4,0,0}},
            {{(__bridge void *)sh,0,tokens*r.n*4},2,{tokens,r.n,1,1},{r.n*4,4,0,0}},
            {{(__bridge void *)sig,0,tokens*4},2,{tokens,1,1,1},{4,4,0,0}},
            {{(__bridge void *)routed,0,tokens*r.n*4},2,{tokens,r.n,1,1},{r.n*4,4,0,0}}};
        Qwen38GpuSpan span={(__bridge void *)route_ids,0,pairs*4};
        uint32_t *flags=word.contents;
        bool fused=!strcmp(r.mode,"fused"), fast=!strcmp(r.mode,"sequence");
        if (!strcmp(r.mode,"oracle")) q38_packed_gpu_set_reference(ctx,true,true);
        struct { uint32_t t,h,k,xpitch,ypitch; } fa={(uint32_t)tokens,(uint32_t)r.n,(uint32_t)slots,(uint32_t)r.n,(uint32_t)r.ypitch};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        double seconds[23];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            flags[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4},false};
            for (size_t rep=0;rep<r.reps && !status;rep++) {
                Qwen38Error err={0};
                if (fused) status=(int)q38_packed_encode_down_reduce(ctx,&cmd,store,binding,&layout,span,xv,&views,yv,&err);
                else {
                    status=(int)(fast ? q38_packed_encode_indexed_fast(ctx,&cmd,store,binding,&layout,span,1,xv,dv,&err) :
                                 q38_packed_encode_indexed(ctx,&cmd,store,binding,&layout,span,1,xv,dv,&err));
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    id<MTLBuffer> in[]={down,sh,sig,scores,identity};
                    [encoder setComputePipelineState:reduce];
                    for (NSUInteger i=0;i<5;i++) [encoder setBuffer:in[i] offset:0 atIndex:i];
                    [encoder setBuffer:output offset:yv.span.offset_bytes atIndex:5];
                    [encoder setBytes:&fa length:sizeof fa atIndex:6];
                    [encoder setBuffer:word offset:0 atIndex:7]; [encoder setBuffer:routed offset:0 atIndex:8];
                    [encoder dispatchThreads:MTLSizeMake(tokens*r.n,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)64,reduce.maxTotalThreadsPerThreadgroup),1,1)];
                }
                if (status && status!=Q38_UNSUPPORTED) { fprintf(stderr,"encode: %d %s\n",status,err.detail); abort(); }
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || (!status && !flags[0]));
        }
        const uint32_t *words=output.contents; /* no write outside y */
        for (size_t i=0;i<yw;i++)
            assert((i>=4 && i<yw-4 && in_view(ry,r.n,r.ypitch,i-4)) || words[i]==0xa5a5a5a5u);
        if (timing) {
            size_t distinct=0;
            for (size_t e=0;e<r.experts;e++) { bool used=false; for (size_t p=0;p<pairs && !used;p++) used=ids[p]==e; distinct+=used; }
            Request tr=r; /* report() adds 4*pairs*(logical+n): keep act, count shared, y and routed */
            tr.bytes=distinct*r.n*read_bytes(r.type,r.logical)+4*3*tokens*r.n-4*pairs*r.n;
            report(tr,seconds,NULL,NULL);
        } else if (!status) {
            for (size_t i=0;i<tokens;i++) assert(fwrite(y+r.offset+i*r.ypitch,4,r.n,stdout)==r.n);
            assert(fwrite(routed.contents,4,tokens*r.n,stdout)==tokens*r.n);
            if (flags[0]) {
                status=(flags[0]&Q38_GPU_NONFINITE) ? Q38_NONFINITE : Q38_GPU_ERROR;
                fprintf(stderr,"qwen38 Metal: error bits %u\n",flags[0]);
            } else write_decoded(store,binding,r);
        }
        q38_packed_gpu_free(ctx); gpu_weights_close(store); assert(close(g.fd)==0);
    }
    assert(gpu_weights_live_mapped_bytes()==initial);
    free(raw); free(ids); free(aux);
    return status;
}

/* R1.5d parallel router top-k (q38_moe_route_fast) against q38f_route:
 *   route <fast|oracle> experts slots tokens norm pitch
 * stdin: logits [tokens, pitch] FP32 (the floats past `experts` in a row are
 * not read). stdout: the error word (uint32), ids uint32 [tokens*slots], weights
 * FP32 [tokens*slots], probs FP32 [tokens*experts]; ids, weights and probs are
 * pre-filled with 0xa5 so a missing write shows. fast = the new kernel, one
 * threadgroup of ROUTE_THREADS per token (the host's dispatch in
 * q38f_routed_encode, which also falls back above 1024 experts: here the
 * kernel runs regardless, to check its own guard); oracle = q38f_route, 64-thread
 * groups. Always exit 0 (the word says what happened).
 *   rtbench <fast|oracle> experts slots tokens reps
 * the same dispatch on synthetic logits, 20 timed per command buffer (rbench's). */
#define ROUTE_THREADS 256 /* Q38F_ROUTE_THREADS of qwen38_f_internal.h */
static int route_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"rtbench");
    assert(argc==(timing ? 7 : 8) &&(!strcmp(argv[2],"fast") || !strcmp(argv[2],"oracle")));
    bool fast=!strcmp(argv[2],"fast");
    size_t experts=number(argv[3]), slots=number(argv[4]), tokens=number(argv[5]);
    size_t reps=timing ? number(argv[6]) : 1, norm=timing ? 1 : number(argv[6]), pitch=timing ? experts : number(argv[7]);
    assert(experts && experts<=65536 && slots && slots<=64 && tokens && tokens<=4096 && reps && reps<=1000);
    size_t pairs=tokens*slots;
    float *input=calloc(tokens*pitch,4); assert(input);
    if (timing) {
        uint64_t seed=3813; /* rbench's synthetic logits */
        for (size_t i=0;i<tokens*experts;i++) input[i]=(float)(mix(&seed)>>40)*0x1p-21f-4.0f;
    } else assert(pitch>=experts && fread(input,4,tokens*pitch,stdin)==tokens*pitch && fgetc(stdin)==EOF);
    double seconds[23];
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:fast ? @"q38_moe_route_fast" : @"q38f_route"] error:&error];
        assert(p && (!fast || (p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=ROUTE_THREADS)));
        id<MTLBuffer> logits=shared(dev,4*tokens*pitch),probs=shared(dev,4*tokens*experts),ids=shared(dev,4*pairs),
            weights=shared(dev,4*pairs),word=shared(dev,4);
        memcpy(logits.contents,input,4*tokens*pitch);
        memset(probs.contents,0xa5,4*tokens*experts); memset(ids.contents,0xa5,4*pairs); memset(weights.contents,0xa5,4*pairs);
        struct { uint32_t t,e,k,norm,pitch; } ra={(uint32_t)tokens,(uint32_t)experts,(uint32_t)slots,(uint32_t)norm,(uint32_t)pitch};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            ((uint32_t *)word.contents)[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            for (size_t rep=0;rep<reps;rep++) {
                [encoder setComputePipelineState:p];
                [encoder setBuffer:logits offset:0 atIndex:0]; [encoder setBuffer:probs offset:0 atIndex:1];
                [encoder setBuffer:ids offset:0 atIndex:2]; [encoder setBuffer:weights offset:0 atIndex:3];
                [encoder setBytes:&ra length:sizeof ra atIndex:4]; [encoder setBuffer:word offset:0 atIndex:5];
                if (fast) [encoder dispatchThreadgroups:MTLSizeMake(tokens,1,1) threadsPerThreadgroup:MTLSizeMake(ROUTE_THREADS,1,1)];
                else [encoder dispatchThreads:MTLSizeMake(tokens,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)64,p.maxTotalThreadsPerThreadgroup),1,1)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || !((uint32_t *)word.contents)[0]);
        }
        if (timing) {
            Request r={.mode=argv[2],.n=slots,.logical=experts,.m=tokens,.reps=reps,.bytes=4*tokens*experts};
            report(r,seconds,NULL,NULL);
        } else {
            assert(fwrite(word.contents,4,1,stdout)==1 && fwrite(ids.contents,4,pairs,stdout)==pairs &&
                   fwrite(weights.contents,4,pairs,stdout)==pairs && fwrite(probs.contents,4,tokens*experts,stdout)==tokens*experts);
        }
    }
    free(input);
    return 0;
}

/* R1.6a RMSNorm through q38c_gpu_encode (op 0 Q38C_RMS, 1 Q38C_GDN_NORM):
 *   rms <oracle|fast|encode> op t groups width gain_rank pitch offset inplace initial
 * oracle/fast force the choice (q38c_gpu_set_reference), encode keeps what
 * Q38_REF_KERNELS selected at creation. x, z (and y unless inplace): [t,groups,
 * width] views, row pitch `pitch` floats, `offset` floats past a 16-byte guard,
 * NaN in every gap; gain [width] (rank 1) or [groups,width]; the error word
 * starts at `initial`. stdin: x, gain, z (op 1), packed. stdout: error word,
 * dispatches, barriers (uint32), then y packed [t,groups,width]; every y float
 * outside the view keeps its 0xa5 bytes. Exit 0 whenever the encode succeeds.
 *   rmsbench <oracle|fast> op t groups width gain_rank reps
 * synthetic x, gain and z, 3 warm-ups + 20 measured command buffers of reps
 * encodes each (check and barriers included, as in the graph). */
static int rms_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"rmsbench"), fast=!strcmp(argv[2],"fast");
    assert(argc==(timing ? 9 : 12) && (fast || !strcmp(argv[2],"oracle") || (!timing && !strcmp(argv[2],"encode"))));
    size_t op=number(argv[3]), t=number(argv[4]), groups=number(argv[5]), width=number(argv[6]), gain_rank=number(argv[7]),
           reps=timing ? number(argv[8]) : 1, pitch=timing ? width : number(argv[8]), offset=timing ? 0 : number(argv[9]),
           inplace=timing ? 0 : number(argv[10]);
    uint32_t initial=timing ? 0 : (uint32_t)number(argv[11]);
    assert(op<=1 && t && t<=4096 && groups && width && groups*width<=1u<<20 && (gain_rank==1 || gain_rank==2) &&
           pitch>=width && offset<=64 && inplace<=1 && reps && reps<=1000);
    size_t rows=t*groups, words=4+offset+(rows-1)*pitch+width+4, gains=(gain_rank==1 ? 1 : groups)*width;
    float *data=malloc(4*(2*rows*width+gains)); assert(data);
    uint64_t seed=3931;
    if (timing) for (size_t i=0;i<rows*width*(1+op)+gains;i++) data[i]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
    else assert(fread(data,4,rows*width*(1+op)+gains,stdin)==rows*width*(1+op)+gains && fgetc(stdin)==EOF);
    double seconds[23];
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38CGpu *ops=NULL;
        assert(q38c_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ops,NULL)==Q38_OK);
        if (strcmp(argv[2],"encode")) q38c_gpu_set_reference(ops,!fast);
        id<MTLBuffer> buf[4]={shared(dev,4*words),shared(dev,4*(gains+8)),shared(dev,4*words),shared(dev,4*words)},
            word=shared(dev,4);
        for (size_t b=0;b<4;b++) for (size_t i=0;i<(b==1 ? gains+8 : words);i++) ((float *)buf[b].contents)[i]=NAN;
        memset(buf[2].contents,0xa5,4*words);
        const float *src[]={data,data+rows*width+gains};
        for (size_t r=0;r<rows;r++) for (size_t j=0;j<width;j++) {
            ((float *)buf[0].contents)[4+offset+r*pitch+j]=src[0][r*width+j];
            if (op) ((float *)buf[3].contents)[4+offset+r*pitch+j]=src[1][r*width+j];
        }
        memcpy((float *)buf[1].contents+4,data+rows*width,4*gains);
        Qwen38GpuF32 v[4];
        for (size_t b=0;b<4;b++) v[b]=(Qwen38GpuF32){{(__bridge void *)buf[b],16+4*offset,4*((rows-1)*pitch+width)},3,
            {t,groups,width,1},{groups*pitch*4,pitch*4,4,0}};
        v[1]=gain_rank==1 ? (Qwen38GpuF32){{(__bridge void *)buf[1],16,4*gains},1,{width,1,1,1},{4,0,0,0}} :
            (Qwen38GpuF32){{(__bridge void *)buf[1],16,4*gains},2,{groups,width,1,1},{width*4,4,0,0}};
        if (inplace) v[2]=v[0];
        Q38CGpuSpec spec={.eps=1e-6f,.gdn={.key_heads=1,.value_heads=groups,.key_dim=1,.value_dim=width,.rms_eps=1e-6f,.l2_eps=1e-6f}};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            ((uint32_t *)word.contents)[0]=initial;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            counter->target=[command computeCommandEncoder]; counter->dispatches=counter->barriers=0;
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)counter,{(__bridge void *)word,0,4},false};
            for (size_t rep=0;rep<reps;rep++)
                assert(q38c_gpu_encode(ops,cmd,op ? Q38C_GDN_NORM : Q38C_RMS,&spec,v,3+op,NULL)==Q38_OK);
            [counter->target endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || !((uint32_t *)word.contents)[0]);
        }
        const uint32_t *raw=buf[inplace ? 0 : 2].contents;
        for (size_t i=0;!inplace && i<words;i++) /* no write outside y */
            assert((i>=4+offset && (i-4-offset)%pitch<width && (i-4-offset)/pitch<rows) || raw[i]==0xa5a5a5a5u);
        if (timing) {
            Request r={.mode=argv[2],.n=width,.logical=width,.m=rows,.reps=reps,.bytes=4*(gains+op*rows*width)};
            char specs[96];
            snprintf(specs,sizeof specs,"op %zu tokens %zu groups %zu width %zu gain_rank %zu",op,t,groups,width,gain_rank);
            report(r,seconds,NULL,specs);
        } else {
            uint32_t head[3]={((uint32_t *)word.contents)[0],counter->dispatches,counter->barriers};
            assert(fwrite(head,4,3,stdout)==3);
            for (size_t r=0;r<rows;r++) assert(fwrite(raw+4+offset+r*pitch,4,width,stdout)==width);
        }
        q38c_gpu_destroy(ops);
    }
    free(data);
    return 0;
}

/* R1.8 GDN prepare through q38c_gpu_encode (op Q38C_GDN_PREP):
 *   prep <oracle|fast|encode> t hk hv dk dv gap offset inplace initial
 * qkv/prepared [t,2hk*dk+hv*dv] and alpha/br/log_g/beta [t,hv]: row pitch
 * width+gap floats; every view (also A, dt [hv]) starts `offset` floats past
 * a 16-byte guard, NaN in every input gap. inplace: prepared, log_g, beta over
 * qkv, alpha, br. stdin: qkv, alpha, br, A, dt packed. stdout: error word,
 * dispatches, barriers (uint32), then prepared, log_g, beta packed; every
 * output float outside its view keeps its 0xa5 bytes.
 *   prepbench <oracle|fast> t hk hv dk dv reps
 * synthetic inputs (A < 0), 3 warm-ups + 20 measured buffers of reps encodes. */
static int prep_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"prepbench"), fast=!strcmp(argv[2],"fast");
    assert(argc==(timing ? 9 : 12) && (fast || !strcmp(argv[2],"oracle") || (!timing && !strcmp(argv[2],"encode"))));
    size_t t=number(argv[3]), hk=number(argv[4]), hv=number(argv[5]), dk=number(argv[6]), dv=number(argv[7]),
           reps=timing ? number(argv[8]) : 1, gap=timing ? 0 : number(argv[8]), offset=timing ? 0 : number(argv[9]),
           inplace=timing ? 0 : number(argv[10]), c=2*hk*dk+hv*dv, total=0;
    uint32_t initial=timing ? 0 : (uint32_t)number(argv[11]);
    assert(t && t<=4096 && hk && hv && hv%hk==0 && dk && dv && c<=1u<<20 && gap<=64 && offset<=64 && inplace<=1 &&
           reps && reps<=1000);
    size_t width[8]={c,hv,hv,hv,hv,c,hv,hv}, rows[8]={t,t,t,1,1,t,t,t}, words[8], pitch[8];
    for (size_t b=0;b<5;b++) total+=rows[b]*width[b];
    float *data=malloc(4*total); assert(data);
    uint64_t seed=3967;
    if (timing) for (size_t i=0;i<total;i++) data[i]=(float)(mix(&seed)>>40)*0x1p-23f-1.0f;
    else assert(fread(data,4,total,stdin)==total && fgetc(stdin)==EOF);
    for (size_t h=0;timing && h<hv;h++) data[t*(c+2*hv)+h]=-fabsf(data[t*(c+2*hv)+h]); /* A <= 0 */
    double seconds[23];
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38CGpu *ops=NULL;
        assert(q38c_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ops,NULL)==Q38_OK);
        if (strcmp(argv[2],"encode")) q38c_gpu_set_reference(ops,!fast);
        id<MTLBuffer> buf[8], word=shared(dev,4); Qwen38GpuF32 v[8]; const float *src=data;
        for (size_t b=0;b<8;b++) {
            pitch[b]=width[b]+(b==3 || b==4 ? 0 : gap); words[b]=8+offset+(rows[b]-1)*pitch[b]+width[b];
            buf[b]=shared(dev,4*words[b]);
            float *f=buf[b].contents;
            if (b>=5) memset(f,0xa5,4*words[b]);
            else for (size_t i=0;i<words[b];i++) f[i]=NAN;
            for (size_t r=0;b<5 && r<rows[b];r++,src+=width[b]) memcpy(f+4+offset+r*pitch[b],src,4*width[b]);
            v[b]=(Qwen38GpuF32){{(__bridge void *)buf[b],16+4*offset,4*((rows[b]-1)*pitch[b]+width[b])},2,
                {rows[b],width[b],1,1},{pitch[b]*4,4,0,0}};
            if (b==3 || b==4) v[b]=(Qwen38GpuF32){v[b].span,1,{width[b],1,1,1},{4,0,0,0}};
        }
        for (size_t b=5;inplace && b<8;b++) { v[b]=v[b-5]; buf[b]=buf[b-5]; }
        Q38CGpuSpec spec={.gdn={.key_heads=hk,.value_heads=hv,.key_dim=dk,.value_dim=dv,.rms_eps=1e-6f,.l2_eps=1e-6f}};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            ((uint32_t *)word.contents)[0]=initial;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            counter->target=[command computeCommandEncoder]; counter->dispatches=counter->barriers=0;
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)counter,{(__bridge void *)word,0,4},false};
            for (size_t rep=0;rep<reps;rep++)
                assert(q38c_gpu_encode(ops,cmd,Q38C_GDN_PREP,&spec,v,8,NULL)==Q38_OK);
            [counter->target endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || !((uint32_t *)word.contents)[0]);
        }
        for (size_t b=5;!inplace && b<8;b++) for (size_t i=0;i<words[b];i++) { /* no write outside the outputs */
            size_t k=i-4-offset;
            assert((i>=4+offset && k/pitch[b]<rows[b] && k%pitch[b]<width[b]) || ((uint32_t *)buf[b].contents)[i]==0xa5a5a5a5u);
        }
        if (timing) {
            Request r={.mode=argv[2],.n=c,.logical=c,.m=t,.reps=reps,.bytes=4*(4*t*hv+2*hv)};
            char specs[96];
            snprintf(specs,sizeof specs,"tokens %zu hk %zu hv %zu dk %zu dv %zu",t,hk,hv,dk,dv);
            report(r,seconds,NULL,specs);
        } else {
            uint32_t head[3]={((uint32_t *)word.contents)[0],counter->dispatches,counter->barriers};
            assert(fwrite(head,4,3,stdout)==3);
            for (size_t b=5;b<8;b++) for (size_t r=0;r<t;r++)
                assert(fwrite((float *)buf[b].contents+4+offset+r*pitch[b],4,width[b],stdout)==width[b]);
        }
        q38c_gpu_destroy(ops);
    }
    free(data);
    return 0;
}

/* R2.1 split attention through q38d_attend_encode:
 *   attend <oracle|fast|encode> h j d n m
 * oracle/fast force the choice, encode follows Q38_REF_KERNELS. stdin: q, gate [h,d],
 * k, v [n,j,d] FP32, ids [m] uint32. stdout: error word, dispatches (uint32), out [h,d].
 *   attbench <oracle|fast> h j d n m: same shapes, seeded data, ids = every n/m-th key. */
static int attend_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"attbench"), fast=!strcmp(argv[2],"fast");
    assert(argc==8 && (fast || !strcmp(argv[2],"oracle") || (!timing && !strcmp(argv[2],"encode"))));
    uint32_t h=(uint32_t)number(argv[3]),j=(uint32_t)number(argv[4]),d=(uint32_t)number(argv[5]),
             n=(uint32_t)number(argv[6]),m=(uint32_t)number(argv[7]);
    assert(h && h<=128 && j && d && d<=256 && n && n<=32768 && m && m<=n);
    size_t sizes[6]={(size_t)h*d,(size_t)h*d,(size_t)n*j*d,(size_t)n*j*d,m,(size_t)h*d};
    double seconds[23];
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38DGpu *g=NULL;
        assert(q38d_gpu_create((__bridge void *)dev,(__bridge void *)lib,&g,NULL)==Q38_OK);
        if (strcmp(argv[2],"encode")) q38d_gpu_set_reference(g,!fast);
        id<MTLBuffer> buf[6], word=shared(dev,4), part=shared(dev,4*(size_t)Q38D_MAX_SPLITS*h*(d+2));
        Qwen38GpuSpan v[6];
        uint64_t seed=4099;
        for (size_t b=0;b<6;b++) {
            buf[b]=shared(dev,4*sizes[b]); v[b]=(Qwen38GpuSpan){(__bridge void *)buf[b],0,4*sizes[b]};
            float *f=buf[b].contents; uint32_t *u=buf[b].contents;
            if (b==5) memset(f,0xa5,4*sizes[b]);
            else if (!timing) assert(fread(f,4,sizes[b],stdin)==sizes[b]);
            else for (size_t i=0;i<sizes[b];i++)
                if (b==4) u[i]=(uint32_t)(i*(n/m)); else f[i]=b==1 ? 0.5f : (float)(mix(&seed)>>40)*0x1p-23f-1.0f;
        }
        assert(timing || fgetc(stdin)==EOF);
        Q38DGpuArgs a={.h=h,.j=j,.d=d,.n=n,.m=m};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            ((uint32_t *)word.contents)[0]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            counter->target=[command computeCommandEncoder]; counter->dispatches=counter->barriers=0;
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)counter,{(__bridge void *)word,0,4},false};
            assert(q38d_attend_encode(g,cmd,&a,v,(Qwen38GpuSpan){(__bridge void *)part,0,part.length},NULL)==Q38_OK);
            [counter->target endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
            assert(!timing || !((uint32_t *)word.contents)[0]);
        }
        if (timing) {
            Request r={.mode=argv[2],.n=h*d,.logical=d,.m=m,.reps=1,.bytes=8*(size_t)m*j*d};
            char specs[96];
            snprintf(specs,sizeof specs,"heads %u kv %u dim %u keys %u of %u",h,j,d,m,n);
            report(r,seconds,NULL,specs);
        } else {
            uint32_t head[2]={((uint32_t *)word.contents)[0],counter->dispatches};
            assert(fwrite(head,4,2,stdout)==2 && fwrite(buf[5].contents,4,sizes[5],stdout)==sizes[5]);
        }
        q38d_gpu_destroy(g);
    }
    return 0;
}

/* R2.4 GDN scan through q38c_gpu_encode (op Q38C_GDN_SCAN):
 *   scan <oracle|fast> t hk hv dk dv initial
 * stdin: qkv [t,2hk*dk+hv*dv], log-decay g [t,hv], beta [t,hv], state [hv,dv,dk] FP32.
 * stdout: error word, dispatches (uint32), out [t,hv,dv], state [hv,dv,dk].
 *   scanbench <oracle|fast> t hk hv dk dv reps: seeded inputs (g < 0), 3 + 20 buffers. */
static int scan_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"scanbench"), fast=!strcmp(argv[2],"fast");
    assert(argc==9 && (fast || !strcmp(argv[2],"oracle")));
    size_t t=number(argv[3]), hk=number(argv[4]), hv=number(argv[5]), dk=number(argv[6]), dv=number(argv[7]),
           reps=timing ? number(argv[8]) : 1, c=2*hk*dk+hv*dv;
    uint32_t initial=timing ? 0 : (uint32_t)number(argv[8]);
    assert(t && t<=1024 && hk && hv && hv%hk==0 && dk && dk<=256 && dv && reps && reps<=1000);
    size_t width[5]={c,hv,hv,dk,dv}, rows[5]={t,t,t,hv*dv,t*hv}, total=0;
    for (size_t b=0;b<4;b++) total+=rows[b]*width[b];
    float *data=malloc(4*total); assert(data);
    uint64_t seed=4157;
    if (timing) for (size_t i=0;i<total;i++) data[i]=((float)(mix(&seed)>>40)*0x1p-23f-1.0f)*0.1f;
    else assert(fread(data,4,total,stdin)==total && fgetc(stdin)==EOF);
    for (size_t i=0;timing && i<t*hv;i++) data[t*c+i]=-fabsf(data[t*c+i]); /* g <= 0 */
    double seconds[23];
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38CGpu *ops=NULL;
        assert(q38c_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ops,NULL)==Q38_OK);
        q38c_gpu_set_reference(ops,!fast);
        id<MTLBuffer> buf[5], word=shared(dev,8); Qwen38GpuF32 v[5]; const float *src=data;
        for (size_t b=0;b<5;b++) {
            buf[b]=shared(dev,4*rows[b]*width[b]);
            if (b<4) { memcpy(buf[b].contents,src,4*rows[b]*width[b]); src+=rows[b]*width[b]; }
            else memset(buf[b].contents,0xa5,buf[b].length);
        }
        Qwen38GpuSpan sp[5]; for (size_t b=0;b<5;b++) sp[b]=(Qwen38GpuSpan){(__bridge void *)buf[b],0,buf[b].length};
        v[0]=(Qwen38GpuF32){sp[0],2,{t,c,1,1},{c*4,4,0,0}};
        v[1]=(Qwen38GpuF32){sp[1],2,{t,hv,1,1},{hv*4,4,0,0}}; v[2]=(Qwen38GpuF32){sp[2],2,{t,hv,1,1},{hv*4,4,0,0}};
        v[3]=(Qwen38GpuF32){sp[3],3,{hv,dv,dk,1},{dv*dk*4,dk*4,4,0}};
        v[4]=(Qwen38GpuF32){sp[4],3,{t,hv,dv,1},{hv*dv*4,dv*4,4,0}};
        Q38CGpuSpec spec={.gdn={.key_heads=hk,.value_heads=hv,.key_dim=dk,.value_dim=dv,.rms_eps=1e-6f,.l2_eps=1e-6f}};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
        for (int run=0;run<(timing ? 23 : 1);run++) {
            ((uint32_t *)word.contents)[0]=initial; ((uint32_t *)word.contents)[1]=0;
            id<MTLCommandBuffer> command=[queue commandBuffer];
            counter->target=[command computeCommandEncoder]; counter->dispatches=counter->barriers=0;
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)counter,{(__bridge void *)word,0,4},false};
            for (size_t rep=0;rep<reps;rep++) assert(q38c_gpu_encode(ops,cmd,Q38C_GDN_SCAN,&spec,v,5,NULL)==Q38_OK);
            [counter->target endEncoding]; [command commit]; [command waitUntilCompleted];
            assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
            seconds[run]=command.GPUEndTime-command.GPUStartTime;
        }
        if (timing) {
            Request r={.mode=argv[2],.n=hv*dv,.logical=dk,.m=t,.reps=reps,.bytes=8*hv*dv*dk};
            char specs[96];
            snprintf(specs,sizeof specs,"tokens %zu hk %zu hv %zu dk %zu dv %zu",t,hk,hv,dk,dv);
            report(r,seconds,NULL,specs);
        } else {
            uint32_t head[2]={((uint32_t *)word.contents)[0],counter->dispatches};
            assert(fwrite(head,4,2,stdout)==2 && fwrite(buf[4].contents,4,t*hv*dv,stdout)==t*hv*dv &&
                   fwrite(buf[3].contents,4,hv*dv*dk,stdout)==hv*dv*dk);
        }
        q38c_gpu_destroy(ops);
    }
    free(data);
    return 0;
}

/* R2.2 exact index scores + device selection (q38d_gpu_encode SCORES, then q38d_select_encode):
 *   index <oracle|fast> h d visible ratio budget
 * stdin: iq [h,d], block keys [visible/ratio,d] FP32. stdout: error word, scores [n], ids
 * [min(n,budget/ratio)*ratio + visible%ratio] (uint32); n = visible/ratio. */
static int index_main(int argc,char **argv,const char *library) {
    assert(argc==8 && (!strcmp(argv[2],"fast") || !strcmp(argv[2],"oracle")));
    uint32_t h=(uint32_t)number(argv[3]),d=(uint32_t)number(argv[4]),visible=(uint32_t)number(argv[5]),
             ratio=(uint32_t)number(argv[6]),budget=(uint32_t)number(argv[7]),n=visible/ratio,
             sel=n<budget/ratio ? n : budget/ratio,count=sel*ratio+visible%ratio;
    assert(h && d && ratio && budget%ratio==0 && n && visible<=32768);
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38DGpu *g=NULL;
        assert(q38d_gpu_create((__bridge void *)dev,(__bridge void *)lib,&g,NULL)==Q38_OK);
        q38d_gpu_set_reference(g,!strcmp(argv[2],"oracle"));
        id<MTLBuffer> q=shared(dev,4*(size_t)h*d), k=shared(dev,4*(size_t)n*d), sc=shared(dev,4*(size_t)n),
                      keep=shared(dev,4*(size_t)n), ids=shared(dev,4*(size_t)count), word=shared(dev,4);
        assert(fread(q.contents,4,(size_t)h*d,stdin)==(size_t)h*d && fread(k.contents,4,(size_t)n*d,stdin)==(size_t)n*d);
        assert(fgetc(stdin)==EOF);
        memset(sc.contents,0xa5,sc.length); memset(ids.contents,0xa5,ids.length); ((uint32_t *)word.contents)[0]=0;
        id<MTLCommandBuffer> command=[[dev newCommandQueue] commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)enc,{(__bridge void *)word,0,4},false};
        Q38DGpuArgs a={.h=h,.d=d,.n=n,.m=sel,.r=ratio};
        Qwen38GpuSpan v[3]={{(__bridge void *)q,0,q.length},{(__bridge void *)k,0,k.length},{(__bridge void *)sc,0,sc.length}};
        assert(q38d_gpu_encode(g,cmd,Q38D_SCORES,&a,v,3,NULL)==Q38_OK);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        assert(q38d_select_encode(g,cmd,visible,ratio,budget,v[2],(Qwen38GpuSpan){(__bridge void *)keep,0,keep.length},
                                  (Qwen38GpuSpan){(__bridge void *)ids,0,ids.length},NULL)==Q38_OK);
        [enc endEncoding]; [command commit]; [command waitUntilCompleted];
        assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
        assert(fwrite(word.contents,4,1,stdout)==1 && fwrite(sc.contents,4,n,stdout)==n &&
               fwrite(ids.contents,4,count,stdout)==count);
        q38d_gpu_destroy(g);
    }
    return 0;
}

/* R2.6 fused HC read tail against the fast sequence (HC_LOW, F16 up, HC_READ, HC_INJECT):
 *   hctail t K h E inject poison: seeded F16 up [h*E, K], low [t,K], normed [t,h*E], il [t,h];
 *     poison 1 puts a NaN in low[t-1, K-1]. Runs both on the same inputs, prints one line:
 *     "status S equal B words W1 W2 dispatches D1 D2" (B: act, gate, mixed, inject byte-equal).
 *   hcbench <fused|sequence> t K h E reps: 3 + 20 command buffers, reps encodes each. */
static int hctail_main(int argc,char **argv,const char *library) {
    bool timing=!strcmp(argv[1],"hcbench");
    assert(argc==8);
    size_t at=2+timing, t=number(argv[at]), k=number(argv[at+1]), h=number(argv[at+2]), width=number(argv[at+3]),
           flag=number(argv[at+4]), reps=timing ? flag : 1;
    bool inj=timing || flag, poison=!timing && number(argv[7]);
    assert(t && t<=64 && k && k<=4096 && h && h<=8 && width && width<=8192 && reps && reps<=1000);
    Request r={.mode="fast",.type=1,.physical=k,.logical=k,.n=h*width,.m=t,.bytes=2*k*h*width};
    unsigned char *raw=malloc(r.bytes); assert(raw);
    uint64_t seed=5101;
    synthesize(raw,r.bytes,&seed,1);
    /* 0 low, 1 normed, 2 il; 3 act, 4 gate, 5 mixed, 6 inject (x2: fused, sequence) */
    const size_t cols[7]={k,h*width,h,k,h*width,width,h};
    Qwen38Status status=Q38_OK;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        NSError *error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(library)] error:&error]; assert(lib);
        Q38PackedGpu *ctx=NULL; Q38CGpu *ops=NULL;
        assert(q38_packed_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ctx,NULL)==Q38_OK);
        assert(q38c_gpu_create((__bridge void *)dev,(__bridge void *)lib,&ops,NULL)==Q38_OK);
        Gguf g={0}; GgufTensor tensor;
        GpuWeightStore *store=make_store(dev,1,&r,raw,&g,&tensor);
        const GgufBinding *binding=&gpu_weights_plan(store)->weights[0];
        Q38PackedLayout layout; assert(q38_packed_layout_init(&binding->tensor,k,&layout,NULL)==Q38_OK);
        id<MTLBuffer> buf[2][7]; Qwen38GpuF32 v[2][7];
        for (size_t s=0;s<2;s++) for (size_t b=0;b<7;b++) {
            buf[s][b]=s && b<3 ? buf[0][b] : shared(dev,4*t*cols[b]);
            if (b>=3) memset(buf[s][b].contents,0xa5,buf[s][b].length);
            v[s][b]=(Qwen38GpuF32){{(__bridge void *)buf[s][b],0,4*t*cols[b]},2,{t,cols[b],1,1},{4*cols[b],4,0,0}};
        }
        for (size_t b=0;b<3;b++) for (size_t i=0;i<t*cols[b];i++) /* low, il in [-4,4), normed in [-1,1) */
            ((float *)buf[0][b].contents)[i]=((float)(mix(&seed)>>40)*0x1p-23f-1.0f)*(b==1 ? 1.0f : 4.0f);
        if (poison) ((float *)buf[0][0].contents)[t*k-1]=NAN;
        Q38CGpuSpec spec={.hc={.streams=h,.hidden_size=width,.low_rank=k,.rms_eps=1e-6f}};
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        id<MTLBuffer> word=shared(dev,8);
        uint32_t words[2]={0,0}, dispatches[2]={0,0};
        double seconds[23];
        for (size_t s=0;s<2;s++) {
            if (timing && strcmp(argv[2],s ? "sequence" : "fused")) continue;
            Qwen38GpuF32 *x=v[s], none={0}, il=inj ? x[2] : none, inject=inj ? x[6] : none;
            for (int run=0;run<(timing ? 23 : 1);run++) {
                memset(word.contents,0,8);
                id<MTLCommandBuffer> command=[queue commandBuffer];
                Q38DispatchCounter *counter=[Q38DispatchCounter alloc];
                counter->target=[command computeCommandEncoder]; counter->dispatches=counter->barriers=0;
                Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)counter,{(__bridge void *)word,0,4},false};
                for (size_t rep=0;rep<reps;rep++) {
                    Qwen38Error err;
                    if (!s) {
                        status=q38_packed_encode_hc_tail(ctx,&cmd,store,binding,&layout,h,x[0],x[1],il,x[3],x[4],x[5],inject,&err);
                        if (status) { fprintf(stderr,"hc tail: %d %s\n",status,err.detail); break; }
                        [counter memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        continue;
                    }
                    Qwen38GpuF32 n3=x[1], g3=x[4];
                    n3.rank=g3.rank=3; n3.dim[1]=g3.dim[1]=h; n3.dim[2]=g3.dim[2]=width;
                    n3.stride_bytes[1]=g3.stride_bytes[1]=width*4; n3.stride_bytes[2]=g3.stride_bytes[2]=4;
                    assert(q38c_gpu_encode(ops,cmd,Q38C_HC_LOW,&spec,(Qwen38GpuF32[]){x[0],x[3]},2,NULL)==Q38_OK);
                    assert(q38_packed_encode_fast(ctx,&cmd,store,binding,&layout,0,x[3],x[4],NULL)==Q38_OK);
                    [counter memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    assert(q38c_gpu_encode(ops,cmd,Q38C_HC_READ,&spec,(Qwen38GpuF32[]){n3,g3,x[5]},3,NULL)==Q38_OK);
                    if (inj) assert(q38c_gpu_encode(ops,cmd,Q38C_HC_INJECT,&spec,(Qwen38GpuF32[]){x[2],x[6]},2,NULL)==Q38_OK);
                }
                [counter->target endEncoding]; [command commit]; [command waitUntilCompleted];
                assert(command.status==MTLCommandBufferStatusCompleted && !command.error);
                seconds[run]=command.GPUEndTime-command.GPUStartTime;
                words[s]=((uint32_t *)word.contents)[0]; dispatches[s]=counter->dispatches;
            }
            if (timing) {
                Request q={.mode=argv[2],.n=h*width,.logical=k,.m=t,.reps=reps,.bytes=r.bytes};
                char specs[96];
                snprintf(specs,sizeof specs,"hc tail tokens %zu K %zu h %zu E %zu",t,k,h,width);
                report(q,seconds,NULL,specs);
            }
        }
        if (!timing) {
            bool equal=true;
            for (size_t b=3;b<7;b++) if (inj || b<6) equal=equal && !memcmp(buf[0][b].contents,buf[1][b].contents,buf[0][b].length);
            printf("status %d equal %d words %u %u dispatches %u %u\n",status,equal,words[0],words[1],dispatches[0],dispatches[1]);
        }
        q38c_gpu_destroy(ops);
        q38_packed_gpu_free(ctx);
        gpu_weights_close(store); assert(close(g.fd)==0);
    }
    free(raw);
    return 0;
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"selector")) return selector_cases();
    const char *lock=getenv("PICOFORGE_GPU_LOCK_FD"), *library=getenv("Q38_PACKED_LIBRARY");
    if (!lock || fcntl((int)number(lock),F_GETFD)<0 || !library) {
        fprintf(stderr,"run through tools/qwen38_gpu_lock.py with Q38_PACKED_LIBRARY set\n"); return 2;
    }
    if (argc>1 && (!strcmp(argv[1],"multi") || !strcmp(argv[1],"mbench"))) return multi_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"indexed") || !strcmp(argv[1],"ibench") || !strcmp(argv[1],"gateup") ||
                   !strcmp(argv[1],"gbench"))) return indexed_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"downreduce") || !strcmp(argv[1],"dbench") || !strcmp(argv[1],"rbench")))
        return down_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"route") || !strcmp(argv[1],"rtbench"))) return route_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"rms") || !strcmp(argv[1],"rmsbench"))) return rms_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"prep") || !strcmp(argv[1],"prepbench"))) return prep_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"attend") || !strcmp(argv[1],"attbench"))) return attend_main(argc,argv,library);
    if (argc>2 && !strcmp(argv[1],"index")) return index_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"scan") || !strcmp(argv[1],"scanbench"))) return scan_main(argc,argv,library);
    if (argc>2 && (!strcmp(argv[1],"hctail") || !strcmp(argv[1],"hcbench"))) return hctail_main(argc,argv,library);
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
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)encoder,{(__bridge void *)word,0,4},false};
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
