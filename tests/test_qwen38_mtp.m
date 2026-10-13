/* Teacher-forced MTP probe plus snapshot/state regressions. No reference math
 * is shared with the independent CPU oracle. Run under the common GPU lock. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_g_internal.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static void check(Qwen38Status rc,Qwen38Error *e) { if (rc) { fprintf(stderr,"MTP status %d: %s\n",rc,e->detail); exit(1); } }
static Qwen38F32Mut view(float *p,size_t n,size_t v) {
    return (Qwen38F32Mut){.data=p,.capacity_bytes=n*v*4,.rank=2,.dim={n,v},.stride_bytes={v*4,4}};
}
static NSData *histories(Qwen38GpuForward *g,uint32_t layers) {
    NSMutableData *out=[NSMutableData new];
    for (uint32_t i=0;i<layers;i++) for (uint32_t j=0;j<Q38GS_COUNT;j++) {
        Qwen38GpuF32 v=q38g_gpu_state(g,i,j);
        if (v.span.buffer) [out appendBytes:(char *)[(__bridge id<MTLBuffer>)v.span.buffer contents]+v.span.offset_bytes length:v.span.bytes];
    }
    return out;
}
typedef struct { Qwen38GpuForward *g; Qwen38GpuCheckpoint s; bool called; } Busy;
static void busy_check(void *user,uint32_t l,Qwen38GraphStage stage,Qwen38F32 v) {
    (void)l; (void)stage; (void)v; Busy *b=user; Qwen38Error e;
    assert(qwen38_gpu_checkpoint_restore(b->g,b->s,&e)==Q38_BUSY); b->called=true;
}
static void state_test(Qwen38GpuForward *g,const int32_t *tokens,uint32_t layers,size_t v) {
    Qwen38Error e; float *a=malloc(v*8),*b=malloc(v*8); assert(a && b);
    Qwen38F32Mut none=view(NULL,0,v); Qwen38GpuCheckpoint s,old;
    check(qwen38_gpu_forward_reset(g,&e),&e);
    check(qwen38_gpu_forward(g,tokens,3,layers,none,NULL,NULL,&e),&e);
    check(qwen38_gpu_mtp(g,0,tokens+1,2,none,&e),&e);
    check(qwen38_gpu_mtp(g,2,tokens+3,1,view(a,1,v),&e),&e);
    check(qwen38_gpu_checkpoint_save(g,&s,&e),&e); old=s;
    NSData *before=histories(g,layers+1); Busy busy={g,s,false};
    check(qwen38_gpu_forward(g,tokens+3,2,layers,view(a,2,v),busy_check,&busy,&e),&e); assert(busy.called);
    Qwen38GpuCheckpoint foreign=s; foreign.owner=tokens;
    assert(qwen38_gpu_checkpoint_restore(g,foreign,&e)==Q38_OWNER);
    Qwen38GpuCheckpoint forged=s; forged.position++;
    assert(qwen38_gpu_checkpoint_restore(g,forged,&e)==Q38_STALE);
    check(qwen38_gpu_checkpoint_restore(g,s,&e),&e);
    assert([before isEqualToData:histories(g,layers+1)]);
    assert(qwen38_gpu_checkpoint_restore(g,old,&e)==Q38_STALE);
    check(qwen38_gpu_forward(g,tokens+3,1,layers,view(b,1,v),NULL,NULL,&e),&e);
    assert(!memcmp(a,b,v*4));
    check(qwen38_gpu_forward(g,tokens+4,1,layers,view(b,1,v),NULL,NULL,&e),&e);
    assert(!memcmp(a+v,b,v*4));
    /* Recompute the whole accepted prefix: catches PLE hash/history rollback. */
    check(qwen38_gpu_forward_reset(g,&e),&e);
    for (size_t i=0;i<5;i++) check(qwen38_gpu_forward(g,tokens+i,1,layers,view(b,1,v),NULL,NULL,&e),&e);
    assert(!memcmp(a+v,b,v*4));
    check(qwen38_gpu_checkpoint_save(g,&s,&e),&e);
    check(qwen38_gpu_forward(g,tokens,3,layers,none,NULL,NULL,&e),&e);
    assert(qwen38_gpu_checkpoint_restore(g,s,&e)==Q38_BAD_STATE);
    check(qwen38_gpu_forward_reset(g,&e),&e);
    assert(qwen38_gpu_checkpoint_restore(g,s,&e)==Q38_STALE);
    check(qwen38_gpu_forward(g,tokens,3,layers,none,NULL,NULL,&e),&e);
    check(qwen38_gpu_mtp(g,0,tokens+1,1,none,&e),&e);
    check(qwen38_gpu_checkpoint_save(g,&s,&e),&e);
    check(qwen38_gpu_mtp(g,1,tokens+2,1,none,&e),&e);
    assert(qwen38_gpu_checkpoint_restore(g,s,&e)==Q38_BAD_STATE);
    assert(qwen38_gpu_mtp(g,1,tokens+2,1,none,&e)==Q38_BAD_STATE);
    /* A late predictor fault poisons both graphs, withholding host output. */
    Qwen38GpuF32 poison=q38g_gpu_state(g,layers,Q38GS_VALUE); assert(poison.span.buffer);
    *(float *)((char *)[(__bridge id<MTLBuffer>)poison.span.buffer contents]+poison.span.offset_bytes)=NAN;
    for (size_t j=0;j<v;j++) b[j]=123;
    assert(qwen38_gpu_mtp(g,2,tokens+3,1,view(b,1,v),&e)==Q38_GPU_ERROR);
    assert(!qwen38_gpu_forward_info(g).valid && !qwen38_gpu_mtp_info(g).valid);
    for (size_t j=0;j<v;j++) assert(b[j]==123);
    assert(qwen38_gpu_checkpoint_restore(g,s,&e)==Q38_BAD_STATE);
    check(qwen38_gpu_forward_reset(g,&e),&e);
    assert(qwen38_gpu_mtp_info(g).valid && !qwen38_gpu_mtp_info(g).next_pos);
    free(a); free(b); puts("snapshot: exact state/logits, PLE replay, busy/foreign/stale/window/predictor/fault PASS");
}
int main(int argc,char **argv) {
    if (argc<7) { fprintf(stderr,"usage: %s GGUF OUTDIR CHUNK CHECK TOKEN...\n",argv[0]); return 2; }
    uint32_t chunk=(uint32_t)strtoul(argv[3],NULL,10); bool checks=atoi(argv[4])!=0;
    size_t n=(size_t)argc-5; assert(chunk>=3 && n>=2 && n<=256);
    int32_t tokens[256]; for (size_t j=0;j<n;j++) tokens[j]=(int32_t)strtol(argv[5+j],NULL,10);
    @autoreleasepool {
        Gguf gguf; gguf_open(argv[1],&gguf); Qwen38Config *c; Qwen38Weights *w; Qwen38Error e;
        check(qwen38_config_load(&gguf,&c,&e),&e); check(qwen38_weights_describe(&gguf,c,&w,&e),&e);
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"picoforge.metallib"] error:NULL]; assert(lib);
        GpuWeightStore *store=gpu_weights_open((__bridge void *)dev,&gguf,0);
        check(qwen38_weights_attach(w,&gguf,gpu_weights_plan(store),&e),&e); Qwen38GpuForward *g;
        const Qwen38ConfigView *cv=qwen38_config_view(c); size_t v=cv->vocab_size;
        check(qwen38_gpu_forward_create(&gguf,c,w,store,(__bridge void *)dev,(__bridge void *)lib,chunk,MAX((uint32_t)n+4,chunk),chunk,&g,&e),&e);
        check(qwen38_gpu_mtp_enable(g,&e),&e);
        float *out=malloc(n*v*4),*again=malloc(n*v*4),*main=malloc(n*v*4); assert(out && again && main);
        for (int pass=0;pass<(checks ? 2 : 1);pass++) {
            check(qwen38_gpu_forward_reset(g,&e),&e); size_t take=pass ? 1 : chunk;
            for (size_t start=0;start<n;start+=take) {
                size_t t=MIN(take,n-start),pairs=MIN(t,n-start-1);
                check(qwen38_gpu_forward(g,tokens+start,t,cv->trunk_blocks,view(main+start*v,t,v),NULL,NULL,&e),&e);
                if (pairs) check(qwen38_gpu_mtp(g,0,tokens+start+1,pairs,view((pass ? again : out)+start*v,pairs,v),&e),&e);
            }
            assert(qwen38_gpu_mtp_info(g).next_pos==n-1);
        }
        if (checks) { assert(!memcmp(out,again,(n-1)*v*4)); assert(n>=6); state_test(g,tokens,cv->trunk_blocks,v); }
        char path[1024]; snprintf(path,sizeof path,"%s/mtp.f32",argv[2]); FILE *f=fopen(path,"wb"); assert(f);
        assert(fwrite(out,4,(n-1)*v,f)==(n-1)*v && !fclose(f));
        snprintf(path,sizeof path,"%s/logits.f32",argv[2]); f=fopen(path,"wb"); assert(f);
        assert(fwrite(main,4,n*v,f)==n*v && !fclose(f));
        printf("{\"tokens\":%zu,\"mtp_rows\":%zu,\"chunk\":%u,\"replay\":%s,\"graph_bytes\":%zu,\"checkpoint_host_bytes\":%zu,\"metal_bytes\":%zu}\n",n,n-1,chunk,checks ? "true":"null",qwen38_gpu_forward_bytes(g),qwen38_gpu_checkpoint_bytes(g),(size_t)dev.currentAllocatedSize);
        free(out); free(again); free(main); qwen38_gpu_forward_destroy(g); qwen38_weights_free(w); gpu_weights_close(store); qwen38_config_free(c); gguf_close(&gguf);
    }
    return 0;
}
