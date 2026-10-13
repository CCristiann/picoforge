/* Bounded QSA attend benchmark, fixed FP32 inputs; run under the common lock. */
#import <Metal/Metal.h>
#include "qwen38_d_internal.h"
#include <assert.h>
#include <math.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
int main(int argc,char **argv) { @autoreleasepool {
    assert(argc==6); /* metallib, fresh output prefix, selected count, repetitions, scalar|cooperative */
    uint32_t n=(uint32_t)strtoul(argv[3],NULL,10),reps=(uint32_t)strtoul(argv[4],NULL,10);
    assert(n && n<=8192 && reps && reps<=1000);
    bool coop=!strcmp(argv[5],"cooperative");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); assert(device);
    id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:NULL]; assert(lib);
    NSString *name=coop ? @"q38d_attend_gpu" : @"q38d_attend_scalar_gpu";
    id<MTLFunction> function=[lib newFunctionWithName:name];
    if (!function && !coop) function=[lib newFunctionWithName:@"q38d_attend_gpu"];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:NULL]; assert(pipeline);
    Q38DGpuArgs a={.h=24,.j=2,.d=256,.n=n,.m=MIN(n,2048u)};
    size_t sizes[]={24*256*4,24*256*4,(size_t)n*2*256*4,(size_t)n*2*256*4,(size_t)a.m*4,24*256*4,4};
    NSMutableArray<id<MTLBuffer>> *b=[NSMutableArray array];
    for (size_t j=0;j<7;j++) {
        id<MTLBuffer> buffer=[device newBufferWithLength:sizes[j] options:MTLResourceStorageModeShared]; assert(buffer); [b addObject:buffer];
        for(size_t k=0;k<sizes[j]/4;k++) ((float *)buffer.contents)[k]=j==1 ? .5f : (float)((int)((k*17+j*13)%101)-50)*.01f;
    }
    for(uint32_t i=0;i<a.m;i++) ((uint32_t *)b[4].contents)[i]=i;
    *(uint32_t *)b[6].contents=0;
    char path[1024]; snprintf(path,sizeof path,"%s.jsonl",argv[2]); FILE *out=fopen(path,"wx"); assert(out);
    id<MTLCommandQueue> queue=[device newCommandQueue]; double warmed=0; uint32_t measured=0;
    while (warmed<.5 || measured<reps) { @autoreleasepool {
        bool warm=warmed<.5; double start=now();
        id<MTLCommandBuffer> cmd=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<6;i++) [enc setBuffer:b[i] offset:0 atIndex:i];
        [enc setBytes:&a length:sizeof a atIndex:6]; [enc setBuffer:b[6] offset:0 atIndex:7];
        if(coop) [enc dispatchThreadgroups:MTLSizeMake(a.h,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        else [enc dispatchThreads:MTLSizeMake(a.h,1,1) threadsPerThreadgroup:MTLSizeMake(a.h,1,1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        assert(cmd.status==MTLCommandBufferStatusCompleted && !*(uint32_t *)b[6].contents);
        double gpu=cmd.GPUEndTime-cmd.GPUStartTime,wall=now()-start;
        fprintf(out,"{\"n\":%u,\"selected\":%u,\"mode\":\"%s\",\"warmup\":%s,\"gpu_s\":%.9f,\"wall_s\":%.9f}\n",n,a.m,argv[5],warm?"true":"false",gpu,wall);
        if(warm) warmed+=gpu; else measured++;
    }}
    fclose(out); snprintf(path,sizeof path,"%s.f32",argv[2]); out=fopen(path,"wbx"); assert(out);
    assert(fwrite(b[5].contents,1,sizes[5],out)==sizes[5]); fclose(out);
} return 0; }
