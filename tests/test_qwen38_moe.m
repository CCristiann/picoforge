/* Private GPU composition harness: same bounded transport and CPU oracle.
 * Compile this ObjC harness separately; never add it to the engine entrypoint. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_f_internal.h"
#include <errno.h>
#include <fcntl.h>
static Qwen38Status metal_forward(const Qwen38MoeConfig *,const Qwen38MoeWeights *,
    const Qwen38Moe *,Qwen38F32,Qwen38F32Mut,Qwen38MoeRoutes *,Qwen38MoeWork *,Qwen38Error *);
#define Q38_F_FORWARD metal_forward
#define main cpu_harness_main
#include "test_qwen38_moe.c"
#undef main

static NSMutableArray *captures;
static id<MTLComputePipelineState> copy_pipeline;
static size_t capture_token,capture_bytes;
static id<MTLBuffer> capture(Qwen38GpuCommand cmd,Qwen38GpuF32 v) {
    id<MTLDevice> dev=(__bridge id<MTLDevice>)cmd.device;
    size_t bytes=v.dim[0]*v.dim[1]*4; capture_bytes+=bytes;
    assert(capture_bytes<=32*1024*1024);
    id<MTLBuffer> out=[dev newBufferWithLength:MAX(bytes,(size_t)4) options:MTLResourceStorageModeShared];
    assert(out);
    id<MTLComputeCommandEncoder> enc=(__bridge id<MTLComputeCommandEncoder>)cmd.compute_encoder;
    uint32_t shape[]={(uint32_t)v.dim[1],(uint32_t)(v.stride_bytes[0]/4)};
    [enc setComputePipelineState:copy_pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)v.span.buffer offset:v.span.offset_bytes atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1]; [enc setBytes:shape length:sizeof shape atIndex:2];
    [enc dispatchThreads:MTLSizeMake(bytes/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return out;
}
static void capture_stage(void *ctx,Qwen38GpuCommand cmd,size_t stage,size_t ex,Qwen38GpuF32 x,Qwen38GpuF32 y) {
    (void)ctx;
    id<MTLBuffer> a=capture(cmd,x),b=capture(cmd,y);
    [captures addObject:@[@(stage),@(ex),@(capture_token),@(x.dim[0]),@(x.dim[1]),@(y.dim[1]),a,b]];
}
static void router_trace(void *ctx,size_t stage,size_t ex,size_t token,Qwen38F32 in,Qwen38F32 out) {
    if (stage==0) emit_trace(ctx,stage,ex,token,in,out);
}
static Qwen38GpuF32 gpu_view(id<MTLBuffer> b,size_t n,size_t h,size_t pitch) {
    return (Qwen38GpuF32){.span={(__bridge void *)b,16,b.length-32},.rank=2,
        .dim={n,h,1,1},.stride_bytes={pitch*4,4,0,0}};
}
static Qwen38Status metal_forward(const Qwen38MoeConfig *c,const Qwen38MoeWeights *weights,
    const Qwen38Moe *m,Qwen38F32 x,Qwen38F32Mut y,Qwen38MoeRoutes *routes,Qwen38MoeWork *work,Qwen38Error *err) {
    assert(getenv("PICOFORGE_GPU_LOCK_FD") && getenv("MTL_DEBUG_LAYER"));
    q38_moe_observe(work,router_trace,NULL);
    Qwen38Status st=qwen38_moe_forward(m,x,y,routes,work,err);
    if (st || !x.dim[0]) return st;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); assert(dev);
        const char *path=getenv("Q38_F_LIBRARY"); assert(path);
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:nil]; assert(lib);
        copy_pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q38f_trace_copy"] error:nil];
        assert(copy_pipeline);
        Q38PackedCpuView v[]={weights->router,weights->gate,weights->up,weights->down,
            weights->shared_gate,weights->shared_up,weights->shared_down,weights->shared_router};
        Gguf g={0}; GgufTensor tensors[8]; char file[]="build/qwen38-f/gpu-XXXXXX";
        g.fd=mkstemp(file); assert(g.fd>=0 && unlink(file)==0);
        size_t page=(size_t)sysconf(_SC_PAGESIZE),end=page,total=0;
        for (size_t j=0;j<8;j++) {
            Q38PackedLayout l=v[j].layout; total+=l.tensor_bytes; assert(total<=64*1024*1024);
            tensors[j]=(GgufTensor){.type=l.gguf_type,.rank=3,.shape={l.k_storage,l.n_out,l.n_experts,1},
                .file_offset=end,.bytes=l.tensor_bytes};
            snprintf(tensors[j].name,sizeof tensors[j].name,"w%zu",j);
            size_t done=0;
            while (done<l.tensor_bytes) {
                ssize_t wrote=pwrite(g.fd,v[j].span.data+done,l.tensor_bytes-done,(off_t)(end+done));
                if (wrote<0 && errno==EINTR) continue;
                assert(wrote>0); done+=(size_t)wrote;
            }
            end=(end+l.tensor_bytes+page-1)/page*page+page;
        }
        assert(ftruncate(g.fd,(off_t)end)==0); g.file_size=end; g.tensors=tensors; g.tensor_count=8;
        GpuWeightStore *store=gpu_weights_open((__bridge void *)dev,&g,0); assert(store);
        const GgufPlan *plan=gpu_weights_plan(store); const GgufBinding *bindings[8];
        for (size_t j=0;j<8;j++) bindings[j]=&plan->weights[j];
        Q38FMoeGpu *gpu=NULL;
        assert(q38f_gpu_create((__bridge void *)dev,(__bridge void *)lib,c,store,bindings,32,1,&gpu,err)==Q38_CAPACITY && !gpu);
        assert(q38f_gpu_create((__bridge void *)dev,(__bridge void *)lib,c,store,bindings,32,32*1024*1024,&gpu,err)==Q38_OK);
        assert(q38f_gpu_bytes(gpu)<=32*1024*1024);
        id<MTLCommandQueue> queue=[dev newCommandQueue]; assert(queue);
        size_t h=c->hidden_size,pitch=h+3;
        for (size_t start=0;start<x.dim[0];start+=32) {
            size_t n=MIN((size_t)32,x.dim[0]-start),bytes=n*pitch*4+32;
            id<MTLBuffer> xb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> yb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> word=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared]; assert(xb && yb && word);
            for (size_t j=0;j<bytes/4;j++) ((float *)xb.contents)[j]=((float *)yb.contents)[j]=-99;
            for (size_t row=0;row<n;row++) memcpy((float *)xb.contents+4+row*pitch,
                x.data+(start+row)*(x.stride_bytes[0]/4),h*4);
            if (getenv("Q38_F_GPU_NONFINITE")) ((float *)xb.contents)[4]=NAN;
            *(uint32_t *)word.contents=0;
            id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            Qwen38GpuCommand cmd={(__bridge void *)dev,(__bridge void *)enc,{(__bridge void *)word,0,4},false};
            Qwen38MoeRoutes r=*routes;
            r.ids.data+=start*(r.ids.row_stride_bytes/4); r.ids.capacity_bytes-=start*r.ids.row_stride_bytes; r.ids.rows=n;
            r.weights.data+=start*(r.weights.stride_bytes[0]/4); r.weights.capacity_bytes-=start*r.weights.stride_bytes[0]; r.weights.dim[0]=n;
            Qwen38GpuF32 gx=gpu_view(xb,n,h,pitch),gy=gpu_view(yb,n,h,pitch);
            assert(q38f_gpu_encode(gpu,cmd,gx,gx,r,err)==Q38_INVALID_ARGUMENT);
            Qwen38GpuF32 short_y=gy; short_y.span.bytes=1;
            assert(q38f_gpu_encode(gpu,cmd,gx,short_y,r,err)==Q38_CAPACITY);
            Qwen38GpuCommand wrong=cmd; wrong.device=NULL;
            assert(q38f_gpu_encode(gpu,wrong,gx,gy,r,err)==Q38_OWNER);
            wrong=cmd; wrong.error_word=gx.span;
            assert(q38f_gpu_encode(gpu,wrong,gx,gy,r,err)==Q38_INVALID_ARGUMENT);
            Qwen38GpuF32 empty_x=gx,empty_y=gy; empty_x.dim[0]=empty_y.dim[0]=0;
            Qwen38MoeRoutes empty_r=r; empty_r.ids.rows=empty_r.weights.dim[0]=0;
            assert(q38f_gpu_encode(gpu,cmd,empty_x,empty_y,empty_r,err)==Q38_OK);
            float saved_weight=r.weights.data[0]; r.weights.data[0]=-1;
            assert(q38f_gpu_encode(gpu,cmd,gx,gy,r,err)==Q38_RANGE);
            r.weights.data[0]=saved_weight;
            if (c->top_k>1) {
                uint32_t saved_id=r.ids.data[1]; r.ids.data[1]=r.ids.data[0];
                assert(q38f_gpu_encode(gpu,cmd,gx,gy,r,err)==Q38_INVALID_ARGUMENT);
                r.ids.data[1]=saved_id;
            }
            captures=[NSMutableArray array]; capture_token=start; capture_bytes=0;
            q38f_gpu_observe(gpu,capture_stage,NULL);
            st=q38f_gpu_encode(gpu,cmd,gx,gy,r,err); assert(st==Q38_OK);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; /* once per microbatch */
            assert(cb.status==MTLCommandBufferStatusCompleted && !cb.error);
            if (getenv("Q38_F_ALLOCATIONS")) {
                FILE *log=fopen(getenv("Q38_F_ALLOCATIONS"),"a"); assert(log);
                fprintf(log,"{\"tokens\":%zu,\"scratch_bytes\":%zu,\"trace_bytes\":%zu,\"metal_allocated_bytes\":%llu}\n",
                    n,q38f_gpu_bytes(gpu),capture_bytes,(unsigned long long)dev.currentAllocatedSize);
                assert(fclose(log)==0);
            }
            if (getenv("Q38_F_GPU_NONFINITE")) {
                assert(*(uint32_t *)word.contents & Q38_GPU_NONFINITE);
                st=Q38_NONFINITE; memset(err,0,sizeof *err); err->code=st; captures=nil; break;
            }
            assert(*(uint32_t *)word.contents==0);
            for (NSArray *record in captures) {
                uint64_t hdr[6]; for (size_t j=0;j<6;j++) hdr[j]=[record[j] unsignedLongLongValue];
                assert(fwrite(hdr,8,6,stdout)==6);
                id<MTLBuffer> a=record[6],b=record[7];
                assert(fwrite(a.contents,1,a.length,stdout)==a.length);
                assert(fwrite(b.contents,1,b.length,stdout)==b.length);
            }
            for (size_t row=0;row<n;row++) {
                memcpy(y.data+(start+row)*(y.stride_bytes[0]/4),(float *)yb.contents+4+row*pitch,h*4);
                for (size_t col=h;col<pitch;col++) assert(((float *)yb.contents)[4+row*pitch+col]==-99);
            }
            for (size_t j=0;j<4;j++) assert(((float *)yb.contents)[j]==-99 && ((float *)yb.contents)[bytes/4-1-j]==-99);
            captures=nil;
        }
        q38f_gpu_destroy(gpu); gpu_weights_close(store); close(g.fd); copy_pipeline=nil;
    }
    return st;
}
int main(int argc,char **argv) { @autoreleasepool { return cpu_harness_main(argc,argv); } }
