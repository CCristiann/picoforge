/* Isolated F harness: clang + qwen38_moe.c, no B/C or GPU dependencies. */
#include "qwen38_moe.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Qwen38F32Mut view(float *p, size_t cap, size_t t, size_t e, size_t pitch) {
    return (Qwen38F32Mut){.data=p, .capacity_bytes=cap, .rank=2,
        .dim={t,e,1,1}, .stride_bytes={pitch,4,0,0}};
}
static Qwen38F32 readonly(Qwen38F32Mut v) {
    Qwen38F32 r = {.data=v.data,.capacity_bytes=v.capacity_bytes,.rank=v.rank};
    memcpy(r.dim,v.dim,sizeof r.dim); memcpy(r.stride_bytes,v.stride_bytes,sizeof r.stride_bytes);
    return r;
}
typedef struct {
    float z[64], p[64], w[64]; uint32_t ids[64];
    Qwen38MoeConfig c; Qwen38F32 logits; Qwen38F32Mut probabilities;
    Qwen38MoeRoutes routes; Qwen38Error error;
} Test;
static void init(Test *t) {
    memset(t,0,sizeof *t);
    t->c=(Qwen38MoeConfig){8,4,2,5,3,5,true};
    for (size_t i=0;i<64;i++) { t->p[i]=t->w[i]=-99.0f; t->ids[i]=UINT32_MAX; }
    t->z[0]=0.0f; t->z[1]=1e-9f; t->z[2]=0.0f; t->z[3]=-1e-9f;
    t->z[7]=0.0f; t->z[8]=2.0f; t->z[9]=2.0f; t->z[10]=-1000.0f;
    t->logits=readonly(view(t->z,sizeof t->z,2,4,28));
    t->probabilities=view(t->p,sizeof t->p,2,4,32);
    t->routes.weights=view(t->w,sizeof t->w,2,2,16);
    t->routes.ids=(Qwen38U32Mut){t->ids,sizeof t->ids,2,2,20};
}
static Qwen38Status run(Test *t) {
    return qwen38_moe_route(&t->c,t->logits,t->routes,t->probabilities,&t->error);
}
static void untouched(const Test *t) {
    for (size_t i=0;i<64;i++) {
        assert(t->p[i]==-99.0f); assert(t->w[i]==-99.0f); assert(t->ids[i]==UINT32_MAX);
    }
}
static void unit_tests(void) {
    Test t;
    init(&t); assert(run(&t)==Q38_OK); assert(t.error.code==Q38_OK);
    assert(t.ids[0]==0 && t.ids[1]==1 && t.ids[5]==1 && t.ids[6]==2);
    assert(t.w[0]==0.5f && t.w[1]==0.5f && t.w[4]==0.5f && t.w[5]==0.5f);
    assert(t.p[0]==0.25f && t.p[11]==0.0f);
    for (size_t j=0;j<64;j++) {
        if (!(j<4 || (j>=8 && j<12))) assert(t.p[j]==-99.0f);
        if (!(j<2 || (j>=4 && j<6))) assert(t.w[j]==-99.0f);
        if (!(j<2 || (j>=5 && j<7))) assert(t.ids[j]==UINT32_MAX);
    }
    init(&t); t.c.norm_topk_prob=false; assert(run(&t)==Q38_OK); assert(t.w[0]==0.25f);
    init(&t); t.z[0]=-3e38f; t.z[1]=3e38f; assert(run(&t)==Q38_OK);
    assert(t.ids[0]==1 && t.ids[1]==0 && t.w[0]==1.0f && t.w[1]==0.0f);
    const Qwen38Status expected[]={Q38_INVALID_ARGUMENT,Q38_SHAPE,Q38_STRIDE,Q38_CAPACITY,
        Q38_STRIDE,Q38_RANGE,Q38_SHAPE,Q38_CAPACITY,Q38_INVALID_ARGUMENT,Q38_INVALID_ARGUMENT,
        Q38_NONFINITE,Q38_NONFINITE,Q38_OVERFLOW,Q38_INVALID_ARGUMENT};
    for (size_t i=0;i<sizeof expected/sizeof *expected;i++) {
        init(&t);
        switch(i) {
        case 0: t.logits.data=NULL; break;
        case 1: t.logits.rank=1; break;
        case 2: t.probabilities.stride_bytes[1]=8; break;
        case 3: t.routes.weights.capacity_bytes=23; break;
        case 4: t.routes.ids.row_stride_bytes=7; break;
        case 5: t.c.top_k=5; break;
        case 6: t.routes.ids.cols=3; break;
        case 7: t.probabilities.capacity_bytes=47; break;
        case 8: t.probabilities.data=t.z+1; break;
        case 9: t.routes.weights.data=t.p; break;
        case 10: t.z[10]=NAN; break;
        case 11: t.z[10]=INFINITY; break;
        case 12: t.logits.dim[0]=SIZE_MAX; break;
        case 13: t.probabilities.data=t.z; break;
        }
        assert(run(&t)==expected[i]); assert(t.error.code==expected[i]); untouched(&t);
    }
    init(&t);
    assert(qwen38_moe_route(NULL,t.logits,t.routes,t.probabilities,&t.error)==Q38_INVALID_ARGUMENT);
    untouched(&t);
    init(&t); t.logits.dim[0]=t.probabilities.dim[0]=t.routes.weights.dim[0]=t.routes.ids.rows=0;
    t.logits.data=NULL; t.probabilities.data=NULL; t.routes.weights.data=NULL; t.routes.ids.data=NULL;
    assert(run(&t)==Q38_OK); untouched(&t);
    t.routes.weights.stride_bytes[1]=8; assert(run(&t)==Q38_STRIDE);
    init(&t); assert(qwen38_moe_route(&t.c,t.logits,t.routes,t.probabilities,NULL)==Q38_OK);
    puts("F routing: ties, stride guards, errors, overflow and empty batch OK");
}

/* Little-endian local fixture protocol: T,E,K,norm then T*E float logits.
 * Output: uint32 status, then row-major probabilities, IDs, weights.
 * All allocations are test-only and bounded; every runtime view has gaps. */
static void stream_route(void) {
    uint32_t h[4];
    assert(fread(h,sizeof *h,4,stdin)==4);
    size_t t=h[0], e=h[1], k=h[2];
    assert(t<=1024 && e>0 && e<=2048 && k>0 && k<=e && h[3]<=1);
    assert(t*e<=1048576);
    size_t zp=e+3, pp=e+5, wp=k+2, ip=k+4;
    float *z=calloc(t*zp+1,sizeof *z), *p=malloc((t*pp+1)*sizeof *p);
    float *w=malloc((t*wp+1)*sizeof *w);
    uint32_t *ids=malloc((t*ip+1)*sizeof *ids);
    assert(z && p && w && ids);
    for (size_t j=0;j<t*pp+1;j++) p[j]=-99.0f;
    for (size_t j=0;j<t*wp+1;j++) w[j]=-99.0f;
    for (size_t j=0;j<t*ip+1;j++) ids[j]=UINT32_MAX;
    for (size_t j=0;j<t;j++) assert(fread(z+j*zp,4,e,stdin)==e);
    Qwen38MoeConfig c={8,h[1],h[2],5,3,5,h[3]!=0};
    Qwen38MoeRoutes routes={.ids={ids,t*ip*4,t,k,ip*4},
                            .weights=view(w,t*wp*4,t,k,wp*4)};
    Qwen38Error error;
    Qwen38Status status=qwen38_moe_route(&c,readonly(view(z,t*zp*4,t,e,zp*4)),routes,
                                        view(p,t*pp*4,t,e,pp*4),&error);
    assert(error.code==status);
    uint32_t code=(uint32_t)status;
    assert(fwrite(&code,4,1,stdout)==1);
    if (status==Q38_OK) {
        for (size_t j=0;j<t;j++) assert(fwrite(p+j*pp,4,e,stdout)==e);
        for (size_t j=0;j<t;j++) assert(fwrite(ids+j*ip,4,k,stdout)==k);
        for (size_t j=0;j<t;j++) assert(fwrite(w+j*wp,4,k,stdout)==k);
    }
    for (size_t j=0;j<t;j++) {
        for (size_t a=(status==Q38_OK?e:0);a<pp;a++) assert(p[j*pp+a]==-99.0f);
        for (size_t a=(status==Q38_OK?k:0);a<wp;a++) assert(w[j*wp+a]==-99.0f);
        for (size_t a=(status==Q38_OK?k:0);a<ip;a++) assert(ids[j*ip+a]==UINT32_MAX);
    }
    assert(p[t*pp]==-99.0f && w[t*wp]==-99.0f && ids[t*ip]==UINT32_MAX);
    free(ids); free(w); free(p); free(z);
}

int main(int argc, char **argv) {
    if (argc==2 && strcmp(argv[1],"--route")==0) stream_route();
    else { assert(argc==1); unit_tests(); }
    return 0;
}
