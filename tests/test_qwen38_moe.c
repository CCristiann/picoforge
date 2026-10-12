/* Isolated F harness: clang + qwen38_moe.c, no B/C or GPU dependencies. */
#include "qwen38_f_internal.h"
#include "qwen38.h"
#include <sys/mman.h>
#include <unistd.h>
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

static void group_tests(void) {
    _Alignas(size_t) uint32_t ids[]={2,0,99,1,2,99,2,0,99};
    size_t offsets[6], cursors[5], p2r[7], r2p[7];
    Qwen38MoeConfig c={8,4,2,5,3,5,true};
    Qwen38U32Mut v={ids,sizeof ids,3,2,12};
    Q38MoeGrouping original={offsets,sizeof offsets,cursors,sizeof cursors,
                            p2r,r2p,sizeof p2r,sizeof r2p};
    const Qwen38Status errors[]={Q38_RANGE,Q38_INVALID_ARGUMENT,Q38_CAPACITY,
        Q38_INVALID_ARGUMENT,Q38_INVALID_ARGUMENT,Q38_STRIDE,Q38_OVERFLOW,
        Q38_SHAPE,Q38_INVALID_ARGUMENT};
    for (size_t i=0;i<sizeof errors/sizeof *errors;i++) {
        for (size_t j=0;j<6;j++) offsets[j]=SIZE_MAX;
        for (size_t j=0;j<5;j++) cursors[j]=SIZE_MAX;
        for (size_t j=0;j<7;j++) p2r[j]=r2p[j]=SIZE_MAX;
        Q38MoeGrouping g=original; Qwen38U32Mut bad=v; Qwen38Error err;
        switch (i) {
        case 0: ids[7]=4; break;
        case 1: ids[7]=2; break;
        case 2: g.row_to_pair_capacity_bytes=6*sizeof(size_t)-1; break;
        case 3: g.pair_to_row=g.row_to_pair; break;
        case 4: g.offsets=NULL; break;
        case 5: bad.row_stride_bytes=7; break;
        case 6: bad.rows=SIZE_MAX; break;
        case 7: bad.cols=1; break;
        case 8: g.offsets=(size_t *)(void *)ids; break;
        }
        assert(q38_moe_group_pairs(&c,bad,g,&err)==errors[i]);
        assert(err.code==errors[i]);
        for (size_t j=0;j<6;j++) assert(offsets[j]==SIZE_MAX);
        for (size_t j=0;j<5;j++) assert(cursors[j]==SIZE_MAX);
        for (size_t j=0;j<7;j++) assert(p2r[j]==SIZE_MAX && r2p[j]==SIZE_MAX);
        ids[7]=0;
    }
    assert(q38_moe_group_pairs(&c,v,original,NULL)==Q38_OK);
    const size_t expected_offsets[]={0,2,3,6,6}, expected_pairs[]={1,5,2,0,3,4};
    assert(memcmp(offsets,expected_offsets,sizeof expected_offsets)==0);
    assert(memcmp(r2p,expected_pairs,sizeof expected_pairs)==0);
    for (size_t j=0;j<6;j++) assert(p2r[r2p[j]]==j);
    assert(offsets[5]==SIZE_MAX && cursors[4]==SIZE_MAX);
    assert(p2r[6]==SIZE_MAX && r2p[6]==SIZE_MAX);
    assert(ids[2]==99 && ids[5]==99 && ids[8]==99);
    v.rows=0; v.data=NULL;
    original.pair_to_row=NULL; original.row_to_pair=NULL;
    original.pair_to_row_capacity_bytes=original.row_to_pair_capacity_bytes=0;
    assert(q38_moe_group_pairs(&c,v,original,NULL)==Q38_OK);
    for (size_t j=0;j<5;j++) assert(offsets[j]==0);
    for (size_t j=0;j<4;j++) assert(cursors[j]==0);
    puts("F grouping: stable pairs, empty experts/batch, preflight and inverse maps OK");
}

/* Local test wire format: uint32 T,E,K, then IDs; uint32 status followed by
 * uint64 offsets[E+1], pair_to_row[T*K], row_to_pair[T*K]. */
static void stream_group(void) {
    uint32_t h[3]; assert(fread(h,4,3,stdin)==3);
    size_t t=h[0], e=h[1], k=h[2], pitch=k+3;
    assert(t<=1024 && e>0 && e<=2048 && k>0 && k<=e && t*k<=1048576);
    size_t n=t*k;
    uint32_t *ids=malloc((t*pitch+1)*sizeof *ids);
    size_t *off=malloc((e+2)*sizeof *off), *cur=malloc((e+1)*sizeof *cur);
    size_t *p2r=malloc((n+1)*sizeof *p2r), *r2p=malloc((n+1)*sizeof *r2p);
    assert(ids && off && cur && p2r && r2p);
    for (size_t j=0;j<t*pitch+1;j++) ids[j]=UINT32_MAX;
    for (size_t j=0;j<e+2;j++) off[j]=SIZE_MAX;
    for (size_t j=0;j<e+1;j++) cur[j]=SIZE_MAX;
    for (size_t j=0;j<n+1;j++) p2r[j]=r2p[j]=SIZE_MAX;
    for (size_t j=0;j<t;j++) assert(fread(ids+j*pitch,4,k,stdin)==k);
    Qwen38MoeConfig c={8,h[1],h[2],5,3,5,true};
    Q38MoeGrouping g={off,(e+1)*sizeof *off,cur,e*sizeof *cur,
                     p2r,r2p,n*sizeof *p2r,n*sizeof *r2p};
    Qwen38Error err;
    Qwen38Status st=q38_moe_group_pairs(&c,(Qwen38U32Mut){ids,t*pitch*4,t,k,pitch*4},g,&err);
    assert(err.code==st);
    uint32_t code=(uint32_t)st; assert(fwrite(&code,4,1,stdout)==1);
    if (st==Q38_OK) {
        size_t *arrays[]={off,p2r,r2p}, lengths[]={e+1,n,n};
        for (size_t a=0;a<3;a++) for (size_t j=0;j<lengths[a];j++) {
            uint64_t item=(uint64_t)arrays[a][j];
            assert(fwrite(&item,8,1,stdout)==1);
        }
    } else {
        for (size_t j=0;j<e+1;j++) assert(off[j]==SIZE_MAX);
        for (size_t j=0;j<e;j++) assert(cur[j]==SIZE_MAX);
        for (size_t j=0;j<n;j++) assert(p2r[j]==SIZE_MAX && r2p[j]==SIZE_MAX);
    }
    for (size_t j=0;j<t;j++) for (size_t a=k;a<pitch;a++) assert(ids[j*pitch+a]==UINT32_MAX);
    assert(off[e+1]==SIZE_MAX && cur[e]==SIZE_MAX && p2r[n]==SIZE_MAX && r2p[n]==SIZE_MAX);
    assert(ids[t*pitch]==UINT32_MAX);
    free(r2p); free(p2r); free(cur); free(off); free(ids);
}

static Q38PackedCpuView float_weight(float *p,size_t in,size_t out,size_t experts) {
    Q38PackedCpuView v={.span={(const unsigned char *)p,in*out*experts*4}};
    GgufTensor t={.type=0,.rank=3,.shape={in,out,experts,1},.bytes=v.span.bytes};
    assert(q38_packed_layout_init(&t,in,&v.layout,NULL)==Q38_OK); return v;
}
static void lifecycle_tests(void) {
    float data[512]={0};
    Qwen38MoeConfig c={8,4,2,5,3,5,true};
    Qwen38MoeWeights v={float_weight(data,8,4,1),float_weight(data,8,5,4),
        float_weight(data,8,5,4),float_weight(data,5,8,4),float_weight(data,8,3,1),
        float_weight(data,8,3,1),float_weight(data,3,8,1),float_weight(data,8,1,1)};
    Qwen38Moe *m=NULL; Qwen38MoeWork *w=NULL; Qwen38Error err; size_t bytes=0,big=0;
    assert(qwen38_moe_create(&c,&v,&m,&err)==Q38_OK);
    c.hidden_size=99; /* config copied, not borrowed */
    assert(qwen38_moe_work_size(m,32,&bytes,&err)==Q38_OK && bytes>0);
    assert(qwen38_moe_work_size(m,1024,&big,&err)==Q38_OK && bytes==big);
    assert(qwen38_moe_work_create(m,1024,bytes-1,&w,&err)==Q38_CAPACITY && !w);
    assert(qwen38_moe_work_create(m,1024,bytes,&w,&err)==Q38_OK);
    assert(qwen38_moe_work_reset(w,&err)==Q38_OK);
    float input[16]={0},output[16];
    for (size_t j=0;j<16;j++) output[j]=-99;
    Qwen38F32 in=readonly(view(input,sizeof input,2,8,32));
    Qwen38F32Mut out=view(output,sizeof output,2,8,32);
    Qwen38F32Mut alias=out; alias.data=input;
    assert(qwen38_moe_forward(m,in,alias,NULL,w,&err)==Q38_INVALID_ARGUMENT);
    alias=out; alias.capacity_bytes=63;
    assert(qwen38_moe_forward(m,in,alias,NULL,w,&err)==Q38_CAPACITY);
    alias=out; alias.stride_bytes[0]=28;
    assert(qwen38_moe_forward(m,in,alias,NULL,w,&err)==Q38_STRIDE);
    alias=out; alias.data=data;
    assert(qwen38_moe_forward(m,in,alias,NULL,w,&err)==Q38_INVALID_ARGUMENT);
    for (size_t j=0;j<16;j++) assert(output[j]==-99);
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_OK);
    for (size_t j=0;j<16;j++) assert(output[j]==0);
    data[0]=NAN; /* borrowed bytes are not copied by create */
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_NONFINITE);
    data[0]=0;
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_BAD_STATE);
    assert(qwen38_moe_work_reset(w,&err)==Q38_OK);
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_OK);
    qwen38_moe_destroy(m); /* retained metadata makes work cleanup safe */
    assert(qwen38_moe_work_reset(w,&err)==Q38_BAD_STATE);
    qwen38_moe_work_destroy(w); m=NULL; w=NULL; c.hidden_size=8;
    v.down.layout.k_logical=4;
    assert(qwen38_moe_create(&c,&v,&m,&err)==Q38_SHAPE && !m);
    v.down.layout.k_logical=5; v.up.span.bytes--;
    assert(qwen38_moe_create(&c,&v,&m,&err)==Q38_CAPACITY && !m);
    v.up.span.bytes++; v.gate.layout.row_bytes++;
    assert(qwen38_moe_create(&c,&v,&m,&err)==Q38_FORMAT && !m);
    v.gate.layout.row_bytes--;
    assert(qwen38_moe_create(&c,&v,&m,&err)==Q38_OK);
    assert(qwen38_moe_work_size(m,0,&bytes,&err)==Q38_OK);
    assert(qwen38_moe_work_create(m,0,bytes,&w,&err)==Q38_OK);
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_CAPACITY);
    in.data=NULL; in.dim[0]=0; out.data=NULL; out.dim[0]=0;
    assert(qwen38_moe_forward(m,in,out,NULL,w,&err)==Q38_OK);
    qwen38_moe_work_destroy(w); qwen38_moe_destroy(m);
    puts("F lifecycle: copied config, budget, shapes, borrowed weights, retained owner OK");
}

static void emit_matrix(Qwen38F32 v) {
    for (size_t j=0;j<v.dim[0];j++)
        assert(fwrite(v.data+j*(v.stride_bytes[0]/4),4,v.dim[1],stdout)==v.dim[1]);
}
static void emit_trace(void *ctx,size_t stage,size_t expert,size_t token,Qwen38F32 in,Qwen38F32 out) {
    (void)ctx;
    uint64_t h[]={stage,expert,token,in.dim[0],in.dim[1],out.dim[1]};
    assert(fwrite(h,8,6,stdout)==6); emit_matrix(in); emit_matrix(out);
}
/* Fixture-only bounded transport. 9 uint32 config/T/trace, eight descriptors
 * (type,Kphysical,N,E) + packed bytes, then X. Pitches exercise real strides. */
static void stream_forward(void) {
    uint32_t h[9]; assert(fread(h,4,9,stdin)==9);
    Qwen38MoeConfig c={h[0],h[1],h[2],h[3],h[4],h[5],h[6]!=0};
    size_t t=h[7], width=h[0],k=h[2];
    assert(t<=1024 && width>0 && width<=2560 && h[1]<=512 && k>0 && k<=h[1]);
    assert(h[3]<=640 && h[4]<=640 && h[5]<=768 && h[6]<=1 && h[8]<=3);
    Q38PackedCpuView v[8]; void *storage[8]; size_t total=0,sizes[8];
    for (size_t j=0;j<8;j++) {
        uint32_t d[4]; assert(fread(d,4,4,stdin)==4);
        const GgufFormat *f=gguf_format(d[0]); assert(f);
        size_t bytes=(size_t)d[1]/f->block*f->bytes*d[2]*d[3];
        sizes[j]=bytes;
        if (h[8]&2) {
            /* Anonymous address reservation, NOT a mapping of checkpoint data.
             * Only explicitly supplied expert pages become readable. */
            assert(bytes<=1024u*1024u*1024u);
            storage[j]=mmap(NULL,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
            assert(storage[j]!=MAP_FAILED);
            uint32_t segments; assert(fread(&segments,4,1,stdin)==1 && segments<=512);
            size_t page=(size_t)sysconf(_SC_PAGESIZE);
            for (size_t seg=0;seg<segments;seg++) {
                uint64_t part[2]; assert(fread(part,8,2,stdin)==2);
                size_t off=(size_t)part[0],len=(size_t)part[1];
                assert(off<=bytes && len<=bytes-off);
                total+=len; assert(total<=64*1024*1024);
                size_t first=off/page*page,extent=(off+len+page-1)/page*page-first;
                assert(mprotect((char *)storage[j]+first,extent,PROT_READ|PROT_WRITE)==0);
                assert(fread((char *)storage[j]+off,1,len,stdin)==len);
                assert(mprotect((char *)storage[j]+first,extent,PROT_READ)==0);
            }
        } else {
            total+=bytes; assert(total<=64*1024*1024);
            storage[j]=malloc(bytes); assert(storage[j]); assert(fread(storage[j],1,bytes,stdin)==bytes);
        }
        GgufTensor tensor={.type=d[0],.rank=3,.shape={d[1],d[2],d[3],1},.bytes=bytes};
        size_t logical=j==3?c.intermediate:d[1];
        v[j].span=(Qwen38CpuSpan){storage[j],bytes};
        assert(q38_packed_layout_init(&tensor,logical,&v[j].layout,NULL)==Q38_OK);
    }
    size_t pitch=width+3;
    float *x=malloc((t*pitch+1)*4),*y=malloc((t*pitch+1)*4),*scores=malloc((t*(k+2)+1)*4);
    uint32_t *ids=malloc((t*(k+3)+1)*4); assert(x && y && scores && ids);
    for (size_t j=0;j<t*pitch+1;j++) x[j]=y[j]=-99.0f;
    for (size_t j=0;j<t*(k+2)+1;j++) scores[j]=-99.0f;
    for (size_t j=0;j<t*(k+3)+1;j++) ids[j]=UINT32_MAX;
    for (size_t j=0;j<t;j++) assert(fread(x+j*pitch,4,width,stdin)==width);
    Qwen38MoeWeights weights={v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]};
    Qwen38Moe *m=NULL; Qwen38MoeWork *w=NULL; Qwen38Error error; size_t budget=0;
    Qwen38Status st=qwen38_moe_create(&c,&weights,&m,&error);
    if (!st) st=qwen38_moe_work_size(m,t,&budget,&error);
    assert(budget<=32*1024*1024);
    if (!st) st=qwen38_moe_work_create(m,t,budget,&w,&error);
    if (!st && (h[8]&1)) q38_moe_observe(w,emit_trace,NULL);
    Qwen38MoeRoutes routes={{ids,t*(k+3)*4,t,k,(k+3)*4},view(scores,t*(k+2)*4,t,k,(k+2)*4)};
    if (!st) st=qwen38_moe_forward(m,readonly(view(x,t*pitch*4,t,width,pitch*4)),
                                   view(y,t*pitch*4,t,width,pitch*4),&routes,w,&error);
    assert(error.code==st);
    uint64_t end[]={UINT64_MAX,(uint64_t)st,budget}; assert(fwrite(end,8,3,stdout)==3);
    if (!st) {
        emit_matrix(readonly(view(y,t*pitch*4,t,width,pitch*4)));
        for (size_t j=0;j<t;j++) assert(fwrite(ids+j*(k+3),4,k,stdout)==k);
        emit_matrix(readonly(routes.weights));
    }
    for (size_t j=0;j<t;j++) {
        for (size_t a=width;a<pitch;a++) assert(y[j*pitch+a]==-99 && x[j*pitch+a]==-99);
        for (size_t a=k;a<k+2;a++) assert(scores[j*(k+2)+a]==-99);
        for (size_t a=k;a<k+3;a++) assert(ids[j*(k+3)+a]==UINT32_MAX);
    }
    assert(y[t*pitch]==-99 && x[t*pitch]==-99 && scores[t*(k+2)]==-99 && ids[t*(k+3)]==UINT32_MAX);
    qwen38_moe_work_destroy(w); qwen38_moe_destroy(m);
    free(ids); free(scores); free(y); free(x);
    for (size_t j=0;j<8;j++) {
        if (h[8]&2) assert(munmap(storage[j],sizes[j])==0); else free(storage[j]);
    }
}

static void describe_slice(const char *path,uint32_t layer) {
    Gguf g; gguf_open(path,&g); Qwen38Config *c=NULL; Qwen38Weights *w=NULL;
    assert(qwen38_config_load(&g,&c,NULL)==Q38_OK);
    assert(qwen38_weights_describe(&g,c,&w,NULL)==Q38_OK);
    Qwen38MoeConfig m=qwen38_config_view(c)->moe;
    printf("{\"config\":[%u,%u,%u,%u,%u,%u,%u],\"tensors\":[",m.hidden_size,m.experts,
           m.top_k,m.intermediate,m.shared_intermediate,m.down_physical,(unsigned)m.norm_topk_prob);
    Qwen38Role roles[]={Q38_MOE_ROUTER,Q38_MOE_GATE,Q38_MOE_UP,Q38_MOE_DOWN,
        Q38_SHARED_GATE,Q38_SHARED_UP,Q38_SHARED_DOWN,Q38_SHARED_ROUTER};
    for (size_t j=0;j<8;j++) {
        const Qwen38Tensor *t=NULL;
        assert(qwen38_weights_get(w,layer,roles[j],&t,NULL)==Q38_OK);
        const GgufTensor *s=&t->storage; Q38PackedLayout layout;
        assert(q38_packed_layout_init(s,(size_t)t->logical_shape[0],&layout,NULL)==Q38_OK);
        printf("%s{\"name\":\"%s\",\"type\":%u,\"offset\":%zu,\"bytes\":%zu,"
               "\"physical\":%zu,\"logical\":%zu,\"rows\":%zu,\"experts\":%zu,\"expert_bytes\":%zu}",
               j?",":"",s->name,s->type,s->file_offset,s->bytes,layout.k_storage,
               layout.k_logical,layout.n_out,layout.n_experts,layout.expert_bytes);
    }
    puts("]}"); qwen38_weights_free(w); qwen38_config_free(c); gguf_close(&g);
}

int main(int argc, char **argv) {
    if (argc==2 && strcmp(argv[1],"--route")==0) stream_route();
    else if (argc==2 && strcmp(argv[1],"--group")==0) stream_group();
    else if (argc==2 && strcmp(argv[1],"--forward")==0) stream_forward();
    else if (argc==4 && strcmp(argv[1],"--describe")==0) describe_slice(argv[2],(uint32_t)strtoul(argv[3],NULL,10));
    else { assert(argc==1); unit_tests(); group_tests(); lifecycle_tests(); }
    return 0;
}
