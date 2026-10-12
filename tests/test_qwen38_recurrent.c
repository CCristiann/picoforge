#include "qwen38_c_internal.h"
#include <assert.h>
#include <stdlib.h>
#include <float.h>

static Qwen38F32Mut view(float *p, size_t cap, size_t rank, const size_t *dims) {
    Qwen38F32Mut v = {.data=p, .capacity_bytes=cap, .rank=rank};
    size_t stride = sizeof(float);
    for (size_t i = rank; i-- > 0;) { v.dim[i]=dims[i]; v.stride_bytes[i]=stride; stride *= dims[i]; }
    return v;
}
static void test_views(void) {
    float data[12] = {0}; Qwen38Error e;
    Qwen38F32 v = q38c_ro(view(data, sizeof(data), 2, (size_t[]){2,3}));
    v.stride_bytes[0] = 8*sizeof(float);
    assert(q38c_view(v, &e) == Q38_OK && q38c_span(v) == 11*sizeof(float));
    assert(q38c_offset(v, 4) == 9);
    v.capacity_bytes = 10*sizeof(float); assert(q38c_view(v, &e) == Q38_CAPACITY);
    assert(e.code == Q38_CAPACITY);
    v.stride_bytes[0] = 4; assert(q38c_view(v, &e) == Q38_STRIDE);
    v.stride_bytes[0] = SIZE_MAX-3; assert(q38c_view(v, &e) == Q38_OVERFLOW);
    v = q38c_ro(view(NULL, 0, 2, (size_t[]){0,3})); assert(q38c_view(v, &e) == Q38_OK);
    v.dim[1]=0; assert(q38c_view(v, &e) == Q38_SHAPE);
}
static Qwen38Memory backing(size_t bytes) {
    Qwen38Memory m={bytes ? aligned_alloc(64,bytes) : NULL, bytes};
    assert(!bytes || m.data); if (bytes) memset(m.data,0,bytes); return m;
}
static void test_conv_lifecycle(void) {
    Qwen38ConvShape shape={2,4,3,true}; Qwen38StateSize size; Qwen38Error e;
    assert(qwen38_conv_state_bytes(&shape,&size,&e)==Q38_OK && size.payload_bytes>=18*sizeof(float));
    Qwen38Memory m=backing(size.metadata_bytes), p=backing(size.payload_bytes);
    Qwen38Memory m2=backing(size.metadata_bytes), p2=backing(size.payload_bytes);
    Qwen38StateKey key={.model=&shape,.layer=1,.epoch=1}; Qwen38ConvState *s=NULL,*copy=NULL;
    assert(qwen38_conv_state_init(&shape,key,m,p,&s,&e)==Q38_OK);
    assert(qwen38_conv_state_init(&shape,key,m2,p2,&copy,&e)==Q38_OK);
    assert(qwen38_conv_state_info(s).valid && qwen38_conv_state_info(s).next_pos==0);
    float out[40]; for (size_t i=0;i<40;++i) out[i]=-99;
    Qwen38F32Mut y=view(out,sizeof(out),2,(size_t[]){9,2}); y.stride_bytes[0]=4*sizeof(float);
    assert(qwen38_conv_state_export(s,y,&e)==Q38_OK);
    for (size_t i=0;i<9;++i) assert(out[4*i]==0 && out[4*i+1]==0 && out[4*i+2]==-99);
    assert(qwen38_conv_state_copy(copy,s,&e)==Q38_OK);
    assert(qwen38_conv_state_copy(s,s,&e)==Q38_INVALID_ARGUMENT);
    assert(qwen38_conv_state_reset(copy,1,&e)==Q38_STALE);
    assert(qwen38_conv_state_reset(copy,2,&e)==Q38_OK);
    assert(qwen38_conv_state_copy(copy,s,&e)==Q38_STALE);
    memcpy(m2.data,m.data,size.metadata_bytes);
    assert(!qwen38_conv_state_info((Qwen38ConvState *)m2.data).valid);
    Qwen38Memory small=p; small.bytes=1;
    assert(qwen38_conv_state_init(&shape,key,m2,small,&copy,&e)==Q38_CAPACITY && !copy);
    key.model=NULL;
    assert(qwen38_conv_state_init(&shape,key,m2,p2,&copy,&e)==Q38_INVALID_ARGUMENT && !copy);
    free(m.data);free(p.data);free(m2.data);free(p2.data);
    shape.kernel=1;
    assert(qwen38_conv_state_bytes(&shape,&size,&e)==Q38_OK && size.payload_bytes==0);
}
static void test_conv_values(void) {
    Qwen38ConvShape shape={1,3,1,false}; Qwen38StateSize size; Qwen38Error e;
    assert(qwen38_conv_state_bytes(&shape,&size,&e)==Q38_OK);
    Qwen38Memory m=backing(size.metadata_bytes), p=backing(size.payload_bytes);
    Qwen38ConvState *s; Qwen38StateKey key={.model=&shape,.epoch=1};
    assert(qwen38_conv_state_init(&shape,key,m,p,&s,&e)==Q38_OK);
    float x[]={1,2,3,4}, taps[]={1,10,100}, out[]={-99,-99,-99,-99}, history[2];
    Qwen38F32 xx=q38c_ro(view(x,sizeof(x),2,(size_t[]){4,1}));
    Qwen38F32 ww=q38c_ro(view(taps,sizeof(taps),2,(size_t[]){1,3}));
    Qwen38F32Mut y=view(out,sizeof(out),2,(size_t[]){4,1});
    assert(qwen38_conv_causal(&shape,xx,ww,y,s,1,&e)==Q38_BAD_STATE && out[0]==-99);
    assert(qwen38_conv_causal(&shape,xx,ww,view(x,sizeof(x),2,xx.dim),s,0,&e)==Q38_INVALID_ARGUMENT);
    x[3]=NAN; assert(qwen38_conv_causal(&shape,xx,ww,y,s,0,&e)==Q38_NONFINITE && out[0]==-99); x[3]=4;
    assert(qwen38_conv_causal(&shape,xx,ww,y,s,0,&e)==Q38_OK);
    assert(out[0]==100 && out[1]==210 && out[2]==321 && out[3]==432);
    assert(qwen38_conv_state_info(s).next_pos==4);
    assert(qwen38_conv_state_export(s,view(history,sizeof(history),2,(size_t[]){2,1}),&e)==Q38_OK);
    assert(history[0]==3 && history[1]==4);
    taps[2]=FLT_MAX;
    assert(qwen38_conv_causal(&shape,xx,ww,y,s,4,&e)==Q38_NONFINITE);
    assert(!qwen38_conv_state_info(s).valid && !qwen38_conv_state_info(s).busy);
    assert(qwen38_conv_causal(&shape,xx,ww,y,s,4,&e)==Q38_BAD_STATE);
    assert(qwen38_conv_state_reset(s,2,&e)==Q38_OK && qwen38_conv_state_info(s).next_pos==0);
    free(m.data);free(p.data);
}
static void test_norm_activation(void) {
    float x[]={-100,0,100}, out[3]={-99,-99,-99}; Qwen38Error e;
    Qwen38F32 a=q38c_ro(view(x,sizeof(x),1,(size_t[]){3}));
    Qwen38F32Mut y=view(out,sizeof(out),1,(size_t[]){3});
    assert(qwen38_activate(Q38_SIGMOID,a,y,&e)==Q38_OK && out[1]==.5f && out[2]==1);
    assert(qwen38_activate(Q38_SOFTPLUS,a,y,&e)==Q38_OK && fabsf(out[1]-logf(2))<1e-7f && out[2]==100);
    assert(qwen38_activate(Q38_SILU,a,y,&e)==Q38_OK && out[1]==0 && out[2]==100);
    assert(qwen38_activate((Qwen38Activation)99,a,y,&e)==Q38_UNSUPPORTED);
    x[2]=NAN; out[0]=-99;
    assert(qwen38_activate(Q38_SIGMOID,a,y,&e)==Q38_NONFINITE && out[0]==-99);
    float raw[]={1,-2,3,1e-6f,-2e-6f,3e-6f}, gain[]={.5f,-1,2}, norm[6];
    Qwen38F32 xx=q38c_ro(view(raw,sizeof(raw),3,(size_t[]){1,2,3}));
    Qwen38F32 ww=q38c_ro(view(gain,sizeof(gain),1,(size_t[]){3}));
    Qwen38F32Mut yy=view(norm,sizeof(norm),3,xx.dim);
    assert(qwen38_rmsnorm_groups(xx,ww,yy,1e-6f,&e)==Q38_OK);
    for (size_t g=0;g<2;++g) for (size_t j=0;j<3;++j) {
        double sum=0; for(size_t k=0;k<3;++k) sum+=(double)raw[g*3+k]*raw[g*3+k];
        double expected=(double)raw[g*3+j]*gain[j]/sqrt(sum/3+1e-6);
        assert(fabs((double)norm[g*3+j]-expected)<1e-5);
    }
    assert(qwen38_rmsnorm_groups(xx,ww,view(raw,sizeof(raw),3,xx.dim),1e-6f,&e)==Q38_OK);
    assert(memcmp(raw,norm,sizeof(raw))==0);
    raw[0]=FLT_MAX; norm[0]=-99;
    assert(qwen38_rmsnorm_groups(xx,ww,yy,1e-6f,&e)==Q38_NONFINITE && norm[0]==-99);
}
static void test_hc(void) {
    Qwen38HcShape shape={2,3,5,1e-6f}; Qwen38Error e;
    float low[5]={0,0,0,0,0}, logits[6]={0}, normed[]={1,2,3,10,20,30}, mixed[3];
    Qwen38F32Mut lo=view(low,sizeof(low),2,(size_t[]){1,5});
    assert(qwen38_hc_lowrank_activate(&shape,q38c_ro(lo),lo,&e)==Q38_OK && low[4]==0);
    Qwen38F32 n=q38c_ro(view(normed,sizeof(normed),3,(size_t[]){1,2,3}));
    Qwen38F32 g=q38c_ro(view(logits,sizeof(logits),3,n.dim));
    assert(qwen38_hc_read(&shape,n,g,view(mixed,sizeof(mixed),2,(size_t[]){1,3}),&e)==Q38_OK);
    assert(mixed[0]==2.75f && mixed[1]==5.5f && mixed[2]==8.25f);
    float inj[2]={0,0}, block[3]={3,-2,1};
    Qwen38F32Mut i=view(inj,sizeof(inj),2,(size_t[]){1,2});
    assert(qwen38_hc_inject_weights(&shape,q38c_ro(i),i,&e)==Q38_OK && inj[0]==1 && inj[1]==1);
    assert(qwen38_hc_add(&shape,view(normed,sizeof(normed),3,n.dim),
        q38c_ro(view(block,sizeof(block),2,(size_t[]){1,3})),q38c_ro(i),&e)==Q38_OK);
    assert(normed[0]==4 && normed[1]==0 && normed[2]==4 && normed[3]==13 && normed[4]==18 && normed[5]==31);
}
static void test_gdn_lifecycle(void) {
    Qwen38GdnShape shape={2,6,3,5,1e-6f,1e-6f}; Qwen38StateSize size; Qwen38Error e;
    assert(qwen38_gdn_state_bytes(&shape,&size,&e)==Q38_OK && size.payload_bytes>=90*sizeof(float));
    Qwen38Memory m=backing(size.metadata_bytes),p=backing(size.payload_bytes);
    Qwen38GdnState *s; Qwen38StateKey key={.model=&shape,.epoch=1};
    assert(qwen38_gdn_state_init(&shape,key,m,p,&s,&e)==Q38_OK);
    float out[90]; for(size_t i=0;i<90;++i) out[i]=-99;
    assert(qwen38_gdn_state_export(s,view(out,sizeof(out),3,(size_t[]){6,5,3}),&e)==Q38_OK);
    for(size_t i=0;i<90;++i) assert(out[i]==0);
    assert(qwen38_gdn_state_info(s).valid && qwen38_gdn_state_info(s).next_pos==0);
    assert(qwen38_gdn_state_reset(s,1,&e)==Q38_STALE);
    assert(qwen38_gdn_state_reset(s,2,&e)==Q38_OK);
    free(m.data);free(p.data);
    shape.value_heads=5; assert(qwen38_gdn_state_bytes(&shape,&size,&e)==Q38_SHAPE && size.payload_bytes==0);
}
static void test_gdn_prepare(void) {
    Qwen38GdnShape shape={1,1,2,1,1e-6f,1e-6f}; Qwen38Error e;
    float qkv[]={3,4,0,0,7}, a[]={0}, b[]={0}, weight[]={0}, dt[]={.25f};
    Qwen38F32Mut x=view(qkv,sizeof(qkv),2,(size_t[]){1,5});
    Qwen38F32Mut aa=view(a,sizeof(a),2,(size_t[]){1,1}), bb=view(b,sizeof(b),2,(size_t[]){1,1});
    Qwen38F32 ww=q38c_ro(view(weight,sizeof(weight),1,(size_t[]){1}));
    Qwen38F32 dd=q38c_ro(view(dt,sizeof(dt),1,(size_t[]){1}));
    assert(qwen38_gdn_prepare(&shape,q38c_ro(x),q38c_ro(aa),q38c_ro(bb),ww,dd,x,aa,bb,&e)==Q38_OK);
    assert(fabsf(qkv[0]-3.0f/sqrtf(25.0f+1e-6f)/sqrtf(2))<1e-6f);
    assert(qkv[2]==0 && qkv[3]==0 && qkv[4]==7 && a[0]==0 && b[0]==.5f);
    float o[]={2}, z[]={0}, gain[]={.5f};
    Qwen38F32Mut output=view(o,sizeof(o),3,(size_t[]){1,1,1});
    assert(qwen38_gdn_norm_gate(&shape,q38c_ro(output),q38c_ro(view(z,sizeof(z),3,output.dim)),
        q38c_ro(view(gain,sizeof(gain),1,(size_t[]){1})),output,&e)==Q38_OK);
    assert(fabsf(o[0]-.25f)<1e-6f);
}
static void test_gdn_scan(void) {
    Qwen38GdnShape shape={1,1,1,1,1e-6f,1e-6f}; Qwen38StateSize size; Qwen38Error e;
    assert(qwen38_gdn_state_bytes(&shape,&size,&e)==Q38_OK);
    Qwen38Memory m=backing(size.metadata_bytes),p=backing(size.payload_bytes);
    Qwen38GdnState *s; Qwen38StateKey key={.model=&shape,.epoch=1};
    assert(qwen38_gdn_state_init(&shape,key,m,p,&s,&e)==Q38_OK);
    float x[]={0,1,2}, g[]={0}, b[]={1}, out[1]={-99}, state[1];
    Qwen38F32 xx=q38c_ro(view(x,sizeof(x),2,(size_t[]){1,3}));
    Qwen38F32 gg=q38c_ro(view(g,sizeof(g),2,(size_t[]){1,1}));
    Qwen38F32 bb=q38c_ro(view(b,sizeof(b),2,gg.dim));
    Qwen38F32Mut y=view(out,sizeof(out),3,(size_t[]){1,1,1});
    assert(qwen38_gdn_scan(&shape,xx,gg,bb,y,s,0,&e)==Q38_OK); /* Public warmup writes S=2. */
    x[0]=.5f; x[2]=4; g[0]=logf(.5f); b[0]=.25f;
    assert(qwen38_gdn_scan(&shape,xx,gg,bb,y,s,1,&e)==Q38_OK && out[0]==.875f);
    assert(qwen38_gdn_state_export(s,view(state,sizeof(state),3,y.dim),&e)==Q38_OK && state[0]==1.75f);
    b[0]=2; out[0]=-99;
    assert(qwen38_gdn_scan(&shape,xx,gg,bb,y,s,2,&e)==Q38_RANGE && out[0]==-99 && qwen38_gdn_state_info(s).valid);
    b[0]=1; x[1]=FLT_MAX;
    assert(qwen38_gdn_scan(&shape,xx,gg,bb,y,s,2,&e)==Q38_NONFINITE && !qwen38_gdn_state_info(s).valid);
    assert(qwen38_gdn_state_reset(s,2,&e)==Q38_OK);
    assert(qwen38_gdn_state_export(s,view(state,sizeof(state),3,y.dim),&e)==Q38_OK && state[0]==0);
    free(m.data);free(p.data);
}
static void test_admission_and_guards(void) {
    Qwen38Error e; float raw[]={1,2,3,4,5,6}, gain[]={1,1,1}, out[10];
    for(size_t i=0;i<10;++i) out[i]=-77;
    Qwen38F32 x=q38c_ro(view(raw,sizeof(raw),3,(size_t[]){1,2,3}));
    Qwen38F32 w=q38c_ro(view(gain,sizeof(gain),1,(size_t[]){3}));
    Qwen38F32Mut y=view(out+1,6*sizeof(float),3,x.dim), short_y=y;
    short_y.capacity_bytes-=1;
    assert(qwen38_rmsnorm_groups(x,w,short_y,1e-6f,&e)==Q38_CAPACITY);
    short_y=y; short_y.stride_bytes[1]=1;
    assert(qwen38_rmsnorm_groups(x,w,short_y,1e-6f,&e)==Q38_STRIDE);
    assert(qwen38_rmsnorm_groups(x,w,y,0,&e)==Q38_RANGE);
    assert(qwen38_rmsnorm_groups(x,w,y,NAN,&e)==Q38_NONFINITE);
    Qwen38F32Mut alias=view(raw+1,6*sizeof(float),3,x.dim);
    assert(qwen38_rmsnorm_groups(x,w,alias,1e-6f,&e)==Q38_INVALID_ARGUMENT);
    for(size_t i=0;i<10;++i) assert(out[i]==-77);
    assert(qwen38_rmsnorm_groups(x,w,y,1e-6f,&e)==Q38_OK && out[0]==-77 && out[7]==-77);
    Qwen38F32 empty=q38c_ro(view(NULL,0,3,(size_t[]){0,2,3}));
    gain[0]=NAN;
    assert(qwen38_rmsnorm_groups(empty,w,view(NULL,0,3,empty.dim),1e-6f,&e)==Q38_OK);
    Qwen38ConvShape shape={1,1,1,false}; Qwen38StateSize size;
    assert(qwen38_conv_state_bytes(&shape,&size,&e)==Q38_OK);
    Qwen38Memory m=backing(size.metadata_bytes),p=backing(size.payload_bytes);
    Qwen38ConvState *state; Qwen38StateKey key={.model=&shape,.epoch=1};
    assert(qwen38_conv_state_init(&shape,key,m,p,&state,&e)==Q38_OK);
    /* Deliberate metadata fault injection: this private test checks admission,
     * never seeds numerical state or introduces a production setter. */
    Q38CState *header=m.data; header->info.next_pos=UINT64_MAX;
    float tap=1;
    Qwen38F32 one=q38c_ro(view(raw,sizeof(raw),2,(size_t[]){1,1}));
    Qwen38F32 wt=q38c_ro(view(&tap,sizeof(tap),2,one.dim));
    assert(qwen38_conv_causal(&shape,one,wt,view(out,sizeof(out),2,one.dim),state,UINT64_MAX,&e)==Q38_OVERFLOW);
    header->info.busy=true; assert(qwen38_conv_state_reset(state,2,&e)==Q38_BUSY);
    header->info.busy=false; header->info.key.epoch=UINT64_MAX;
    assert(qwen38_conv_state_reset(state,1,&e)==Q38_OVERFLOW);
    free(m.data);free(p.data);
}
/* This CLI exposes A's catalog to the Python fixture builder, never re-parses
 * model dimensions or tensor names in C4. No tensor payload is mapped here. */
static int describe_layer(const char *path, uint32_t index) {
    Gguf g; gguf_open(path,&g); Qwen38Config *config=NULL; Qwen38Weights *weights=NULL;
    assert(qwen38_config_load(&g,&config,NULL)==Q38_OK);
    assert(qwen38_weights_describe(&g,config,&weights,NULL)==Q38_OK);
    Q38CLayer layer; assert(q38c_layer_plan(config,weights,index,&layer,NULL)==Q38_OK);
    const Qwen38ConfigView *c=&layer.config;
    printf("{\"layer\":%u,\"hidden\":%u,\"streams\":%zu,\"rank\":%zu,"
        "\"hk\":%zu,\"hv\":%zu,\"dk\":%zu,\"dv\":%zu,\"kernel\":%u,\"eps\":%.9g,\"weights\":[",
        index,c->hidden_size,c->hc.streams,c->hc.low_rank,c->gdn.key_heads,c->gdn.value_heads,
        c->gdn.key_dim,c->gdn.value_dim,c->gdn_conv_kernel,(double)c->hc.rms_eps);
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        GgufTensor *t=&layer.tensor[i].storage; Q38PackedLayout *p=&layer.weight[i].layout;
        printf("%s{\"slot\":%zu,\"role\":%u,\"name\":\"%s\",\"type\":%u,\"columns\":%zu,"
            "\"rows\":%zu,\"offset\":%zu,\"bytes\":%zu}",i ? "," : "",i,(unsigned)q38c_layer_roles[i],
            t->name,t->type,p->k_storage,p->n_out,t->file_offset,t->bytes);
    }
    puts("]}"); qwen38_weights_free(weights); qwen38_config_free(config); gguf_close(&g); return 0;
}
/* Bounded fixture transport. Packed data stays borrowed until A/C are released. */
static FILE *fixture_file(const char *directory, const char *name, const char *mode) {
    char path[4096]; int n=snprintf(path,sizeof(path),"%s/%s",directory,name);
    assert(n>0 && (size_t)n<sizeof(path)); FILE *f=fopen(path,mode); assert(f); return f;
}
static void fixture_read(const char *directory, const char *name, void *data, size_t bytes) {
    FILE *f=fixture_file(directory,name,"rb"); assert(fread(data,1,bytes,f)==bytes);
    assert(fgetc(f)==EOF && !ferror(f)); assert(!fclose(f));
}
static void fixture_write(const char *directory, const char *name, const void *data, size_t bytes) {
    FILE *f=fixture_file(directory,name,"wb"); assert(fwrite(data,1,bytes,f)==bytes); assert(!fclose(f));
}
static void compose_ok(Qwen38Status rc, Qwen38Error *e) {
    if (rc!=Q38_OK) fprintf(stderr,"C4 status=%d: %s\n",rc,e->detail);
    assert(rc==Q38_OK);
}
static int compose_layer(const char *path, uint32_t index, const char *input,
        const char *output, size_t tokens, size_t chunk) {
    assert(tokens && tokens<=129 && chunk<=129); Qwen38Error e;
    Gguf gguf; gguf_open(path,&gguf); Qwen38Config *config=NULL; Qwen38Weights *weights=NULL;
    compose_ok(qwen38_config_load(&gguf,&config,&e),&e);
    compose_ok(qwen38_weights_describe(&gguf,config,&weights,&e),&e);
    Q38CLayer l; compose_ok(q38c_layer_plan(config,weights,index,&l,&e),&e);
    Qwen38CpuSpan packed[Q38CW_COUNT]; size_t total=0;
    for (size_t i=0;i<Q38CW_COUNT;++i) {
        size_t bytes=l.weight[i].layout.tensor_bytes; total+=bytes; assert(total<=96*1024*1024);
        void *p=malloc(bytes); assert(p); char name[64]; snprintf(name,sizeof(name),"weight-%zu.bin",i);
        fixture_read(input,name,p,bytes); packed[i]=(Qwen38CpuSpan){p,bytes};
    }
    Qwen38Memory constants=backing((l.constant_bytes+63)&~(size_t)63);
    compose_ok(q38c_layer_bind(&l,packed,constants,&e),&e);
    assert(q38c_layer_bind(&l,packed,constants,&e)==Q38_BUSY);
    size_t work_bytes; compose_ok(q38c_work_size(&l,tokens,&work_bytes,&e),&e);
    Qwen38Memory memory=backing(work_bytes); Q38CWork work;
    compose_ok(q38c_work_init(&l,tokens,memory,&work,&e),&e);
    Qwen38StateSize cs,gs; compose_ok(qwen38_conv_state_bytes(&l.conv,&cs,&e),&e);
    compose_ok(qwen38_gdn_state_bytes(&l.config.gdn,&gs,&e),&e);
    Qwen38Memory cm=backing(cs.metadata_bytes),cp=backing(cs.payload_bytes),gm=backing(gs.metadata_bytes),gp=backing(gs.payload_bytes);
    Qwen38ConvState *conv; Qwen38GdnState *gdn; Qwen38StateKey key={.model=config,.layer=index,.epoch=1};
    compose_ok(qwen38_conv_state_init(&l.conv,key,cm,cp,&conv,&e),&e);
    compose_ok(qwen38_gdn_state_init(&l.config.gdn,key,gm,gp,&gdn,&e),&e);
    Q38CBlockState state; compose_ok(q38c_block_init(&l,conv,gdn,&state,&e),&e);
    size_t h=l.config.hc.streams, width=l.config.hidden_size, bytes=tokens*h*width*4;
    float *r=malloc(bytes),*saved=malloc(bytes); assert(r && saved);
    Qwen38F32Mut residual=view(r,bytes,3,(size_t[]){tokens,h,width});
    Qwen38F32Mut empty=residual; empty.dim[0]=0;
    assert(q38c_block_cpu(&l,&work,empty,&state,1,&e)==Q38_BAD_STATE);
    compose_ok(q38c_block_cpu(&l,&work,empty,&state,0,&e),&e);
    state.busy=true; assert(q38c_block_cpu(&l,&work,empty,&state,0,&e)==Q38_BUSY); state.busy=false;
    Qwen38F32Mut original=work.trace[Q38CT_LOW]; work.trace[Q38CT_LOW].data=r;
    assert(q38c_block_cpu(&l,&work,residual,&state,0,&e)==Q38_INVALID_ARGUMENT); work.trace[Q38CT_LOW]=original;
    assert(qwen38_conv_state_info(conv).next_pos==0 && qwen38_gdn_state_info(gdn).next_pos==0);
    for (size_t replay=0;replay<2;++replay) {
        fixture_read(input,"input.f32",r,bytes); FILE *files[Q38CT_COUNT];
        for (size_t j=0;j<Q38CT_COUNT;++j) {
            char name[64]; snprintf(name,sizeof(name),"%s.f32",q38c_stage_names[j]);
            files[j]=fixture_file(output,name,replay ? "rb" : "wb");
        }
        for (size_t pos=0;pos<tokens;) {
            size_t n=chunk ? 1+(pos%chunk) : tokens; if (n>tokens-pos) n=tokens-pos;
            Qwen38F32Mut part=residual; part.data=r+pos*h*width; part.capacity_bytes=bytes-pos*h*width*4; part.dim[0]=n;
            compose_ok(q38c_block_cpu(&l,&work,part,&state,pos,&e),&e);
            empty.dim[0]=0; compose_ok(q38c_block_cpu(&l,&work,empty,&state,pos+n,&e),&e);
            for (size_t j=0;j<Q38CT_COUNT;++j) {
                size_t count=n*work.trace[j].dim[1];
                if (!replay) assert(fwrite(work.trace[j].data,4,count,files[j])==count);
                else for (size_t k=0;k<count;++k) { float old; assert(fread(&old,4,1,files[j])==1); assert(!memcmp(&old,work.trace[j].data+k,4)); }
            }
            pos+=n;
        }
        for (size_t j=0;j<Q38CT_COUNT;++j) assert(!fclose(files[j]));
        assert(qwen38_conv_state_info(conv).next_pos==tokens && qwen38_gdn_state_info(gdn).next_pos==tokens);
        if (!replay) { memcpy(saved,r,bytes); compose_ok(q38c_block_reset(&state,2,&e),&e); }
        else assert(!memcmp(saved,r,bytes));
    }
    fixture_write(output,"residual.f32",r,bytes);
    size_t history_n=(l.conv.kernel-1)*l.conv.channels;
    size_t state_n=l.config.gdn.value_heads*l.config.gdn.value_dim*l.config.gdn.key_dim;
    float *history=malloc(history_n*4),*final=malloc(state_n*4); assert(history && final);
    compose_ok(qwen38_conv_state_export(conv,view(history,history_n*4,2,(size_t[]){l.conv.kernel-1,l.conv.channels}),&e),&e);
    compose_ok(qwen38_gdn_state_export(gdn,view(final,state_n*4,3,(size_t[]){l.config.gdn.value_heads,l.config.gdn.value_dim,l.config.gdn.key_dim}),&e),&e);
    fixture_write(output,"history.f32",history,history_n*4); fixture_write(output,"state.f32",final,state_n*4);
    assert(q38c_block_reset(&state,2,&e)==Q38_STALE); compose_ok(q38c_block_reset(&state,3,&e),&e);
    /* Fault after conv/scan advanced: reject continuation until a newer reset. */
    fixture_read(input,"input.f32",r,bytes); float *gain=(float *)l.constant[Q38CW_GN].data, old_gain=*gain; *gain=NAN;
    Qwen38F32Mut one=residual; one.dim[0]=1;
    assert(q38c_block_cpu(&l,&work,one,&state,0,&e)==Q38_NONFINITE && !state.valid); *gain=old_gain;
    assert(qwen38_conv_state_info(conv).next_pos==1 && qwen38_gdn_state_info(gdn).next_pos==1);
    assert(q38c_block_cpu(&l,&work,empty,&state,1,&e)==Q38_BAD_STATE);
    compose_ok(q38c_block_reset(&state,4,&e),&e);
    compose_ok(qwen38_gdn_state_export(gdn,view(final,state_n*4,3,(size_t[]){l.config.gdn.value_heads,l.config.gdn.value_dim,l.config.gdn.key_dim}),&e),&e);
    for (size_t i=0;i<state_n;++i) assert(final[i]==0);
    printf("{\"tokens\":%zu,\"workspace_bytes\":%zu,\"packed_bytes\":%zu,\"state_bytes\":%zu,\"reset_replay_bitwise\":true,\"post_update_error_reset\":true}\n",
        tokens,work_bytes,total,cp.bytes+gp.bytes);
    free(history);free(final);free(r);free(saved);free(cm.data);free(cp.data);free(gm.data);free(gp.data);free(memory.data);free(constants.data);
    for (size_t i=0;i<Q38CW_COUNT;++i) free((void *)packed[i].data);
    qwen38_weights_free(weights);qwen38_config_free(config);gguf_close(&gguf);return 0;
}
int main(int argc, char **argv) {
    if (argc==4 && !strcmp(argv[1],"--describe")) {
        char *end; unsigned long layer=strtoul(argv[3],&end,10);
        assert(*argv[3] && !*end && layer<=UINT32_MAX);
        return describe_layer(argv[2],(uint32_t)layer);
    }
    if (argc==8 && !strcmp(argv[1],"--compose"))
        return compose_layer(argv[2],(uint32_t)strtoul(argv[3],NULL,10),argv[4],argv[5],
            strtoul(argv[6],NULL,10),strtoul(argv[7],NULL,10));
    assert(argc==1);
    test_views(); test_conv_lifecycle(); test_conv_values(); test_norm_activation();
    test_hc(); test_gdn_lifecycle(); test_gdn_prepare(); test_gdn_scan();
    test_admission_and_guards();
    puts("PASS: Qwen3.8 C conv, GDN, RMS, activations, HC and state"); return 0;
}
