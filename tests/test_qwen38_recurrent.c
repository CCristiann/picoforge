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
int main(void) {
    test_views(); test_conv_lifecycle(); test_conv_values(); test_norm_activation();
    test_hc(); puts("PASS: Qwen3.8 C conv, state, RMS, activations and HC"); return 0;
}
