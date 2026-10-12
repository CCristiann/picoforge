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
int main(void) {
    test_views(); test_conv_lifecycle(); test_conv_values(); puts("PASS: Qwen3.8 C views, conv and lifecycle"); return 0;
}
