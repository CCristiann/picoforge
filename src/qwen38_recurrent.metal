#include <metal_stdlib>
using namespace metal;
#include "qwen38_c_internal.h"
static_assert(sizeof(Q38CGpuView)==40 && sizeof(Q38CGpuArgs)==352,"C/Metal argument ABI");
#define Q38C_BUFFERS device float *v0 [[buffer(0)]], device float *v1 [[buffer(1)]], \
    device float *v2 [[buffer(2)]], device float *v3 [[buffer(3)]], \
    device float *v4 [[buffer(4)]], device float *v5 [[buffer(5)]], \
    device float *v6 [[buffer(6)]], device float *v7 [[buffer(7)]], \
    constant Q38CGpuArgs &a [[buffer(8)]], device atomic_uint *error [[buffer(9)]], \
    uint i [[thread_position_in_grid]]
static uint at(constant Q38CGpuView &v, uint index) {
    uint offset=0;
    for (uint j=v.rank;j-- >0;) { offset+=(index%v.dim[j])*v.stride[j]; index/=v.dim[j]; }
    return offset;
}
static float sigmoid(float x) {
    if (x>=0.0f) return 1.0f/(1.0f+exp(-x));
    float z=exp(x); return z/(1.0f+z);
}
/* MSL has no log1p. Compensate the rounding of 1+x (here 0<=x<=1),
 * retaining the small positive tail when 1+x rounds to exactly one. */
static float softplus(float x) {
    float tail=exp(-abs(x)), z=1.0f+tail;
    return max(x,0.0f)+(log(z)-((z-1.0f)-tail)/z);
}
static bool valid(float x, device atomic_uint *error) {
    if (isfinite(x)) return true;
    atomic_fetch_or_explicit(error,1u,memory_order_relaxed); return false;
}
kernel void q38c_check(Q38C_BUFFERS) {
    device float *v[]={v0,v1,v2,v3,v4,v5,v6,v7};
    for (uint j=0;j<8;++j)
        if ((a.read_mask&(1u<<j)) && i<a.v[j].count) valid(v[j][at(a.v[j],i)],error);
}
kernel void q38c_activate(Q38C_BUFFERS) {
    if (i>=a.v[0].count || atomic_load_explicit(error,memory_order_relaxed)) return;
    float x=v0[at(a.v[0],i)], y;
    if (a.p[0]==0) y=sigmoid(x);
    else if (a.p[0]==1) y=x*sigmoid(x);
    else y=softplus(x);
    if (valid(y,error)) v1[at(a.v[1],i)]=y;
}
