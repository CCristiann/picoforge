#include <metal_stdlib>
using namespace metal;
kernel void q38e_gate_gpu(device const float *key [[buffer(0)]],
    device const float *query [[buffer(1)]], device const float *value [[buffer(2)]],
    device float *gate [[buffer(3)]], device float *u [[buffer(4)]],
    constant uint4 &a [[buffer(6)]], device atomic_uint *err [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i>=a.x*a.y || atomic_load_explicit(err,memory_order_relaxed)) return;
    float dot=0;
    for (uint d=0;d<a.z;++d) dot+=key[i*a.z+d]*query[i*a.z+d];
    dot/=sqrt(float(a.z));
    if (!isfinite(dot)) { atomic_fetch_or_explicit(err,1u,memory_order_relaxed); return; }
    float decay=exp(-sqrt(max(abs(dot),1e-6f)));
    float g=dot==0 ? 0.5f : dot>0 ? 1.0f/(1.0f+decay) : decay/(1.0f+decay);
    gate[i]=g;
    for (uint d=0;d<a.z;++d) {
        float v=g*value[(i/a.y)*a.z+d];
        if (!isfinite(v)) atomic_fetch_or_explicit(err,1u,memory_order_relaxed);
        u[i*a.z+d]=v;
    }
}
kernel void q38e_add_gpu(device const float *x [[buffer(0)]], device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint4 &a [[buffer(6)]],
    device atomic_uint *err [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i>=a.x || atomic_load_explicit(err,memory_order_relaxed)) return;
    float v=x[i]+y[i];
    if (!isfinite(v)) atomic_fetch_or_explicit(err,1u,memory_order_relaxed);
    out[i]=v;
}
