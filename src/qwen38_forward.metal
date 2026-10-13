#include <metal_stdlib>
using namespace metal;
/* Graph wiring only: copy snapshots, or repeat each embedding over H streams. */
kernel void q38g_copy(device const float *x [[buffer(0)]],device float *y [[buffer(1)]],
    constant uint4 &a [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if (i<a.x) y[i]=x[a.w ? (i/(a.y*a.z))*a.y+i%a.y : i];
}
/* [e,0], [0,h_0], ..., [0,h_H-1] for each token. */
kernel void q38g_mtp_cat(device const float *en [[buffer(0)]],device const float *hn [[buffer(1)]],
    device float *cat [[buffer(2)]],constant uint4 &a [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if (i>=a.x) return;
    uint d=i%(2*a.y), row=i/(2*a.y), stream=row%(a.z+1), t=row/(a.z+1);
    cat[i]=stream ? (d<a.y ? 0 : hn[(t*a.z+stream-1)*a.y+d-a.y]) : (d<a.y ? en[t*a.y+d] : 0);
}
kernel void q38g_mtp_sum(device const float *proj [[buffer(0)]],device float *r [[buffer(1)]],
    constant uint4 &a [[buffer(2)]],device atomic_uint *error [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if (i>=a.x) return;
    uint d=i%a.y, stream=(i/a.y)%a.z, t=i/(a.y*a.z);
    float value=proj[(t*(a.z+1)+stream+1)*a.y+d]+proj[t*(a.z+1)*a.y+d];
    if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    r[i]=value;
}
