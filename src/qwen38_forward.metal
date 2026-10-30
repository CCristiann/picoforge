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
/* One threadgroup over one logits row (T15c). Each thread keeps the first maximum of its
 * strided slice; the tree keeps the larger value, and on a tie the smaller index, so the
 * result is the FIRST maximum whatever the slice split. Non-finite logits never compete
 * and set error bit 1. Idea: ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_argmax. */
kernel void q38g_argmax(device const float *x [[buffer(0)]],device int *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],device atomic_uint *error [[buffer(3)]],uint t [[thread_index_in_threadgroup]]) {
    threadgroup float bv[1024]; threadgroup uint bi[1024];
    float v=-INFINITY; uint idx=0xffffffffu; bool bad=false;
    for (uint i=t;i<n;i+=1024) {
        float c=x[i];
        if (!isfinite(c)) { bad=true; continue; }
        if (idx==0xffffffffu || c>v) { v=c; idx=i; }
    }
    bv[t]=v; bi[t]=idx;
    for (uint s=512;s>0;s>>=1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t<s && (bv[t+s]>bv[t] || (bv[t+s]==bv[t] && bi[t+s]<bi[t]))) { bv[t]=bv[t+s]; bi[t]=bi[t+s]; }
    }
    if (bad) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    if (t==0) out[0]=bi[0]==0xffffffffu ? 0 : (int)bi[0];
}
