#include <metal_stdlib>
using namespace metal;
struct FArgs { uint t,h,k,xpitch,ypitch; };
kernel void q38f_gather(device const float *x [[buffer(0)]],device float *out [[buffer(1)]],
    device const uint *map [[buffer(2)]],constant FArgs &a [[buffer(3)]],
    device atomic_uint *error [[buffer(4)]],uint index [[thread_position_in_grid]]) {
    uint row=index/a.h,col=index%a.h;
    if (row>=a.t*a.k) return;
    uint pair=map[row];
    if (pair>=a.t*a.k) { atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return; }
    float v=x[(pair/a.k)*a.xpitch+col];
    if (!isfinite(v)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    out[index]=v;
}
kernel void q38f_product(device float *act [[buffer(0)]],device const float *up [[buffer(1)]],
    device atomic_uint *error [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    float v=act[index]*up[index];
    if (!isfinite(v)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    act[index]=v;
}
kernel void q38f_reduce(device const float *down [[buffer(0)]],device const float *shared [[buffer(1)]],
    device const float *sigmoid [[buffer(2)]],device const float *scores [[buffer(3)]],
    device const uint *map [[buffer(4)]],device float *y [[buffer(5)]],
    constant FArgs &a [[buffer(6)]],device atomic_uint *error [[buffer(7)]],
    device float *routed [[buffer(8)]],uint index [[thread_position_in_grid]]) {
    uint t=index/a.h,col=index%a.h;
    if (t>=a.t) return;
    float sum=0;
    for (uint s=0;s<a.k;s++) {
        uint pair=t*a.k+s,row=map[pair];
        if (row>=a.t*a.k) { atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return; }
        sum+=scores[pair]*down[row*a.h+col];
    }
    routed[index]=sum;
    sum+=sigmoid[t]*shared[index];
    if (!isfinite(sum)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    y[t*a.ypitch+col]=sum;
}
/* Private diagnostic copy: the harness allocates and retains destinations. */
kernel void q38f_trace_copy(device const float *x [[buffer(0)]],device float *y [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    y[index]=x[(index/shape.x)*shape.y+index%shape.x];
}
/* Device routing, same order as qwen38_moe_route: max, exp-sum, divide, then
 * K ascending scans with strict > (equal probabilities keep the smaller ID). */
struct RouteArgs { uint t,e,k,norm,pitch; };
kernel void q38f_route(device const float *logits [[buffer(0)]],device float *probs [[buffer(1)]],
    device uint *ids [[buffer(2)]],device float *weights [[buffer(3)]],
    constant RouteArgs &a [[buffer(4)]],device atomic_uint *error [[buffer(5)]],
    uint t [[thread_position_in_grid]]) {
    if (t>=a.t) return;
    device const float *z=logits+t*a.pitch; device float *p=probs+t*a.e;
    device uint *id=ids+t*a.k; device float *w=weights+t*a.k;
    for (uint e=0;e<a.e;e++) if (!isfinite(z[e])) {
        atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        for (uint s=0;s<a.k;s++) { id[s]=s; w[s]=0; }
        return;
    }
    float maximum=z[0],sum=0;
    for (uint e=1;e<a.e;e++) if (z[e]>maximum) maximum=z[e];
    for (uint e=0;e<a.e;e++) { p[e]=exp(z[e]-maximum); sum+=p[e]; }
    for (uint e=0;e<a.e;e++) p[e]/=sum;
    float selected=0;
    for (uint s=0;s<a.k;s++) {
        uint best=0; bool found=false;
        for (uint e=0;e<a.e;e++) {
            bool used=false;
            for (uint j=0;j<s;j++) used|=id[j]==e;
            if (!used && (!found || p[e]>p[best])) { best=e; found=true; }
        }
        id[s]=best; w[s]=p[best]; selected+=w[s];
    }
    if (a.norm) for (uint s=0;s<a.k;s++) w[s]/=selected;
}
/* Rows are pairs directly (no grouping): the reduce map is the identity. */
kernel void q38f_iota(device uint *map [[buffer(0)]],uint i [[thread_position_in_grid]]) { map[i]=i; }
