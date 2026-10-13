/* Conservative FP32 stages. No half cache, fast math or per-tile normalization. */
#include <metal_stdlib>
using namespace metal;
#include "qwen38_d_internal.h"
static void fail(device atomic_uint *e, uint bit) {
    atomic_fetch_or_explicit(e,bit,memory_order_relaxed);
}
kernel void q38d_rope_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant float *frequencies [[buffer(8)]],
    uint h [[thread_position_in_grid]]) {
    for (uint d=0;d<a.d;++d) {
        float value=x[h*a.d+d];
        if (!isfinite(value)) { fail(error,1); return; }
        if (d<a.r) {
            uint i=d%(a.r/2); float frequency=frequencies[i];
            float angle=float(a.pos)*frequency;
            float other=x[h*a.d+(d<a.r/2 ? d+a.r/2 : d-a.r/2)];
            value=value*cos(angle)+(d<a.r/2 ? -other : other)*sin(angle);
        }
        if (!isfinite(value)) { fail(error,1); return; }
        y[h*a.d+d]=value;
    }
}
kernel void q38d_pool_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint d [[thread_position_in_grid]]) {
    float sum=0;
    for (uint t=0;t<a.n;++t) {
        float x0=x[t*a.d+d]; if (!isfinite(x0)) { fail(error,1); return; } sum+=x0;
    }
    float v=sum/float(a.n); if (!isfinite(v)) { fail(error,1); return; } y[d]=v;
}
kernel void q38d_scores_gpu(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device float *out [[buffer(2)]], constant Q38DGpuArgs &a [[buffer(6)]],
    device atomic_uint *error [[buffer(7)]], uint b [[thread_position_in_grid]]) {
    float score=0;
    for (uint h=0;h<a.h;++h) {
        float dot=0;
        for (uint d=0;d<a.d;++d) {
            float x=q[h*a.d+d],y=k[b*a.d+d];
            if (!isfinite(x)||!isfinite(y)) { fail(error,1); return; } dot+=x*y;
        }
        if (!isfinite(dot)) { fail(error,1); return; } score+=max(0.0f,dot);
    }
    if (!isfinite(score)) { fail(error,1); return; } out[b]=score;
}
kernel void q38d_attend_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint h [[thread_position_in_grid]]) {
    for (uint i=0;i<a.m;++i)
        if (ids[i]>=a.n || (i && ids[i]<=ids[i-1])) { fail(error,2); return; }
    float acc[256],logits[16],maximum=-INFINITY,denominator=0;
    for (uint d=0;d<a.d;++d) {
        float x=q[h*a.d+d],g=gate[h*a.d+d];
        if (!isfinite(x)||!isfinite(g)) { fail(error,1); return; }
        if (g<0 || g>1) { fail(error,2); return; } acc[d]=0;
    }
    float scale=1.0f/sqrt(float(a.d));
    for (uint base=0;base<a.m;base+=16) {
        uint tile=min(16u,a.m-base); float next=maximum;
        for (uint i=0;i<tile;++i) {
            uint at=(ids[base+i]*a.j+h/(a.h/a.j))*a.d; float dot=0;
            for (uint d=0;d<a.d;++d) {
                float key=k[at+d]; if (!isfinite(key)) { fail(error,1); return; }
                dot+=q[h*a.d+d]*key;
            }
            logits[i]=dot*scale;
            if (!isfinite(logits[i])) { fail(error,1); return; } next=max(next,logits[i]);
        }
        float rescale=denominator==0 ? 0 : exp(maximum-next);
        denominator*=rescale;
        for (uint d=0;d<a.d;++d) acc[d]*=rescale;
        for (uint i=0;i<tile;++i) {
            float weight=exp(logits[i]-next); denominator+=weight;
            uint at=(ids[base+i]*a.j+h/(a.h/a.j))*a.d;
            for (uint d=0;d<a.d;++d) {
                float value=v[at+d]; if (!isfinite(value)) { fail(error,1); return; }
                acc[d]+=weight*value;
            }
        }
        maximum=next;
    }
    if (!isfinite(denominator)||denominator<=0) { fail(error,1); return; }
    for (uint d=0;d<a.d;++d) {
        float y=acc[d]/denominator*gate[h*a.d+d];
        if (!isfinite(y)) { fail(error,1); return; } out[h*a.d+d]=y;
    }
}
/* Device block selection, equivalent to q38d_select: block b is kept when
 * fewer than a.m blocks beat it (higher score, or equal score and smaller ID).
 * Kept blocks are then expanded in time order, followed by the causal tail. */
kernel void q38d_rank_gpu(device const float *scores [[buffer(0)]], device uint *keep [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint b [[thread_position_in_grid]]) {
    if (b>=a.n) return;
    keep[b]=0;
    float s=scores[b]; uint rank=0;
    for (uint c=0;c<a.n;++c) {
        float o=scores[c];
        /* Every lane checks the same scores. NaN must not inflate the kept
         * set beyond the admitted ID capacity or reuse an old keep flag. */
        if (!isfinite(o)) { fail(error,1); return; }
        rank+=(o>s || (o==s && c<b)) ? 1u : 0u;
    }
    keep[b]=rank<a.m ? 1u : 0u;
}
kernel void q38d_compact_gpu(device const uint *keep [[buffer(0)]], device uint *ids [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint b [[thread_position_in_grid]]) {
    if (b>=a.n || !keep[b]) return;
    uint slot=0;
    for (uint c=0;c<b;++c) slot+=keep[c];
    if (slot>=a.m) { fail(error,2); return; }
    for (uint r=0;r<a.r;++r) ids[slot*a.r+r]=b*a.r+r;
}
/* a.m = first tail slot, a.n = first tail token, a.pos = last visible token. */
kernel void q38d_tail_gpu(device uint *ids [[buffer(0)]], constant Q38DGpuArgs &a [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (a.n+i<=a.pos) ids[a.m+i]=a.n+i;
}
kernel void q38d_copy_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=a.n) return;
    float v=x[i];
    if (!isfinite(v)) fail(error,1);
    y[i]=v;
}
