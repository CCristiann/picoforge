#include <metal_stdlib>
using namespace metal;
#include "qwen38_c_internal.h"
static_assert(sizeof(Q38CGpuView)==40 && sizeof(Q38CGpuArgs)==352,"C/Metal argument ABI");
static_assert(__builtin_offsetof(Q38CGpuArgs,p)==320 && __builtin_offsetof(Q38CGpuArgs,eps)==348,"C/Metal offsets");
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
    for (uint j=0;j<8;++j) if ((a.read_mask&(1u<<j)) && i<a.v[j].count) {
        float x=v[j][at(a.v[j],i)]; valid(x,error);
        if (a.p[4]==1 && j==3 && x>0) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        if (a.p[4]==2 && ((j==1 && x>0) || (j==2 && (x<0 || x>1))))
            atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
    }
}
kernel void q38c_activate(Q38C_BUFFERS) {
    if (i>=a.v[0].count || atomic_load_explicit(error,memory_order_relaxed)) return;
    float x=v0[at(a.v[0],i)]/float(a.p[1]), y;
    if (a.p[0]==0) y=sigmoid(x);
    else if (a.p[0]==1) y=x*sigmoid(x);
    else y=softplus(x);
    y*=float(a.p[2]);
    if (valid(y,error)) v1[at(a.v[1],i)]=y;
}
kernel void q38c_rms(Q38C_BUFFERS) {
    uint width=a.v[0].dim[2], groups=a.v[0].dim[1];
    if (i>=a.v[0].count/width || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint xi=at(a.v[0],i*width), yi=at(a.v[2],i*width);
    uint wi=a.v[1].rank==1 ? 0 : (i%groups)*a.v[1].stride[0];
    float sum=0;
    for (uint j=0;j<width;++j) sum+=v0[xi+j]*v0[xi+j];
    float variance=sum/float(width)+a.eps;
    if (!valid(sum,error) || !valid(variance,error)) return;
    float scale=1.0f/sqrt(variance);
    for (uint j=0;j<width;++j) {
        float y=(v0[xi+j]*scale)*v1[wi+j];
        if (!valid(y,error)) return;
        if (a.p[0]) y*=sigmoid(v3[at(a.v[3],i*width+j)]);
        v2[yi+j]=y;
    }
}
/* R1.6a: q38c_rms with one Q38C_RMS_THREADS threadgroup per row instead of
 * one thread (at one token the HC norm [1,4,2560] was 4 threads, each a
 * serial pass over 2560 floats).
 * The sum order depends only on width, never on rows or position: thread tid
 * adds x[j]^2 for j = tid, tid+256, ... in ascending j, simd_sum joins its 32
 * lanes, thread 0 adds the 8 simdgroup sums in ascending order. Variance,
 * scale, y and the GDN gate are q38c_rms's expressions. The input check is
 * folded in: each row reads all its x (and z) and a whole gain row, the rows
 * together every element, so a nonfinite input sets bit 1 as q38c_check does,
 * even when the word is already set; the op's own bits (sum, variance, y)
 * and the writes are skipped when the word was set before, as in q38c_rms.
 * Outputs are unspecified when the word ends nonzero (a nonfinite input: the
 * oracle writes nothing, this kernel may write other rows and elements); the
 * session then discards the command. The sum order differs from q38c_rms, so
 * the two can disagree on overflow (bit 1) only when the sum of squares is
 * within about width*2^-24 (relative) of FLT_MAX; y has a 1-ulp-of-scale band.
 * Idea (threadgroup per row, strided squares, simd_sum, ordered simdgroup
 * combine): ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_hc_norm. */
kernel void q38c_rms_fast(device float *v0 [[buffer(0)]], device float *v1 [[buffer(1)]],
        device float *v2 [[buffer(2)]], device float *v3 [[buffer(3)]],
        constant Q38CGpuArgs &a [[buffer(8)]], device atomic_uint *error [[buffer(9)]],
        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[Q38C_RMS_THREADS/32+1]; /* simdgroup sums, then the row's sum */
    uint width=a.v[0].dim[2], groups=a.v[0].dim[1];
    if (row>=a.v[0].count/width) return; /* whole threadgroups only: no barrier is skipped by part of one */
    bool skip=atomic_load_explicit(error,memory_order_relaxed)!=0, bad=false;
    uint xi=at(a.v[0],row*width), yi=at(a.v[2],row*width), zi=a.p[0] ? at(a.v[3],row*width) : 0;
    uint wi=a.v[1].rank==1 ? 0 : (row%groups)*a.v[1].stride[0];
    float sum=0;
    for (uint j=tid;j<width;j+=Q38C_RMS_THREADS) { float x=v0[xi+j]; bad|=!isfinite(x); sum+=x*x; }
    sum=simd_sum(sum);
    if (!lane) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup); /* also: every x of the row is read before any y */
    if (!tid) { float s=0; for (uint k=0;k<Q38C_RMS_THREADS/32;++k) s+=partial[k]; partial[Q38C_RMS_THREADS/32]=s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=partial[Q38C_RMS_THREADS/32];
    float variance=total/float(width)+a.eps;
    bool ok=!skip && isfinite(total) && isfinite(variance);
    if (!skip && !tid && !ok) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    float scale=1.0f/sqrt(variance);
    for (uint j=tid;j<width;j+=Q38C_RMS_THREADS) {
        float g=v1[wi+j], z=a.p[0] ? v3[zi+j] : 0.0f;
        bad|=!isfinite(g) || !isfinite(z);
        float y=(v0[xi+j]*scale)*g;
        if (!ok) continue;
        if (!valid(y,error)) continue;
        if (a.p[0]) y*=sigmoid(z);
        v2[yi+j]=y;
    }
    if (bad) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
}
kernel void q38c_hc_read(Q38C_BUFFERS) {
    if (i>=a.v[2].count || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint width=a.v[0].dim[2], h=a.v[0].dim[1], t=i/width, j=i%width; float sum=0;
    for (uint k=0;k<h;++k) {
        uint n=(t*h+k)*width+j; sum+=v0[at(a.v[0],n)]*sigmoid(v1[at(a.v[1],n)]);
    }
    if (valid(sum,error)) v2[at(a.v[2],i)]=sum/float(h);
}
kernel void q38c_hc_add(Q38C_BUFFERS) {
    if (i>=a.v[0].count || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint width=a.v[0].dim[2], h=a.v[0].dim[1], th=i/width;
    float y=v0[at(a.v[0],i)]+v2[at(a.v[2],th)]*v1[at(a.v[1],(th/h)*width+i%width)];
    if (valid(y,error)) v0[at(a.v[0],i)]=y;
}
kernel void q38c_conv(Q38C_BUFFERS) {
    if (i>=a.v[0].count || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint c=a.v[0].dim[1], t=i/c, channel=i%c, l=a.v[3].dim[0]; float sum=0;
    for (uint j=0;j<a.p[0];++j) {
        ulong row=ulong(t)+ulong(j)*a.p[1];
        float x=row<l ? v3[at(a.v[3],uint(row)*c+channel)] : v0[at(a.v[0],uint(row-l)*c+channel)];
        sum+=v1[at(a.v[1],channel*a.p[0]+j)]*x;
    }
    if (!valid(sum,error)) return;
    float y=a.p[2] ? sum*sigmoid(sum) : sum;
    if (valid(y,error)) v2[at(a.v[2],i)]=y;
}
/* One thread owns a channel. Ascending history rows read strictly later old
 * rows, so a short append needs no scratch copy and cannot race another lane. */
kernel void q38c_history(Q38C_BUFFERS) {
    uint c=a.v[0].dim[1], t=a.v[0].dim[0], l=a.v[3].dim[0];
    if (i>=c || atomic_load_explicit(error,memory_order_relaxed)) return;
    for (uint j=0;j<l;++j) {
        ulong row=ulong(t)+j;
        float x=row<l ? v3[at(a.v[3],uint(row)*c+i)] : v0[at(a.v[0],uint(row-l)*c+i)];
        v3[at(a.v[3],j*c+i)]=x;
    }
}
kernel void q38c_gdn_prepare(Q38C_BUFFERS) {
    if (i>=a.v[0].dim[0] || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint hk=a.p[0], hv=a.p[1], dk=a.p[2], dv=a.p[3];
    uint src=i*a.v[0].stride[0], dst=i*a.v[5].stride[0];
    float qscale=1.0f/sqrt(float(dk));
    for (uint h=0;h<2*hk;++h) {
        uint begin=h*dk; float sum=0;
        for (uint j=0;j<dk;++j) sum+=v0[src+begin+j]*v0[src+begin+j];
        float squared=sum+a.eps;
        if (!valid(squared,error)) return;
        float inv=1.0f/sqrt(squared), scale=h<hk ? qscale : 1.0f;
        for (uint j=0;j<dk;++j) v5[dst+begin+j]=(v0[src+begin+j]*inv)*scale;
    }
    for (uint j=0;j<hv*dv;++j) v5[dst+2*hk*dk+j]=v0[src+2*hk*dk+j];
    for (uint h=0;h<hv;++h) {
        float value=v1[at(a.v[1],i*hv+h)]+v4[h];
        if (!valid(value,error)) return;
        float g=v3[h]*softplus(value);
        if (!valid(g,error)) return;
        v6[at(a.v[6],i*hv+h)]=g;
        v7[at(a.v[7],i*hv+h)]=sigmoid(v2[at(a.v[2],i*hv+h)]);
    }
}
/* R1.8: q38c_gdn_prepare on a [ceil(2hk/8), t] grid of Q38C_RMS_THREADS
 * threadgroups instead of one thread per token (at one token a single thread
 * did 32 L2 norms of 128 floats, a 6144-float copy and 48 gates, serially).
 * Simdgroup s of a token's groups owns q/k head s: lane l adds x[j]^2 for
 * j = l, l+32, ... in ascending j and simd_sum joins the 32 lanes, so the
 * order depends only on dk, never on t or position. The value copy and the
 * gates are strided over the token's threads. Expressions (sum+eps, 1/sqrt,
 * (x*inv)*scale, softplus, sigmoid) are q38c_gdn_prepare's. Every element is
 * read and written by one thread, so the admitted in-place aliases need no
 * barrier. The input check is folded in as in q38c_rms_fast: a nonfinite
 * qkv, alpha, br, A or dt element sets bit 1 and A > 0 bit 4 even when the
 * word is already set; the op's own bits and the writes are skipped when the
 * word was set before. Outputs are unspecified when the word ends nonzero
 * (the oracle stops a token at its first fault, this kernel writes the other
 * heads); the word itself can differ only when A > 0 meets an overflow of the
 * op (5 here, 4 for the oracle, which then skips the op).
 * Idea (GDN front per head, simdgroup L2 norm): ds4 fc80bd6, MIT,
 * metal/qwen4.metal:kernel_qwen4_gdn_front and kernel_qwen4_gdn_prep. */
kernel void q38c_gdn_prepare_fast(device float *v0 [[buffer(0)]], device float *v1 [[buffer(1)]],
        device float *v2 [[buffer(2)]], device float *v3 [[buffer(3)]], device float *v4 [[buffer(4)]],
        device float *v5 [[buffer(5)]], device float *v6 [[buffer(6)]], device float *v7 [[buffer(7)]],
        constant Q38CGpuArgs &a [[buffer(8)]], device atomic_uint *error [[buffer(9)]],
        uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint hk=a.p[0], hv=a.p[1], dk=a.p[2], dv=a.p[3], t=group.y, keys=2*hk*dk;
    uint src=t*a.v[0].stride[0], dst=t*a.v[5].stride[0];
    uint first=group.x*Q38C_RMS_THREADS+tid, threads=groups.x*Q38C_RMS_THREADS;
    bool skip=atomic_load_explicit(error,memory_order_relaxed)!=0, bad=false, fault=false;
    float qscale=1.0f/sqrt(float(dk));
    for (uint h=first/32;h<2*hk;h+=threads/32) { /* uniform per simdgroup: simd_sum is safe */
        float sum=0;
        for (uint j=lane;j<dk;j+=32) { float x=v0[src+h*dk+j]; bad|=!isfinite(x); sum+=x*x; }
        float squared=simd_sum(sum)+a.eps;
        bool ok=!skip && isfinite(squared);
        fault|=!skip && !ok;
        float inv=1.0f/sqrt(squared), scale=h<hk ? qscale : 1.0f;
        for (uint j=lane;ok && j<dk;j+=32) v5[dst+h*dk+j]=(v0[src+h*dk+j]*inv)*scale;
    }
    for (uint j=first;j<hv*dv;j+=threads) {
        float x=v0[src+keys+j]; bad|=!isfinite(x);
        if (!skip) v5[dst+keys+j]=x;
    }
    for (uint h=first;h<hv;h+=threads) {
        uint n=t*hv+h; float A=v3[h], dt=v4[h], alpha=v1[at(a.v[1],n)], br=v2[at(a.v[2],n)];
        bad|=!isfinite(A) || !isfinite(dt) || !isfinite(alpha) || !isfinite(br);
        if (A>0) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        float value=alpha+dt, g=A*softplus(value);
        if (skip) continue;
        if (!isfinite(value) || !isfinite(g)) { fault=true; continue; }
        v6[at(a.v[6],n)]=g;
        v7[at(a.v[7],n)]=sigmoid(br);
    }
    if (bad || fault) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
}
/* Each lane owns an entire (value-head,value-channel) state row. FP32 and
 * sequential feature/time order deliberately mirror C before optimization. */
kernel void q38c_gdn_scan(Q38C_BUFFERS) {
    uint hk=a.p[0], hv=a.p[1], dk=a.p[2], dv=a.p[3];
    if (i>=hv*dv || atomic_load_explicit(error,memory_order_relaxed)) return;
    uint h=i/dv, v=i%dv, row=at(a.v[3],i*dk);
    for (uint t=0;t<a.v[0].dim[0];++t) {
        uint q=t*a.v[0].stride[0]+(h%hk)*dk, k=q+hk*dk;
        uint value=t*a.v[0].stride[0]+2*hk*dk+h*dv+v;
        float decay=exp(v1[at(a.v[1],t*hv+h)]), beta=v2[at(a.v[2],t*hv+h)], prediction=0;
        for (uint j=0;j<dk;++j) { v3[row+j]*=decay; prediction+=v3[row+j]*v0[k+j]; }
        float delta=(v0[value]-prediction)*beta, result=0;
        if (!valid(prediction,error) || !valid(delta,error)) return;
        for (uint j=0;j<dk;++j) {
            v3[row+j]+=delta*v0[k+j];
            if (!valid(v3[row+j],error)) return;
            result+=v3[row+j]*v0[q+j];
        }
        if (!valid(result,error)) return;
        v4[at(a.v[4],(t*hv+h)*dv+v)]=result;
    }
}
