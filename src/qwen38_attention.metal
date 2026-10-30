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
kernel void q38d_attend_scalar_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
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
/* One threadgroup per head, one accumulator per feature. Products and the 16 dots are
 * parallel; each dot is summed in the original increasing-d order. Online
 * softmax tiles, denominator and per-feature accumulation retain that exact
 * FP32 order. No narrowing, tree reduction or private 256-float accumulator. */
kernel void q38d_attend_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint h [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float products[16*256], weights[16], rescale, denominator;
    threadgroup atomic_uint invalid;
    if (!d) {
        atomic_store_explicit(&invalid,0,memory_order_relaxed); denominator=0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=d;i<a.m;i+=256)
        if (ids[i]>=a.n || (i && ids[i]<=ids[i-1])) atomic_store_explicit(&invalid,1,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&invalid,memory_order_relaxed)) { if (!d) fail(error,2); return; }
    float query=d<a.d ? q[h*a.d+d] : 0, g=d<a.d ? gate[h*a.d+d] : 0,acc=0;
    if (!isfinite(query) || !isfinite(g)) fail(error,1);
    if (g<0 || g>1) fail(error,2);
    float maximum=-INFINITY,scale=1.0f/sqrt(float(a.d));
    for (uint base=0;base<a.m;base+=16) {
        uint tile=min(16u,a.m-base);
        for (uint i=0;i<tile;i++) if (d<a.d) {
            uint at=(ids[base+i]*a.j+h/(a.h/a.j))*a.d+d;
            float key=k[at]; if (!isfinite(key)) fail(error,1);
            products[d*16+i]=query*key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d<tile) {
            float dot=0;
            for (uint feature=0;feature<a.d;feature++) dot+=products[feature*16+d];
            weights[d]=dot*scale;
            if (!isfinite(weights[d])) fail(error,1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!d) {
            float next=maximum;
            for (uint i=0;i<tile;i++) next=max(next,weights[i]);
            rescale=denominator==0 ? 0 : exp(maximum-next); denominator*=rescale;
            for (uint i=0;i<tile;i++) { weights[i]=exp(weights[i]-next); denominator+=weights[i]; }
            maximum=next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d<a.d) {
            acc*=rescale;
            for (uint i=0;i<tile;i++) {
                uint at=(ids[base+i]*a.j+h/(a.h/a.j))*a.d+d;
                float value=v[at]; if (!isfinite(value)) fail(error,1);
                acc+=weights[i]*value;
            }
        }
        // Every reader must finish before lane 0 overwrites weights next tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!isfinite(denominator) || denominator<=0) { if (!d) fail(error,1); return; }
    if (d<a.d) {
        float y=acc/denominator*g; if (!isfinite(y)) fail(error,1);
        out[h*a.d+d]=y;
    }
}
/* R2.1 fast decode attention, split over the selected keys ("flash decoding"; the
 * layout follows ds4 fc80bd6, MIT, metal/qwen4.metal:kernel_qwen4_attn_decode and
 * kernel_qwen4_attn_merge, rewritten for the FP32 cache). The oracle above runs one
 * threadgroup per head over all m keys in tiles of 16 with four barriers each: 24
 * threadgroups and a serial chain, 2.4 ms per layer at 8K. Here threadgroup (s, kv)
 * owns keys [s*w, min(m, s*w+w)) of one KV head (a.r = w keys per split, a.pos =
 * split count, both functions of m only, so every token, chunk and MTP row adds in
 * the same order). Its 4 simdgroups share the K/V rows and own disjoint query heads
 * of the group (<= 4 each); lane l owns features l*P..l*P+P-1, P = d/32 <= 8. Per
 * key: P products in increasing feature order, simd_sum, the oracle's scale, then
 * online softmax per key (the oracle rescales per tile of 16: rounding differs, see
 * attend_fast in tests/qwen38_kernel_tolerances.json). One split writes acc/l*gate
 * like the oracle; more write (max, sum, acc[d]) partials that the merge combines in
 * split order. Same error bits as the oracle (1 nonfinite, 2 ids or gate range). */
#define Q38D_SPLIT_HEADS 4
kernel void q38d_attend_split_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device float *part [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint group=a.h/a.j, hps=(group+3)/4, g0=sg*hps, P=a.d/32, kv=tg.y;
    if (g0>=group) return;
    const uint ng=min(hps,group-g0), k0=tg.x*a.r, k1=min(a.m,k0+a.r);
    const float scale=1.0f/sqrt(float(a.d));
    float qv[Q38D_SPLIT_HEADS][8], acc[Q38D_SPLIT_HEADS][8], mx[Q38D_SPLIT_HEADS], l[Q38D_SPLIT_HEADS];
    bool bad=false, range=false;
    for (uint g=0;g<Q38D_SPLIT_HEADS;g++) {
        uint h=kv*group+g0+min(g,ng-1);
        for (uint i=0;i<8;i++) {
            float x=i<P ? q[h*a.d+lane*P+i] : 0, gt=i<P ? gate[h*a.d+lane*P+i] : 0;
            bad|=!isfinite(x) || !isfinite(gt); range|=gt<0 || gt>1;
            qv[g][i]=x; acc[g][i]=0;
        }
        mx[g]=-INFINITY; l[g]=0;
    }
    for (uint idx=k0;idx<k1;idx++) {
        uint id=ids[idx];
        if (id>=a.n || (idx && id<=ids[idx-1])) { range=true; continue; }
        uint at=(id*a.j+kv)*a.d+lane*P;
        float kx[8], vx[8];
        for (uint i=0;i<8;i++) {
            kx[i]=i<P ? k[at+i] : 0; vx[i]=i<P ? v[at+i] : 0;
            bad|=!isfinite(kx[i]) || !isfinite(vx[i]);
        }
        for (uint g=0;g<Q38D_SPLIT_HEADS;g++) if (g<ng) {   /* uniform per simdgroup */
            float dot=0;
            for (uint i=0;i<8;i++) if (i<P) dot+=qv[g][i]*kx[i];
            float logit=simd_sum(dot)*scale; bad|=!isfinite(logit);
            float next=max(mx[g],logit), rescale=l[g]==0 ? 0 : exp(mx[g]-next), weight=exp(logit-next);
            l[g]=l[g]*rescale+weight;
            for (uint i=0;i<8;i++) acc[g][i]=acc[g][i]*rescale+weight*vx[i];
            mx[g]=next;
        }
    }
    for (uint g=0;g<Q38D_SPLIT_HEADS;g++) if (g<ng) {   /* constant bounds keep the arrays in registers */
        uint h=kv*group+g0+g;
        if (a.pos==1) {
            bad|=!isfinite(l[g]) || l[g]<=0;
            for (uint i=0;i<8;i++) if (i<P) {
                float y=acc[g][i]/l[g]*gate[h*a.d+lane*P+i]; bad|=!isfinite(y);
                out[h*a.d+lane*P+i]=y;
            }
        } else {
            device float *p=part+((size_t)tg.x*a.h+h)*(a.d+2);
            if (!lane) { p[0]=mx[g]; p[1]=l[g]; }
            for (uint i=0;i<8;i++) if (i<P) p[2+lane*P+i]=acc[g][i];
        }
    }
    if (bad) fail(error,1);
    if (range) fail(error,2);
}
/* Threadgroup h, thread f: combine the a.pos split partials of head h in split order. */
kernel void q38d_attend_merge_gpu(device const float *gate [[buffer(1)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device const float *part [[buffer(8)]], uint h [[threadgroup_position_in_grid]],
    uint f [[thread_index_in_threadgroup]]) {
    if (f>=a.d) return;
    const size_t stride=(size_t)a.h*(a.d+2);
    device const float *p=part+(size_t)h*(a.d+2);
    float mx=-INFINITY, l=0, o=0;
    for (uint s=0;s<a.pos;s++) mx=max(mx,p[s*stride]);
    for (uint s=0;s<a.pos;s++) {
        float c=exp(p[s*stride]-mx);
        l=l+p[s*stride+1]*c; o=o+p[s*stride+2+f]*c;
    }
    if (!isfinite(l) || l<=0) { fail(error,1); return; }
    float y=o/l*gate[h*a.d+f]; if (!isfinite(y)) fail(error,1);
    out[h*a.d+f]=y;
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
/* R2.2 exact fast variants of scores, rank and compact: same results, bit for bit
 * (scores keep every dot's increasing-d order and the head order of the sum; rank and
 * slot are integer counts). The oracles are latency bound at 8K: one thread per block
 * runs a 512-step dependent chain (scores) or a 2048-step loop over device memory (rank),
 * 0.8 ms per layer for 2048 blocks. Scores (R2.8): Q38D_SCORE_LANES threads per block,
 * lane h runs head h's dot (a 128-step chain instead of the R2.2 kernel's 4 interleaved
 * ones in one thread: 31 us even at 3 blocks), then the block's first lane adds the
 * clamped dots in head order. Rank and compact: one simdgroup per block, lanes count
 * strided candidates, simd_sum adds them. */
kernel void q38d_scores_fast_gpu(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device float *out [[buffer(2)]], constant Q38DGpuArgs &a [[buffer(6)]],
    device atomic_uint *error [[buffer(7)]], uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint b=i/Q38D_SCORE_LANES, h=i%Q38D_SCORE_LANES, first=lane-h;
    float dot=0; bool bad=false;
    if (b<a.n && h<a.h)
        for (uint d=0;d<a.d;++d) {
            float x=q[h*a.d+d],y=k[b*a.d+d];
            bad|=!isfinite(x)||!isfinite(y); dot+=x*y;
        }
    bad|=!isfinite(dot);
    float score=0;
    for (uint g=0;g<Q38D_SCORE_LANES;++g) { /* every lane shuffles: uniform control */
        float v=simd_shuffle(dot,ushort(first+g)); uint w=simd_shuffle(uint(bad),ushort(first+g));
        if (g<a.h) { score+=max(0.0f,v); bad|=w!=0; }
    }
    if (h || b>=a.n) return;
    if (bad || !isfinite(score)) { fail(error,1); return; }
    out[b]=score;
}
/* R2.8: q38d_rope_gpu with one thread per element instead of one per head (24 threads
 * each running 256 precise cos/sin in series: 33 us per call at any context). Each
 * element's arithmetic is the oracle's, so y is bit-identical; on a nonfinite value the
 * other elements are still written (outputs are unspecified once the word is set). */
kernel void q38d_rope_fast_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant float *frequencies [[buffer(8)]], uint e [[thread_position_in_grid]]) {
    if (e>=a.h*a.d) return;
    uint h=e/a.d, d=e%a.d;
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
kernel void q38d_rank_fast_gpu(device const float *scores [[buffer(0)]], device uint *keep [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort sn [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint b=g*sn+sg;
    if (b>=a.n) return;
    float s=scores[b]; uint rank=0; bool bad=false;
    for (uint c=lane;c<a.n;c+=32) {
        float o=scores[c]; bad|=!isfinite(o);
        rank+=(o>s || (o==s && c<b)) ? 1u : 0u;
    }
    rank=simd_sum(rank);
    if (simd_any(bad)) { if (!lane) { keep[b]=0; fail(error,1); } return; }
    if (!lane) keep[b]=rank<a.m ? 1u : 0u;
}
kernel void q38d_compact_fast_gpu(device const uint *keep [[buffer(0)]], device uint *ids [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort sn [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint b=g*sn+sg;
    if (b>=a.n || !keep[b]) return;
    uint slot=0;
    for (uint c=lane;c<b;c+=32) slot+=keep[c];
    slot=simd_sum(slot);
    if (slot>=a.m) { if (!lane) fail(error,2); return; }
    for (uint r=lane;r<a.r;r+=32) ids[slot*a.r+r]=b*a.r+r;
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
