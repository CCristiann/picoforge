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
static inline void pool(device const float *x, device float *y, constant Q38DGpuArgs &a,
    device atomic_uint *error, uint d) {
    float sum=0;
    for (uint t=0;t<a.n;++t) {
        float x0=x[t*a.d+d]; if (!isfinite(x0)) { fail(error,1); return; } sum+=x0;
    }
    float v=sum/float(a.n); if (!isfinite(v)) { fail(error,1); return; } y[d]=v;
}
kernel void q38d_pool_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint d [[thread_position_in_grid]]) {
    pool(x,y,a,error,d);
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
/* Body shared with the R3.3 batch kernel: Q38DSplit carries the per-call m, n,
 * keys per split (r) and split count (pos) that a.* holds for one token. */
struct Q38DSplit { uint h, j, d, n, m, r, pos; };
static inline void attend_split(device const float *q, device const float *gate, device const float *k,
    device const float *v, device const uint *ids, device float *out, Q38DSplit a, device atomic_uint *error,
    device float *part, uint2 tg, ushort sg, ushort lane) {
    const uint group=a.h/a.j, hps=(group+3)/4, g0=sg*hps, P=a.d/32, kv=tg.y;
    /* An invalid id anywhere: the oracle sets bit 2 alone and writes nothing (R2 review I1). */
    bool invalid=false;
    for (uint i=lane;i<a.m;i+=32) invalid|=ids[i]>=a.n || (i && ids[i]<=ids[i-1]);
    if (simd_any(invalid)) { if (!tg.x && !tg.y && !sg && !lane) fail(error,2); return; }
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
        if (id>=a.n || (idx && id<=ids[idx-1])) continue; /* excluded above */
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
kernel void q38d_attend_split_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device float *part [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    attend_split(q,gate,k,v,ids,out,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,part,tg,sg,lane);
}
/* Threadgroup h, thread f: combine the a.pos split partials of head h in split order. */
static inline void attend_merge(device const float *gate, device const uint *ids, device float *out,
    Q38DSplit a, device atomic_uint *error, device const float *part, uint h, uint f, uint tn,
    threadgroup atomic_uint &invalid) {   /* the split kernel set bit 2 alone and wrote no partials */
    if (!f) atomic_store_explicit(&invalid,0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=f;i<a.m;i+=tn)
        if (ids[i]>=a.n || (i && ids[i]<=ids[i-1])) atomic_store_explicit(&invalid,1,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&invalid,memory_order_relaxed) || f>=a.d) return;
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
kernel void q38d_attend_merge_gpu(device const float *gate [[buffer(1)]], device const uint *ids [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device const float *part [[buffer(8)]], uint h [[threadgroup_position_in_grid]],
    uint f [[thread_index_in_threadgroup]], uint tn [[threads_per_threadgroup]]) {
    threadgroup atomic_uint invalid;
    attend_merge(gate,ids,out,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,part,h,f,tn,invalid);
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
static inline void scores_fast(device const float *q, device const float *k, device float *out,
    Q38DSplit a, device atomic_uint *error, uint i, uint lane) {
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
kernel void q38d_scores_fast_gpu(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device float *out [[buffer(2)]], constant Q38DGpuArgs &a [[buffer(6)]],
    device atomic_uint *error [[buffer(7)]], uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    scores_fast(q,k,out,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,i,lane);
}
/* R2.8: q38d_rope_gpu with one thread per element instead of one per head (24 threads
 * each running 256 precise cos/sin in series: 33 us per call at any context). Each
 * element's arithmetic is the oracle's, so y is bit-identical; on a nonfinite value the
 * other elements are still written (outputs are unspecified once the word is set). */
static inline void rope_fast(device const float *x, device float *y, constant Q38DGpuArgs &a, uint pos,
    device atomic_uint *error, constant float *frequencies, uint e) {
    if (e>=a.h*a.d) return;
    uint h=e/a.d, d=e%a.d;
    float value=x[h*a.d+d];
    if (!isfinite(value)) { fail(error,1); return; }
    if (d<a.r) {
        uint i=d%(a.r/2); float frequency=frequencies[i];
        float angle=float(pos)*frequency;
        float other=x[h*a.d+(d<a.r/2 ? d+a.r/2 : d-a.r/2)];
        value=value*cos(angle)+(d<a.r/2 ? -other : other)*sin(angle);
    }
    if (!isfinite(value)) { fail(error,1); return; }
    y[h*a.d+d]=value;
}
kernel void q38d_rope_fast_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant float *frequencies [[buffer(8)]], uint e [[thread_position_in_grid]]) {
    rope_fast(x,y,a,a.pos,error,frequencies,e);
}
static inline void rank_fast(device const float *scores, device uint *keep, Q38DSplit a,
    device atomic_uint *error, uint g, ushort sg, ushort sn, ushort lane) {
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
kernel void q38d_rank_fast_gpu(device const float *scores [[buffer(0)]], device uint *keep [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort sn [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    rank_fast(scores,keep,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,g,sg,sn,lane);
}
static inline void compact_fast(device const uint *keep, device uint *ids, Q38DSplit a,
    device atomic_uint *error, uint g, ushort sg, ushort sn, ushort lane) {
    uint b=g*sn+sg;
    if (b>=a.n || !keep[b]) return;
    uint slot=0;
    for (uint c=lane;c<b;c+=32) slot+=keep[c];
    slot=simd_sum(slot);
    if (slot>=a.m) { if (!lane) fail(error,2); return; }
    for (uint r=lane;r<a.r;r+=32) ids[slot*a.r+r]=b*a.r+r;
}
kernel void q38d_compact_fast_gpu(device const uint *keep [[buffer(0)]], device uint *ids [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    uint g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort sn [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    compact_fast(keep,ids,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,g,sg,sn,lane);
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

/* ---- R3.3 batched QSA prefill. The per-token loop of q38d_layer_encode ran
 * ~9.5 dispatches per token, each behind a barrier. Here every stage covers
 * the whole append: grid z is the token (or the completed block), and each
 * kernel derives the token's position, visible blocks n, selection sel and
 * tail exactly as the host loop does, then runs the same inline body as the
 * per-token kernel: the bits are the loop's. Causality holds because a token
 * reads only keys <= p and blocks < (p+1)/ratio, all written by earlier stages. */
#define Q38D_MAX_SPLITS 64u /* mirror of qwen38_d_internal.h */
struct Q38DBatch { uint pos0, step, x_stride, y_stride, ratio, budget, scores_stride, ids_stride, first, count; };
static inline uint q38d_visible_blocks(constant Q38DBatch &b, uint z) { return (b.pos0+z+1)/b.ratio; }
static inline uint q38d_selected(constant Q38DBatch &b, uint z) {
    const uint n=q38d_visible_blocks(b,z), kb=b.budget/b.ratio; return n<kb ? n : kb;
}
kernel void q38d_rope_batch_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant float *frequencies [[buffer(8)]], constant Q38DBatch &b [[buffer(9)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.y>=b.count) return;
    rope_fast(x+(ulong)i.y*b.x_stride,y+(ulong)i.y*b.y_stride,a,b.pos0+i.y*b.step,error,frequencies,i.x);
}
kernel void q38d_pool_batch_gpu(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant Q38DBatch &b [[buffer(9)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x>=a.d || i.y>=b.count) return;
    pool(x+(ulong)i.y*b.x_stride,y+(ulong)i.y*b.y_stride,a,error,i.x);
}
kernel void q38d_scores_batch_gpu(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device float *out [[buffer(2)]], constant Q38DGpuArgs &a [[buffer(6)]],
    device atomic_uint *error [[buffer(7)]], constant Q38DBatch &b [[buffer(9)]],
    uint2 i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (i.y>=b.count) return; /* uniform over the simdgroup: the grid's x is a multiple of 32 */
    const uint n=q38d_visible_blocks(b,i.y);
    scores_fast(q+(ulong)i.y*b.x_stride,k,out+(ulong)i.y*b.scores_stride,Q38DSplit{a.h,a.j,a.d,n,0,0,0},error,i.x,lane);
}
kernel void q38d_rank_batch_gpu(device const float *scores [[buffer(0)]], device uint *keep [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant Q38DBatch &b [[buffer(9)]], uint2 g [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort sn [[simdgroups_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (g.y>=b.count) return;
    const uint n=q38d_visible_blocks(b,g.y);
    rank_fast(scores+(ulong)g.y*b.scores_stride,keep+(ulong)g.y*b.scores_stride,
              Q38DSplit{0,0,0,n,q38d_selected(b,g.y),b.ratio,0},error,g.x,sg,sn,lane);
}
kernel void q38d_compact_batch_gpu(device const uint *keep [[buffer(0)]], device uint *ids [[buffer(1)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    constant Q38DBatch &b [[buffer(9)]], uint2 g [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort sn [[simdgroups_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (g.y>=b.count) return;
    const uint n=q38d_visible_blocks(b,g.y);
    compact_fast(keep+(ulong)g.y*b.scores_stride,ids+(ulong)g.y*b.ids_stride,
                 Q38DSplit{0,0,0,n,q38d_selected(b,g.y),b.ratio,0},error,g.x,sg,sn,lane);
}
/* The causal tail after the selected blocks: as q38d_tail_gpu with m = sel*ratio, n = blocks*ratio. */
kernel void q38d_tail_batch_gpu(device uint *ids [[buffer(0)]], constant Q38DBatch &b [[buffer(9)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.y>=b.count) return;
    const uint visible=b.pos0+i.y+1, n=visible/b.ratio;
    if (n*b.ratio+i.x<=visible-1) ids[(ulong)i.y*b.ids_stride+q38d_selected(b,i.y)*b.ratio+i.x]=n*b.ratio+i.x;
}
/* Split attention of tokens first .. first+count-1 (grid z, one part slab each). */
static inline Q38DSplit q38d_token_split(constant Q38DGpuArgs &a, constant Q38DBatch &b, uint z) {
    const uint visible=b.pos0+z+1, m=q38d_selected(b,z)*b.ratio+visible%b.ratio;
    uint s=(m+31)/32; s=s<1 ? 1 : (s>Q38D_MAX_SPLITS ? Q38D_MAX_SPLITS : s); /* q38d_attend_splits */
    const uint keys=(m+s-1)/s;
    return Q38DSplit{a.h,a.j,a.d,visible,m,keys,(m+keys-1)/keys};
}
kernel void q38d_split_batch_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device float *part [[buffer(8)]], constant Q38DBatch &b [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint z=b.first+tg.z;
    if (tg.z>=b.count) return;
    const Q38DSplit s=q38d_token_split(a,b,z);
    if (tg.x>=s.pos) return;
    const ulong row=(ulong)z*a.h*a.d;
    attend_split(q+row,gate+row,k,v,ids+(ulong)z*b.ids_stride,out+row,s,error,
                 part+(ulong)tg.z*Q38D_MAX_SPLITS*a.h*(a.d+2),tg.xy,sg,lane);
}
kernel void q38d_merge_batch_gpu(device const float *gate [[buffer(1)]], device const uint *ids [[buffer(4)]],
    device float *out [[buffer(5)]], constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device const float *part [[buffer(8)]], constant Q38DBatch &b [[buffer(9)]], uint2 h [[threadgroup_position_in_grid]],
    uint f [[thread_index_in_threadgroup]], uint2 tn [[threads_per_threadgroup]]) {
    threadgroup atomic_uint invalid;
    const uint z=b.first+h.y;
    if (h.y>=b.count) return;
    const Q38DSplit s=q38d_token_split(a,b,z);
    if (s.pos==1) return; /* one split wrote out itself */
    const ulong row=(ulong)z*a.h*a.d;
    attend_merge(gate+row,ids+(ulong)z*b.ids_stride,out+row,s,error,part+(ulong)h.y*Q38D_MAX_SPLITS*a.h*(a.d+2),
                 h.x,f,tn.x,invalid);
}

/* ---- P5.1 split attention on simdgroup matrices (FP32 MMA, no reduced precision).
 * attend_split runs one key at a time: per key and head a 256-term dot as 8 products
 * per lane plus a simd_sum, then a rescale of the accumulator: ~0.75 TFLOPS, 5.7 s of
 * the 8K prefill. Here one threadgroup (4 simdgroups) is still one split of one KV
 * head, but its G query heads (8 <= G <= 16) are the rows of two 8x8 tiles, A = rows
 * 0..7 and B = rows G-8..G-1 (they overlap when G < 16, never past the group), loaded
 * straight from q; each simdgroup owns one quarter (64) of the d = 256 features:
 *   1. per chunk of 8 keys: S = Q K^T over the quarter (K staged by the simdgroup),
 *      the four partial tiles added by one thread per element into the logits;
 *   2. one thread per (row, 8 columns): scale, row max over the whole split, exp
 *      weights and their sum: no per-key rescale;
 *   3. O = P V, each simdgroup its 64 columns (V staged like K), stored straight to
 *      the partials (or to out, then scaled in place by 1/l and the gate).
 * 13 KB of threadgroup memory (a first version staged Q and O there, 29 KB, and was
 * slower than attend_split: one threadgroup per core). Same partials (max, sum, acc),
 * split layout and error bits as attend_split; the sums run in another order (8-term
 * MMA steps, four quarters), so the gate is the attend_fast tolerance, not its bits.
 * Q38D_MMA_D (256) and Q38D_MMA_KEYS (40 = 5 chunks of 8) come from qwen38_d_internal.h. */
/* P5.2: one threadgroup now covers Q38D_MMA_GROUP consecutive R2.1 splits (at most
 * 4 x 40 keys), one sub-block of r keys at a time with an online softmax: per sub-block
 * the row max M grows, O is rescaled by exp(M_old - M) with an MMA against a diagonal
 * tile, then O += P V. The merge reads 4x fewer partials (it was ~0.84 s at 8K) and the
 * fixed cost of a threadgroup (id scan, Q tiles, barriers) is paid once per 4 splits.
 * Still a function of m only. */
#define Q38D_MMA_FLOATS (4u*512u + 512u + 16u*Q38D_MMA_KEYS + 36u + 128u)
static inline void attend_split_mma(device const float *q, device const float *gate, device const float *k,
    device const float *v, device const uint *ids, device float *out, Q38DSplit a, device atomic_uint *error,
    device float *part, uint2 tg, ushort sg, ushort lane, threadgroup float *mem) {
    const uint group=a.h/a.j, kv=tg.y, t=sg*32u+lane, c0=sg*64u, rb=group-8u;
    threadgroup float *own=mem+sg*512u;             /* 8 keys x 64 features of K, then of V */
    threadgroup float *sp=mem+2048u;                /* [4 quarters][16 rows][8] partial S of a chunk */
    threadgroup float *pw=sp+512u;                  /* [16][KEYS] logits, then weights, of a sub-block */
    threadgroup float *st=pw+16u*Q38D_MMA_KEYS;     /* [16] row max, [16] row sum, [4] invalid flags */
    threadgroup float *dg=st+36u;                   /* two 8x8 diagonal rescale tiles (rows A, rows B) */
    bool invalid=false;   /* the 128 threads share the id scan; the flags make the return uniform */
    for (uint i=t;i<a.m;i+=128u) invalid|=ids[i]>=a.n || (i && ids[i]<=ids[i-1]);
    invalid=simd_any(invalid);
    if (!lane) st[32u+sg]=invalid ? 1.0f : 0.0f;
    dg[t]=0.0f;                                     /* off-diagonal zeros, written once */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (st[32u]+st[33u]+st[34u]+st[35u]!=0.0f) { if (!tg.x && !tg.y && !t) fail(error,2); return; }
    const uint groups=(a.pos+Q38D_MMA_GROUP-1u)/Q38D_MMA_GROUP;
    const uint b0=tg.x*Q38D_MMA_GROUP*a.r, b1=min(a.m,b0+Q38D_MMA_GROUP*a.r);
    bool bad=false, range=false;
    device const float *qg=q+(ulong)kv*group*a.d+c0;
    for (uint i=lane;i<group*64u;i+=32) bad|=!isfinite(qg[(i/64u)*a.d+i%64u]);
    simdgroup_float8x8 qa[8], qb[8], oa[8], ob[8];
    for (uint ks=0;ks<8;ks++) { simdgroup_load(qa[ks],qg+ks*8u,a.d); simdgroup_load(qb[ks],qg+(ulong)rb*a.d+ks*8u,a.d); }
    for (uint ct=0;ct<8;ct++) { oa[ct]=make_filled_simdgroup_matrix<float,8,8>(0.0f); ob[ct]=oa[ct]; }
    float M=-INFINITY, L=0.0f;                      /* row t/8's running max and sum */
    for (uint k0=b0;k0<b1;k0+=a.r) {
    const uint nk=min(b1,k0+a.r)-k0, chunks=(nk+7)/8;
    for (uint c=0;c<chunks;c++) {                   /* 1. logits, chunk of 8 keys */
        for (uint i=lane;i<512u;i+=32) {            /* 8 keys x 64 features, missing keys zero */
            const uint r=c*8u+i/64u; float x=0.0f;
            if (r<nk) { x=k[((ulong)ids[k0+r]*a.j+kv)*a.d+c0+i%64u]; bad|=!isfinite(x); }
            own[i]=x;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s0=make_filled_simdgroup_matrix<float,8,8>(0.0f), s1=s0;
        for (uint ks=0;ks<8;ks++) {
            simdgroup_float8x8 kt; simdgroup_load(kt,own+ks*8u,64,ulong2(0,0),true);
            simdgroup_multiply_accumulate(s0,qa[ks],kt,s0);
            simdgroup_multiply_accumulate(s1,qb[ks],kt,s1);
        }
        simdgroup_store(s0,sp+sg*128u,8); simdgroup_store(s1,sp+sg*128u+rb*8u,8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t/8u<group) {                           /* row t/8, key t%8: quarters in order */
            const uint r=t/8u, col=t%8u;
            pw[r*Q38D_MMA_KEYS+c*8u+col]=((sp[r*8u+col]+sp[128u+r*8u+col])+sp[256u+r*8u+col])+sp[384u+r*8u+col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {                                               /* 2. thread t: row t/8, columns t%8 + 8i */
        const uint r=t/8u, c=t%8u; const float scale=1.0f/sqrt(float(a.d));
        float x[Q38D_MMA_KEYS/8u], mb=-INFINITY, l=0.0f;
        for (uint i=0;i<Q38D_MMA_KEYS/8u;i++) {
            const uint col=c+8u*i; x[i]=-INFINITY;
            if (col<nk && r<group) { const float s=pw[r*Q38D_MMA_KEYS+col]*scale; bad|=!isfinite(s); x[i]=s; mb=max(mb,s); }
        }
        for (ushort o=1;o<8;o<<=1) mb=max(mb,simd_shuffle_xor(mb,o));
        const float mn=max(M,mb), alpha=M==-INFINITY ? 0.0f : exp(M-mn);
        for (uint i=0;i<Q38D_MMA_KEYS/8u;i++) {
            const uint col=c+8u*i; const float w=col<nk && r<group ? exp(x[i]-mn) : 0.0f;
            pw[r*Q38D_MMA_KEYS+col]=w; l+=w;
        }
        for (ushort o=1;o<8;o<<=1) l+=simd_shuffle_xor(l,o);
        L=L*alpha+l; M=mn;
        if (!c && r<8u) dg[r*9u]=alpha;
        if (!c && r>=rb && r<rb+8u) dg[64u+(r-rb)*9u]=alpha;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0!=b0) {                                   /* O *= exp(M_old - M) per row */
        simdgroup_float8x8 da, db; simdgroup_load(da,dg,8); simdgroup_load(db,dg+64u,8);
        for (uint ct=0;ct<8;ct++) { simdgroup_multiply(oa[ct],da,oa[ct]); simdgroup_multiply(ob[ct],db,ob[ct]); }
    }
    for (uint c=0;c<chunks;c++) {                   /* 3. O += P V, chunk of 8 keys */
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=lane;i<512u;i+=32) {
            const uint r=c*8u+i/64u; float x=0.0f;
            if (r<nk) { x=v[((ulong)ids[k0+r]*a.j+kv)*a.d+c0+i%64u]; bad|=!isfinite(x); }
            own[i]=x;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 pa, pb;
        simdgroup_load(pa,pw+c*8u,Q38D_MMA_KEYS); simdgroup_load(pb,pw+rb*Q38D_MMA_KEYS+c*8u,Q38D_MMA_KEYS);
        for (uint ct=0;ct<8;ct++) {
            simdgroup_float8x8 vt; simdgroup_load(vt,own+ct*8u,64);
            simdgroup_multiply_accumulate(oa[ct],pa,vt,oa[ct]);
            simdgroup_multiply_accumulate(ob[ct],pb,vt,ob[ct]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  /* pw and dg are rewritten by the next sub-block */
    }
    if (t%8u==0u) { st[t/8u]=M; st[16u+t/8u]=L; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pitch=groups==1u ? a.d : a.d+2;      /* acc rows of the group: out, or the partials */
    device float *dst=groups==1u ? out+(ulong)kv*group*a.d : part+((ulong)tg.x*a.h+kv*group)*(a.d+2)+2;
    for (uint ct=0;ct<8;ct++) {                     /* B after A: shared rows get the same values */
        simdgroup_store(oa[ct],dst+c0+ct*8u,pitch); simdgroup_store(ob[ct],dst+(ulong)rb*pitch+c0+ct*8u,pitch);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i=t;i<group*a.d;i+=128u) {
        const uint r=i/a.d, f=i%a.d, h=kv*group+r;
        const float gt=gate[h*a.d+f], l=st[16u+r];
        bad|=!isfinite(gt); range|=gt<0 || gt>1;
        if (groups==1u) {
            bad|=!isfinite(l) || l<=0;
            const float y=out[h*a.d+f]/l*gt; bad|=!isfinite(y);
            out[h*a.d+f]=y;
        } else if (!f) {
            device float *p=part+((ulong)tg.x*a.h+h)*(a.d+2);
            p[0]=st[r]; p[1]=l;
        }
    }
    if (bad) fail(error,1);
    if (range) fail(error,2);
}
kernel void q38d_attend_split_mma_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device float *part [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float mem[Q38D_MMA_FLOATS];
    attend_split_mma(q,gate,k,v,ids,out,Q38DSplit{a.h,a.j,a.d,a.n,a.m,a.r,a.pos},error,part,tg,sg,lane,mem);
}
kernel void q38d_split_mma_batch_gpu(device const float *q [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
    device const uint *ids [[buffer(4)]], device float *out [[buffer(5)]],
    constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device float *part [[buffer(8)]], constant Q38DBatch &b [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float mem[Q38D_MMA_FLOATS];
    const uint z=b.first+tg.z;
    if (tg.z>=b.count) return;
    const Q38DSplit s=q38d_token_split(a,b,z);
    if (tg.x>=(s.pos+Q38D_MMA_GROUP-1u)/Q38D_MMA_GROUP) return;
    const ulong row=(ulong)z*a.h*a.d;
    attend_split_mma(q+row,gate+row,k,v,ids+(ulong)z*b.ids_stride,out+row,s,error,
                     part+(ulong)tg.z*(Q38D_MAX_SPLITS/Q38D_MMA_GROUP)*a.h*(a.d+2),tg.xy,sg,lane,mem);
}
/* The merge of the P5.2 MMA split: as q38d_merge_batch_gpu over its ceil(splits/4) partials. */
kernel void q38d_merge_mma_batch_gpu(device const float *gate [[buffer(1)]], device const uint *ids [[buffer(4)]],
    device float *out [[buffer(5)]], constant Q38DGpuArgs &a [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
    device const float *part [[buffer(8)]], constant Q38DBatch &b [[buffer(9)]], uint2 h [[threadgroup_position_in_grid]],
    uint f [[thread_index_in_threadgroup]], uint2 tn [[threads_per_threadgroup]]) {
    threadgroup atomic_uint invalid;
    const uint z=b.first+h.y;
    if (h.y>=b.count) return;
    Q38DSplit s=q38d_token_split(a,b,z);
    s.pos=(s.pos+Q38D_MMA_GROUP-1u)/Q38D_MMA_GROUP;
    if (s.pos==1) return; /* one threadgroup wrote out itself */
    const ulong row=(ulong)z*a.h*a.d;
    attend_merge(gate+row,ids+(ulong)z*b.ids_stride,out+row,s,error,part+(ulong)h.y*(Q38D_MAX_SPLITS/Q38D_MMA_GROUP)*a.h*(a.d+2),
                 h.x,f,tn.x,invalid);
}
