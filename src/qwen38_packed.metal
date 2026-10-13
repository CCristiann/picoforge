/* Correctness baseline: FP32, no TensorOps narrowing or full dequantization. */
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct PackedArgs {
    uint type, k_storage, k_logical, n_out, rows, reserved;
    ulong row_bytes, weight_offset, weight_end, x_stride, y_stride, x_bytes, y_bytes;
};
static_assert(sizeof(PackedArgs) == 80, "host/Metal wire record size");
struct IndexedArgs { PackedArgs base; ulong expert_bytes; uint experts, x_divisor; };
static_assert(sizeof(IndexedArgs) == 96, "indexed wire record size");
constant ulong iq2_grid[256] = {
#include "qwen38_iq2_grid.inc"
};
inline uint le16(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint le32(device const uchar *p) { return le16(p) | (le16(p+2) << 16); }
inline float half_value(device const uchar *p) {
    uint h=le16(p), sign=(h&0x8000u)<<16, exponent=(h>>10)&31u, fraction=h&1023u, bits;
    if (!exponent && fraction) {
        exponent=113;
        while (!(fraction&1024u)) { fraction<<=1; exponent--; }
        bits=sign|(exponent<<23)|((fraction&1023u)<<13);
    } else if (!exponent) bits=sign;
    else bits=sign|((exponent==31 ? 255u : exponent+112u)<<23)|(fraction<<13);
    return as_type<float>(bits);
}
inline float packed_value(device const uchar *row, uint column, uint type) {
    if (type==0) return as_type<float>(le32(row+ulong(column)*4));
    if (type==1) return half_value(row+ulong(column)*2);
    if (type==30) return as_type<float>(le16(row+ulong(column)*2)<<16);
    if (type==8) {
        device const uchar *b=row+ulong(column/32)*34;
        int code=int(b[2+column%32]);
        return half_value(b)*float(code>=128 ? code-256 : code);
    }
    if (type==10) {
        device const uchar *b=row+ulong(column/256)*84;
        uint lane=column%256, scale=b[lane/16];
        uint code=(b[16+lane/128*32+lane%32]>>(2*(lane%128/32)))&3u;
        return (half_value(b+80)*float(scale&15u))*float(code)-half_value(b+82)*float(scale>>4);
    }
    device const uchar *b=row+ulong(column/256)*66;
    uint lane=column%256, slot=lane%32/8;
    device const uchar *group=b+2+lane/32*8;
    uint bits=le32(group+4), signs=(bits>>(slot*7))&127u;
    signs|=(popcount(signs)&1u)<<7;
    uint magnitude=uint((iq2_grid[group[slot]]>>(8*(lane%8)))&255u);
    float value=(half_value(b)*float(2*(bits>>28)+1)*0.125f)*float(magnitude);
    return signs&(1u<<(lane%8)) ? -value : value;
}
inline bool row_bounds(constant PackedArgs &p, ulong row, device atomic_uint *error) {
    if (row>p.weight_end || p.row_bytes>p.weight_end-row) {
        atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return false;
    }
    return true;
}
/* Private diagnostic kernel: same decoder used by the public linear kernel. */
kernel void q38_packed_decode(device const uchar *weights [[buffer(0)]],
        device float *out [[buffer(1)]], device atomic_uint *error [[buffer(2)]],
        constant PackedArgs &p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
    if (at.x>=p.k_storage || at.y>=p.n_out) return;
    ulong row=p.weight_offset+ulong(at.y)*p.row_bytes;
    if (!row_bounds(p,row,error)) return;
    ulong index=ulong(at.y)*p.k_storage+at.x;
    if (index>=p.y_bytes/4) { atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return; }
    float value=packed_value(weights+row,at.x,p.type);
    if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    out[index]=value;
}
kernel void q38_packed_linear(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant PackedArgs &p [[buffer(4)]],
        uint2 at [[thread_position_in_grid]]) {
    if (at.x>=p.n_out || at.y>=p.rows) return;
    ulong row=p.weight_offset+ulong(at.x)*p.row_bytes;
    if (!row_bounds(p,row,error)) return;
    ulong xi=ulong(at.y)*(p.x_stride/4), yi=ulong(at.y)*(p.y_stride/4)+at.x;
    if (xi>p.x_bytes/4 || p.k_logical>p.x_bytes/4-xi || yi>=p.y_bytes/4) {
        atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return;
    }
    float sum=0;
    for (uint k=0;k<p.k_logical;k++) {
        float value=packed_value(weights+row,k,p.type), input=x[xi+k];
        float product=value*input;
        sum+=product;
        if (!isfinite(value) || !isfinite(input) || !isfinite(product) || !isfinite(sum))
            atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    }
    y[yi]=sum;
}
/* Same decoder and FP32 accumulation order as q38_packed_linear; only the
 * expert (from the GPU route buffer) and the input row are per output row. */
kernel void q38_packed_linear_indexed(device const uchar *weights [[buffer(0)]],
        device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
        device atomic_uint *error [[buffer(3)]], constant IndexedArgs &a [[buffer(4)]],
        device const uint *ids [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
    constant PackedArgs &p=a.base;
    if (at.x>=p.n_out || at.y>=p.rows) return;
    uint expert=ids[at.y];
    if (expert>=a.experts) { atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return; }
    ulong row=p.weight_offset+ulong(expert)*a.expert_bytes+ulong(at.x)*p.row_bytes;
    if (!row_bounds(p,row,error)) return;
    ulong xi=ulong(at.y/a.x_divisor)*(p.x_stride/4), yi=ulong(at.y)*(p.y_stride/4)+at.x;
    if (xi>p.x_bytes/4 || p.k_logical>p.x_bytes/4-xi || yi>=p.y_bytes/4) {
        atomic_fetch_or_explicit(error,2u,memory_order_relaxed); return;
    }
    float sum=0;
    for (uint k=0;k<p.k_logical;k++) {
        float value=packed_value(weights+row,k,p.type), input=x[xi+k];
        float product=value*input;
        sum+=product;
        if (!isfinite(value) || !isfinite(input) || !isfinite(product) || !isfinite(sum))
            atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    }
    y[yi]=sum;
}
