#include <metal_stdlib>
using namespace metal;
/* Graph wiring only: copy snapshots, or repeat each embedding over H streams. */
kernel void q38g_copy(device const float *x [[buffer(0)]],device float *y [[buffer(1)]],
    constant uint4 &a [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if (i<a.x) y[i]=x[a.w ? (i/(a.y*a.z))*a.y+i%a.y : i];
}
