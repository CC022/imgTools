//
//  ExposureCompensation.metal
//  panoDev
//
//  Buffer-backed Brown-Lowe (2007) gain-compensation kernels.
//

#include <metal_stdlib>
using namespace metal;

constant uint TG  = 16;
constant uint TG2 = TG * TG;        // 256 threads / threadgroup

struct BufDims { uint width; uint height; };

// MARK: - overlapStatsReducePair
// Per-threadgroup tree reduction of overlap statistics for one (A, B) pair.
// Overlap = both alphas exceed 0.5 (binary inside / outside the source rect).
//
// partialsA[k] = (ΣR_A, ΣG_A, ΣB_A, count)  per threadgroup k
// partialsB[k] = (ΣR_B, ΣG_B, ΣB_B, count)  ditto for B

kernel void overlapStatsReducePair(
    device const half4*    bufA      [[buffer(0)]],
    device const half4*    bufB      [[buffer(1)]],
    device       float4*   partialsA [[buffer(2)]],
    device       float4*   partialsB [[buffer(3)]],
    constant     BufDims&  dim       [[buffer(4)]],
    uint2 gid       [[thread_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint2 tgid      [[threadgroup_position_in_grid]],
    uint2 tgPerGrid [[threadgroups_per_grid]])
{
    threadgroup float4 sA[TG2];
    threadgroup float4 sB[TG2];

    float4 lA = float4(0);
    float4 lB = float4(0);
    if (gid.x < dim.width && gid.y < dim.height) {
        uint i = gid.y * dim.width + gid.x;
        half4 a = bufA[i];
        half4 b = bufB[i];
        if (a.a > 0.5h && b.a > 0.5h) {
            lA = float4(float3(a.rgb), 1.0f);
            lB = float4(float3(b.rgb), 1.0f);
        }
    }
    sA[tid] = lA;
    sB[tid] = lB;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = TG2 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sA[tid] += sA[tid + s];
            sB[tid] += sB[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        uint k = tgid.y * tgPerGrid.x + tgid.x;
        partialsA[k] = sA[0];
        partialsB[k] = sB[0];
    }
}

// MARK: - applyExposureGain
// In-place RGB *= gain. Alpha preserved.

kernel void applyExposureGain(
    device       half4*    buf  [[buffer(0)]],
    constant     BufDims&  dim  [[buffer(1)]],
    constant     float3&   gain [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = gid.y * dim.width + gid.x;
    half4 c = buf[i];
    c.rgb = half3(float3(c.rgb) * gain);
    buf[i] = c;
}
