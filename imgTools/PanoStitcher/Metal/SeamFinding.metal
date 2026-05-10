//
//  SeamFinding.metal
//  panoDev
//
//  Voronoi-by-camera-centre coverage assignment.
//
//  Each canvas pixel is owned by the camera whose optical-axis projection on
//  the canvas is closest. A per-image dispatch zeros this image's α channel
//  for any covered pixel where another camera's centre is closer — yielding
//  a binary partition of unity over the canvas. The downstream Laplacian-
//  pyramid blend's Gaussian-blur of the α channel then produces the multi-
//  band feathering at every level (canonical Burt–Adelson on a binary mask).
//
//  This replaces the old chain-of-pairs DP-seam algorithm, which broke down
//  for high-overlap captures (3+ images covering the same canvas region):
//  long-range pair overlaps were never seamed and the otherBuf-α guard
//  refused to mask images already zeroed by a previous pair.
//

#include <metal_stdlib>
using namespace metal;

struct BufDims { uint width; uint height; };

// MARK: - voronoiCoverage

kernel void voronoiCoverage(
    device       half4*    warpedBuf [[buffer(0)]],
    device const float2*   centers   [[buffer(1)]],
    constant     BufDims&  dim       [[buffer(2)]],
    constant     uint&     nCameras  [[buffer(3)]],
    constant     uint&     selfIndex [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;

    // Early out: pixel not covered by THIS image — nothing to do regardless
    // of who else might be closest.
    uint i = gid.y * dim.width + gid.x;
    half4 c = warpedBuf[i];
    if (c.a < 1e-3h) return;

    float2 p = float2(gid);
    float bestDist = INFINITY;
    uint  bestCam  = 0;
    for (uint k = 0; k < nCameras; ++k) {
        float2 d = centers[k] - p;
        float dist = dot(d, d);   // squared Euclidean — argmin is monotone
        if (dist < bestDist) { bestDist = dist; bestCam = k; }
    }

    if (bestCam != selfIndex) {
        c.a = 0.0h;
        warpedBuf[i] = c;
    }
}
