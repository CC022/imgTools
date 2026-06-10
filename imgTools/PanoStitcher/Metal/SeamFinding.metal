//
//  SeamFinding.metal
//  panoDev
//
//  Seam masking kernels. Two variants:
//
//  • applySeamLabels — preferred path. A low-resolution per-pixel label map
//    (computed on the CPU: Voronoi init + pairwise DP seams on image content,
//    see SeamFinder.computeLabels) names the single owning camera for every
//    covered canvas pixel. Each per-image dispatch zeros this image's α
//    wherever the label names someone else. Unlike the old chain-of-pairs
//    DP-seam (which mutated α buffers pair-by-pair and broke down for 3+
//    image overlaps), ownership is resolved globally before any α is touched.
//
//  • voronoiCoverage — content-agnostic fallback. Each canvas pixel is owned
//    by the camera whose optical-axis projection on the canvas is closest.
//
//  Either way the result is a binary partition of unity over the canvas; the
//  downstream Laplacian-pyramid blend's Gaussian-blur of the α channel then
//  produces the multi-band feathering at every level (canonical Burt–Adelson
//  on a binary mask).
//

#include <metal_stdlib>
using namespace metal;

struct BufDims { uint width; uint height; };

// MARK: - applySeamLabels

struct SeamLabelParams {
    uint selfIndex;
    uint scale;     // full-res pixels per label-map pixel
    uint lowW;
    uint lowH;
};

// Zero this image's α wherever the low-res label map names another owner.
// Nearest-neighbour label lookup is sufficient: the pyramid blend feathers
// the binary mask at every band. Label 255 = "no owner" (uncovered in the
// low-res prepass); α is kept there and the blend's normalisation averages
// whichever images actually cover the pixel at full resolution.
kernel void applySeamLabels(
    device       half4*           warpedBuf [[buffer(0)]],
    device const uchar*           labels    [[buffer(1)]],
    constant     BufDims&         dim       [[buffer(2)]],
    constant     SeamLabelParams& P         [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;

    uint i = gid.y * dim.width + gid.x;
    half4 c = warpedBuf[i];
    if (c.a < 1e-3h) return;

    uint lx = min(gid.x / P.scale, P.lowW - 1);
    uint ly = min(gid.y / P.scale, P.lowH - 1);
    uchar lab = labels[ly * P.lowW + lx];
    if (lab != uchar(P.selfIndex) && lab != 255) {
        c.a = 0.0h;
        warpedBuf[i] = c;
    }
}

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
