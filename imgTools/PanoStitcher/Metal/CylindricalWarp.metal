//
//  CylindricalWarp.metal
//  panoDev
//
//  Per-image inverse-warp from a source texture into a canvas-sized buffer
//  (rgba16f, row-major). Buffer-backed so the canvas is not constrained by
//  the 16384-px Metal 2D texture limit.
//
//  Critical detail for clean multi-band blending:
//    • RGB is sampled with `clamp_to_edge` *every* canvas pixel, including
//      pixels whose inverse projection lands outside the source rectangle.
//      Outside the source rect we therefore get the extrapolated edge color,
//      not zero — eliminating the sharp "real → 0" boundary that previously
//      contaminated the Laplacian pyramid.
//    • Alpha is the *coverage mask*: 1 inside the source rect, 0 outside.
//      The Laplacian-pyramid blend uses this α as the per-frequency weight;
//      its Gaussian-blurred mass at every level provides the canonical
//      Burt-Adelson per-band blending mask.
//

#include <metal_stdlib>
using namespace metal;

// Layout MUST match CylindricalWarpParams in CylindricalCanvas.swift.
struct Params {
    float3x3 R;             // world → camera, orthonormal
    float    focal;         // pixel focal length
    float    canvasRadius;  // pixels per radian
    float2   principalPoint;
    float2   canvasOrigin;  // (θ₀, h₀) of canvas pixel (0, 0)
    float2   imageSize;     // (W, H) of the input image
    float2   canvasSize;    // (W, H) of the output canvas
};

// MARK: - cylindricalWarp

kernel void cylindricalWarp(
    texture2d<half, access::sample>  inTex   [[texture(0)]],
    device   half4*                  outBuf  [[buffer(0)]],
    constant Params&                 P       [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint W = uint(P.canvasSize.x);
    uint H = uint(P.canvasSize.y);
    if (gid.x >= W || gid.y >= H) return;

    uint idx = gid.y * W + gid.x;

    // Canvas pixel → (θ, h)
    float2 c     = float2(gid);
    float  theta = P.canvasOrigin.x + c.x / P.canvasRadius;
    float  h     = P.canvasOrigin.y + c.y / P.canvasRadius;

    // World ray on the cylinder
    float3 worldRay = float3(sin(theta), h, cos(theta));

    // World → camera
    float3 camRay = P.R * worldRay;

    // Behind-camera (or singular near grazing) → fully transparent.
    if (camRay.z <= 1e-3) {
        outBuf[idx] = half4(0);
        return;
    }

    // Project to image pixel
    float2 imgPx = camRay.xy / camRay.z * P.focal + P.principalPoint;

    // Sample with clamp_to_edge — extrapolates the source's edge color where
    // imgPx falls outside the rectangle. Crucial for Laplacian-pyramid hygiene.
    constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_edge);
    half4 colour = inTex.sample(s, imgPx);

    bool inside = (imgPx.x >= 0.0 && imgPx.x < P.imageSize.x &&
                   imgPx.y >= 0.0 && imgPx.y < P.imageSize.y);
    half alpha = inside ? 1.0h : 0.0h;

    outBuf[idx] = half4(colour.rgb, alpha);
}
