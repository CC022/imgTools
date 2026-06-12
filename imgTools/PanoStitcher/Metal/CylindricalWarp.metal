//
//  CylindricalWarp.metal
//  panoDev
//
//  Per-image inverse-warp from a source texture into a canvas-sized buffer
//  (rgba16f, row-major). Buffer-backed so the canvas is not constrained by
//  the 16384-px Metal 2D texture limit. Two entry points:
//
//    • cylindricalWarp       — plain warp (low-res prepass, single image).
//    • cylindricalWarpMasked — warp fused with per-image exposure gain and
//      seam-label α masking, used by the streaming blend. One pass instead
//      of warp + gain + mask, saving two full-canvas read-modify-writes
//      per image.
//
//  Critical detail for clean multi-band blending:
//    • RGB is sampled with `clamp_to_edge` *every* canvas pixel, including
//      pixels whose inverse projection lands outside the source rectangle.
//      Outside the source rect we therefore get the extrapolated edge color,
//      not zero — eliminating the sharp "real → 0" boundary that previously
//      contaminated the Laplacian pyramid.
//    • Alpha is the *coverage mask*: 1 inside the source rect, 0 outside
//      (and, for the masked variant, 0 where the seam labels name another
//      owner). The Laplacian-pyramid blend uses this α as the per-frequency
//      weight; its Gaussian-blurred mass at every level provides the
//      canonical Burt-Adelson per-band blending mask.
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
    float2   canvasSize;    // (W, H) of the output buffer (ROI or full canvas)
    float2   roiOrigin;     // canvas px of output pixel (0, 0) — 0 for full warps
};

// Layout MUST match WarpMaskParams in CylindricalCanvas.swift.
struct MaskParams {
    float3 gain;            // per-image RGB exposure gain
    uint   selfIndex;       // this image's seam label
    uint   labelScale;      // full-res pixels per label-map pixel
    uint   lowW;
    uint   lowH;
};

// Canvas pixel → cylinder ray → source pixel; extrapolated edge colour,
// α = coverage.
static half4 warpPixel(texture2d<half, access::sample> inTex,
                       constant Params& P,
                       uint2 gid)
{
    float2 c     = float2(gid) + P.roiOrigin;
    float  theta = P.canvasOrigin.x + c.x / P.canvasRadius;
    float  h     = P.canvasOrigin.y + c.y / P.canvasRadius;

    float3 camRay = P.R * float3(sin(theta), h, cos(theta));

    // Behind-camera (or singular near grazing) → fully transparent.
    if (camRay.z <= 1e-3) return half4(0);

    float2 imgPx = camRay.xy / camRay.z * P.focal + P.principalPoint;

    constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_edge);
    half4 colour = inTex.sample(s, imgPx);

    bool inside = (imgPx.x >= 0.0 && imgPx.x < P.imageSize.x &&
                   imgPx.y >= 0.0 && imgPx.y < P.imageSize.y);
    return half4(colour.rgb, inside ? 1.0h : 0.0h);
}

// MARK: - cylindricalWarp

kernel void cylindricalWarp(
    texture2d<half, access::sample>  inTex   [[texture(0)]],
    device   half4*                  outBuf  [[buffer(0)]],
    constant Params&                 P       [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(P.canvasSize.x) || gid.y >= uint(P.canvasSize.y)) return;
    outBuf[gid.y * uint(P.canvasSize.x) + gid.x] = warpPixel(inTex, P, gid);
}

// MARK: - cylindricalWarpMasked

kernel void cylindricalWarpMasked(
    texture2d<half, access::sample>  inTex   [[texture(0)]],
    device   half4*                  outBuf  [[buffer(0)]],
    device const uchar*              labels  [[buffer(1)]],
    constant Params&                 P       [[buffer(2)]],
    constant MaskParams&             M       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(P.canvasSize.x) || gid.y >= uint(P.canvasSize.y)) return;

    half4 c = warpPixel(inTex, P, gid);
    // Gain applies to every pixel (including extrapolated edge colour, which
    // leaks into the Laplacian bands and must match the covered interior).
    c.rgb = half3(float3(c.rgb) * M.gain);

    // Zero α where the seam label map names another owner. Nearest-neighbour
    // lookup is sufficient: the pyramid blend feathers the binary mask at
    // every band. Label 255 = "no owner" — α is kept and the blend's
    // normalisation averages whichever images cover the pixel.
    if (c.a > 0.0h) {
        uint gx = gid.x + uint(P.roiOrigin.x);      // labels live in global
        uint gy = gid.y + uint(P.roiOrigin.y);      // canvas coordinates
        uint lx = min(gx / M.labelScale, M.lowW - 1);
        uint ly = min(gy / M.labelScale, M.lowH - 1);
        uchar lab = labels[ly * M.lowW + lx];
        if (lab != uchar(M.selfIndex) && lab != 255) c.a = 0.0h;
    }
    outBuf[gid.y * uint(P.canvasSize.x) + gid.x] = c;
}
