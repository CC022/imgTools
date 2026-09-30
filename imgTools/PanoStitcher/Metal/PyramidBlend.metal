//
//  PyramidBlend.metal
//  panoDev
//
//  Burt-Adelson Laplacian-pyramid blending, buffer-backed (rgba16f packed
//  rgb+α, row-major) so canvas dimensions are not bounded by the 16384-px
//  Metal 2D texture limit. Shape pairs `(W, H)` are passed per-kernel via
//  `BufDims` constants — each pyramid level is a distinct buffer with its
//  own dimensions.
//
//  Streaming kernels used by the live pipeline (one image at a time, fused
//  expand + laplacian + accumulate so no per-image Laplacian pyramid and no
//  full-canvas `temps` pyramid are ever allocated):
//    • pyramidReduce             — 5×5 Gaussian + 2× decimate (half → half)
//    • pyramidLapAccExpanded     — fuse: G[k] − Expand(G[k+1]) → α-weighted
//                                  add into acc[k]
//    • pyramidLapAccDC           — coarsest-level fuse: lap = G[N-1]
//    • pyramidNormalize          — acc.rgb / acc.α at coarsest level
//    • pyramidCollapseAddExpanded — fuse: Expand(coarseResult) + acc[k]
//                                   detail → fineOut
//    • pyramidFinalise           — blend domain → linear light, float32 →
//                                  half16 final output
//
//  Every level holds rgb in the blend domain (BlendDomain.h: asinh, i.e.
//  log above the noise floor), not linear light: the warp encodes and
//  `pyramidFinalise` decodes. The kernels in between are linear and
//  domain-agnostic, so the bands' low-frequency correction between frames
//  is a smooth multiplicative gain field rather than an additive offset
//  (which in linear light lifts / crushes shadows).
//
//  ROI support: each image's fine pyramid levels cover only that image's
//  canvas footprint (origin-aligned sub-rect), so per-image blend work is
//  O(footprint), not O(canvas). `pyramidReduce` takes an output origin/size
//  (used by the ROI→global hand-off level), and `pyramidLapAccExpanded`
//  takes the fine/coarse buffers' global origins so it can write the
//  globally-indexed accumulator from ROI-local buffers. Full-canvas callers
//  pass zero origins and identical dims — the maths reduce to the original
//  kernels exactly.
//

#include <metal_stdlib>
#include "BlendDomain.h"
using namespace metal;

// Burt-Adelson 5-tap generating kernel: [1, 4, 6, 4, 1] / 16
constant float kW[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

struct BufDims { uint width; uint height; };

static inline uint idx2(uint x, uint y, uint w) { return y * w + x; }
static inline int  iclampi(int v, int lo, int hi) { return min(max(v, lo), hi); }

// MARK: - pyramidReduce

// `outOrigin`/`outSize` define the written region inside `outBuf` (the
// ROI→global hand-off writes a footprint rect into a zeroed global level);
// plain calls pass origin (0,0) and size == outDim. Source reads stay
// out-region-local: in/out origins are kept consistent by the caller
// (inOrigin == 2·(outOrigin + gid base), pyramid-grid aligned).
kernel void pyramidReduce(
    device const half4*      inBuf     [[buffer(0)]],
    device       half4*      outBuf    [[buffer(1)]],
    constant     BufDims&    inDim     [[buffer(2)]],
    constant     BufDims&    outDim    [[buffer(3)]],
    constant     uint2&      outOrigin [[buffer(4)]],
    constant     uint2&      outSize   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outSize.x || gid.y >= outSize.y) return;

    int Wi = int(inDim.width);
    int Hi = int(inDim.height);
    int2 center = int2(gid) * 2;

    float4 sum = float4(0);
    for (int dy = -2; dy <= 2; ++dy) {
        int  yy = iclampi(center.y + dy, 0, Hi - 1);
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = iclampi(center.x + dx, 0, Wi - 1);
            float4 v = float4(inBuf[idx2(uint(xx), uint(yy), uint(Wi))]);
            sum += kW[dx + 2] * kW[dy + 2] * v;
        }
    }
    outBuf[idx2(gid.x + outOrigin.x, gid.y + outOrigin.y, outDim.width)] = half4(sum);
}

// MARK: - pyramidLapAccExpanded
// Streaming, fused: compute (G[k] − Expand(G[k+1])) inline at this pixel
// and accumulate it into acc[k], α-weighted by the fine Gaussian's α.
// Replaces the split (pyramidExpand → laplacianBuild → laplacianAccumulate)
// pipeline so neither a per-image Laplacian pyramid nor a per-level expand
// `temps` buffer is needed.

// `fineOrigin`/`coarseOrigin` are the global canvas origins of the (possibly
// ROI-local) fine/coarse buffers; `accDim` is the global accumulator's dims.
// Expansion parity is evaluated in global coordinates so ROI-local pyramids
// stay on the global pyramid grid. Full-canvas callers pass zero origins and
// accDim == fineDim.
kernel void pyramidLapAccExpanded(
    device const half4*    fineGaussBuf   [[buffer(0)]],
    device const half4*    coarseGaussBuf [[buffer(1)]],
    device       float4*   accBuf         [[buffer(2)]],
    constant     BufDims&  fineDim        [[buffer(3)]],
    constant     BufDims&  coarseDim      [[buffer(4)]],
    constant     BufDims&  accDim         [[buffer(5)]],
    constant     uint2&    fineOrigin     [[buffer(6)]],
    constant     uint2&    coarseOrigin   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= fineDim.width || gid.y >= fineDim.height) return;

    // Global canvas coordinates at this pyramid level.
    int x = int(gid.x + fineOrigin.x);
    int y = int(gid.y + fineOrigin.y);
    if (uint(x) >= accDim.width || uint(y) >= accDim.height) return;

    int Wc = int(coarseDim.width);
    int Hc = int(coarseDim.height);
    int Cx = int(coarseOrigin.x);
    int Cy = int(coarseOrigin.y);

    float4 expanded = float4(0);
    for (int m = -2; m <= 2; ++m) {
        if (((x - m) & 1) != 0) continue;
        int xx = iclampi((x - m) / 2 - Cx, 0, Wc - 1);
        for (int n = -2; n <= 2; ++n) {
            if (((y - n) & 1) != 0) continue;
            int yy = iclampi((y - n) / 2 - Cy, 0, Hc - 1);
            float4 v = float4(coarseGaussBuf[idx2(uint(xx), uint(yy), uint(Wc))]);
            expanded += 4.0f * kW[m + 2] * kW[n + 2] * v;
        }
    }

    float4 fine = float4(fineGaussBuf[idx2(gid.x, gid.y, fineDim.width)]);
    float4 lap  = fine - expanded;
    float  w    = fine.a;
    uint   ai   = idx2(uint(x), uint(y), accDim.width);
    float4 acc  = accBuf[ai];
    acc.rgb += lap.rgb * w;
    acc.a   += w;
    accBuf[ai] = acc;
}

// MARK: - pyramidExtendRect
// Clamp-extends rgb outside `rect` from the nearest rect-edge pixel, with
// α = 0. Run on the ROI→global hand-off level right after its reduce: it
// restores the clamp-to-edge rgb extrapolation field that the hand-off
// truncates at the ROI border. Without it, the coarser levels' wide-support
// reduces mix zeros — not edge colour — into rgb wherever their footprint
// nears the ROI rect, while α (spreading identically in both pipelines)
// stays full: the normalised coarse bands then darken visibly near every
// image's footprint. Interior pixels are never written, so the in-place
// exterior writes race nothing.

kernel void pyramidExtendRect(
    device       half4*    buf        [[buffer(0)]],
    constant     BufDims&  dim        [[buffer(1)]],
    constant     uint2&    rectOrigin [[buffer(2)]],
    constant     uint2&    rectSize   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    bool insideX = gid.x >= rectOrigin.x && gid.x < rectOrigin.x + rectSize.x;
    bool insideY = gid.y >= rectOrigin.y && gid.y < rectOrigin.y + rectSize.y;
    if (insideX && insideY) return;

    uint sx = clamp(gid.x, rectOrigin.x, rectOrigin.x + rectSize.x - 1);
    uint sy = clamp(gid.y, rectOrigin.y, rectOrigin.y + rectSize.y - 1);
    half4 v = buf[idx2(sx, sy, dim.width)];
    buf[idx2(gid.x, gid.y, dim.width)] = half4(v.rgb, 0.0h);
}

// MARK: - pyramidLapAccDC
// Streaming, fused: coarsest-level term lap = G[N-1]; α-weighted add into
// acc[N-1].

kernel void pyramidLapAccDC(
    device const half4*    gaussBuf [[buffer(0)]],
    device       float4*   accBuf   [[buffer(1)]],
    constant     BufDims&  dim      [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    float4 g = float4(gaussBuf[i]);
    float  w = g.a;
    float4 acc = accBuf[i];
    acc.rgb += g.rgb * w;
    acc.a   += w;
    accBuf[i] = acc;
}

// MARK: - pyramidNormalize
// At coarsest level: rgb = sum(L·w)/sum(w). α stays as the total weight so
// pyramidFinalise can mark uncovered pixels transparent.

kernel void pyramidNormalize(
    device const float4*   accBuf [[buffer(0)]],
    device       float4*   outBuf [[buffer(1)]],
    constant     BufDims&  dim    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    float4 acc = accBuf[i];
    float3 rgb = (acc.a > 1e-6f) ? (acc.rgb / acc.a) : float3(0);
    outBuf[i] = float4(rgb, acc.a);
}

// MARK: - pyramidCollapseAddExpanded
// Streaming, fused collapse: expand the coarser result inline at this pixel
// and add this level's normalised detail. Replaces the split
// (pyramidExpandFloat → pyramidCollapseAdd) pipeline so the collapse pass
// only needs two adjacent pyramid levels in flight at once instead of two
// full canvas-sized pyramids.

kernel void pyramidCollapseAddExpanded(
    device const float4*   accBuf       [[buffer(0)]],
    device const float4*   coarseBuf    [[buffer(1)]],
    device       float4*   fineBuf      [[buffer(2)]],
    constant     BufDims&  fineDim      [[buffer(3)]],
    constant     BufDims&  coarseDim    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= fineDim.width || gid.y >= fineDim.height) return;

    int Wc = int(coarseDim.width);
    int Hc = int(coarseDim.height);
    int x  = int(gid.x), y = int(gid.y);

    float4 base = float4(0);
    for (int m = -2; m <= 2; ++m) {
        if (((x - m) & 1) != 0) continue;
        int xx = iclampi((x - m) / 2, 0, Wc - 1);
        for (int n = -2; n <= 2; ++n) {
            if (((y - n) & 1) != 0) continue;
            int yy = iclampi((y - n) / 2, 0, Hc - 1);
            base += 4.0f * kW[m + 2] * kW[n + 2]
                  * coarseBuf[idx2(uint(xx), uint(yy), uint(Wc))];
        }
    }

    uint i = idx2(gid.x, gid.y, fineDim.width);
    float4 acc = accBuf[i];
    float3 detail = (acc.a > 1e-6f) ? (acc.rgb / acc.a) : float3(0);
    fineBuf[i] = float4(detail + base.rgb, acc.a);
}

// MARK: - pyramidFinalise

kernel void pyramidFinalise(
    device const float4*   inBuf  [[buffer(0)]],
    device       half4*    outBuf [[buffer(1)]],
    constant     BufDims&  dim    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    float4 c = inBuf[i];
    half4 out = (c.a > 0.5f) ? half4(half3(decodeBlendDomain(c.rgb)), 1.0h)
                             : half4(0.0h);
    outBuf[i] = out;
}
