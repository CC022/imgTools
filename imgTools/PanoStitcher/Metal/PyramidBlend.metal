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
//    • pyramidFinalise           — float32 → half16 final output
//

#include <metal_stdlib>
using namespace metal;

// Burt-Adelson 5-tap generating kernel: [1, 4, 6, 4, 1] / 16
constant float kW[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

struct BufDims { uint width; uint height; };

static inline uint idx2(uint x, uint y, uint w) { return y * w + x; }
static inline int  iclampi(int v, int lo, int hi) { return min(max(v, lo), hi); }

// MARK: - pyramidReduce

kernel void pyramidReduce(
    device const half4*      inBuf  [[buffer(0)]],
    device       half4*      outBuf [[buffer(1)]],
    constant     BufDims&    inDim  [[buffer(2)]],
    constant     BufDims&    outDim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outDim.width || gid.y >= outDim.height) return;

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
    outBuf[idx2(gid.x, gid.y, outDim.width)] = half4(sum);
}

// MARK: - pyramidLapAccExpanded
// Streaming, fused: compute (G[k] − Expand(G[k+1])) inline at this pixel
// and accumulate it into acc[k], α-weighted by the fine Gaussian's α.
// Replaces the split (pyramidExpand → laplacianBuild → laplacianAccumulate)
// pipeline so neither a per-image Laplacian pyramid nor a per-level expand
// `temps` buffer is needed.

kernel void pyramidLapAccExpanded(
    device const half4*    fineGaussBuf   [[buffer(0)]],
    device const half4*    coarseGaussBuf [[buffer(1)]],
    device       float4*   accBuf         [[buffer(2)]],
    constant     BufDims&  fineDim        [[buffer(3)]],
    constant     BufDims&  coarseDim      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= fineDim.width || gid.y >= fineDim.height) return;

    int Wc = int(coarseDim.width);
    int Hc = int(coarseDim.height);
    int x  = int(gid.x), y = int(gid.y);

    float4 expanded = float4(0);
    for (int m = -2; m <= 2; ++m) {
        if (((x - m) & 1) != 0) continue;
        int xx = iclampi((x - m) / 2, 0, Wc - 1);
        for (int n = -2; n <= 2; ++n) {
            if (((y - n) & 1) != 0) continue;
            int yy = iclampi((y - n) / 2, 0, Hc - 1);
            float4 v = float4(coarseGaussBuf[idx2(uint(xx), uint(yy), uint(Wc))]);
            expanded += 4.0f * kW[m + 2] * kW[n + 2] * v;
        }
    }

    uint i = idx2(gid.x, gid.y, fineDim.width);
    float4 fine = float4(fineGaussBuf[i]);
    float4 lap  = fine - expanded;
    float  w    = fine.a;
    float4 acc  = accBuf[i];
    acc.rgb += lap.rgb * w;
    acc.a   += w;
    accBuf[i] = acc;
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

// MARK: - pyramidExpand  (legacy — superseded by pyramidLapAccExpanded)
// Reads half4 at the coarser level, writes float4 at the next-finer level.

kernel void pyramidExpand(
    device const half4*    inBuf  [[buffer(0)]],
    device       float4*   outBuf [[buffer(1)]],
    constant     BufDims&  inDim  [[buffer(2)]],
    constant     BufDims&  outDim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outDim.width || gid.y >= outDim.height) return;

    int Wi = int(inDim.width);
    int Hi = int(inDim.height);
    int x  = int(gid.x), y = int(gid.y);

    float4 sum = float4(0);
    for (int m = -2; m <= 2; ++m) {
        if (((x - m) & 1) != 0) continue;
        int xx = iclampi((x - m) / 2, 0, Wi - 1);
        for (int n = -2; n <= 2; ++n) {
            if (((y - n) & 1) != 0) continue;
            int yy = iclampi((y - n) / 2, 0, Hi - 1);
            float4 v = float4(inBuf[idx2(uint(xx), uint(yy), uint(Wi))]);
            sum += 4.0f * kW[m + 2] * kW[n + 2] * v;
        }
    }
    outBuf[idx2(gid.x, gid.y, outDim.width)] = sum;
}

// MARK: - pyramidExpandFloat

kernel void pyramidExpandFloat(
    device const float4*  inBuf  [[buffer(0)]],
    device       float4*  outBuf [[buffer(1)]],
    constant     BufDims& inDim  [[buffer(2)]],
    constant     BufDims& outDim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outDim.width || gid.y >= outDim.height) return;

    int Wi = int(inDim.width);
    int Hi = int(inDim.height);
    int x  = int(gid.x), y = int(gid.y);

    float4 sum = float4(0);
    for (int m = -2; m <= 2; ++m) {
        if (((x - m) & 1) != 0) continue;
        int xx = iclampi((x - m) / 2, 0, Wi - 1);
        for (int n = -2; n <= 2; ++n) {
            if (((y - n) & 1) != 0) continue;
            int yy = iclampi((y - n) / 2, 0, Hi - 1);
            sum += 4.0f * kW[m + 2] * kW[n + 2]
                 * inBuf[idx2(uint(xx), uint(yy), uint(Wi))];
        }
    }
    outBuf[idx2(gid.x, gid.y, outDim.width)] = sum;
}

// MARK: - pyramidHalfToFloat

kernel void pyramidHalfToFloat(
    device const half4*    inBuf  [[buffer(0)]],
    device       float4*   outBuf [[buffer(1)]],
    constant     BufDims&  dim    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    outBuf[i] = float4(inBuf[i]);
}

// MARK: - laplacianBuild

kernel void laplacianBuild(
    device const half4*    gaussBuf    [[buffer(0)]],
    device const float4*   expandedBuf [[buffer(1)]],
    device       float4*   lapBuf      [[buffer(2)]],
    constant     BufDims&  dim         [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    lapBuf[i] = float4(gaussBuf[i]) - expandedBuf[i];
}

// MARK: - laplacianAccumulate
// acc.rgb += L.rgb · w;  acc.α += w
// where w is the α-channel of THIS image's gauss[k] (the per-band blend mask).

kernel void laplacianAccumulate(
    device const float4*   lapBuf   [[buffer(0)]],
    device const half4*    gaussBuf [[buffer(1)]],
    device       float4*   accBuf   [[buffer(2)]],
    constant     BufDims&  dim      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);

    float4 lap = lapBuf[i];
    float  w   = float(gaussBuf[i].a);
    float4 acc = accBuf[i];
    acc.rgb += lap.rgb * w;
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

// MARK: - pyramidCollapseAdd

kernel void pyramidCollapseAdd(
    device const float4*   accBuf      [[buffer(0)]],
    device const float4*   expandedBuf [[buffer(1)]],
    device       float4*   outBuf      [[buffer(2)]],
    constant     BufDims&  dim         [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = idx2(gid.x, gid.y, dim.width);
    float4 acc = accBuf[i];
    float3 detail = (acc.a > 1e-6f) ? (acc.rgb / acc.a) : float3(0);
    float3 base   = expandedBuf[i].rgb;
    outBuf[i] = float4(detail + base, acc.a);
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
    half4 out = (c.a > 0.5f) ? half4(half3(c.rgb), 1.0h) : half4(0.0h);
    outBuf[i] = out;
}
