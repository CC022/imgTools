//
//  MetalImageDisplay.metal
//  panoDev
//
//  Houses everything needed to *show* and *edit* the canvas:
//
//    1. `EditParams` struct (shared layout with Swift)
//    2. `applyEdits()` — per-pixel post-processing math, in scene-linear
//        extendedLinearSRGB. Algorithms model RawTherapee / Lightroom
//        conventions: highlights / shadows / brightness operate on
//        luminance and are applied as a uniform RGB scale (the "luminance-
//        preserving" pattern), which avoids the hue shifts that channel-
//        wise gain or contrast curves introduce.
//    3. `metalImageVertex` + `metalImageFragment` — display path. Fragment
//        applies `applyEdits` inline, so the on-screen preview is what the
//        export kernel produces (same code, no disparity by construction).
//    4. `applyEditsKernel` — export path. Bakes the same `applyEdits` into
//        a destination CanvasBuffer for HEIF10 save.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - EditParams (mirrors Swift `EditParams`)
//
// 10 scalar floats followed by three float3 colour-balance tints. The 8
// bytes of trailing pad after `tint` align the first float3 to a 16-byte
// boundary, matching what Swift's SIMD3<Float> insertion produces.

// Per-hue (h, s, l) shifts for the eight Selective Color bands. Each
// component is in [-1, 1]; range scales the bandwidth of every band
// uniformly. Centers are at the standard photo-app hues (see
// `kSelHueCenters` below).
struct SelectiveColorParams {
    float3 red;       // hue shift, sat scale, luma scale
    float3 orange;
    float3 yellow;
    float3 green;
    float3 aqua;
    float3 blue;
    float3 purple;
    float3 magenta;
    float  range;     // bandwidth multiplier, default 1.0
};

struct EditParams {
    float  exposure;        // stops, [-3, 3]
    float  highlights;      // [-1, 1]
    float  shadows;         // [-1, 1]
    float  brightness;      // [-1, 1]
    float  contrast;        // [-1, 1]
    float  blackPoint;      // [-1, 1]
    float  saturation;      // [-1, 1]
    float  vibrance;        // [-1, 1]
    float  temperature;     // [-1, 1]
    float  tint;            // [-1, 1]
    float3 shadowsTint;     // RGB shift for shadow region (≈ ±0.1)
    float3 midtonesTint;    // RGB shift for midtones
    float3 highlightsTint;  // RGB shift for highlights
    SelectiveColorParams selective;  // 8 × float3 + 1 float (with pad)
};

constant float3 kLumaWeights = float3(0.2126f, 0.7152f, 0.0722f);

// Standard photo-app hue centers (degrees on the 0–360 wheel). Eight bands,
// non-uniform spacing because human-named colours aren't equally spaced
// (red→orange→yellow are bunched, then a long gap to green).
constant float kSelHueCenters[8] = {
    0.0f, 30.0f, 60.0f, 120.0f, 180.0f, 220.0f, 280.0f, 320.0f
};

// HSV from a normalised RGB (max channel == 1). Hue in [0, 360).
inline float3 rgbToHsvNorm(float3 c) {
    float M = max(max(c.r, c.g), c.b);
    float m = min(min(c.r, c.g), c.b);
    float C = M - m;
    float h = 0.0f;
    if (C > 1e-6f) {
        if (M == c.r)      h = fmod((c.g - c.b) / C, 6.0f);
        else if (M == c.g) h = (c.b - c.r) / C + 2.0f;
        else               h = (c.r - c.g) / C + 4.0f;
        h *= 60.0f;
        if (h < 0.0f) h += 360.0f;
    }
    float s = (M > 1e-6f) ? (C / M) : 0.0f;
    return float3(h, s, M);
}

// HSV → RGB. h in [0, 360), s,v in [0, 1].
inline float3 hsvToRgbNorm(float3 hsv) {
    float h = hsv.x / 60.0f;
    float c = hsv.z * hsv.y;
    float x = c * (1.0f - fabs(fmod(h, 2.0f) - 1.0f));
    float m = hsv.z - c;
    float3 rgb;
    if      (h < 1.0f) rgb = float3(c, x, 0);
    else if (h < 2.0f) rgb = float3(x, c, 0);
    else if (h < 3.0f) rgb = float3(0, c, x);
    else if (h < 4.0f) rgb = float3(0, x, c);
    else if (h < 5.0f) rgb = float3(x, 0, c);
    else               rgb = float3(c, 0, x);
    return rgb + float3(m);
}

// Smallest signed difference (a - b) on a circular hue axis, in [-180, 180].
inline float hueDelta(float a, float b) {
    float d = fmod(a - b + 540.0f, 360.0f) - 180.0f;
    return d;
}

// Per-pixel selective colour adjustment. HDR-safe: factors out the max
// channel as `M`, runs hue/sat math on the chroma-normalised RGB, and
// reapplies `M` (with a luma scale) at the end so highlights above 1.0
// keep their over-range energy.
inline float3 applySelectiveColor(float3 rgb, SelectiveColorParams sc) {
    float M = max(max(rgb.r, rgb.g), rgb.b);
    if (M < 1e-5f) return rgb;
    float3 norm = rgb / M;                  // max channel == 1
    float3 hsv  = rgbToHsvNorm(norm);       // (hue, sat, 1)
    float h = hsv.x;
    float s = hsv.y;
    if (s < 1e-4f) return rgb;              // pure gray → no hue to bias

    // Band σ in degrees. Range slider scales it; default (1.0) gives
    // σ = 30°. Gaussian rolloff means adjacent bands — sitting 30°–60°
    // apart on a non-uniform circle — overlap smoothly at their
    // midpoints instead of both dropping to zero (which a smoothstep
    // with the same half-width does, leaving "dead-zone" hues that no
    // band can touch).
    float sigma = max(5.0f, 30.0f * sc.range);

    float3 picks[8] = {
        sc.red, sc.orange, sc.yellow, sc.green,
        sc.aqua, sc.blue, sc.purple, sc.magenta
    };

    float dh = 0.0f;   // hue rotation, degrees
    float ds = 0.0f;   // saturation shift, [-1..]
    float dl = 0.0f;   // luminance shift, [-1..]

    for (int i = 0; i < 8; ++i) {
        float diff = fabs(hueDelta(h, kSelHueCenters[i]));
        // Truncate at 3σ — beyond that the Gaussian weight is < 0.0001
        // and not worth the multiply.
        if (diff >= sigma * 3.0f) continue;
        // Gaussian roll-off: 1 at the center, ≈0.37 at one σ, smoothly
        // tapering past the band. Weighted by current saturation so
        // desaturated near-grays barely move.
        float t = diff / sigma;
        float w = exp(-t * t) * s;
        dh += w * picks[i].x * 30.0f;     // up to ±30° hue shift
        ds += w * picks[i].y;             // ±1 → up to ±100% sat
        dl += w * picks[i].z;             // ±1 → up to ±1 stop
    }

    if (dh == 0.0f && ds == 0.0f && dl == 0.0f) return rgb;

    float newH = fmod(h + dh + 720.0f, 360.0f);
    float newS = clamp(s * (1.0f + ds), 0.0f, 1.0f);
    float3 newNorm = hsvToRgbNorm(float3(newH, newS, 1.0f));
    return newNorm * M * exp2(dl);
}

// MARK: - applyEdits  (the math)

inline float3 applyEdits(float3 rgb, EditParams p) {
    // 1. Exposure — clean multiplicative gain in stops. Linear-domain natural fit.
    rgb *= exp2(p.exposure);

    // 2. Temperature — R/B per-channel gain (warm/cool). White balance is the
    //    raw-→-display interpretation step and conventionally runs *before*
    //    tone and saturation, so all downstream luma/chroma math sees the
    //    corrected colour.
    rgb.r *= 1.0f + p.temperature * 0.5f;
    rgb.b *= 1.0f - p.temperature * 0.5f;

    // 3. Tint — green/magenta axis on G.
    rgb.g *= 1.0f - p.tint * 0.2f;

    float luma = dot(rgb, kLumaWeights);

    // 4. Highlights — log-space gain weighted by bright luminance.
    //    Luminance-preserving: scale RGB by the same factor → no hue shift.
    //    Direction matches Lightroom: + brightens highlights, − recovers them.
    if (p.highlights != 0.0f) {
        float w = smoothstep(0.4f, 1.5f, luma);   // 0 below mid, 1 in highlights
        float g = exp2(p.highlights * w * 1.5f);  // up to ±1.5 stops in pure highlights
        rgb  *= g;
        luma *= g;
    }

    // 5. Shadows — symmetric on the dark end.
    if (p.shadows != 0.0f) {
        float w = 1.0f - smoothstep(0.0f, 0.4f, luma);
        float g = exp2(p.shadows * w * 1.5f);
        rgb  *= g;
        luma *= g;
    }

    // 6. Black point — small additive shift on the dark anchor.
    //    + crushes blacks, − lifts them. Clamped to ≥ 0 immediately so the
    //    HSV-based steps below (selective colour) and the chroma ratio in
    //    vibrance never see a negative max-channel.
    if (p.blackPoint != 0.0f) {
        rgb  = max(rgb - p.blackPoint * 0.05f, 0.0f);
        luma = dot(rgb, kLumaWeights);
    }

    // 7. Brightness — midtone-targeted exp2 gain. Effect peaks at L ≈ 0.5
    //    (perceptual mid in displayable range), tapers to identity at L=0
    //    and L=1+, so HDR highlights are untouched. This is the "Brightness"
    //    behaviour Photoshop / Lightroom users expect: midtones move, the
    //    extremes do not.
    if (p.brightness != 0.0f) {
        float Lc = clamp(luma, 0.0f, 1.0f);
        float w  = 4.0f * Lc * (1.0f - Lc);   // 0 at 0/1, 1 at Lc=0.5
        float g  = exp2(p.brightness * w);
        rgb  *= g;
        luma *= g;
    }

    // 8. Contrast — sine-based parametric S-curve. Anchored at 0, 0.5, 1
    //    so blacks-stay-black and whites-stay-white at any amount; provably
    //    monotonic for |amount| ≤ 1 (with A = 0.15, min derivative is
    //    1 − 2π·A ≈ 0.058 > 0). Avoids the dead-black clipping a linear
    //    stretch around mid produces, and matches the photographic-curve
    //    shape used by Lightroom / RawTherapee. HDR (luma > 1) is passed
    //    through unchanged because sin(2π·1) = 0.
    if (p.contrast != 0.0f) {
        const float TWO_PI = 6.28318530718f;
        const float A      = 0.15f;
        float xc   = clamp(luma, 0.0f, 1.0f);
        float adj  = sin(TWO_PI * xc) * p.contrast * A;
        float L_new = luma - adj;
        if (luma > 1e-4f) {
            rgb  *= L_new / luma;
            luma  = L_new;
        }
    }

    // 9. Saturation — chroma scale around grayscale.
    {
        rgb  = mix(float3(luma), rgb, 1.0f + p.saturation);
        luma = dot(rgb, kLumaWeights);
    }

    // 10. Vibrance — saturation weighted by `1 − chroma`. Already-saturated
    //     pixels (sky, skin) move less than dull ones.
    if (p.vibrance != 0.0f) {
        float maxC   = max(max(rgb.r, rgb.g), rgb.b);
        float minC   = min(min(rgb.r, rgb.g), rgb.b);
        float chroma = (maxC > 1e-4f) ? (maxC - minC) / maxC : 0.0f;
        float boost  = p.vibrance * (1.0f - chroma);
        rgb = mix(float3(luma), rgb, 1.0f + boost);
    }

    // 11. Three-zone colour balance.
    //     Each tint is an RGB push (already scaled in Swift from the
    //     wheel position). We weight by region: shadows for low luma,
    //     midtones for a mid bell, highlights for high luma. Weights
    //     sum to 1 over the displayable [0, 1] range so a uniform tint
    //     across all three wheels behaves like a flat colour push,
    //     while individual wheels affect only their region.
    //
    //     Clamp ≥ 0 after the additive push so a negative-tinted wheel
    //     on already-dark pixels doesn't propagate negative RGB into the
    //     HSV math in step 12.
    {
        float Lc = clamp(luma, 0.0f, 1.0f);
        float wS = 1.0f - smoothstep(0.0f, 0.5f, Lc);  // 1 at black, 0 above mid
        float wH = smoothstep(0.5f, 1.0f, Lc);         // 0 below mid, 1 at white
        float wM = 1.0f - wS - wH;                     // bell, peaks at 0.5
        rgb += p.shadowsTint    * wS;
        rgb += p.midtonesTint   * wM;
        rgb += p.highlightsTint * wH;
        rgb  = max(rgb, 0.0f);
    }

    // 12. Selective Color — per-hue HSL bias. Runs last so the hue bands
    //     it tests against see the post-balance pixel; no-op when all
    //     band shifts are zero.
    rgb = applySelectiveColor(rgb, p.selective);

    return rgb;
}

// MARK: - Display path (vertex + fragment)

struct VertexOut {
    float4 position [[position]];
};

struct MetalImageUniforms {
    float2 viewSize;
    float2 imageSize;     // full canvas-buffer size in pixels
    float2 offset;
    float  scale;
    float2 cropOrigin;    // sub-rect top-left in image pixels (0,0 for no crop)
    float2 cropSize;      // sub-rect size in image pixels (== imageSize for no crop)
};

vertex VertexOut metalImageVertex(uint vid [[vertex_id]]) {
    float2 pos[3] = {
        float2(-1.0,  -1.0),
        float2( 3.0,  -1.0),
        float2(-1.0,   3.0),
    };
    VertexOut out;
    out.position = float4(pos[vid], 0.0, 1.0);
    return out;
}

fragment half4 metalImageFragment(VertexOut in [[stage_in]],
                                  device const half4*        image  [[buffer(0)]],
                                  constant MetalImageUniforms& u    [[buffer(1)]],
                                  constant EditParams&        edits [[buffer(2)]]) {
    if (u.viewSize.x <= 0.0 || u.viewSize.y <= 0.0 ||
        u.imageSize.x <= 0.0 || u.imageSize.y <= 0.0 ||
        u.cropSize.x <= 0.0 || u.cropSize.y <= 0.0) {
        return half4(0, 0, 0, 1);
    }

    // Pan/zoom and fit-to-view operate over the *crop* sub-rect, so the
    // visible region exactly fills the viewport when no crop is active
    // (cropSize == imageSize) or zooms into the chosen sub-rect otherwise.
    float fit = min(u.viewSize.x / u.cropSize.x, u.viewSize.y / u.cropSize.y);
    float2 displayedSize = u.cropSize * fit * u.scale;
    float2 center = u.viewSize * 0.5 + u.offset;

    float2 p = in.position.xy;
    float2 uv = (p - (center - displayedSize * 0.5)) / displayedSize;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        return half4(0, 0, 0, 1);
    }
    float2 srcPx = u.cropOrigin + uv * u.cropSize;
    uint x = min(uint(srcPx.x), uint(u.imageSize.x - 1.0));
    uint y = min(uint(srcPx.y), uint(u.imageSize.y - 1.0));
    half4 c = image[y * uint(u.imageSize.x) + x];

    float3 edited = applyEdits(float3(c.rgb), edits);
    return half4(half3(edited), 1.0h);
}

// MARK: - Export path (compute kernel)
//
// Slot order matches Swift `Compute.encode` convention:
//   buffers: [src, dst]   → 0, 1
//   dims:    [dim]        → 2
//   bytes:   [params]     → 3

struct EditDims { uint width; uint height; };

kernel void applyEditsKernel(
    device const half4*    src   [[buffer(0)]],
    device       half4*    dst   [[buffer(1)]],
    constant     EditDims& dim   [[buffer(2)]],
    constant     EditParams& p   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = gid.y * dim.width + gid.x;

    half4 c = src[i];
    if (c.a < 0.5h) { dst[i] = c; return; }   // pass through panorama void

    // Clamp negatives on the export path so HLG/PQ encoders — whose
    // transfer curves are undefined for x < 0 — never see sub-black
    // values from blackPoint or color-balance subtraction. The display
    // path leaves negatives intact because the linear extendedLinearSRGB
    // color space handles them gracefully.
    float3 edited = max(applyEdits(float3(c.rgb), p), 0.0f);
    dst[i] = half4(half3(edited), c.a);
}
