//
//  MetalImageDisplay.metal
//  panoDev
//
//  Houses everything needed to *show* and *edit* the canvas:
//
//    1. `EditParams` struct (shared layout with Swift)
//    2. `applyEdits()` — per-pixel post-processing math on scene-linear
//        extendedLinearSRGB pixels. Conventions follow pro raw editors:
//        • the *tonal axis* (curve pivots, zone weights) lives in
//          sRGB-encoded luma, so "midtones" means perceptual mid grey
//          (0.18 linear ≈ 0.46 encoded), not 0.5 linear (≈ 74% encoded);
//        • white balance is a Bradford chromatic adaptation along the
//          blackbody locus (mired-linear slider), not naive R/B gains;
//        • saturation / vibrance scale chroma in OKLab, preserving hue
//          and perceived lightness;
//        • highlights / shadows weight by a *low-frequency* luma proxy
//          (a coarse mip of the image), so recovering highlights keeps
//          local contrast instead of flattening bright regions.
//    3. `metalImageVertex` + `metalImageFragment` — display path with
//        mip-chain bilinear/trilinear sampling (no more nearest-neighbour
//        aliasing when zoomed out). The fragment applies `applyEdits`
//        inline, so the preview is what the export kernel produces.
//    4. `applyEditsKernel` — export path. Bakes the same `applyEdits` into
//        a destination CanvasBuffer for HEIF10 save, sampling the same
//        zone-luma proxy so preview and export stay identical.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - EditParams (mirrors Swift `EditParams`)

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
    float  exposure;        // stops, [-6, 6]
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

// How strongly the highlights/shadows zone weights follow the blurred
// low-frequency luma instead of the pixel's own luma. 0 = pure per-pixel
// (flat look when recovering), 1 = pure local mask (max local contrast,
// some halo risk at hard edges).
constant float kZoneMix = 0.6f;

// MARK: - Tonal-axis helpers (sRGB encode/decode)

inline float srgbEncode(float y) {
    y = clamp(y, 0.0f, 1.0f);
    return (y <= 0.0031308f) ? 12.92f * y
                             : 1.055f * powr(y, 1.0f / 2.4f) - 0.055f;
}

inline float srgbDecode(float e) {
    e = clamp(e, 0.0f, 1.0f);
    return (e <= 0.04045f) ? e / 12.92f
                           : powr((e + 0.055f) / 1.055f, 2.4f);
}

// MARK: - OKLab (saturation / vibrance work here)

inline float safeCbrt(float x) {
    return sign(x) * powr(fabs(x), 1.0f / 3.0f);
}

inline float3 linearSRGBToOKLab(float3 c) {
    float l = dot(c, float3(0.4122214708f, 0.5363325363f, 0.0514459929f));
    float m = dot(c, float3(0.2119034982f, 0.6806995451f, 0.1073969566f));
    float s = dot(c, float3(0.0883024619f, 0.2817188376f, 0.6299787005f));
    float l_ = safeCbrt(l), m_ = safeCbrt(m), s_ = safeCbrt(s);
    return float3(0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
                  1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
                  0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_);
}

inline float3 oklabToLinearSRGB(float3 lab) {
    float l_ = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
    float m_ = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
    float s_ = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return float3(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                  -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                  -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
}

// MARK: - White balance (Bradford adaptation)

// Slider semantics match Lightroom: temperature + warms the image,
// tint + pushes magenta. Implemented as Bradford adaptation from an
// assumed scene illuminant (mired-linear in the slider) to D65.
inline float3x3 whiteBalanceMatrix(float temperature, float tint) {
    // Assumed scene illuminant: D65 is ~153.8 mired; ±1 slider sweeps
    // ∓89 mired (≈ 15400 K blue … 4100 K amber). Correcting a cool
    // illuminant warms the image and vice versa.
    float mired = 153.846f - temperature * 89.0f;
    float T = clamp(1.0e6f / mired, 2000.0f, 25000.0f);

    // CCT → blackbody xy (Kim et al. cubic spline approximation).
    float x;
    if (T < 4000.0f) {
        x = -0.2661239e9f / (T*T*T) - 0.2343589e6f / (T*T)
          + 0.8776956e3f / T + 0.179910f;
    } else {
        x = -3.0258469e9f / (T*T*T) + 2.1070379e6f / (T*T)
          + 0.2226347e3f / T + 0.240390f;
    }
    float x2 = x * x, x3 = x2 * x;
    float y;
    if (T < 2222.0f) {
        y = -1.1063814f*x3 - 1.34811020f*x2 + 2.18555832f*x - 0.20219683f;
    } else if (T < 4000.0f) {
        y = -0.9549476f*x3 - 1.37418593f*x2 + 2.09137015f*x - 0.16748867f;
    } else {
        y = +3.0817580f*x3 - 5.87338670f*x2 + 3.75112997f*x - 0.37001483f;
    }
    // Tint: green–magenta offset perpendicular-ish to the locus. + slider
    // assumes a greener illuminant, so the correction pushes magenta.
    y += tint * 0.04f;

    // xyY (Y = 1) → XYZ for the scene white; D65 reference.
    float3 srcW = float3(x / y, 1.0f, (1.0f - x - y) / y);
    const float3 d65W = float3(0.95047f, 1.0f, 1.08883f);

    // Bradford cone response.
    const float3x3 B = float3x3(float3( 0.8951f, -0.7502f,  0.0389f),
                                float3( 0.2664f,  1.7135f, -0.0685f),
                                float3(-0.1614f,  0.0367f,  1.0296f));
    const float3x3 Binv = float3x3(float3( 0.9869929f,  0.4323053f, -0.0085287f),
                                   float3(-0.1470543f,  0.5183603f,  0.0400428f),
                                   float3( 0.1599627f,  0.0492912f,  0.9684867f));
    // sRGB (D65) ↔ XYZ.
    const float3x3 toXYZ = float3x3(float3(0.4124564f, 0.2126729f, 0.0193339f),
                                    float3(0.3575761f, 0.7151522f, 0.1191920f),
                                    float3(0.1804375f, 0.0721750f, 0.9503041f));
    const float3x3 toRGB = float3x3(float3( 3.2404542f, -0.9692660f,  0.0556434f),
                                    float3(-1.5371385f,  1.8760108f, -0.2040259f),
                                    float3(-0.4985314f,  0.0415560f,  1.0572252f));

    float3 lmsSrc = B * srcW;
    float3 lmsDst = B * d65W;
    float3 g = lmsDst / max(lmsSrc, float3(1e-6f));
    float3x3 diag = float3x3(float3(g.x, 0, 0), float3(0, g.y, 0), float3(0, 0, g.z));
    return toRGB * (Binv * (diag * (B * toXYZ)));
}

// MARK: - Selective color

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
//
// `zoneLuma` is the *pre-edit* low-frequency luminance at this pixel (a
// coarse mip sample); pass the pixel's own luma when no proxy is available
// and the behaviour degrades to plain per-pixel weighting.

inline float3 applyEdits(float3 rgb, float zoneLuma, EditParams p) {
    // 1. Exposure — clean multiplicative gain in stops.
    float ev = exp2(p.exposure);
    rgb      *= ev;
    zoneLuma *= ev;

    // 2+3. White balance — Bradford chromatic adaptation (see above).
    if (p.temperature != 0.0f || p.tint != 0.0f) {
        rgb = whiteBalanceMatrix(p.temperature, p.tint) * rgb;
    }

    float luma = dot(rgb, kLumaWeights);

    // Tonal axis for the zone-weighted ops: blend of the pixel's own luma
    // and the blurred local luma (local contrast preservation), in encoded
    // (perceptual) space. HDR (≥ 1 linear) saturates to 1.
    float zl    = mix(luma, max(zoneLuma, 0.0f), kZoneMix);
    float eZone = srgbEncode(zl);

    // 4. Highlights — log-space gain keyed by the *local* bright-zone
    //    weight: recovering highlights darkens bright regions while each
    //    pixel keeps its contrast relative to the neighbourhood.
    //    Direction matches Lightroom: + brightens, − recovers.
    if (p.highlights != 0.0f) {
        float w = smoothstep(0.60f, 1.0f, eZone);
        float g = exp2(p.highlights * w * 1.5f);
        rgb  *= g;
        luma *= g;
    }

    // 5. Shadows — symmetric on the dark end (below perceptual mid).
    if (p.shadows != 0.0f) {
        float w = 1.0f - smoothstep(0.0f, 0.45f, eZone);
        float g = exp2(p.shadows * w * 1.5f);
        rgb  *= g;
        luma *= g;
    }

    // 6. Black point — remap of the encoded black anchor. + crushes
    //    ([b, 1] → [0, 1]), − lifts (classic lift, white stays anchored).
    //    Applied as a luminance ratio so hue is preserved; for near-black
    //    pixels (no ratio to scale) the lift is added achromatically.
    if (p.blackPoint != 0.0f) {
        float eL   = srgbEncode(luma);
        float b    = p.blackPoint * 0.25f;
        float eNew = (b >= 0.0f) ? max((eL - b) / (1.0f - b), 0.0f)
                                 : eL + (-b) * (1.0f - eL);
        float Lnew = srgbDecode(eNew);
        if (luma > 1.0f) {
            // HDR pass-through (matches contrast below).
        } else if (luma > 1e-3f) {
            rgb *= Lnew / luma;
        } else {
            rgb += float3(Lnew - luma);
        }
        luma = dot(rgb, kLumaWeights);
    }

    // 7. Brightness — midtone-targeted gain. The bell peaks at perceptual
    //    mid grey (0.5 encoded ≈ 0.18 linear) and tapers to identity at
    //    black and at 1.0+, so HDR highlights are untouched.
    if (p.brightness != 0.0f) {
        float eL = srgbEncode(luma);
        float w  = 4.0f * eL * (1.0f - eL);
        float g  = exp2(p.brightness * w);
        rgb  *= g;
        luma *= g;
    }

    // 8. Contrast — sine-based S-curve on *encoded* luma, pivoting at
    //    perceptual mid grey. Anchored at 0, 0.5, 1 encoded; monotonic for
    //    |amount| ≤ 1 (A = 0.15 → min slope 1 − 2πA ≈ 0.058 > 0). HDR
    //    (luma > 1) passes through: the curve is identity at encoded 1.
    if (p.contrast != 0.0f) {
        const float TWO_PI = 6.28318530718f;
        const float A      = 0.15f;
        float eL   = srgbEncode(luma);
        float eNew = eL - sin(TWO_PI * eL) * p.contrast * A;
        float Lnew = srgbDecode(eNew);
        if (luma > 1e-4f && luma <= 1.0f) {
            rgb *= Lnew / luma;
            luma = Lnew;
        }
    }

    // 9+10. Saturation & vibrance — chroma scaling in OKLab: hue and
    //    perceived lightness are preserved exactly, so big moves stay
    //    clean instead of shifting colours. Vibrance weights by
    //    1 − normalised chroma (dull colours move more) and protects the
    //    skin-tone hue range from positive boosts.
    if (p.saturation != 0.0f || p.vibrance != 0.0f) {
        float3 lab = linearSRGBToOKLab(rgb);
        float  C   = length(lab.yz);
        float  f   = 1.0f + p.saturation;
        if (p.vibrance != 0.0f && C > 1e-6f) {
            float cNorm = clamp(C / 0.18f, 0.0f, 1.0f);
            float boost = p.vibrance * (1.0f - cNorm);
            if (p.vibrance > 0.0f) {
                // Skin protection: hues around ~50° in OKLab move less.
                float hueDeg = atan2(lab.z, lab.y) * 57.2957795f;
                float dSkin  = fabs(fmod(hueDeg - 50.0f + 540.0f, 360.0f) - 180.0f);
                boost *= 1.0f - 0.7f * exp(-(dSkin * dSkin) / (2.0f * 25.0f * 25.0f));
            }
            f *= 1.0f + boost;
        }
        lab.yz *= max(f, 0.0f);
        rgb  = oklabToLinearSRGB(lab);
        luma = dot(rgb, kLumaWeights);
    }

    // 11. Three-zone colour balance. Each tint is an RGB push (already
    //     scaled in Swift from the wheel position), weighted by encoded-
    //     luma zones: shadows below perceptual mid, highlights above,
    //     midtones a bell peaking at mid grey. Weights sum to 1 over the
    //     displayable range. Clamp ≥ 0 after the additive push so negative
    //     tints never propagate negative RGB into the HSV math below.
    {
        float eL = srgbEncode(luma);
        float wS = 1.0f - smoothstep(0.0f, 0.5f, eL);
        float wH = smoothstep(0.5f, 1.0f, eL);
        float wM = 1.0f - wS - wH;
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

// MARK: - Bilinear buffer sampling (shared by display + export)

inline float3 sampleBilinear(device const half4* buf, float2 dims, float2 px) {
    float2 p  = clamp(px, float2(0.0f), dims - 1.0f);
    float2 p0 = floor(p);
    float2 f  = p - p0;
    int x0 = int(p0.x), y0 = int(p0.y);
    int x1 = min(x0 + 1, int(dims.x) - 1);
    int y1 = min(y0 + 1, int(dims.y) - 1);
    uint W = uint(dims.x);
    float3 c00 = float3(buf[uint(y0) * W + uint(x0)].rgb);
    float3 c01 = float3(buf[uint(y0) * W + uint(x1)].rgb);
    float3 c10 = float3(buf[uint(y1) * W + uint(x0)].rgb);
    float3 c11 = float3(buf[uint(y1) * W + uint(x1)].rgb);
    return mix(mix(c00, c01, f.x), mix(c10, c11, f.x), f.y);
}

// Base-pixel coordinate → mip-level coordinate (pixel centres aligned).
inline float2 mipCoord(float2 basePx, float invScale) {
    return (basePx + 0.5f) * invScale - 0.5f;
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
    float2 levelADims;    // bound mip A dims
    float2 levelBDims;    // bound mip B dims (== A when no trilinear blend)
    float2 zoneDims;      // zone-luma mip dims
    float  invA;          // base px → mip A px (1 / 2^kA)
    float  invB;
    float  invZone;
    float  levelBlend;    // trilinear fraction between A and B
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
                                  device const half4*        mipA  [[buffer(0)]],
                                  device const half4*        mipB  [[buffer(1)]],
                                  device const half4*        zone  [[buffer(2)]],
                                  constant MetalImageUniforms& u   [[buffer(3)]],
                                  constant EditParams&       edits [[buffer(4)]]) {
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

    // Trilinear: bilinear within the two mips bracketing the minification
    // factor, lerped by `levelBlend` (Swift picks the levels per frame —
    // zoom is uniform across the view).
    float3 c = sampleBilinear(mipA, u.levelADims, mipCoord(srcPx, u.invA));
    if (u.levelBlend > 0.0f) {
        float3 cB = sampleBilinear(mipB, u.levelBDims, mipCoord(srcPx, u.invB));
        c = mix(c, cB, u.levelBlend);
    }

    float zoneLuma = dot(sampleBilinear(zone, u.zoneDims, mipCoord(srcPx, u.invZone)),
                         kLumaWeights);

    float3 edited = applyEdits(c, zoneLuma, edits);
    return half4(half3(edited), 1.0h);
}

// MARK: - Export path (compute kernel)
//
// Slot order matches Swift `Compute.encode` convention:
//   buffers: [src, dst, zone] → 0, 1, 2
//   dims:    [dim, zoneDim]   → 3, 4
//   bytes:   [params, invZone]→ 5, 6

struct EditDims { uint width; uint height; };

kernel void applyEditsKernel(
    device const half4*    src     [[buffer(0)]],
    device       half4*    dst     [[buffer(1)]],
    device const half4*    zone    [[buffer(2)]],
    constant     EditDims& dim     [[buffer(3)]],
    constant     EditDims& zoneDim [[buffer(4)]],
    constant     EditParams& p     [[buffer(5)]],
    constant     float&    invZone [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim.width || gid.y >= dim.height) return;
    uint i = gid.y * dim.width + gid.x;

    half4 c = src[i];
    if (c.a < 0.5h) { dst[i] = c; return; }   // pass through panorama void

    float2 zDims = float2(float(zoneDim.width), float(zoneDim.height));
    float zoneLuma = dot(sampleBilinear(zone, zDims,
                                        mipCoord(float2(gid), invZone)),
                         kLumaWeights);

    // Clamp negatives on the export path so HLG/PQ encoders — whose
    // transfer curves are undefined for x < 0 — never see sub-black
    // values. The display path leaves negatives intact because the linear
    // extendedLinearSRGB color space handles them gracefully.
    float3 edited = max(applyEdits(float3(c.rgb), zoneLuma, p), 0.0f);
    dst[i] = half4(half3(edited), c.a);
}
