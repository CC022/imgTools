//
//  BlendDomain.h
//  panoDev
//
//  The encoding the Laplacian-pyramid blend operates in. Encoded by the
//  fused warp (`cylindricalWarpMasked`), decoded by `pyramidFinalise`; every
//  pyramid kernel in between is linear and domain-agnostic.
//
//  Why not linear light
//  ────────────────────
//  Multi-band blending hides photometric mismatch between frames by mixing
//  their low-frequency bands, and that correction is *additive* in whatever
//  domain the pyramid runs in. Exposure / tone mismatch is multiplicative,
//  so in linear light the correction lands on every pixel as an absolute
//  offset sized by the bright content (sky): +0.015 is 5% on sky but ×15 on
//  a 0.001 shadow — lifted blacks where a frame is darker than its
//  neighbours, crushed (even negative) blacks where it is brighter.
//
//  asinh(x / 2ε) is ln(x / ε) for x ≫ ε, so the additive band correction
//  becomes a smooth multiplicative gain field: black stays black, shadows
//  and highlights scale together. Below ε (the noise floor, where ratios
//  carry no information) it is linear. It is odd and defined for all reals,
//  so wide-gamut negatives (P3 colours outside sRGB) and sub-black decode
//  noise pass through without clamping, and it inverts exactly.
//
//  ε ≈ one 8-bit sRGB code value above black. The mid-grey pivot is
//  subtracted so rgba16f storage (densest near zero) keeps ~0.1% relative
//  precision around diffuse white instead of ~0.8%.
//

#pragma once
#include <metal_stdlib>

namespace blend_domain {
    constant float kToe   = 1.0f / 4096.0f;
    constant float kPivot = 6.602969578f;   // asinh(0.18 / (2·kToe))
}

static inline float3 encodeBlendDomain(float3 linearRGB) {
    return metal::asinh(linearRGB * (0.5f / blend_domain::kToe)) - blend_domain::kPivot;
}

static inline float3 decodeBlendDomain(float3 encoded) {
    return (2.0f * blend_domain::kToe) * metal::sinh(encoded + blend_domain::kPivot);
}
