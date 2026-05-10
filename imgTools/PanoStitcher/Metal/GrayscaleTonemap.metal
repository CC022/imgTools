//
//  GrayscaleTonemap.metal
//  panoDev
//
//  Converts an rgba16Float texture (scene-linear extendedLinearSRGB) to a flat
//  float32 buffer in row-major order, tone-mapped and gamma-encoded to [0, 1].
//  The buffer is laid out as [height × width] floats matching MLMultiArray [1,1,H,W].
//

#include <metal_stdlib>
using namespace metal;

/// Standard sRGB gamma (IEC 61966-2-1)
static float linearToSRGB(float y) {
    return (y <= 0.0031308f)
        ? 12.92f * y
        : 1.055f * powr(y, 1.0f / 2.4f) - 0.055f;
}

kernel void grayscaleTonemap(
    texture2d<float, access::read> inTexture  [[texture(0)]],
    device float*                  outBuffer  [[buffer(0)]],
    constant uint2&                dims       [[buffer(1)]],   // (width, height)
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = inTexture.read(gid);

    // Scene-linear luminance (BT.709 / sRGB primaries)
    float Y = 0.2126f * rgba.r + 0.7152f * rgba.g + 0.0722f * rgba.b;

    // Reinhard tone-map then sRGB gamma → [0, 1], matching SuperPoint training domain
    float y = Y / (1.0f + Y);
    float encoded = linearToSRGB(clamp(y, 0.0f, 1.0f));

    outBuffer[gid.y * dims.x + gid.x] = encoded;
}
