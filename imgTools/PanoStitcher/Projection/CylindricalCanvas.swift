//
//  CylindricalCanvas.swift
//  panoDev
//
//  Cylindrical projection helpers and the output-canvas geometry.
//
//  Cylinder coordinates (θ, h):
//      θ = atan2(X, Z)             in radians, anchor camera looks along +Z so θ ≈ 0
//      h = Y / sqrt(X² + Z²)       dimensionless cylinder height
//  Canvas pixel:
//      px = (θ - origin.θ) · radius
//      py = (h - origin.h) · radius
//  Inverse (used by the warp kernel):
//      θ = origin.θ + px / radius,   h = origin.h + py / radius
//      worldRay = (sin θ, h, cos θ)
//

import simd
import Foundation

// MARK: - Canvas

struct CylindricalCanvas: Sendable {
    /// Pixels per radian along the equator. Defaults to anchor's pixel focal length.
    var radius: Float
    /// (θ, h) of canvas pixel (0, 0).
    var origin: SIMD2<Float>
    /// (width, height) in pixels.
    var size: SIMD2<Int>

    /// Build canvas dimensions from per-image poses + their pixel sizes.
    /// `radius` defaults to the anchor's pixel focal length, which keeps
    /// 1 canvas pixel ≈ 1 input pixel along the equator.
    static func compute(poses: [CameraPose],
                        imageSizes: [SIMD2<Float>],
                        anchorIndex: Int) -> CylindricalCanvas {
        precondition(poses.count == imageSizes.count && !poses.isEmpty)
        let radius = poses[anchorIndex].intrinsics.focal

        var thMin: Float =  .infinity
        var thMax: Float = -.infinity
        var hMin:  Float =  .infinity
        var hMax:  Float = -.infinity

        for (pose, size) in zip(poses, imageSizes) {
            let corners: [SIMD2<Float>] = [
                SIMD2(0,       0),
                SIMD2(size.x,  0),
                SIMD2(size.x,  size.y),
                SIMD2(0,       size.y),
            ]
            for c in corners {
                let p = projectImagePixelToCylinder(c, pose: pose)
                thMin = min(thMin, p.x); thMax = max(thMax, p.x)
                hMin  = min(hMin,  p.y); hMax  = max(hMax,  p.y)
            }
        }

        // 1-pixel pad in cylinder units.
        let pad: Float = 1.0 / radius
        thMin -= pad; thMax += pad
        hMin  -= pad; hMax  += pad

        if thMax - thMin > 2 * .pi - 0.1 {
            dbg("[Canvas] WARNING: θ-span ≈ \(thMax - thMin) rad — wraparound not handled yet")
        }

        let width  = max(1, Int(((thMax - thMin) * radius).rounded(.up)))
        let height = max(1, Int(((hMax  - hMin)  * radius).rounded(.up)))

        return CylindricalCanvas(
            radius: radius,
            origin: SIMD2<Float>(thMin, hMin),
            size:   SIMD2<Int>(width, height)
        )
    }
}

// MARK: - Projection

/// Forward-project an image-space pixel to cylinder (θ, h) via its camera pose.
/// `pose.rotation` is world→camera, so camera→world is `pose.rotation.transpose`.
func projectImagePixelToCylinder(_ pixel: SIMD2<Float>,
                                  pose: CameraPose) -> SIMD2<Float> {
    let camRay   = pose.intrinsics.Kinv * SIMD3<Float>(pixel.x, pixel.y, 1)
    let worldRay = pose.rotation.transpose * camRay
    let theta    = atan2(worldRay.x, worldRay.z)
    let radial   = sqrt(worldRay.x * worldRay.x + worldRay.z * worldRay.z)
    let h        = worldRay.y / max(radial, 1e-9)
    return SIMD2<Float>(theta, h)
}

// MARK: - GPU params

/// Mirrors the `Params` struct in `CylindricalWarp.metal`. Field order and types
/// must stay in sync; see that file for layout notes.
///
/// Layout (must match Metal's padded struct):
///   R               48 bytes  (float3x3 is 3 × 16-byte rows in MSL)
///   focal            4
///   canvasRadius     4
///   principalPoint   8
///   canvasOrigin     8
///   imageSize        8
///   canvasSize       8
///   _pad             8  ← brings total to 96, matching Metal's 16-byte-aligned struct
struct CylindricalWarpParams {
    var R: simd_float3x3
    var focal: Float
    var canvasRadius: Float
    var principalPoint: SIMD2<Float>
    var canvasOrigin: SIMD2<Float>
    var imageSize: SIMD2<Float>
    var canvasSize: SIMD2<Float>
    var _pad: SIMD2<Float> = .zero   // padding to reach 96 bytes (Metal struct alignment)
}
