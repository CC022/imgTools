//
//  CameraPose.swift
//  panoDev
//
//  Coordinate conventions used throughout camera / projection code:
//
//    • Pixel space: origin top-left, x right, y down. Matches the rgba16Float
//      texture layout produced by ImageLoader.
//    • Camera space: x right, y down, z forward (into the scene). A 3-D point
//      `P_cam` projects to pixels as `K · P_cam` followed by homogeneous divide.
//    • World rotations: each camera has rotation `R` such that
//          P_cam_i = R_i · P_world.
//      The anchor camera has R = I.
//    • Homography convention (matches Stage 4): `H_ij : pSrc_i → pDst_j`.
//      Equivalently `K_j · R_j · R_iᵀ · K_i⁻¹`.
//

import simd
import Foundation

// MARK: - Intrinsics

struct Intrinsics: Sendable {
    /// Focal length in pixel units (NOT mm).
    var focal: Float
    /// Principal point in pixel coords. Defaults to the image centre.
    var principalPoint: SIMD2<Float>

    init(focalPixels: Float, principalPoint: SIMD2<Float>) {
        self.focal = focalPixels
        self.principalPoint = principalPoint
    }

    /// Convert a 35mm-equivalent focal length to pixel focal length.
    /// `f_px = f_mm35 · max(W, H) / 36`  — the long-edge convention.
    static func fromMM35(focalMM35: Float, imageSize: SIMD2<Float>) -> Intrinsics {
        let f = focalMM35 * max(imageSize.x, imageSize.y) / 36.0
        return Intrinsics(focalPixels: f,
                          principalPoint: imageSize * 0.5)
    }

    var K: simd_float3x3 {
        simd_float3x3(
            SIMD3<Float>(focal, 0,     0),
            SIMD3<Float>(0,     focal, 0),
            SIMD3<Float>(principalPoint.x, principalPoint.y, 1)
        )
    }

    var Kinv: simd_float3x3 {
        let invF = 1 / focal
        return simd_float3x3(
            SIMD3<Float>(invF, 0,    0),
            SIMD3<Float>(0,    invF, 0),
            SIMD3<Float>(-principalPoint.x * invF,
                         -principalPoint.y * invF,
                         1)
        )
    }
}

// MARK: - Pose

struct CameraPose: Sendable {
    var intrinsics: Intrinsics
    /// Orthonormal rotation; world → camera.
    var rotation: simd_float3x3

    init(intrinsics: Intrinsics, rotation: simd_float3x3 = matrix_identity_float3x3) {
        self.intrinsics = intrinsics
        self.rotation = rotation
    }
}

// MARK: - Free helpers

/// Project an unnormalised 3×3 matrix to the closest rotation in SO(3) via SVD.
/// Required after computing R̂ = K⁻¹ · H · K, which is generally not orthonormal.
func projectToSO3(_ M: simd_float3x3) -> simd_float3x3 {
    // simd has no SVD, but for 3×3 we can write it via eigendecomposition of MᵀM.
    // Simpler: iterative orthonormalisation via Newton's method on the polar
    // decomposition — converges in 4–6 iterations and avoids SVD entirely.
    //   X_{k+1} = ½ (X_k + (X_kᵀ)⁻¹)
    // Result is the orthonormal factor of the polar decomposition of M.
    var X = M
    for _ in 0..<10 {
        let Xt_inv = X.transpose.inverse
        let next = (X + Xt_inv) * 0.5
        let diff = simd_norm_inf(next.columns.0 - X.columns.0)
                 + simd_norm_inf(next.columns.1 - X.columns.1)
                 + simd_norm_inf(next.columns.2 - X.columns.2)
        X = next
        if diff < 1e-7 { break }
    }
    // Ensure proper rotation (det = +1, not -1).
    if simd_determinant(X) < 0 {
        X.columns.2 = -X.columns.2
    }
    return X
}

/// Decompose `H_ij` into the implied relative rotation `R_ij`, given both
/// cameras' intrinsics. Output is projected to SO(3).
///
///     P_cam_j = R_ij · P_cam_i             (rotation only between i and j)
///     H_ij    = K_j · R_ij · K_i⁻¹
///   ⇒ R_ij    = K_j⁻¹ · H_ij · K_i
func relativeRotation(from H: simd_float3x3,
                      Ki: Intrinsics,
                      Kj: Intrinsics) -> simd_float3x3 {
    projectToSO3(Kj.Kinv * H * Ki.K)
}

/// Convert axis-angle (3-vector, magnitude = angle in rad) to a 3×3 rotation.
/// Used by RotationRefiner to parameterise updates.
func rotationFromAxisAngle(_ aa: SIMD3<Float>) -> simd_float3x3 {
    let theta = simd_length(aa)
    if theta < 1e-9 { return matrix_identity_float3x3 }
    let k = aa / theta
    let c = cos(theta), s = sin(theta), C = 1 - c
    // Rodrigues' formula
    return simd_float3x3(
        SIMD3<Float>(c + k.x*k.x*C,
                     k.y*k.x*C + k.z*s,
                     k.z*k.x*C - k.y*s),
        SIMD3<Float>(k.x*k.y*C - k.z*s,
                     c + k.y*k.y*C,
                     k.z*k.y*C + k.x*s),
        SIMD3<Float>(k.x*k.z*C + k.y*s,
                     k.y*k.z*C - k.x*s,
                     c + k.z*k.z*C)
    )
}

/// Inverse of `rotationFromAxisAngle`. Returns axis-angle (3-vector).
func axisAngleFromRotation(_ R: simd_float3x3) -> SIMD3<Float> {
    // Stable conversion via the matrix-log form.
    let trace = R.columns.0.x + R.columns.1.y + R.columns.2.z
    let cos_t = simd_clamp((trace - 1) / 2, -1, 1)
    let theta = acos(cos_t)
    if theta < 1e-7 { return SIMD3<Float>(0, 0, 0) }
    let s = sin(theta)
    if abs(s) < 1e-7 {
        // theta ≈ π — fall back to diagonal sqrt
        let m00 = R.columns.0.x, m11 = R.columns.1.y, m22 = R.columns.2.z
        let x = sqrt(max(0, (m00 + 1) / 2))
        let y = sqrt(max(0, (m11 + 1) / 2)) * (R.columns.1.x >= 0 ? 1 : -1)
        let z = sqrt(max(0, (m22 + 1) / 2)) * (R.columns.2.x >= 0 ? 1 : -1)
        return SIMD3<Float>(x, y, z) * theta
    }
    let axis = SIMD3<Float>(
        R.columns.1.z - R.columns.2.y,
        R.columns.2.x - R.columns.0.z,
        R.columns.0.y - R.columns.1.x
    ) / (2 * s)
    return axis * theta
}
