//
//  Homography.swift
//  panoDev
//
//  Thin value-type wrapper around `simd_float3x3` representing a projective
//  transform between two image planes. Convention used throughout panoDev:
//
//      pB = H · pA            (H maps from image A's pixel space → image B)
//
//  i.e. for a Match (kpA, kpB), the relation `H · kpA ≈ kpB` should hold (up to
//  homogeneous-divide). To warp image B back into image A's frame, use `H.inverse`.
//

import simd
import Foundation

struct Homography: Sendable {

    /// Column-major 3×3 matrix. Bottom-right element is generally normalised
    /// to 1 by `HomographyEstimator`, but callers should not rely on that.
    var matrix: simd_float3x3

    static let identity = Homography(matrix: matrix_identity_float3x3)

    /// Apply the homography to a 2D point: `H · [x, y, 1]ᵀ`, then dehomogenise.
    func apply(_ p: SIMD2<Float>) -> SIMD2<Float> {
        let h = matrix * SIMD3<Float>(p.x, p.y, 1)
        let w = h.z
        // Avoid division by ~0; surface as "point at infinity" mapped to a huge
        // coordinate. Caller decides what to do with it.
        guard abs(w) > 1e-12 else { return SIMD2<Float>(repeating: .infinity) }
        return SIMD2<Float>(h.x / w, h.y / w)
    }

    var inverse: Homography {
        Homography(matrix: matrix.inverse)
    }

    var determinant: Float {
        simd_determinant(matrix)
    }

    /// Returns a single-line debug string: `[h11 h12 h13; h21 h22 h23; h31 h32 h33]`
    var debugDescription: String {
        let m = matrix
        return String(
            format: "[% .4f % .4f % .4f; % .4f % .4f % .4f; % .4f % .4f % .4f]",
            m.columns.0.x, m.columns.1.x, m.columns.2.x,
            m.columns.0.y, m.columns.1.y, m.columns.2.y,
            m.columns.0.z, m.columns.1.z, m.columns.2.z
        )
    }
}
