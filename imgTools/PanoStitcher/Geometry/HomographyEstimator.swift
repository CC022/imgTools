//
//  HomographyEstimator.swift
//  panoDev
//
//  Robust homography estimation from a list of feature correspondences.
//
//  Pipeline:
//      1. Confidence-weighted RANSAC with 4-point Direct Linear Transform.
//      2. After convergence, refit using all inliers (overdetermined DLT).
//
//  All linear algebra is plain Swift over `simd` types; for the dense
//  least-squares step we use Gaussian elimination on the 8×8 normal-equation
//  matrix. The normal-equation conditioning is fine here because every DLT
//  call is preceded by Hartley normalisation, which keeps coordinates
//  near unit scale.
//

import simd
import Foundation

// MARK: - Result

struct RANSACResult: Sendable {
    /// Best-scoring homography (already refit on all inliers).
    let homography: Homography
    /// Indices into the input arrays that survived the inlier test.
    let inlierIndices: [Int]
    /// Number of RANSAC iterations actually performed (≤ maxIterations).
    let iterationsUsed: Int
}

// MARK: - Estimator

enum HomographyEstimator {

    /// Estimate a homography mapping `srcPoints[i] → dstPoints[i]` using
    /// confidence-weighted RANSAC and a 4-point DLT minimal solver.
    ///
    /// - Parameters:
    ///   - srcPoints, dstPoints: paired correspondences in pixel coords. Same length.
    ///   - weights:              per-correspondence quality in [0, 1] (e.g. LightGlue confidence).
    ///   - threshold:            symmetric reprojection-error cutoff in pixels for inliers.
    ///   - successProb:          desired probability that at least one all-inlier sample is drawn.
    ///   - maxIterations:        hard cap on RANSAC trials.
    ///   - minIterations:        minimum trials even if adaptive estimator says stop early.
    /// - Returns: best result, or nil if no homography with ≥ 4 inliers was found.
    static func estimate(srcPoints: [SIMD2<Float>],
                         dstPoints: [SIMD2<Float>],
                         weights: [Float],
                         threshold: Float = 3.0,
                         successProb: Float = 0.999,
                         maxIterations: Int = 2000,
                         minIterations: Int = 50) -> RANSACResult? {

        let n = srcPoints.count
        guard n >= 4, dstPoints.count == n, weights.count == n else { return nil }

        // ── Build cumulative weight distribution for sampling ──────────────────
        // Use w² to bias more strongly toward high-confidence matches.
        let cumulative: [Float] = {
            var acc: Float = 0
            return weights.map { w in acc += w * w; return acc }
        }()
        guard let totalWeight = cumulative.last, totalWeight > 0 else { return nil }

        let thresholdSq = threshold * threshold
        var rng = SystemRandomNumberGenerator()

        var bestInliers: [Int] = []
        var bestH: simd_float3x3 = matrix_identity_float3x3
        var iter = 0
        var k = maxIterations   // adaptive — shrinks as we find better models

        while iter < min(k, maxIterations) {
            iter += 1

            // Sample 4 distinct correspondences, weighted.
            guard let sample = sampleDistinct(
                cumulative: cumulative, total: totalWeight, k: 4, rng: &rng
            ) else { continue }

            // Solve 4-point DLT.
            let srcSamp = sample.map { srcPoints[$0] }
            let dstSamp = sample.map { dstPoints[$0] }
            guard let H = solveDLT(src: srcSamp, dst: dstSamp),
                  isReasonable(H) else { continue }

            // Need H⁻¹ for symmetric error.
            let det = simd_determinant(H)
            guard abs(det) > 1e-8 else { continue }
            let Hinv = H.inverse

            // Inlier count via symmetric reprojection error
            //   err = max( ||H·a - b||²,  ||H⁻¹·b - a||² )
            var inliers: [Int] = []
            inliers.reserveCapacity(n)
            for i in 0..<n {
                let predDst = applyHomography(H, srcPoints[i])
                let predSrc = applyHomography(Hinv, dstPoints[i])
                let errFwd = simd_distance_squared(predDst, dstPoints[i])
                let errBwd = simd_distance_squared(predSrc, srcPoints[i])
                if max(errFwd, errBwd) < thresholdSq {
                    inliers.append(i)
                }
            }

            // Update best + adaptive K
            if inliers.count > bestInliers.count {
                bestInliers = inliers
                bestH = H

                // Adaptive iteration count:
                //   K = ln(1 - p) / ln(1 - w⁴)
                //   where w = current best inlier ratio, p = desired success prob.
                let w = Float(inliers.count) / Float(n)
                if w > 0, w < 1 {
                    let pInlierSample = pow(w, Float(4))
                    let denom = log(max(1 - pInlierSample, Float.leastNormalMagnitude))
                    let kAdaptive = log(1 - successProb) / denom
                    let kClamped  = max(Float(minIterations),
                                        min(Float(maxIterations), kAdaptive))
                    k = Int(kClamped.rounded(.up))
                } else if w >= 1 {
                    break    // perfect — done
                }
            }
        }

        guard bestInliers.count >= 4 else { return nil }

        // ── Final refit on all inliers ─────────────────────────────────────────
        let inlierSrc = bestInliers.map { srcPoints[$0] }
        let inlierDst = bestInliers.map { dstPoints[$0] }
        var refined = bestH
        if let H = solveDLT(src: inlierSrc, dst: inlierDst), isReasonable(H) {
            refined = H
        }

        return RANSACResult(
            homography: Homography(matrix: refined),
            inlierIndices: bestInliers,
            iterationsUsed: iter
        )
    }

    // MARK: - 4-point / N-point DLT (Hartley-normalised, h33=1 form)

    /// Direct Linear Transform fit using the h33=1 parameterisation.
    /// Works for both the minimal 4-point case and overdetermined N>4.
    static func solveDLT(src: [SIMD2<Float>], dst: [SIMD2<Float>]) -> simd_float3x3? {
        guard src.count >= 4, src.count == dst.count else { return nil }

        // Hartley normalisation — *required* for stability of the h33=1 form.
        guard let (Tsrc, srcN) = hartleyNormalise(src),
              let (Tdst, dstN) = hartleyNormalise(dst) else { return nil }

        let n = src.count
        // Each correspondence gives 2 equations in 8 unknowns (h11 … h32).
        // Per the cross-multiplied homography:
        //   row 1: [ x,  y,  1,  0,  0,  0, -x'·x, -x'·y ] · h = x'
        //   row 2: [ 0,  0,  0,  x,  y,  1, -y'·x, -y'·y ] · h = y'
        var A = [Float](repeating: 0, count: 2 * n * 8)
        var b = [Float](repeating: 0, count: 2 * n)

        for i in 0..<n {
            let p = srcN[i], q = dstN[i]
            let r1 = 2 * i, r2 = 2 * i + 1

            A[r1 * 8 + 0] = p.x
            A[r1 * 8 + 1] = p.y
            A[r1 * 8 + 2] = 1
            A[r1 * 8 + 6] = -q.x * p.x
            A[r1 * 8 + 7] = -q.x * p.y
            b[r1] = q.x

            A[r2 * 8 + 3] = p.x
            A[r2 * 8 + 4] = p.y
            A[r2 * 8 + 5] = 1
            A[r2 * 8 + 6] = -q.y * p.x
            A[r2 * 8 + 7] = -q.y * p.y
            b[r2] = q.y
        }

        // Solve via normal equations: (AᵀA) h = (Aᵀb)
        // 2N × 8 → 8 × 8 — well-conditioned after Hartley.
        let AtA = matMulATA(A, rows: 2 * n, cols: 8)
        let Atb = matMulATb(A, b, rows: 2 * n, cols: 8)
        guard let h = solveLinearSystem(AtA, Atb, n: 8) else { return nil }

        // Pack into 3×3 matrix in normalised space (column-major).
        let Hnorm = simd_float3x3(
            SIMD3<Float>(h[0], h[3], h[6]),
            SIMD3<Float>(h[1], h[4], h[7]),
            SIMD3<Float>(h[2], h[5], 1)
        )

        // Denormalise: H = T_dst⁻¹ · H_norm · T_src
        return Tdst.inverse * Hnorm * Tsrc
    }

    // MARK: - Hartley normalisation

    /// Returns (T, T·points). Shifts centroid to origin and scales so that the
    /// mean distance from the origin is √2. Required preconditioning for DLT.
    private static func hartleyNormalise(_ pts: [SIMD2<Float>])
        -> (simd_float3x3, [SIMD2<Float>])?
    {
        guard !pts.isEmpty else { return nil }
        let n = Float(pts.count)
        let centroid = pts.reduce(SIMD2<Float>(0, 0), +) / n

        var sumDist: Float = 0
        for p in pts { sumDist += simd_distance(p, centroid) }
        let meanDist = sumDist / n
        guard meanDist > 1e-8 else { return nil }   // all points coincident

        let scale: Float = sqrt(2.0) / meanDist
        let T = simd_float3x3(
            SIMD3<Float>( scale, 0,     0),
            SIMD3<Float>( 0,     scale, 0),
            SIMD3<Float>(-scale * centroid.x, -scale * centroid.y, 1)
        )

        let normalised = pts.map { p -> SIMD2<Float> in
            let v = T * SIMD3<Float>(p.x, p.y, 1)
            return SIMD2<Float>(v.x / v.z, v.y / v.z)
        }
        return (T, normalised)
    }

    // MARK: - Sanity check

    private static func isReasonable(_ H: simd_float3x3) -> Bool {
        let det = simd_determinant(H)
        return det.isFinite && abs(det) > 1e-6 && abs(det) < 1e6
    }

    // MARK: - Sampling

    /// Pick `k` distinct indices weighted by `cumulative`. Returns nil if it
    /// can't find `k` distinct samples within a few attempts (degenerate input).
    private static func sampleDistinct(cumulative: [Float],
                                       total: Float,
                                       k: Int,
                                       rng: inout some RandomNumberGenerator) -> [Int]? {
        var picked = Set<Int>()
        var result: [Int] = []
        result.reserveCapacity(k)

        var attempts = 0
        let maxAttempts = k * 10 + 20
        while result.count < k && attempts < maxAttempts {
            attempts += 1
            let r = Float.random(in: 0..<total, using: &rng)
            // Lower-bound binary search on the cumulative array.
            var lo = 0
            var hi = cumulative.count - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if cumulative[mid] <= r { lo = mid + 1 } else { hi = mid }
            }
            if picked.insert(lo).inserted {
                result.append(lo)
            }
        }
        return result.count == k ? result : nil
    }

    // MARK: - Linear algebra helpers (row-major flat arrays)

    /// Apply 3×3 homography to a 2D point with homogeneous divide.
    private static func applyHomography(_ H: simd_float3x3,
                                         _ p: SIMD2<Float>) -> SIMD2<Float> {
        let h = H * SIMD3<Float>(p.x, p.y, 1)
        return SIMD2<Float>(h.x / h.z, h.y / h.z)
    }

    /// Computes AᵀA for a row-major matrix `A` of shape (rows × cols).
    /// Returns a (cols × cols) row-major flat array.
    private static func matMulATA(_ A: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cols * cols)
        for i in 0..<cols {
            for j in i..<cols {
                var s: Float = 0
                for k in 0..<rows {
                    s += A[k * cols + i] * A[k * cols + j]
                }
                out[i * cols + j] = s
                out[j * cols + i] = s   // symmetric
            }
        }
        return out
    }

    /// Computes Aᵀb for row-major A (rows × cols) and vector b (length rows).
    private static func matMulATb(_ A: [Float],
                                   _ b: [Float],
                                   rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cols)
        for i in 0..<cols {
            var s: Float = 0
            for k in 0..<rows {
                s += A[k * cols + i] * b[k]
            }
            out[i] = s
        }
        return out
    }

    /// In-place Gaussian elimination with partial pivoting on the augmented
    /// system [A | b]. `A` is row-major (n × n); `b` is length-n RHS.
    /// Returns x such that A·x = b, or nil if singular.
    private static func solveLinearSystem(_ AIn: [Float], _ bIn: [Float], n: Int)
        -> [Float]?
    {
        var A = AIn
        var b = bIn

        for col in 0..<n {
            // Find pivot row (largest abs value in current column).
            var pivotRow = col
            var pivotMag = abs(A[col * n + col])
            for r in (col + 1)..<n {
                let v = abs(A[r * n + col])
                if v > pivotMag {
                    pivotMag = v
                    pivotRow = r
                }
            }
            guard pivotMag > 1e-12 else { return nil }

            if pivotRow != col {
                for c in 0..<n {
                    A.swapAt(col * n + c, pivotRow * n + c)
                }
                b.swapAt(col, pivotRow)
            }

            // Eliminate below.
            let pivot = A[col * n + col]
            for r in (col + 1)..<n {
                let factor = A[r * n + col] / pivot
                if factor != 0 {
                    for c in col..<n {
                        A[r * n + c] -= factor * A[col * n + c]
                    }
                    b[r] -= factor * b[col]
                }
            }
        }

        // Back-substitution.
        var x = [Float](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<n {
                s -= A[r * n + c] * x[c]
            }
            x[r] = s / A[r * n + r]
        }
        return x
    }
}
