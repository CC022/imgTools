//
//  ExposureCompensator.swift
//  panoDev
//
//  Brown & Lowe (2007) per-image gain compensation. Three independent N×N
//  solves (one per RGB channel) → exposure AND white-balance correction in
//  a single pass.
//
//  Statistics come from one CPU scan of the low-res prepass warps (rgba16f,
//  `.shared` storage): pairwise overlap means at ~1 MP are statistically
//  identical to full resolution, and a single scan covers all n(n−1)/2
//  pairs at once. The solved gains are applied on the GPU by the fused
//  `cylindricalWarpMasked` kernel during the streaming blend.
//

import Foundation
import Metal
import simd

nonisolated enum ExposureCompensator {

    private static let sigmaN: Double = 10.0 / 255.0
    private static let sigmaG: Double = 0.10

    /// Returns per-image RGB gain factors (length `warps.count`). Identity
    /// gains on trivial input (`n < 2`). All warps share one canvas size.
    static func estimateGains(warps: [CanvasBuffer]) -> [SIMD3<Float>] {
        let n = warps.count
        let identity = Array(repeating: SIMD3<Float>(repeating: 1), count: n)
        guard n >= 2, let first = warps.first else { return identity }
        let W = first.width, H = first.height

        let px: [UnsafePointer<Float16>] = warps.map {
            UnsafePointer($0.buffer.contents()
                .bindMemory(to: Float16.self, capacity: W * H * 4))
        }

        // Pairwise overlap stats: I[i][j] = mean RGB of i in overlap with j.
        // Overlap = both alphas exceed 0.5 (binary inside / outside masks).
        var I = Array(repeating: Array(repeating: SIMD3<Double>.zero, count: n), count: n)
        var N = Array(repeating: Array(repeating: 0,                   count: n), count: n)
        var covered = [Int](repeating: 0, count: n)
        for idx in 0..<(W * H) {
            var m = 0
            for k in 0..<n where Float(px[k][idx * 4 + 3]) > 0.5 {
                covered[m] = k
                m += 1
            }
            guard m >= 2 else { continue }
            for a in 0..<(m - 1) {
                let i = covered[a]
                let rgbI = SIMD3<Double>(Double(px[i][idx * 4]),
                                         Double(px[i][idx * 4 + 1]),
                                         Double(px[i][idx * 4 + 2]))
                for b in (a + 1)..<m {
                    let j = covered[b]
                    I[i][j] += rgbI
                    I[j][i] += SIMD3<Double>(Double(px[j][idx * 4]),
                                             Double(px[j][idx * 4 + 1]),
                                             Double(px[j][idx * 4 + 2]))
                    N[i][j] += 1
                    N[j][i] += 1
                }
            }
        }
        for i in 0..<n {
            for j in 0..<n where N[i][j] > 0 { I[i][j] /= Double(N[i][j]) }
        }

        // Per-channel solve.
        var gainsD = Array(repeating: SIMD3<Double>(repeating: 1), count: n)
        for c in 0..<3 {
            var Ic = Array(repeating: Array(repeating: 0.0, count: n), count: n)
            for i in 0..<n { for j in 0..<n { Ic[i][j] = I[i][j][c] } }
            for (i, g) in solveGains(n: n, I: Ic, N: N).enumerated() {
                gainsD[i][c] = g
            }
        }

        let gains = gainsD.map { SIMD3<Float>(Float($0.x), Float($0.y), Float($0.z)) }
        let report = gains.enumerated().map { i, g in
            String(format: "%d=(%.3f,%.3f,%.3f)", i, g.x, g.y, g.z)
        }.joined(separator: " ")
        dbg("[Exposure] gains: \(report)")
        return gains
    }

    // MARK: - Linear solve (Brown-Lowe gain system)

    private static func solveGains(n: Int, I: [[Double]], N: [[Int]]) -> [Double] {
        let lambda = (sigmaN * sigmaN) / (sigmaG * sigmaG)
        var A = Array(repeating: Array(repeating: 0.0, count: n), count: n)
        var b = Array(repeating: 0.0, count: n)
        for k in 0..<n {
            for j in 0..<n where j != k {
                let Nkj = Double(N[k][j]), Ikj = I[k][j], Ijk = I[j][k]
                A[k][k] += Nkj * (Ikj * Ikj + lambda)
                A[k][j] -= Nkj * Ikj * Ijk
                b[k]    += Nkj * lambda
            }
            if A[k][k] < 1e-12 { A[k][k] = 1; b[k] = 1 }
        }
        return gaussSolve(A: &A, b: &b, n: n)
    }

    private static func gaussSolve(A: inout [[Double]], b: inout [Double], n: Int) -> [Double] {
        for k in 0..<n {
            var pivot = k
            for r in (k + 1)..<n where abs(A[r][k]) > abs(A[pivot][k]) { pivot = r }
            if pivot != k { A.swapAt(k, pivot); b.swapAt(k, pivot) }
            guard abs(A[k][k]) > 1e-15 else { return Array(repeating: 1, count: n) }
            for r in (k + 1)..<n {
                let f = A[r][k] / A[k][k]
                if f == 0 { continue }
                for c in k..<n { A[r][c] -= f * A[k][c] }
                b[r] -= f * b[k]
            }
        }
        var x = Array(repeating: 0.0, count: n)
        for k in stride(from: n - 1, through: 0, by: -1) {
            var s = b[k]
            for c in (k + 1)..<n { s -= A[k][c] * x[c] }
            x[k] = s / A[k][k]
        }
        return x
    }
}
