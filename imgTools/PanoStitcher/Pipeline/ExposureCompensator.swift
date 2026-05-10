//
//  ExposureCompensator.swift
//  panoDev
//
//  Brown & Lowe (2007) per-image gain compensation. Buffer-backed.
//  Three independent N×N solves (one per RGB channel) → exposure AND
//  white-balance correction in a single pass.
//
//  Streaming-friendly split:
//
//    • `estimateGains(imageCount:canvas:makeWarped:)` — pairwise overlap
//      stats + Gauss elim, returning per-image gains. The warped buffers
//      are produced lazily by the caller's `makeWarped(i)` callback so
//      the estimator only ever holds the two warps for the current pair
//      plus the small partial-sum reduction buffers.
//
//    • `apply(gain:to:)` — single in-place RGB scale dispatch, used by
//      the streaming blend on each transient warp before accumulation.
//

import Metal
import simd
import Foundation

enum ExposureCompensator {

    private static let sigmaN: Double = 10.0 / 255.0
    private static let sigmaG: Double = 0.10

    // MARK: - Estimate

    /// Returns per-image RGB gain factors (length `n`). Identity gains on
    /// failure or trivial input (`n < 2`).
    static func estimateGains(imageCount n: Int,
                              canvas: CylindricalCanvas,
                              makeWarped: (Int) -> CanvasBuffer?) -> [SIMD3<Float>] {
        let identity = Array(repeating: SIMD3<Float>(repeating: 1), count: n)
        guard n >= 2 else { return identity }
        let ctx = PanoContext.shared
        guard let statsPSO = ctx.loadPSO("overlapStatsReducePair") else { return identity }

        let W = canvas.size.x, H = canvas.size.y
        let tgX = (W + 15) / 16, tgY = (H + 15) / 16, tgCount = tgX * tgY
        let bufBytes = tgCount * MemoryLayout<SIMD4<Float>>.stride

        guard let bufA = ctx.device.makeBuffer(length: bufBytes, options: .storageModeShared),
              let bufB = ctx.device.makeBuffer(length: bufBytes, options: .storageModeShared)
        else { return identity }

        // Pairwise overlap stats: I[i][j] = mean RGB of i in overlap with j.
        var I = Array(repeating: Array(repeating: SIMD3<Double>.zero, count: n), count: n)
        var N = Array(repeating: Array(repeating: 0,                   count: n), count: n)

        for i in 0..<n {
            for j in (i + 1)..<n {
                guard let warpedA = makeWarped(i),
                      let warpedB = makeWarped(j) else { continue }
                Compute.run { cb in
                    Compute.encode(cb, statsPSO,
                        buffers: [warpedA.buffer, warpedB.buffer, bufA, bufB],
                        dims:    [warpedA.dimsPacked],
                        gridW:   tgX * 16, gridH: tgY * 16)
                }
                let (sA, count) = sumPartials(buf: bufA, count: tgCount)
                let (sB, _    ) = sumPartials(buf: bufB, count: tgCount)
                if count > 0 {
                    I[i][j] = sA / Double(count)
                    I[j][i] = sB / Double(count)
                }
                N[i][j] = count; N[j][i] = count
            }
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

    // MARK: - Apply

    /// In-place RGB gain scale on a single warped buffer. No-op for identity.
    static func apply(gain: SIMD3<Float>, to warped: CanvasBuffer) {
        if gain == SIMD3<Float>(1, 1, 1) { return }
        let ctx = PanoContext.shared
        guard let applyPSO = ctx.loadPSO("applyExposureGain") else { return }
        var g = gain
        Compute.run { cb in
            withUnsafeBytes(of: &g) { raw in
                Compute.encode(cb, applyPSO,
                    buffers: [warped.buffer],
                    dims:    [warped.dimsPacked],
                    bytes:   [(raw.baseAddress!, raw.count)],
                    gridW: warped.width, gridH: warped.height)
            }
        }
    }

    // MARK: - CPU partial-sum reduction

    private static func sumPartials(buf: MTLBuffer, count: Int) -> (SIMD3<Double>, Int) {
        let p = buf.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        var sR = 0.0, sG = 0.0, sB = 0.0, total = 0.0
        for k in 0..<count {
            let v = p[k]
            sR += Double(v.x); sG += Double(v.y); sB += Double(v.z); total += Double(v.w)
        }
        return (SIMD3<Double>(sR, sG, sB), Int(total))
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
