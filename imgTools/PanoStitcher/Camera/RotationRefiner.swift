//
//  RotationRefiner.swift
//  panoDev
//
//  Levenberg-Marquardt bundle adjustment for a rotational panorama:
//  refines per-non-anchor-node rotations *and* a single shared focal length.
//
//  State: per-non-anchor 3-vec axis-angle (anchor R = I gauge-fixes the system),
//         plus one scalar f (shared across all cameras).
//  Residuals: symmetric reprojection error in pixels for every inlier match
//             across every edge:
//      r_ij(m) = π( K_j · R_j · R_iᵀ · K_i⁻¹ · pᵢ ) − pⱼ
//      r_ji(m) = π( K_i · R_i · R_jᵀ · K_j⁻¹ · pⱼ ) − pᵢ
//      where K_i = diag(f, f, 1) with the per-image principal point.
//
//  Self-calibrating focal length is the standard fix for residual ghosting
//  in panorama overlap regions: a 1 % focal-length error shifts edge pixels
//  by 1 % of half-image-width (≈ 30 px on a 5712-px shot), which the
//  rotation-only LM can't compensate.
//
//  Numerical Jacobian, normal-equation solve. Plenty fast for ≤ 20 images
//  and ≤ 10k inliers.
//

import simd
import Foundation

enum RotationRefiner {

    static func refine(graph: inout PanoGraph,
                       anchor: Int,
                       maxIters: Int = 30) {
        guard graph.nodes.count > 1, !graph.edges.isEmpty else { return }

        // Map node index → parameter slot. Anchor gets no slot.
        var slotOf = [Int: Int](minimumCapacity: graph.nodes.count - 1)
        var nextSlot = 0
        for i in graph.nodes.indices where i != anchor {
            slotOf[i] = nextSlot
            nextSlot += 1
        }
        let rotParamCount = nextSlot * 3
        let P = rotParamCount + 1                  // +1 for the shared focal
        let fSlot = rotParamCount                  // index of `f` in x

        // Pre-extract observations.
        struct Obs {
            let i: Int; let j: Int
            let pi: SIMD2<Float>; let pj: SIMD2<Float>
        }
        var obs: [Obs] = []
        for e in graph.edges {
            for inlierIdx in e.inliers {
                let m   = e.matches[inlierIdx]
                let kpi = graph.nodes[e.src].keypoints[m.indexA]
                let kpj = graph.nodes[e.dst].keypoints[m.indexB]
                obs.append(Obs(
                    i: e.src, j: e.dst,
                    pi: SIMD2(kpi.x, kpi.y),
                    pj: SIMD2(kpj.x, kpj.y)
                ))
            }
        }
        guard !obs.isEmpty else { return }
        let M = obs.count * 4    // 2 directions × 2 components

        // Per-node principal points (only `f` is shared & refined).
        let principalPoints: [SIMD2<Float>] = graph.nodes.map { $0.pose!.intrinsics.principalPoint }
        let f0 = graph.nodes[anchor].pose!.intrinsics.focal

        // Initial parameter vector.
        var x = [Float](repeating: 0, count: P)
        for (nodeIdx, slot) in slotOf {
            let aa = axisAngleFromRotation(graph.nodes[nodeIdx].pose!.rotation)
            x[slot * 3 + 0] = aa.x
            x[slot * 3 + 1] = aa.y
            x[slot * 3 + 2] = aa.z
        }
        x[fSlot] = f0

        // Closure: residual vector for a parameter setting.
        func residuals(_ x: [Float]) -> [Float] {
            let f = x[fSlot]

            // K^{-1} for each image at current f. K itself isn't materialised —
            // forward projection uses scalar f + principal point directly.
            var Ki  = [simd_float3x3](); Ki.reserveCapacity(graph.nodes.count)
            for pp in principalPoints {
                Ki.append(Intrinsics(focalPixels: f, principalPoint: pp).Kinv)
            }

            // Rotations.
            var R = [simd_float3x3](repeating: matrix_identity_float3x3,
                                    count: graph.nodes.count)
            for (nodeIdx, slot) in slotOf {
                R[nodeIdx] = rotationFromAxisAngle(SIMD3<Float>(
                    x[slot * 3], x[slot * 3 + 1], x[slot * 3 + 2]
                ))
            }

            var r = [Float](repeating: 0, count: M)
            for (k, o) in obs.enumerated() {
                let Ri = R[o.i],   Rj = R[o.j]
                let KiI = Ki[o.i], KjI = Ki[o.j]
                let ppI = principalPoints[o.i]
                let ppJ = principalPoints[o.j]

                // i → j
                let camI  = KiI * SIMD3<Float>(o.pi.x, o.pi.y, 1)
                let world = Ri.transpose * camI
                let camJ  = Rj * world
                if abs(camJ.z) > 1e-6 {
                    let predJ = SIMD2<Float>(camJ.x / camJ.z, camJ.y / camJ.z)
                                * f + ppJ
                    r[k * 4 + 0] = predJ.x - o.pj.x
                    r[k * 4 + 1] = predJ.y - o.pj.y
                }

                // j → i
                let camJ2  = KjI * SIMD3<Float>(o.pj.x, o.pj.y, 1)
                let worldJ = Rj.transpose * camJ2
                let camI2  = Ri * worldJ
                if abs(camI2.z) > 1e-6 {
                    let predI = SIMD2<Float>(camI2.x / camI2.z, camI2.y / camI2.z)
                                * f + ppI
                    r[k * 4 + 2] = predI.x - o.pi.x
                    r[k * 4 + 3] = predI.y - o.pi.y
                }
            }
            return r
        }

        func sumSq(_ v: [Float]) -> Float {
            var s: Float = 0
            for x in v { s += x * x }
            return s
        }

        // Per-parameter finite-difference step.
        // Rotations live in radians (~1 rad scale); focal is ~thousands of pixels,
        // so a uniform `h` would underflow on the focal column.
        let hRot: Float = 1e-4
        func hFor(slot: Int, currentX: [Float]) -> Float {
            if slot == fSlot { return max(0.5, 1e-4 * currentX[fSlot]) }
            return hRot
        }

        var lambda: Float = 1e-3
        var r0 = residuals(x)
        var cost = sumSq(r0)
        let cost0 = cost
        dbg("[LM] init cost=\(String(format: "%.1f", cost)) over \(M) residuals, \(P) params, f₀=\(String(format: "%.1f", f0)) px")

        for iter in 0..<maxIters {
            // Numerical Jacobian (forward differences) with per-column step.
            var J = [Float](repeating: 0, count: M * P)   // M rows × P cols
            for k in 0..<P {
                let h = hFor(slot: k, currentX: x)
                let saved = x[k]
                x[k] = saved + h
                let rk = residuals(x)
                x[k] = saved
                let invH = 1 / h
                for m in 0..<M {
                    J[m * P + k] = (rk[m] - r0[m]) * invH
                }
            }

            // JᵀJ (P × P) and Jᵀr (P).
            var JtJ = [Float](repeating: 0, count: P * P)
            var Jtr = [Float](repeating: 0, count: P)
            for i in 0..<P {
                var ji: Float = 0
                for m in 0..<M { ji += J[m * P + i] * r0[m] }
                Jtr[i] = -ji
                for j in i..<P {
                    var s: Float = 0
                    for m in 0..<M { s += J[m * P + i] * J[m * P + j] }
                    JtJ[i * P + j] = s
                    JtJ[j * P + i] = s
                }
            }
            for i in 0..<P { JtJ[i * P + i] *= (1 + lambda) }

            guard let dx = solveSymmetricLinear(JtJ, Jtr, n: P) else {
                lambda *= 10; continue
            }

            // Trial step. Clamp focal to a sane band (½f₀ … 2f₀).
            var xNew = x
            for i in 0..<P { xNew[i] += dx[i] }
            xNew[fSlot] = simd_clamp(xNew[fSlot], 0.5 * f0, 2.0 * f0)

            let rNew  = residuals(xNew)
            let costN = sumSq(rNew)

            if costN < cost {
                let rel = (cost - costN) / max(cost, 1e-12)
                x = xNew; r0 = rNew; cost = costN
                lambda *= 0.1
                if rel < 1e-5 {
                    dbg("[LM] converged @iter \(iter+1), cost \(String(format: "%.1f→%.1f", cost0, cost))")
                    break
                }
            } else {
                lambda *= 10
                if lambda > 1e10 {
                    dbg("[LM] λ blew up; stopping @iter \(iter+1)")
                    break
                }
            }
        }

        let fRefined = x[fSlot]
        let pct = (fRefined - f0) / f0 * 100
        dbg("[LM] focal: \(String(format: "%.1f → %.1f px (%+.2f %%)", f0, fRefined, pct))")

        // Write back rotations + refined focal to every node's intrinsics.
        for (nodeIdx, slot) in slotOf {
            let aa = SIMD3<Float>(x[slot * 3], x[slot * 3 + 1], x[slot * 3 + 2])
            graph.nodes[nodeIdx].pose?.rotation =
                projectToSO3(rotationFromAxisAngle(aa))
        }
        for i in graph.nodes.indices {
            graph.nodes[i].pose?.intrinsics.focal = fRefined
        }
    }

    // MARK: - Wave correction (global up-vector alignment)
    //
    // Sequential spanning-tree pose recovery + LM refinement leaves one
    // unconstrained gauge degree of freedom: a global rotation about the
    // origin. For a horizontal pan, the cameras' local X-axes (image-x in
    // world frame, R_iᵀ·e₀) should all lie close to a common plane; the
    // normal of that plane is the true world-up. We solve the smallest-
    // eigenvector problem on M = Σ xᵢ xᵢᵀ, then rotate every camera so
    // that normal aligns with world Y. Pairwise reprojection residuals
    // are invariant under this global rotation — only the canvas tilt
    // changes. (Brown & Lowe 2007 / OpenCV waveCorrect HORIZ.)
    static func waveCorrectHorizontal(graph: inout PanoGraph) {
        guard graph.nodes.count >= 2 else { return }

        var M = simd_float3x3(0)
        var count = 0
        for n in graph.nodes {
            guard let R = n.pose?.rotation else { continue }
            // Camera X-axis expressed in world: R^T · (1,0,0) = row 0 of R.
            let x = SIMD3<Float>(R.columns.0.x, R.columns.1.x, R.columns.2.x)
            M += simd_float3x3(columns: (x * x.x, x * x.y, x * x.z))
            count += 1
        }
        guard count >= 2 else { return }

        let (eigVecs, eigVals) = jacobiEigen3x3(M)
        // Smallest-eigenvalue eigenvector ≈ world up.
        var smallest = 0
        for i in 1..<3 where eigVals[i] < eigVals[smallest] { smallest = i }
        var u = eigVecs[smallest]
        let n2 = simd_length(u)
        guard n2 > 1e-6 else { return }
        u /= n2
        // Sign convention: pick u with positive Y so R_g is the smaller rotation.
        if u.y < 0 { u = -u }

        // Build R_g such that R_g · (0,1,0) = u  (Rodrigues from Y to u).
        let y = SIMD3<Float>(0, 1, 0)
        let dotYU = simd_clamp(simd_dot(y, u), -1, 1)
        let Rg: simd_float3x3
        if dotYU > 0.999999 {
            Rg = matrix_identity_float3x3
        } else if dotYU < -0.999999 {
            // 180° about Z (any axis ⟂ Y works).
            Rg = simd_float3x3(diagonal: SIMD3<Float>(-1, -1, 1))
        } else {
            let axis = simd_normalize(simd_cross(y, u))
            let angle = acos(dotYU)
            Rg = rotationFromAxisAngle(axis * angle)
        }

        // Apply: R_i' = R_i · R_g  (rotates the world frame; cam-frame unchanged).
        for i in graph.nodes.indices {
            guard let R = graph.nodes[i].pose?.rotation else { continue }
            graph.nodes[i].pose?.rotation = projectToSO3(R * Rg)
        }

        let degrees = acos(dotYU) * 180 / .pi
        dbg("[wave] global up tilt corrected: \(String(format: "%.3f°", degrees)) (\(count) cameras, λ_min=\(String(format: "%.4g", eigVals[smallest])))")
    }

    /// Symmetric 3×3 eigendecomposition via Jacobi rotations.
    /// Returns (eigenvectors as columns, eigenvalues) — paired by index.
    private static func jacobiEigen3x3(_ A: simd_float3x3)
        -> ([SIMD3<Float>], [Float]) {
        var a = [[Float]](repeating: [0, 0, 0], count: 3)
        for r in 0..<3 { for c in 0..<3 { a[r][c] = A[c][r] } }
        var v = [[Float]](repeating: [0, 0, 0], count: 3)
        for i in 0..<3 { v[i][i] = 1 }

        for _ in 0..<32 {
            var p = 0, q = 1
            var maxOff: Float = abs(a[0][1])
            if abs(a[0][2]) > maxOff { p = 0; q = 2; maxOff = abs(a[0][2]) }
            if abs(a[1][2]) > maxOff { p = 1; q = 2; maxOff = abs(a[1][2]) }
            if maxOff < 1e-10 { break }

            let app = a[p][p], aqq = a[q][q], apq = a[p][q]
            let theta = (aqq - app) / (2 * apq)
            let t: Float = (theta >= 0)
                ? 1 / (theta + sqrt(1 + theta * theta))
                : 1 / (theta - sqrt(1 + theta * theta))
            let c = 1 / sqrt(1 + t * t)
            let s = t * c

            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = 0; a[q][p] = 0
            for r in 0..<3 where r != p && r != q {
                let arp = a[r][p], arq = a[r][q]
                a[r][p] = c * arp - s * arq; a[p][r] = a[r][p]
                a[r][q] = s * arp + c * arq; a[q][r] = a[r][q]
            }
            for r in 0..<3 {
                let vrp = v[r][p], vrq = v[r][q]
                v[r][p] = c * vrp - s * vrq
                v[r][q] = s * vrp + c * vrq
            }
        }

        let vecs = [
            SIMD3<Float>(v[0][0], v[1][0], v[2][0]),
            SIMD3<Float>(v[0][1], v[1][1], v[2][1]),
            SIMD3<Float>(v[0][2], v[1][2], v[2][2]),
        ]
        let vals = [a[0][0], a[1][1], a[2][2]]
        return (vecs, vals)
    }

    // MARK: - Linear solve

    private static func solveSymmetricLinear(_ AIn: [Float],
                                              _ bIn: [Float],
                                              n: Int) -> [Float]? {
        var A = AIn; var b = bIn
        for col in 0..<n {
            var pivotRow = col
            var pivotMag = abs(A[col * n + col])
            for r in (col + 1)..<n {
                let v = abs(A[r * n + col])
                if v > pivotMag { pivotMag = v; pivotRow = r }
            }
            guard pivotMag > 1e-12 else { return nil }
            if pivotRow != col {
                for c in 0..<n { A.swapAt(col * n + c, pivotRow * n + c) }
                b.swapAt(col, pivotRow)
            }
            let pivot = A[col * n + col]
            for r in (col + 1)..<n {
                let factor = A[r * n + col] / pivot
                if factor != 0 {
                    for c in col..<n { A[r * n + c] -= factor * A[col * n + c] }
                    b[r] -= factor * b[col]
                }
            }
        }
        var x = [Float](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<n { s -= A[r * n + c] * x[c] }
            x[r] = s / A[r * n + r]
        }
        return x
    }
}
