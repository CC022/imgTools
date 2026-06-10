//
//  SeamFinder.swift
//  panoDev
//
//  Canvas-pixel ownership ("who provides this pixel"). Two strategies; both
//  hand the downstream multi-band pyramid blend a binary partition of unity
//  that its per-level Gaussian α downsample converts into feathered weights.
//
//  • Content-aware DP labels (preferred) — `computeLabels` builds a global
//    low-resolution label map on the CPU from the seam-prepass warps:
//    coverage-restricted Voronoi initialisation, then pairwise Efros–Freeman
//    minimal-error-boundary-cut refinement of each contested overlap, nearest
//    camera pairs first. Seams route through regions where the overlapping
//    images *agree*, so parallax-misaligned foreground objects are no longer
//    cut through — the visible-seam failure of the pure Voronoi split.
//    Ownership is resolved globally before any α is touched, which is what
//    the old (removed) chain-of-pairs α-mutating DP got wrong for 3+ image
//    overlaps. `applyLabels` is the per-image full-res α-masking dispatch.
//
//  • Voronoi by camera centre (fallback) — each canvas pixel is owned by the
//    camera whose optical axis projects nearest to it; content-agnostic.
//    `makeCoverageState` + `applyVoronoi`, used when the prepass or label
//    computation fails.
//

import Foundation
import Metal
import simd

/// Which seam strategy the blend should apply to each warped image.
enum SeamMasking {
    case labels(SeamFinder.LabelState)
    case voronoi(SeamFinder.CoverageState)
    case none
}

enum SeamFinder {

    struct CoverageState {
        let pso: MTLComputePipelineState
        let centersBuffer: MTLBuffer
        let cameraCount: UInt32
    }

    // MARK: - Prepare

    /// Build reusable Voronoi state. Returns nil if there's nothing to do
    /// (single-image graph) or the kernel can't load.
    static func makeCoverageState(graph: PanoGraph,
                                  canvas: CylindricalCanvas) -> CoverageState? {
        guard graph.nodes.count >= 2 else { return nil }
        let ctx = PanoContext.shared
        guard let pso = ctx.loadPSO("voronoiCoverage") else { return nil }

        // Project each camera's optical axis to canvas (cx, cy).
        let centers: [SIMD2<Float>] = graph.nodes.map { node in
            guard let pose = node.pose else { return .zero }
            let opt = pose.rotation.transpose * SIMD3<Float>(0, 0, 1)
            let theta = atan2(opt.x, opt.z)
            let radial = max(sqrt(opt.x * opt.x + opt.z * opt.z), 1e-9)
            let h = opt.y / radial
            return SIMD2<Float>(
                (theta - canvas.origin.x) * canvas.radius,
                (h     - canvas.origin.y) * canvas.radius
            )
        }

        let centerBytes = centers.count * MemoryLayout<SIMD2<Float>>.stride
        guard let centersBuf = ctx.device.makeBuffer(bytes: centers,
                                                      length: centerBytes,
                                                      options: .storageModeShared)
        else { return nil }

        let summary = centers.enumerated()
            .map { i, c in String(format: "%d=(%.0f,%.0f)", i, c.x, c.y) }
            .joined(separator: " ")
        dbg("[Seam] Voronoi centres: \(summary)")

        return CoverageState(pso: pso,
                             centersBuffer: centersBuf,
                             cameraCount: UInt32(centers.count))
    }

    // MARK: - Apply

    /// Single-buffer dispatch: zero this image's α where another centre is
    /// closer. No-op when `state` is nil (single-image graph).
    static func applyVoronoi(state: CoverageState?,
                             warped: CanvasBuffer,
                             selfIndex: UInt32) {
        guard let state else { return }
        var n = state.cameraCount
        var s = selfIndex
        Compute.run { cb in
            withUnsafeBytes(of: &n) { nRaw in
                withUnsafeBytes(of: &s) { sRaw in
                    Compute.encode(cb, state.pso,
                        buffers: [warped.buffer, state.centersBuffer],
                        dims:    [warped.dimsPacked],
                        bytes:   [(nRaw.baseAddress!, nRaw.count),
                                  (sRaw.baseAddress!, sRaw.count)],
                        gridW:   warped.width,
                        gridH:   warped.height)
                }
            }
        }
    }

    // MARK: - Content-aware DP labels

    struct LabelState {
        let pso: MTLComputePipelineState
        /// uint8 per low-res pixel: owning camera index, or 255 = uncovered.
        let labelBuffer: MTLBuffer
        let lowWidth: Int
        let lowHeight: Int
        /// Full-res canvas pixels per label-map pixel.
        let scale: Int
    }

    /// Mirrors `SeamLabelParams` in SeamFinding.metal.
    private struct SeamLabelParams {
        var selfIndex: UInt32
        var scale: UInt32
        var lowW: UInt32
        var lowH: UInt32
    }

    private static let noOwner: UInt8 = 255

    /// Build the global ownership label map from gain-corrected low-res warps.
    /// Returns nil on any failure so the caller can fall back to Voronoi.
    static func computeLabels(graph: PanoGraph,
                              canvas: CylindricalCanvas,
                              prepass: PanoPipeline.SeamPrepass) -> LabelState? {
        let n = graph.nodes.count
        guard n >= 2, n < Int(noOwner), prepass.warps.count == n else { return nil }
        let low = prepass.lowCanvas
        let lowW = low.size.x, lowH = low.size.y
        guard lowW > 1, lowH > 1 else { return nil }
        let t0 = Date()

        // Camera optical-axis projections in low-res canvas pixels (same
        // formula as makeCoverageState, in the low canvas's coordinates).
        let centers: [SIMD2<Float>] = graph.nodes.map { node in
            guard let pose = node.pose else { return .zero }
            let opt = pose.rotation.transpose * SIMD3<Float>(0, 0, 1)
            let theta = atan2(opt.x, opt.z)
            let radial = max(sqrt(opt.x * opt.x + opt.z * opt.z), 1e-9)
            let h = opt.y / radial
            return SIMD2<Float>((theta - low.origin.x) * low.radius,
                                (h     - low.origin.y) * low.radius)
        }

        // The prepass warps are rgba16f with .shared storage and every GPU
        // pass on them ran through Compute.run (waitUntilCompleted) — safe to
        // read directly. Pixel (x, y) channel c is px[k][(y·lowW + x)·4 + c].
        let px: [UnsafePointer<Float16>] = prepass.warps.map {
            UnsafePointer($0.buffer.contents()
                .bindMemory(to: Float16.self, capacity: lowW * lowH * 4))
        }

        // 1 — Coverage-restricted Voronoi init + per-image coverage bboxes.
        //     (minX, minY, maxX, maxY); empty coverage stays inverted.
        var labels = [UInt8](repeating: noOwner, count: lowW * lowH)
        var bbox = [SIMD4<Int>](repeating: SIMD4(Int.max, Int.max, Int.min, Int.min),
                                count: n)
        for y in 0..<lowH {
            for x in 0..<lowW {
                let idx = y * lowW + x
                var best = -1
                var bestD = Float.infinity
                for k in 0..<n where Float(px[k][idx * 4 + 3]) > 0.5 {
                    let dx = centers[k].x - Float(x)
                    let dy = centers[k].y - Float(y)
                    let d = dx * dx + dy * dy
                    if d < bestD { bestD = d; best = k }
                    bbox[k] = SIMD4(min(bbox[k].x, x), min(bbox[k].y, y),
                                    max(bbox[k].z, x), max(bbox[k].w, y))
                }
                if best >= 0 { labels[idx] = UInt8(best) }
            }
        }

        // 2 — Pairs whose coverage bboxes intersect, nearest centres first.
        //     Later pairs operate on the running label state, so 3+-way
        //     overlaps are settled by successive pairwise relabelings.
        var pairs: [(i: Int, j: Int, d: Float)] = []
        for i in 0..<(n - 1) {
            for j in (i + 1)..<n {
                guard bbox[i].x <= bbox[j].z, bbox[j].x <= bbox[i].z,
                      bbox[i].y <= bbox[j].w, bbox[j].y <= bbox[i].w
                else { continue }
                pairs.append((i, j, simd_distance_squared(centers[i], centers[j])))
            }
        }
        pairs.sort { $0.d < $1.d }

        // 3 — Per-pair DP seam refinement.
        for (i, j, _) in pairs {
            refinePairDP(i: i, j: j, centers: centers, px: px,
                         labels: &labels, lowW: lowW,
                         covI: bbox[i], covJ: bbox[j])
        }

        guard let pso = PanoContext.shared.loadPSO("applySeamLabels"),
              let buf = PanoContext.shared.device.makeBuffer(
                  bytes: labels, length: labels.count, options: .storageModeShared)
        else { return nil }

        dbg("[Seam] DP labels: \(lowW)×\(lowH), \(pairs.count) pair(s), \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
        return LabelState(pso: pso, labelBuffer: buf,
                          lowWidth: lowW, lowHeight: lowH, scale: prepass.scale)
    }

    /// Single-buffer dispatch: zero this image's α wherever the label map
    /// names another owner.
    static func applyLabels(state: LabelState,
                            warped: CanvasBuffer,
                            selfIndex: UInt32) {
        var p = SeamLabelParams(selfIndex: selfIndex,
                                scale: UInt32(state.scale),
                                lowW: UInt32(state.lowWidth),
                                lowH: UInt32(state.lowHeight))
        Compute.run { cb in
            withUnsafeBytes(of: &p) { raw in
                Compute.encode(cb, state.pso,
                    buffers: [warped.buffer, state.labelBuffer],
                    dims:    [warped.dimsPacked],
                    bytes:   [(raw.baseAddress!, raw.count)],
                    gridW:   warped.width,
                    gridH:   warped.height)
            }
        }
    }

    // MARK: - Pairwise DP seam (Efros–Freeman minimal error boundary cut)

    /// Re-split the contested overlap of images `i` and `j` along a minimal
    /// colour-difference path. Only pixels currently owned by `i` or `j` AND
    /// covered by both are relabeled, so third-image territory and uncovered
    /// gaps are never touched.
    private static func refinePairDP(i: Int, j: Int,
                                     centers: [SIMD2<Float>],
                                     px: [UnsafePointer<Float16>],
                                     labels: inout [UInt8],
                                     lowW: Int,
                                     covI: SIMD4<Int>, covJ: SIMD4<Int>) {
        let largeCost: Float = 1e6
        let minRegion = 64

        // Intersection of the two coverage bboxes.
        let x0 = max(covI.x, covJ.x), x1 = min(covI.z, covJ.z)
        let y0 = max(covI.y, covJ.y), y1 = min(covI.w, covJ.w)
        guard x0 <= x1, y0 <= y1 else { return }

        let li = UInt8(i), lj = UInt8(j)
        @inline(__always) func covered(_ k: Int, _ idx: Int) -> Bool {
            Float(px[k][idx * 4 + 3]) > 0.5
        }

        // Contested set C = overlap ∩ label∈{i,j}; track its bbox.
        var overlapCount = 0, contestedCount = 0
        var cx0 = Int.max, cy0 = Int.max, cx1 = Int.min, cy1 = Int.min
        for y in y0...y1 {
            for x in x0...x1 {
                let idx = y * lowW + x
                guard covered(i, idx), covered(j, idx) else { continue }
                overlapCount += 1
                let l = labels[idx]
                guard l == li || l == lj else { continue }
                contestedCount += 1
                cx0 = min(cx0, x); cy0 = min(cy0, y)
                cx1 = max(cx1, x); cy1 = max(cy1, y)
            }
        }
        guard overlapCount >= minRegion, contestedCount >= minRegion else { return }

        let bw = cx1 - cx0 + 1, bh = cy1 - cy0 + 1
        // Vertical seam (DP top→bottom) when the pair is side-by-side,
        // horizontal seam (DP left→right, transposed grid) when stacked.
        let vertical = abs(centers[i].x - centers[j].x) >= abs(centers[i].y - centers[j].y)
        let (gw, gh) = vertical ? (bw, bh) : (bh, bw)

        // Squared RGB difference where both images cover the pixel; LARGE
        // elsewhere so the path can traverse gaps of a non-convex overlap
        // (relabeling stays restricted to C, so this is purely a bridge).
        var cost = [Float](repeating: 0, count: gw * gh)
        for y in 0..<bh {
            for x in 0..<bw {
                let idx = (cy0 + y) * lowW + (cx0 + x)
                var d = largeCost
                if covered(i, idx), covered(j, idx) {
                    d = 0
                    for c in 0..<3 {
                        let diff = Float(px[i][idx * 4 + c]) - Float(px[j][idx * 4 + c])
                        d += diff * diff
                    }
                }
                cost[vertical ? (y * gw + x) : (x * gw + y)] = d
            }
        }

        let seam = dpSeamColumns(cost: &cost, w: gw, h: gh)

        // Relabel contested pixels on each side of the seam. "first" is the
        // image whose centre is on the lower-coordinate side of the cut axis.
        let first: UInt8 = vertical
            ? (centers[i].x <= centers[j].x ? li : lj)
            : (centers[i].y <= centers[j].y ? li : lj)
        let second: UInt8 = first == li ? lj : li
        var relabeled = 0
        for y in 0..<bh {
            for x in 0..<bw {
                let idx = (cy0 + y) * lowW + (cx0 + x)
                let l = labels[idx]
                guard l == li || l == lj,
                      covered(i, idx), covered(j, idx) else { continue }
                let before = vertical ? (x < seam[y]) : (y < seam[x])
                let newLabel = before ? first : second
                if newLabel != l { relabeled += 1 }
                labels[idx] = newLabel
            }
        }
        dbg("[Seam] DP \(i)-\(j): \(vertical ? "vertical" : "horizontal"), overlap=\(overlapCount), contested=\(contestedCount), relabeled=\(relabeled)")
    }

    /// Minimal-cost top→bottom path with (-1, 0, +1) column moves per row.
    /// Accumulates in place, then backtracks. Returns the seam column per row.
    private static func dpSeamColumns(cost: inout [Float], w: Int, h: Int) -> [Int] {
        for r in 1..<h {
            let row = r * w, prev = row - w
            for c in 0..<w {
                var m = cost[prev + c]
                if c > 0     { m = min(m, cost[prev + c - 1]) }
                if c < w - 1 { m = min(m, cost[prev + c + 1]) }
                cost[row + c] += m
            }
        }
        var seam = [Int](repeating: 0, count: h)
        let last = (h - 1) * w
        var best = 0
        for c in 1..<w where cost[last + c] < cost[last + best] { best = c }
        seam[h - 1] = best
        for r in stride(from: h - 2, through: 0, by: -1) {
            let row = r * w
            var nc = best
            if best > 0,     cost[row + best - 1] < cost[row + nc] { nc = best - 1 }
            if best < w - 1, cost[row + best + 1] < cost[row + nc] { nc = best + 1 }
            best = nc
            seam[r] = best
        }
        return seam
    }
}
