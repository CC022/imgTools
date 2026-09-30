//
//  SeamFinder.swift
//  panoDev
//
//  Content-aware canvas-pixel ownership ("who provides this pixel").
//
//  `computeLabels` builds a global low-resolution label map on the CPU from
//  the seam-prepass warps: coverage-restricted Voronoi initialisation, then
//  pairwise Efros–Freeman minimal-error-boundary-cut refinement of each
//  contested overlap, nearest camera pairs first. Seams route through
//  regions where the overlapping images *agree*, so parallax-misaligned
//  foreground objects are not cut through. Ownership is resolved globally
//  before any α is touched, which is what the old (removed) chain-of-pairs
//  α-mutating DP got wrong for 3+ image overlaps.
//
//  The label map is consumed by the fused `cylindricalWarpMasked` kernel
//  during the streaming blend: each image's α is zeroed wherever the labels
//  name another owner, handing the multi-band pyramid blend a binary
//  partition of unity that its per-level Gaussian α downsample converts
//  into feathered weights.
//
//  CPU on purpose: the label map is decision logic — a sequential DP and
//  label bookkeeping over ~1 MP of data. All O(full-resolution) per-pixel
//  work stays on the GPU in the warp/blend kernels.
//

import Foundation
import Metal
import simd

nonisolated enum SeamFinder {

    /// Low-res ownership map, ready for the masked warp kernel.
    struct LabelState {
        /// uint8 per low-res pixel: owning camera index, or 255 = uncovered.
        let labelBuffer: MTLBuffer
        let lowWidth: Int
        let lowHeight: Int
        /// Full-res canvas pixels per label-map pixel.
        let scale: Int

        /// 1×1 map naming no owner anywhere: the masked warp keeps plain
        /// coverage α. Used when there are no seams to find (single image,
        /// or prepass failure) so the blend still runs the masked warp,
        /// which is where rgb enters the blend domain.
        static func unowned() -> LabelState? {
            var none = noOwner
            guard let buf = PanoContext.shared.device.makeBuffer(
                bytes: &none, length: 1, options: .storageModeShared)
            else { return nil }
            return LabelState(labelBuffer: buf, lowWidth: 1, lowHeight: 1, scale: 1)
        }
    }

    private static let noOwner: UInt8 = 255

    /// Build the global ownership label map from the low-res prepass warps.
    /// `gains` are applied on the fly when computing seam costs so the seam
    /// sees final (gain-corrected) colours. Returns nil on any failure.
    static func computeLabels(graph: PanoGraph,
                              prepass: PanoPipeline.SeamPrepass,
                              gains: [SIMD3<Float>]) -> LabelState? {
        let n = graph.nodes.count
        guard n >= 2, n < Int(noOwner),
              prepass.warps.count == n, gains.count == n else { return nil }
        let low = prepass.lowCanvas
        let lowW = low.size.x, lowH = low.size.y
        guard lowW > 1, lowH > 1 else { return nil }
        let t0 = Date()

        // Camera optical-axis projections in low-res canvas pixels.
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
            refinePairDP(i: i, j: j, centers: centers, px: px, gains: gains,
                         labels: &labels, lowW: lowW,
                         covI: bbox[i], covJ: bbox[j])
        }

        guard let buf = PanoContext.shared.device.makeBuffer(
            bytes: labels, length: labels.count, options: .storageModeShared)
        else { return nil }

        dbg("[Seam] DP labels: \(lowW)×\(lowH), \(pairs.count) pair(s), \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
        return LabelState(labelBuffer: buf,
                          lowWidth: lowW, lowHeight: lowH, scale: prepass.scale)
    }

    // MARK: - Pairwise DP seam (Efros–Freeman minimal error boundary cut)

    /// Re-split the contested overlap of images `i` and `j` along a minimal
    /// colour-difference path. Only pixels currently owned by `i` or `j` AND
    /// covered by both are relabeled, so third-image territory and uncovered
    /// gaps are never touched.
    private static func refinePairDP(i: Int, j: Int,
                                     centers: [SIMD2<Float>],
                                     px: [UnsafePointer<Float16>],
                                     gains: [SIMD3<Float>],
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

        // Squared gain-corrected RGB difference where both images cover the
        // pixel; LARGE elsewhere so the path can traverse gaps of a
        // non-convex overlap (relabeling stays restricted to C, so this is
        // purely a bridge).
        let gi = gains[i], gj = gains[j]
        var cost = [Float](repeating: 0, count: gw * gh)
        for y in 0..<bh {
            for x in 0..<bw {
                let idx = (cy0 + y) * lowW + (cx0 + x)
                var d = largeCost
                if covered(i, idx), covered(j, idx) {
                    d = 0
                    for c in 0..<3 {
                        let diff = gi[c] * Float(px[i][idx * 4 + c])
                                 - gj[c] * Float(px[j][idx * 4 + c])
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
