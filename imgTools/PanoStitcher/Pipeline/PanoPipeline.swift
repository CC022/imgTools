//
//  PanoPipeline.swift
//  panoDev
//
//  End-to-end orchestrator. Buffer-backed canvas storage (CanvasBuffer)
//  removes the 16384-px Metal 2D-texture limit; canvas is bounded only by
//  available GPU memory.
//
//  Memory strategy (streaming):
//    • Steps 1–8 run on detection-size decodes (≤ 2048 px long edge) and a
//      ~1 MP low-res prepass; no full-res pixels are resident. Full-res
//      decodes happen one image at a time inside the blend's `warpNode`.
//    • Warped buffers and per-image Gaussian pyramids are *transient* —
//      materialised inside `warpNode` / `accumulate` and released at the end
//      of each iteration — and sized to the image's canvas footprint (ROI),
//      not the whole canvas, so per-image blend cost is O(footprint) and the
//      whole blend is O(n · image), not O(n · canvas).
//    • The only canvas-sized allocation that survives across images is the
//      float32 `acc` pyramid. The `collapse` pass keeps two adjacent
//      pyramid levels in flight at any moment.
//

import Foundation
import simd
import Metal
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import AppKit

// MARK: - Result types

nonisolated struct PanoramaResult: @unchecked Sendable {
    let graph: PanoGraph
    let canvas: CylindricalCanvas
    /// Final blended panorama (rgba16f, .shared storage so the CPU can read it).
    let blendedBuffer: CanvasBuffer
}

nonisolated struct PipelineProgress: Sendable {
    let completedUnits: Int
    let totalUnits: Int
    let message: String

    var fraction: Double {
        guard totalUnits > 0 else { return 0 }
        return min(1, max(0, Double(completedUnits) / Double(totalUnits)))
    }
}

// MARK: - Pipeline

nonisolated enum PanoPipeline {

    static func process(urls: [URL],
                        focalLengthMM35: Float = 24.0,
                        progress: (@Sendable (PipelineProgress) -> Void)? = nil) async -> PanoramaResult? {
        dbg("[Pipeline] start: \(urls.count) image(s), f_mm35=\(focalLengthMM35)")
        let t0 = Date()
        let tracker = ProgressTracker(totalUnits: progressUnitCount(imageCount: urls.count),
                                      handler: progress)
        await tracker.update("Preparing \(urls.count) image\(urls.count == 1 ? "" : "s")")

        // 1+2 — Load + detect. `loadAndDetect` owns the `[ImageTexture]`
        // array (detection-size decodes) and lets it fall out of scope on
        // return; the rest of the pipeline runs on metadata only.
        guard let (images, allKps) = await loadAndDetect(urls: urls, t0: t0, progress: tracker)
        else { return nil }

        // 3 — All-pairs LightGlue + RANSAC.
        let t2 = Date()
        let edges = await matchAllPairs(images: images, allKps: allKps, progress: tracker)
        dbg("[Pipeline] matching+RANSAC: \(edges.count) edge(s) survived (\(elapsed(t2)) ms)")
        guard !edges.isEmpty else {
            dbg("[Pipeline] no valid edges — aborting")
            return nil
        }

        // 4 — Assemble graph.
        let nodes = images.enumerated().map { i, img in
            PanoNode(id: i, image: img, keypoints: allKps[i])
        }
        var graph = PanoGraph(nodes: nodes, edges: edges)
        for warning in graph.validate() { dbg("[Pipeline] WARN: \(warning)") }
        var anchor = graph.anchorIndex()
        dbg("[Pipeline] anchor = node \(anchor)")
        await tracker.advance("Built pose graph")

        // 5 — Camera recovery via spanning tree.
        recoverCameras(graph: &graph, anchor: anchor, focalMM35: focalLengthMM35)
        // Drop disconnected nodes (no pose) and reindex.
        pruneToAnchorComponent(graph: &graph, anchor: &anchor)
        guard graph.nodes.count >= 1 else {
            dbg("[Pipeline] no nodes survived pruning — aborting")
            return nil
        }
        await tracker.advance("Recovered camera poses")

        // 6 — LM rotation + focal refinement.
        let t3 = Date()
        RotationRefiner.refine(graph: &graph, anchor: anchor)
        dbg("[Pipeline] LM refinement (\(elapsed(t3)) ms)")
        RotationRefiner.waveCorrectHorizontal(graph: &graph)
        await tracker.advance("Refined rotations")

        // 7 — Cylindrical canvas.
        let canvas = CylindricalCanvas.compute(
            poses:      graph.nodes.map { $0.pose! },
            imageSizes: graph.nodes.map { SIMD2(Float($0.image.width), Float($0.image.height)) },
            anchorIndex: anchor
        )
        dbg("[Pipeline] canvas: \(canvas.size.x)×\(canvas.size.y) px, radius=\(canvas.radius)")
        await tracker.advance("Computed canvas")

        // 8 + 8.5 — Low-res prepass shared by Brown-Lowe exposure estimation
        //     and content-aware seam labels. Both are applied per image by
        //     the fused warp kernel during the streaming blend.
        let t_exp = Date()
        await tracker.update("Estimating exposure")
        let (exposureGains, seamLabels) = prepareExposureAndSeams(graph: graph,
                                                                  canvas: canvas)
        dbg("[Pipeline] exposure + seam preparation (\(elapsed(t_exp)) ms)")
        await tracker.advance("Estimated exposure")
        await tracker.advance("Prepared seams")

        // 9 — Burt-Adelson Laplacian-pyramid blend, streaming one image at a
        //     time into the accumulator.
        let t5 = Date()
        await tracker.update("Blending panorama")
        guard let blended = await blendStream(graph: graph,
                                               canvas: canvas,
                                               exposureGains: exposureGains,
                                               seamLabels: seamLabels,
                                               progress: tracker) else {
            dbg("[Pipeline] blend dispatch failed")
            return nil
        }
        dbg("[Pipeline] blended \(graph.nodes.count) image(s) (\(elapsed(t5)) ms)")
        await tracker.advance("Finished")

        dbg("[Pipeline] complete in \(elapsed(t0)) ms")
        return PanoramaResult(graph: graph, canvas: canvas, blendedBuffer: blended)
    }

    // MARK: - Steps 1+2: load + detect (detection-size decodes only)

    /// Long edge for working-resolution decodes (SuperPoint detection and
    /// the low-res seam/exposure prepass). Full-res decodes happen only
    /// per-image inside the streaming blend.
    private static let workingLongEdge = 2048

    /// Decode every URL at detection size, run SuperPoint on each, and
    /// return only the metadata + keypoints (scaled to original pixels).
    /// The detection textures live for the duration of this function and
    /// are released on return.
    private static func loadAndDetect(urls: [URL],
                                       t0: Date,
                                       progress: ProgressTracker)
                                       async -> ([PanoImage], [[Keypoint]])? {
        let loaded = await loadAll(urls: urls, progress: progress)
        guard !loaded.isEmpty else { return nil }
        dbg("[Pipeline] loaded \(loaded.count) detection-size image(s) in \(elapsed(t0)) ms")

        let t_det = Date()
        let detector = SuperPointDetector()
        var allKps: [[Keypoint]] = []
        allKps.reserveCapacity(loaded.count)
        for (i, item) in loaded.enumerated() {
            let scale = Float(item.image.width) / Float(item.texture.width)
            let kps = detector.detect(in: item).map { kp in
                var k = kp
                k.x *= scale; k.y *= scale; k.scale *= scale
                return k
            }
            allKps.append(kps)
            await progress.advance("Detected features \(i + 1)/\(loaded.count)")
        }
        dbg("[Pipeline] keypoints: \(allKps.map { String($0.count) }.joined(separator: ", ")) (\(elapsed(t_det)) ms)")

        return (loaded.map(\.image), allKps)
    }

    /// Decode all URLs at working resolution with bounded parallelism —
    /// RAW decodes hold large transient buffers, so unbounded fan-out can
    /// spike memory on big sets.
    private static func loadAll(urls: [URL],
                                 progress: ProgressTracker) async -> [ImageTexture] {
        await withTaskGroup(of: (Int, ImageTexture?).self) { group in
            var next = 0
            func enqueue() {
                guard next < urls.count else { return }
                let i = next
                let url = urls[i]
                next += 1
                group.addTask { (i, try? loadImageTexture(url: url, maxLongEdge: workingLongEdge)) }
            }
            for _ in 0..<min(3, urls.count) { enqueue() }

            var slots = [ImageTexture?](repeating: nil, count: urls.count)
            var done = 0
            for await (i, img) in group {
                slots[i] = img
                done += 1
                await progress.advance("Loaded \(done)/\(urls.count)")
                enqueue()
            }
            return slots.compactMap { $0 }
        }
    }

    // MARK: - Step 3: match all pairs

    private static func matchAllPairs(images: [PanoImage],
                                       allKps: [[Keypoint]],
                                       progress: ProgressTracker) async -> [PanoEdge] {
        guard images.count >= 2 else { return [] }
        let matcher = LightGlueMatcher()
        var edges: [PanoEdge] = []
        var pairIndex = 0
        let pairTotal = images.count * (images.count - 1) / 2
        for i in 0..<(images.count - 1) {
            for j in (i + 1)..<images.count {
                pairIndex += 1
                let matches = matcher.match(
                    kpA: allKps[i], imgWidthA: images[i].width, imgHeightA: images[i].height,
                    kpB: allKps[j], imgWidthB: images[j].width, imgHeightB: images[j].height
                )
                guard matches.count >= 30 else {
                    dbg("[Pipeline] edge \(i)→\(j): only \(matches.count) matches, skipping")
                    await progress.advance("Matched \(i)→\(j) (\(pairIndex)/\(pairTotal))")
                    continue
                }
                let srcPts = matches.map { SIMD2(allKps[i][$0.indexA].x, allKps[i][$0.indexA].y) }
                let dstPts = matches.map { SIMD2(allKps[j][$0.indexB].x, allKps[j][$0.indexB].y) }
                let weights = matches.map(\.confidence)

                guard let r = HomographyEstimator.estimate(
                    srcPoints: srcPts, dstPoints: dstPts, weights: weights
                ), r.inlierIndices.count >= 30 else {
                    dbg("[Pipeline] edge \(i)→\(j): RANSAC dropped (insufficient inliers)")
                    await progress.advance("Matched \(i)→\(j) (\(pairIndex)/\(pairTotal))")
                    continue
                }
                dbg("[Pipeline] edge \(i)→\(j): \(r.inlierIndices.count)/\(matches.count) inliers")
                edges.append(PanoEdge(src: i, dst: j, matches: matches,
                                       homography: r.homography, inliers: r.inlierIndices))
                await progress.advance("Matched \(i)→\(j) (\(pairIndex)/\(pairTotal))")
            }
        }
        return edges
    }

    // MARK: - Step 5: camera recovery

    private static func recoverCameras(graph: inout PanoGraph,
                                        anchor: Int,
                                        focalMM35: Float) {
        let tree = graph.maximumSpanningTree { Float($0.inlierCount) }
        var adj = [[(child: Int, edge: PanoEdge)]](repeating: [], count: graph.nodes.count)
        for e in tree {
            adj[e.src].append((e.dst, e))
            adj[e.dst].append((e.src, e.reversed))
        }

        graph.nodes[anchor].pose = CameraPose(
            intrinsics: makeIntrinsics(node: graph.nodes[anchor], focalMM35: focalMM35),
            rotation: matrix_identity_float3x3
        )

        var visited = Set([anchor]); var queue = [anchor]
        while !queue.isEmpty {
            let parent = queue.removeFirst()
            let R_p = graph.nodes[parent].pose!.rotation
            let K_p = graph.nodes[parent].pose!.intrinsics
            for (child, edge) in adj[parent] where !visited.contains(child) {
                visited.insert(child)
                let K_c = makeIntrinsics(node: graph.nodes[child], focalMM35: focalMM35)
                let R_rel = relativeRotation(from: edge.homography.matrix, Ki: K_p, Kj: K_c)
                graph.nodes[child].pose = CameraPose(intrinsics: K_c,
                                                      rotation: projectToSO3(R_rel * R_p))
                queue.append(child)
            }
        }
    }

    private static func makeIntrinsics(node: PanoNode, focalMM35: Float) -> Intrinsics {
        Intrinsics.fromMM35(focalMM35: focalMM35,
                            imageSize: SIMD2(Float(node.image.width), Float(node.image.height)))
    }

    private static func pruneToAnchorComponent(graph: inout PanoGraph,
                                                anchor: inout Int) {
        let surviving = graph.nodes.indices.filter { graph.nodes[$0].pose != nil }
        guard surviving.count != graph.nodes.count else { return }

        var oldToNew = [Int: Int]()
        oldToNew.reserveCapacity(surviving.count)
        for (newIdx, oldIdx) in surviving.enumerated() {
            oldToNew[oldIdx] = newIdx
        }

        let newNodes = surviving.map { graph.nodes[$0] }
        let newEdges: [PanoEdge] = graph.edges.compactMap { e in
            guard let s = oldToNew[e.src], let d = oldToNew[e.dst] else { return nil }
            return PanoEdge(src: s, dst: d,
                            matches: e.matches,
                            homography: e.homography,
                            inliers: e.inliers)
        }

        let dropped = graph.nodes.count - surviving.count
        let droppedNames = graph.nodes.indices
            .filter { graph.nodes[$0].pose == nil }
            .map { graph.nodes[$0].image.sourceURL.lastPathComponent }
        dbg("[Pipeline] dropping \(dropped) image(s) outside anchor's connected component: \(droppedNames.joined(separator: ", "))")

        graph = PanoGraph(nodes: newNodes, edges: newEdges)
        anchor = oldToNew[anchor] ?? 0
    }

    // MARK: - Streaming warp helper

    /// Per-image inputs for the fused `cylindricalWarpMasked` kernel.
    private typealias WarpMask = (labels: SeamFinder.LabelState,
                                  gain: SIMD3<Float>,
                                  selfIndex: UInt32)

    /// Materialise a single canvas-sized warped buffer for `node`. The source
    /// texture is loaded inside this function and released when it returns;
    /// only the output `CanvasBuffer` survives. Returns nil if the node has
    /// no pose or any allocation/decode fails.
    ///
    /// Pass a downscaled `canvas` (same origin, radius/s, size/s) and a
    /// `maxSourceLongEdge` for low-resolution warps — the kernel's
    /// `θ = origin + x/radius` mapping needs no other change. Intrinsics are
    /// rescaled to the loaded texture the same way the detection pass does.
    ///
    /// With `mask` non-nil, `pso` must be `cylindricalWarpMasked`: exposure
    /// gain, seam-label α masking and the blend-domain encode are fused into
    /// the warp dispatch, so the output rgb is blend-domain, not linear.
    /// Without `mask` the output is linear light (low-res prepass).
    ///
    /// With `roi` non-nil, only that canvas sub-rect is warped and the
    /// returned buffer has the ROI's dimensions (origin recorded in the
    /// kernel params) — per-image cost scales with the image's footprint
    /// instead of the whole canvas.
    private static func warpNode(_ node: PanoNode,
                                  canvas: CylindricalCanvas,
                                  pso: MTLComputePipelineState,
                                  storage: MTLStorageMode = .private,
                                  maxSourceLongEdge: Int? = nil,
                                  mask: WarpMask? = nil,
                                  roi: BlendROI? = nil) -> CanvasBuffer? {
        let ctx = PanoContext.shared
        let outW = roi?.size.x ?? canvas.size.x
        let outH = roi?.size.y ?? canvas.size.y
        guard let pose = node.pose,
              let imgTex = try? node.image.loadTexture(maxLongEdge: maxSourceLongEdge),
              let outBuf = ctx.makeHalfBuffer(width: outW,
                                               height: outH,
                                               storage: storage) else {
            dbg("[Pipeline] warp: missing pose, source texture, or output buffer for node \(node.id)")
            return nil
        }

        let srcScale = Float(imgTex.width) / Float(node.image.width)
        var params = CylindricalWarpParams(
            R:              pose.rotation,
            focal:          pose.intrinsics.focal * srcScale,
            canvasRadius:   canvas.radius,
            principalPoint: pose.intrinsics.principalPoint * srcScale,
            canvasOrigin:   canvas.origin,
            imageSize:      SIMD2(Float(imgTex.width), Float(imgTex.height)),
            canvasSize:     SIMD2(Float(outW), Float(outH)),
            roiOrigin:      SIMD2(Float(roi?.origin.x ?? 0), Float(roi?.origin.y ?? 0))
        )
        guard Compute.run({ cb in
            withUnsafeBytes(of: &params) { raw in
                if let mask {
                    var m = WarpMaskParams(gain: mask.gain,
                                           selfIndex: mask.selfIndex,
                                           labelScale: UInt32(mask.labels.scale),
                                           lowW: UInt32(mask.labels.lowWidth),
                                           lowH: UInt32(mask.labels.lowHeight))
                    withUnsafeBytes(of: &m) { mRaw in
                        Compute.encode(cb, pso,
                            buffers:  [outBuf.buffer, mask.labels.labelBuffer],
                            textures: [imgTex.texture],
                            bytes:    [(raw.baseAddress!, raw.count),
                                       (mRaw.baseAddress!, mRaw.count)],
                            gridW: outW, gridH: outH)
                    }
                } else {
                    Compute.encode(cb, pso,
                        buffers:  [outBuf.buffer],
                        textures: [imgTex.texture],
                        bytes:    [(raw.baseAddress!, raw.count)],
                        gridW: outW, gridH: outH)
                }
            }
        }) else { return nil }
        return outBuf
        // imgTex (and its source texture) is released when this function returns.
    }

    // MARK: - Low-res seam prepass (shared by exposure + seam finding)

    /// All images warped onto a downscaled canvas, CPU-readable. Built once,
    /// used for exposure-gain estimation and content-aware seam labels, then
    /// released before the full-res streaming blend.
    struct SeamPrepass {
        let lowCanvas: CylindricalCanvas
        /// Integer downscale factor relative to the full-res canvas.
        let scale: Int
        /// One rgba16f `.shared` warp per graph node.
        let warps: [CanvasBuffer]
    }

    /// Target ~1.2 MP total for the low-res canvas — plenty for seam routing
    /// and overlap statistics, cheap enough to keep all n warps resident.
    private static func makeSeamPrepass(graph: PanoGraph,
                                        canvas: CylindricalCanvas) -> SeamPrepass? {
        guard graph.nodes.count >= 2,
              let pso = PanoContext.shared.loadPSO("cylindricalWarp")
        else { return nil }

        let W = canvas.size.x, H = canvas.size.y
        let s = max(1, Int((Double(W * H) / 1_200_000.0).squareRoot().rounded(.up)))
        let low = CylindricalCanvas(radius: canvas.radius / Float(s),
                                    origin: canvas.origin,
                                    size: SIMD2<Int>(max(1, W / s), max(1, H / s)))

        var warps: [CanvasBuffer] = []
        warps.reserveCapacity(graph.nodes.count)
        for node in graph.nodes {
            guard let w = warpNode(node, canvas: low, pso: pso,
                                   storage: .shared,
                                   maxSourceLongEdge: workingLongEdge) else {
                dbg("[Seam] prepass: warp failed for node \(node.id)")
                return nil
            }
            warps.append(w)
        }
        dbg("[Seam] prepass: scale=\(s), low canvas \(low.size.x)×\(low.size.y)")
        return SeamPrepass(lowCanvas: low, scale: s, warps: warps)
    }

    /// Steps 8 + 8.5: exposure gains and seam labels, both from one CPU
    /// analysis of the low-res prepass. Scoped so the n prepass warps are
    /// released on return — only the 1 B/px label buffer survives into the
    /// blend. Identity gains + nil labels for a single image (or on the
    /// never-in-practice prepass failure), which the blend handles with an
    /// ownerless label map (plain coverage α).
    private static func prepareExposureAndSeams(graph: PanoGraph,
                                                canvas: CylindricalCanvas)
                                                -> ([SIMD3<Float>], SeamFinder.LabelState?) {
        let identity = Array(repeating: SIMD3<Float>(repeating: 1), count: graph.nodes.count)
        guard let prepass = makeSeamPrepass(graph: graph, canvas: canvas) else {
            return (identity, nil)
        }
        let gains = ExposureCompensator.estimateGains(warps: prepass.warps)
        let labels = SeamFinder.computeLabels(graph: graph, prepass: prepass, gains: gains)
        return (gains, labels)
    }

    // MARK: - Pyramid helpers

    /// Number of Gaussian-pyramid levels for a canvas. Coarsest level shrinks
    /// to ~16 px, giving wide low-frequency blending in low-texture regions.
    private static func pyramidLevelCount(_ w: Int, _ h: Int) -> Int {
        min(12, max(1, Int(log2(Double(min(w, h))))))
    }

    private static func levelSize(_ w: Int, _ h: Int, _ k: Int) -> (Int, Int) {
        (max(1, w >> k), max(1, h >> k))
    }

    // MARK: - Per-image blend ROI

    /// Canvas sub-rect covering one image's footprint (plus pyramid-spread
    /// padding). Pyramid levels `0..<levels` are computed ROI-locally; the
    /// origin is aligned to `1 << levels` so every ROI level sits exactly on
    /// the global pyramid grid (`origin >> k` is even for all k < levels).
    ///
    /// Accuracy contract (verified numerically): blend weights (α) and all
    /// levels ≤ `levels` are bit-identical to the full-canvas path — the pad
    /// guarantees ROI-border clamping only touches zero-weight pixels. The
    /// coarser full-canvas levels rely on `pyramidExtendRect` re-extending
    /// the rgb clamp-extrapolation field after the hand-off; with it they
    /// match the full-canvas pipeline's extrapolation semantics exactly
    /// (without it, deep bands mix zeros and darken near footprints by tens
    /// of percent — visible as large smooth brightness patches).
    struct BlendROI {
        let origin: SIMD2<Int>
        let size: SIMD2<Int>
        /// Pyramid levels 0..<levels are ROI-local; coarser levels (cheap)
        /// are full-canvas.
        let levels: Int
    }

    /// This image's coverage bbox in canvas pixels, from densely sampled
    /// source-rect edges (cylindrical projection of straight edges bulges,
    /// so corners alone are not enough).
    private static func footprint(of node: PanoNode,
                                  canvas: CylindricalCanvas) -> (lo: SIMD2<Float>, hi: SIMD2<Float>)? {
        guard let pose = node.pose else { return nil }
        let w = Float(node.image.width), h = Float(node.image.height)
        var lo = SIMD2<Float>(repeating: .infinity)
        var hi = SIMD2<Float>(repeating: -.infinity)
        let steps = 32
        for t in 0...steps {
            let f = Float(t) / Float(steps)
            for p in [SIMD2(f * w, 0), SIMD2(f * w, h),
                      SIMD2(0, f * h), SIMD2(w, f * h)] {
                let cyl = projectImagePixelToCylinder(p, pose: pose)
                let px = (cyl - canvas.origin) * canvas.radius
                lo = simd_min(lo, px)
                hi = simd_max(hi, px)
            }
        }
        guard lo.x.isFinite, lo.y.isFinite, hi.x.isFinite, hi.y.isFinite,
              hi.x > lo.x, hi.y > lo.y else { return nil }
        return (lo, hi)
    }

    /// Build the blend ROI for `node`, or nil to use the plain full-canvas
    /// path (degenerate pyramids, footprint failure, or no real saving).
    private static func makeBlendROI(node: PanoNode,
                                     canvas: CylindricalCanvas,
                                     nLevels: Int) -> BlendROI? {
        let R = min(6, nLevels - 1)
        guard R >= 1, let fp = footprint(of: node, canvas: canvas) else { return nil }
        let align = 1 << R
        // Pad covers the cumulative Gaussian α spread across the ROI levels
        // (≈ 2^(R+2) px) with margin, so ROI-border clamping only ever
        // touches zero-weight pixels and the blend stays bit-identical.
        let pad = 8 * align
        let W = canvas.size.x, H = canvas.size.y
        let x0 = max(0, Int(fp.lo.x.rounded(.down)) - pad) & ~(align - 1)
        let y0 = max(0, Int(fp.lo.y.rounded(.down)) - pad) & ~(align - 1)
        let x1 = min(W, (Int(fp.hi.x.rounded(.up)) + pad + align - 1) & ~(align - 1))
        let y1 = min(H, (Int(fp.hi.y.rounded(.up)) + pad + align - 1) & ~(align - 1))
        guard x1 > x0, y1 > y0 else { return nil }
        // ROI must save real work — otherwise keep the single-path simplicity.
        guard Double((x1 - x0) * (y1 - y0)) < 0.9 * Double(W * H) else { return nil }
        return BlendROI(origin: SIMD2(x0, y0), size: SIMD2(x1 - x0, y1 - y0), levels: R)
    }

    /// The ROI's rect at pyramid level `k` (0...levels), clamped to the
    /// global level dimensions. Origin alignment makes `origin >> k` exact.
    private static func roiLevelRect(_ roi: BlendROI, _ k: Int,
                                     _ W: Int, _ H: Int) -> (origin: SIMD2<Int>, size: SIMD2<Int>) {
        let (gw, gh) = levelSize(W, H, k)
        let ox = min(roi.origin.x >> k, gw - 1)
        let oy = min(roi.origin.y >> k, gh - 1)
        let ex = min((roi.origin.x + roi.size.x + (1 << k) - 1) >> k, gw)
        let ey = min((roi.origin.y + roi.size.y + (1 << k) - 1) >> k, gh)
        return (SIMD2(ox, oy), SIMD2(max(1, ex - ox), max(1, ey - oy)))
    }

    private static func makePyramid(_ W: Int, _ H: Int, nLevels: Int,
                                     factory: (Int, Int) -> CanvasBuffer?) -> [CanvasBuffer]? {
        var levels: [CanvasBuffer] = []
        levels.reserveCapacity(nLevels)
        for k in 0 ..< nLevels {
            let (lw, lh) = levelSize(W, H, k)
            guard let b = factory(lw, lh) else { return nil }
            levels.append(b)
        }
        return levels
    }

    // MARK: - Streaming blend

    private struct BlendKernels {
        let reduce: MTLComputePipelineState
        let extendRect: MTLComputePipelineState
        let lapAccExpanded: MTLComputePipelineState
        let lapAccDC: MTLComputePipelineState
        let norm: MTLComputePipelineState
        let collapseExpanded: MTLComputePipelineState
        let finalize: MTLComputePipelineState
    }

    private static func loadBlendKernels(ctx: PanoContext) -> BlendKernels? {
        guard let r   = ctx.loadPSO("pyramidReduce"),
              let er  = ctx.loadPSO("pyramidExtendRect"),
              let lae = ctx.loadPSO("pyramidLapAccExpanded"),
              let lad = ctx.loadPSO("pyramidLapAccDC"),
              let n   = ctx.loadPSO("pyramidNormalize"),
              let ce  = ctx.loadPSO("pyramidCollapseAddExpanded"),
              let f   = ctx.loadPSO("pyramidFinalise") else { return nil }
        return BlendKernels(reduce: r, extendRect: er, lapAccExpanded: lae, lapAccDC: lad,
                            norm: n, collapseExpanded: ce, finalize: f)
    }

    private static func blendStream(graph: PanoGraph,
                                     canvas: CylindricalCanvas,
                                     exposureGains: [SIMD3<Float>],
                                     seamLabels: SeamFinder.LabelState?,
                                     progress: ProgressTracker) async -> CanvasBuffer? {
        let ctx = PanoContext.shared
        let W = canvas.size.x, H = canvas.size.y
        let nLevels = pyramidLevelCount(W, H)
        dbg("[Blend] streaming Laplacian pyramid: \(nLevels) levels, canvas \(W)×\(H)")

        let floatFactory: (Int, Int) -> CanvasBuffer? = { ctx.makeFloatBuffer(width: $0, height: $1) }
        // Always the masked warp: it is where rgb enters the blend domain
        // that `pyramidFinalise` decodes (BlendDomain.h).
        guard let labels  = seamLabels ?? SeamFinder.LabelState.unowned(),
              let warpPSO = ctx.loadPSO("cylindricalWarpMasked"),
              let k       = loadBlendKernels(ctx: ctx),
              let acc     = makePyramid(W, H, nLevels: nLevels, factory: floatFactory)
        else { return nil }

        // Zero the accumulator (private buffers are not documented to zero-init).
        guard Compute.run({ cb in
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            for level in acc { blit.fill(buffer: level.buffer, range: 0..<level.byteCount, value: 0) }
            blit.endEncoding()
        }) else { return nil }

        let total = graph.nodes.count
        for (i, node) in graph.nodes.enumerated() {
            let mask: WarpMask = (labels: labels,
                                  gain: i < exposureGains.count ? exposureGains[i]
                                                                : SIMD3<Float>(repeating: 1),
                                  selfIndex: UInt32(i))
            let roi = makeBlendROI(node: node, canvas: canvas, nLevels: nLevels)
            if let roi {
                dbg("[Blend] node \(node.id): ROI \(roi.size.x)×\(roi.size.y) @(\(roi.origin.x),\(roi.origin.y)) of \(W)×\(H)")
            }
            guard let warped = warpNode(node, canvas: canvas, pso: warpPSO,
                                        mask: mask, roi: roi) else {
                dbg("[Pipeline] warp: skipping node \(node.id)")
                continue
            }

            guard accumulate(image: warped, roi: roi, into: acc,
                              W: W, H: H, nLevels: nLevels, k: k, ctx: ctx)
            else { return nil }

            await progress.advance("Blended image \(i + 1)/\(total)")
            // `warped` and its per-image Gaussian pyramid drop out of scope here.
        }

        return collapse(acc: acc, W: W, H: H, nLevels: nLevels, k: k, ctx: ctx)
    }

    /// Build this image's Gaussian pyramid and accumulate fused Laplacian
    /// bands into `acc`. `gauss[0]` is `src` directly (no copy). Per-image
    /// allocations: rgba16f Gaussian pyramid only — no Laplacian pyramid,
    /// no expand `temps`.
    ///
    /// With `roi` non-nil, `src` and levels `0..<roi.levels` are ROI-sized;
    /// the reduce into level `roi.levels` writes the footprint rect of an
    /// otherwise-zeroed global buffer, and the remaining (cheap) levels run
    /// full-canvas exactly as before.
    private static func accumulate(image src: CanvasBuffer,
                                    roi: BlendROI?,
                                    into acc: [CanvasBuffer],
                                    W: Int, H: Int, nLevels: Int,
                                    k: BlendKernels,
                                    ctx: PanoContext) -> Bool {
        let R = roi?.levels ?? 0
        var gauss: [CanvasBuffer] = [src]
        for level in 1 ..< nLevels {
            let (lw, lh): (Int, Int)
            if let roi, level < R {
                let r = roiLevelRect(roi, level, W, H)
                (lw, lh) = (r.size.x, r.size.y)
            } else {
                (lw, lh) = levelSize(W, H, level)
            }
            guard let g = ctx.makeHalfBuffer(width: lw, height: lh) else { return false }
            gauss.append(g)
        }

        let zero = SIMD2<UInt32>(repeating: 0)
        func packed(_ v: SIMD2<Int>) -> SIMD2<UInt32> { SIMD2(UInt32(v.x), UInt32(v.y)) }

        return Compute.run { cb in
            // 1. gauss[k+1] = reduce(gauss[k]) — ROI-local below the hand-off
            //    level, ROI-rect into global at it (followed by a clamp-extend
            //    of rgb into the rest of that level, α = 0, preserving the
            //    extrapolation field for the coarser reduces), full-canvas
            //    above it.
            for i in 0 ..< (nLevels - 1) {
                var outOrigin = zero
                var outSize = gauss[i + 1].dimsPacked
                let isHandOff = roi != nil && i + 1 == R
                if let roi, isHandOff {
                    let r = roiLevelRect(roi, i + 1, W, H)
                    outOrigin = packed(r.origin)
                    outSize = packed(r.size)
                }
                Compute.encode(cb, k.reduce,
                    buffers: [gauss[i].buffer, gauss[i + 1].buffer],
                    dims:    [gauss[i].dimsPacked, gauss[i + 1].dimsPacked,
                              outOrigin, outSize],
                    gridW: Int(outSize.x), gridH: Int(outSize.y))
                if isHandOff {
                    Compute.encode(cb, k.extendRect,
                        buffers: [gauss[i + 1].buffer],
                        dims:    [gauss[i + 1].dimsPacked, outOrigin, outSize],
                        gridW: gauss[i + 1].width, gridH: gauss[i + 1].height)
                }
            }
            // 2. DC term: acc[N-1] += gauss[N-1] * α (always full-canvas)
            let dc = gauss[nLevels - 1]
            Compute.encode(cb, k.lapAccDC,
                buffers: [dc.buffer, acc[nLevels - 1].buffer],
                dims:    [dc.dimsPacked],
                gridW: dc.width, gridH: dc.height)
            // 3. Bandpass: acc[i] += (gauss[i] − Expand(gauss[i+1])) * α,
            //    dispatched over the fine buffer (ROI or full canvas).
            for i in stride(from: nLevels - 2, through: 0, by: -1) {
                var fineOrigin = zero
                var coarseOrigin = zero
                if let roi {
                    if i < R     { fineOrigin   = packed(roiLevelRect(roi, i,     W, H).origin) }
                    if i + 1 < R { coarseOrigin = packed(roiLevelRect(roi, i + 1, W, H).origin) }
                }
                Compute.encode(cb, k.lapAccExpanded,
                    buffers: [gauss[i].buffer, gauss[i + 1].buffer, acc[i].buffer],
                    dims:    [gauss[i].dimsPacked, gauss[i + 1].dimsPacked,
                              acc[i].dimsPacked, fineOrigin, coarseOrigin],
                    gridW: gauss[i].width, gridH: gauss[i].height)
            }
        }
    }

    /// Collapse coarse → fine, finalise to a half16 shared-storage buffer.
    /// Streaming: only two adjacent pyramid levels are alive at any time.
    private static func collapse(acc: [CanvasBuffer],
                                  W: Int, H: Int, nLevels: Int,
                                  k: BlendKernels, ctx: PanoContext) -> CanvasBuffer? {
        let (cw, ch) = levelSize(W, H, nLevels - 1)
        guard var current = ctx.makeFloatBuffer(width: cw, height: ch),
              let out     = ctx.makeHalfBuffer(width: W, height: H, storage: .shared)
        else { return nil }

        guard Compute.run({ cb in
            Compute.encode(cb, k.norm,
                buffers: [acc[nLevels - 1].buffer, current.buffer],
                dims:    [current.dimsPacked],
                gridW: current.width, gridH: current.height)
        }) else { return nil }

        for i in stride(from: nLevels - 2, through: 0, by: -1) {
            let (fw, fh) = levelSize(W, H, i)
            guard let fine = ctx.makeFloatBuffer(width: fw, height: fh) else { return nil }
            guard Compute.run({ cb in
                Compute.encode(cb, k.collapseExpanded,
                    buffers: [acc[i].buffer, current.buffer, fine.buffer],
                    dims:    [fine.dimsPacked, current.dimsPacked],
                    gridW: fine.width, gridH: fine.height)
            }) else { return nil }
            current = fine     // previous `current` released here
        }

        guard Compute.run({ cb in
            Compute.encode(cb, k.finalize,
                buffers: [current.buffer, out.buffer],
                dims:    [out.dimsPacked],
                gridW: W, gridH: H)
        }) else { return nil }
        return out
    }

    // MARK: - Display & export

    /// Convert blended buffer → display-ready CGImage in HLG HDR space.
    static func renderPreview(_ result: PanoramaResult) -> CGImage? {
        guard let ci = result.blendedBuffer.linearCIImage() else { return nil }
        return PanoContext.shared.hdrCIContext.createCGImage(
            ci, from: ci.extent,
            format: .RGBAh,
            colorSpace: PanoContext.hdrColorSpace
        )
    }

    /// Save a `CanvasBuffer` as HEIF10 HDR in HLG space.
    ///
    /// `edits` are baked in via the same `applyEditsKernel` the display
    /// fragment shader uses (preview and exported pixels are bit-identical
    /// for any given EditParams). If `crop` is non-nil the edited buffer is
    /// trimmed to that canvas-pixel rectangle before export.
    ///
    /// Metadata: if `exifSource` is non-nil its EXIF/TIFF/XMP is spliced in
    /// losslessly via `CGImageDestinationCopyImageSource`. GPano XMP for the
    /// final dimensions is added on top when `writeGPano` is true (skip it
    /// for non-panorama images — e.g. a standalone open the user is just
    /// using to test crop).
    static func save(buffer: CanvasBuffer,
                     to url: URL,
                     edits: EditParams = .identity,
                     crop: CGRect? = nil,
                     exifSource: URL? = nil,
                     writeGPano: Bool = true,
                     quality: Double = 0.95,
                     format: ImageExportFormat = .heifHLG) throws {
        let edited = ImageEditor.apply(edits, to: buffer, storage: .shared)
                  ?? buffer
        let final: CanvasBuffer = {
            guard let crop else { return edited }
            return ImageEditor.crop(edited, to: crop) ?? edited
        }()
        guard let ci = final.linearCIImage() else {
            throw PanoError.exportFailed("CGImage from blended buffer failed")
        }

        try saveCIImage(ci, to: url, format: format,
                        ciContext: PanoContext.shared.hdrCIContext,
                        quality: quality)

        let w = final.width, h = final.height
        injectMetadata(url: url, exifSource: exifSource, writeGPano: writeGPano,
                        width: w, height: h)

        let cropTag = crop.map { String(format: " crop=%.0f×%.0f", $0.width, $0.height) } ?? ""
        dbg("[Export] saved \(w)×\(h) px \(format.displayName) (edits=\(edits.isIdentity ? "none" : "applied"))\(cropTag) → \(url.lastPathComponent)")
    }

    /// Convenience overload: save a stitched panorama result — uses its
    /// first source as EXIF source and always writes GPano metadata.
    static func save(result: PanoramaResult,
                     to url: URL,
                     edits: EditParams = .identity,
                     crop: CGRect? = nil,
                     quality: Double = 0.95,
                     format: ImageExportFormat = .heifHLG) throws {
        try save(buffer: result.blendedBuffer, to: url,
                  edits: edits, crop: crop,
                  exifSource: result.graph.nodes.first?.image.sourceURL,
                  writeGPano: true,
                  quality: quality,
                  format: format)
    }

    // MARK: - Metadata splice (EXIF passthrough + GPano XMP)

    private static func injectMetadata(url: URL,
                                        exifSource: URL?,
                                        writeGPano: Bool,
                                        width: Int, height: Int) {
        // Skip the rewrite entirely if we have nothing to add — saves an
        // I/O round-trip for standalone-image saves with no EXIF source.
        guard exifSource != nil || writeGPano else { return }
        guard let src  = CGImageSourceCreateWithURL(url as CFURL, nil),
              let type = CGImageSourceGetType(src) else { return }

        let meta: CGMutableImageMetadata = {
            if let srcURL  = exifSource,
               let srcMeta = EXIF.metadata(for: srcURL),
               let copy    = CGImageMetadataCreateMutableCopy(srcMeta) {
                return copy
            }
            return CGImageMetadataCreateMutable()
        }()
        if writeGPano { addGPanoTags(meta, width: width, height: height) }

        let tmp = url.deletingLastPathComponent()
            .appendingPathComponent(".tmp_\(url.lastPathComponent)")
        guard let dest = CGImageDestinationCreateWithURL(tmp as CFURL, type, 1, nil) else { return }

        let opts: [CFString: Any] = [
            kCGImageDestinationMetadata:      meta,
            kCGImageDestinationMergeMetadata: true
        ]
        var err: Unmanaged<CFError>?
        guard CGImageDestinationCopyImageSource(dest, src, opts as CFDictionary, &err) else {
            dbg("[Export] metadata inject failed: \(String(describing: err))")
            try? FileManager.default.removeItem(at: tmp)
            return
        }
        try? FileManager.default.replaceItem(at: url, withItemAt: tmp,
                                              backupItemName: nil, options: [], resultingItemURL: nil)
    }

    private static func addGPanoTags(_ meta: CGMutableImageMetadata,
                                      width: Int, height: Int) {
        let ns  = "http://ns.google.com/photos/1.0/panorama/" as CFString
        let pfx = "GPano" as CFString

        func set(_ tag: String, _ value: String) {
            guard let t = CGImageMetadataTagCreate(ns, pfx, tag as CFString,
                                                   .string, value as CFTypeRef) else { return }
            CGImageMetadataSetTagWithPath(meta, nil, "\(pfx):\(tag)" as CFString, t)
        }
        set("ProjectionType",              "cylindrical")
        set("FullPanoWidthPixels",         "\(width)")
        set("FullPanoHeightPixels",        "\(height)")
        set("CroppedAreaImageWidthPixels", "\(width)")
        set("CroppedAreaImageHeightPixels","\(height)")
        set("CroppedAreaLeftPixels",       "0")
        set("CroppedAreaTopPixels",        "0")
    }

    // MARK: - Misc

    private static func elapsed(_ t: Date) -> Int {
        Int(Date().timeIntervalSince(t) * 1000)
    }

    private static func progressUnitCount(imageCount n: Int) -> Int {
        let pairs = max(0, n * (n - 1) / 2)
        // Per image: load, detect, streaming blend. Fixed: graph, poses,
        // rotations, canvas, exposure, seams, finished = 7.
        return max(1, 3 * n + pairs + 7)
    }
}

// MARK: - ProgressTracker

private actor ProgressTracker {
    private let totalUnits: Int
    private let handler: (@Sendable (PipelineProgress) -> Void)?
    private var completedUnits = 0

    init(totalUnits: Int, handler: (@Sendable (PipelineProgress) -> Void)?) {
        self.totalUnits = totalUnits
        self.handler = handler
    }

    func update(_ message: String) { emit(message) }

    func advance(_ message: String, units: Int = 1) {
        completedUnits = min(totalUnits, completedUnits + units)
        emit(message)
    }

    private func emit(_ message: String) {
        guard let handler else { return }
        handler(PipelineProgress(completedUnits: completedUnits,
                                  totalUnits: totalUnits,
                                  message: message))
    }
}
