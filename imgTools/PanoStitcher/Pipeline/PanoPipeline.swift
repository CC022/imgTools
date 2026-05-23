//
//  PanoPipeline.swift
//  panoDev
//
//  End-to-end orchestrator. Buffer-backed canvas storage (CanvasBuffer)
//  removes the 16384-px Metal 2D-texture limit; canvas is bounded only by
//  available GPU memory.
//
//  Memory strategy (streaming):
//    • Source full-res `ImageTexture`s exist only during step 1+2 (load +
//      detection); they are dropped once `[PanoImage]` metadata is in the
//      graph.
//    • Warped canvas buffers, exposure-gain estimation warps, and per-image
//      Gaussian pyramids are *transient* — materialised inside `warpNode`
//      / `accumulate` and released at the end of each iteration.
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

struct PanoramaResult: @unchecked Sendable {
    let graph: PanoGraph
    let canvas: CylindricalCanvas
    /// Final blended panorama (rgba16f, .shared storage so the CPU can read it).
    let blendedBuffer: CanvasBuffer
}

struct PipelineProgress: Sendable {
    let completedUnits: Int
    let totalUnits: Int
    let message: String

    var fraction: Double {
        guard totalUnits > 0 else { return 0 }
        return min(1, max(0, Double(completedUnits) / Double(totalUnits)))
    }
}

// MARK: - Pipeline

enum PanoPipeline {

    static func process(urls: [URL],
                        focalLengthMM35: Float = 24.0,
                        progress: (@Sendable (PipelineProgress) -> Void)? = nil) async -> PanoramaResult? {
        dbg("[Pipeline] start: \(urls.count) image(s), f_mm35=\(focalLengthMM35)")
        let t0 = Date()
        let tracker = ProgressTracker(totalUnits: progressUnitCount(imageCount: urls.count),
                                      handler: progress)
        await tracker.update("Preparing \(urls.count) image\(urls.count == 1 ? "" : "s")")

        // 1+2 — Load + detect. `loadAndDetect` owns the `[ImageTexture]`
        // array and lets it (and its full-res GPU textures) fall out of
        // scope on return; the rest of the pipeline runs on metadata only.
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

        // 8 — Brown-Lowe exposure / white-balance compensation. Estimate from
        //     pairwise on-demand warps, then apply each gain during the
        //     streaming blend.
        let t_exp = Date()
        await tracker.update("Estimating exposure")
        let exposureGains = estimateExposureGains(graph: graph, canvas: canvas)
        dbg("[Pipeline] exposure estimation (\(elapsed(t_exp)) ms)")
        await tracker.advance("Estimated exposure")

        // 8.5 — Seam state. Mask is applied to each warped image just before
        //       pyramid accumulation.
        let t_seam = Date()
        await tracker.update("Preparing seams")
        let seamState = SeamFinder.makeCoverageState(graph: graph, canvas: canvas)
        dbg("[Pipeline] seam preparation (\(elapsed(t_seam)) ms)")
        await tracker.advance("Prepared seams")

        // 9 — Burt-Adelson Laplacian-pyramid blend, streaming one image at a
        //     time into the accumulator.
        let t5 = Date()
        await tracker.update("Blending panorama")
        guard let blended = await blendStream(graph: graph,
                                               canvas: canvas,
                                               exposureGains: exposureGains,
                                               seamState: seamState,
                                               progress: tracker) else {
            dbg("[Pipeline] blend dispatch failed")
            return nil
        }
        dbg("[Pipeline] blended \(graph.nodes.count) image(s) (\(elapsed(t5)) ms)")
        await tracker.advance("Finished")

        dbg("[Pipeline] complete in \(elapsed(t0)) ms")
        return PanoramaResult(graph: graph, canvas: canvas, blendedBuffer: blended)
    }

    // MARK: - Steps 1+2: load + detect (full-res textures scoped here only)

    /// Decode every URL to a full-res `ImageTexture`, run SuperPoint on a
    /// 2 K detection-size copy of each, and return only the metadata +
    /// keypoints. The full-res textures live for the duration of this
    /// function and are released on return — they never coexist with the
    /// canvas-sized blend pyramids allocated in step 9.
    private static func loadAndDetect(urls: [URL],
                                       t0: Date,
                                       progress: ProgressTracker)
                                       async -> ([PanoImage], [[Keypoint]])? {
        let loaded = await loadAll(urls: urls, progress: progress)
        guard !loaded.isEmpty else { return nil }
        dbg("[Pipeline] loaded \(loaded.count) full-res image(s) in \(elapsed(t0)) ms")

        let t_det = Date()
        let detector = SuperPointDetector()
        var allKps: [[Keypoint]] = []
        allKps.reserveCapacity(loaded.count)
        for (i, full) in loaded.enumerated() {
            guard let smallTex = TextureScaler.scaledDown(full.texture,
                                                           maxLongEdge: 2048) else {
                allKps.append([])
                await progress.advance("Detected features \(i + 1)/\(loaded.count)")
                continue
            }
            let detItem = ImageTexture(image: full.image, texture: smallTex)
            let scale = Float(full.image.width) / Float(smallTex.width)
            let kps = detector.detect(in: detItem).map { kp in
                var k = kp
                k.x *= scale; k.y *= scale; k.scale *= scale
                return k
            }
            allKps.append(kps)
            await progress.advance("Detected features \(i + 1)/\(loaded.count)")
        }
        dbg("[Pipeline] keypoints: \(allKps.map { String($0.count) }.joined(separator: ", ")) (\(elapsed(t_det)) ms)")

        return (loaded.map(\.image), allKps)
        // `loaded` and every `ImageTexture` it owns drop out of scope here,
        // releasing all full-res source GPU textures before step 3 runs.
    }

    private static func loadAll(urls: [URL],
                                 progress: ProgressTracker) async -> [ImageTexture] {
        await withTaskGroup(of: (Int, ImageTexture?).self) { group in
            for (i, url) in urls.enumerated() {
                group.addTask { (i, try? loadImageTexture(url: url, maxLongEdge: nil)) }
            }
            var slots = [ImageTexture?](repeating: nil, count: urls.count)
            for await (i, img) in group {
                slots[i] = img
                await progress.advance("Loaded full-res \(i + 1)/\(urls.count)")
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

    /// Materialise a single canvas-sized warped buffer for `node`. The source
    /// texture is loaded inside this function and released when it returns;
    /// only the output `CanvasBuffer` survives. Returns nil if the node has
    /// no pose or any allocation/decode fails.
    private static func warpNode(_ node: PanoNode,
                                  canvas: CylindricalCanvas,
                                  pso: MTLComputePipelineState) -> CanvasBuffer? {
        let ctx = PanoContext.shared
        guard let pose = node.pose,
              let imgTex = try? node.image.loadTexture(),
              let outBuf = ctx.makeHalfBuffer(width: canvas.size.x,
                                               height: canvas.size.y) else {
            dbg("[Pipeline] warp: missing pose, source texture, or output buffer for node \(node.id)")
            return nil
        }

        var params = CylindricalWarpParams(
            R:              pose.rotation,
            focal:          pose.intrinsics.focal,
            canvasRadius:   canvas.radius,
            principalPoint: pose.intrinsics.principalPoint,
            canvasOrigin:   canvas.origin,
            imageSize:      SIMD2(Float(imgTex.width), Float(imgTex.height)),
            canvasSize:     SIMD2(Float(canvas.size.x), Float(canvas.size.y))
        )
        guard Compute.run({ cb in
            withUnsafeBytes(of: &params) { raw in
                Compute.encode(cb, pso,
                    buffers:  [outBuf.buffer],
                    textures: [imgTex.texture],
                    bytes:    [(raw.baseAddress!, raw.count)],
                    gridW: canvas.size.x, gridH: canvas.size.y)
            }
        }) else { return nil }
        return outBuf
        // imgTex (and its source texture) is released when this function returns.
    }

    private static func estimateExposureGains(graph: PanoGraph,
                                               canvas: CylindricalCanvas) -> [SIMD3<Float>] {
        let identity = Array(repeating: SIMD3<Float>(repeating: 1), count: graph.nodes.count)
        guard graph.nodes.count >= 2,
              let pso = PanoContext.shared.loadPSO("cylindricalWarp")
        else { return identity }

        return ExposureCompensator.estimateGains(
            imageCount: graph.nodes.count,
            canvas: canvas,
            makeWarped: { idx in
                guard graph.nodes.indices.contains(idx) else { return nil }
                return warpNode(graph.nodes[idx], canvas: canvas, pso: pso)
            }
        )
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
        let lapAccExpanded: MTLComputePipelineState
        let lapAccDC: MTLComputePipelineState
        let norm: MTLComputePipelineState
        let collapseExpanded: MTLComputePipelineState
        let finalize: MTLComputePipelineState
    }

    private static func loadBlendKernels(ctx: PanoContext) -> BlendKernels? {
        guard let r   = ctx.loadPSO("pyramidReduce"),
              let lae = ctx.loadPSO("pyramidLapAccExpanded"),
              let lad = ctx.loadPSO("pyramidLapAccDC"),
              let n   = ctx.loadPSO("pyramidNormalize"),
              let ce  = ctx.loadPSO("pyramidCollapseAddExpanded"),
              let f   = ctx.loadPSO("pyramidFinalise") else { return nil }
        return BlendKernels(reduce: r, lapAccExpanded: lae, lapAccDC: lad,
                            norm: n, collapseExpanded: ce, finalize: f)
    }

    private static func blendStream(graph: PanoGraph,
                                     canvas: CylindricalCanvas,
                                     exposureGains: [SIMD3<Float>],
                                     seamState: SeamFinder.CoverageState?,
                                     progress: ProgressTracker) async -> CanvasBuffer? {
        let ctx = PanoContext.shared
        let W = canvas.size.x, H = canvas.size.y
        let nLevels = pyramidLevelCount(W, H)
        dbg("[Blend] streaming Laplacian pyramid: \(nLevels) levels, canvas \(W)×\(H)")

        let floatFactory: (Int, Int) -> CanvasBuffer? = { ctx.makeFloatBuffer(width: $0, height: $1) }
        guard let warpPSO = ctx.loadPSO("cylindricalWarp"),
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
            guard let warped = warpNode(node, canvas: canvas, pso: warpPSO) else {
                dbg("[Pipeline] warp: skipping node \(node.id)")
                continue
            }
            if i < exposureGains.count {
                ExposureCompensator.apply(gain: exposureGains[i], to: warped)
            }
            SeamFinder.applyVoronoi(state: seamState, warped: warped, selfIndex: UInt32(i))

            guard accumulate(image: warped, into: acc,
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
    private static func accumulate(image src: CanvasBuffer,
                                    into acc: [CanvasBuffer],
                                    W: Int, H: Int, nLevels: Int,
                                    k: BlendKernels,
                                    ctx: PanoContext) -> Bool {
        var gauss: [CanvasBuffer] = [src]
        for level in 1 ..< nLevels {
            let (lw, lh) = levelSize(W, H, level)
            guard let g = ctx.makeHalfBuffer(width: lw, height: lh) else { return false }
            gauss.append(g)
        }

        return Compute.run { cb in
            // 1. gauss[k+1] = reduce(gauss[k])
            for i in 0 ..< (nLevels - 1) {
                Compute.encode(cb, k.reduce,
                    buffers: [gauss[i].buffer, gauss[i + 1].buffer],
                    dims:    [gauss[i].dimsPacked, gauss[i + 1].dimsPacked],
                    gridW: gauss[i + 1].width, gridH: gauss[i + 1].height)
            }
            // 2. DC term: acc[N-1] += gauss[N-1] * α
            let dc = gauss[nLevels - 1]
            Compute.encode(cb, k.lapAccDC,
                buffers: [dc.buffer, acc[nLevels - 1].buffer],
                dims:    [dc.dimsPacked],
                gridW: dc.width, gridH: dc.height)
            // 3. Bandpass: acc[i] += (gauss[i] − Expand(gauss[i+1])) * α
            for i in stride(from: nLevels - 2, through: 0, by: -1) {
                Compute.encode(cb, k.lapAccExpanded,
                    buffers: [gauss[i].buffer, gauss[i + 1].buffer, acc[i].buffer],
                    dims:    [gauss[i].dimsPacked, gauss[i + 1].dimsPacked],
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
