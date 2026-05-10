//
//  SeamFinder.swift
//  panoDev
//
//  Voronoi-by-camera-centre coverage assignment. Each canvas pixel is owned
//  by the camera whose optical axis projects nearest to that pixel on the
//  canvas; everyone else's α at that pixel is zeroed. The downstream
//  multi-band pyramid blend converts this binary partition into per-level
//  feathered weights via its Gaussian α downsample.
//
//  Streaming-friendly split:
//
//    • `makeCoverageState(graph:canvas:)` — projects every camera centre
//      to canvas pixel coordinates and uploads them once into a small
//      shared buffer. Returned `CoverageState` is reused for every image.
//
//    • `applyVoronoi(state:warped:selfIndex:)` — single per-image dispatch
//      that zeros this image's α where some other camera centre is closer.
//

import Foundation
import Metal
import simd

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
}
