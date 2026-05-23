//
//  PanoContext.swift
//  panoDev
//

import Metal
import CoreImage
import CoreGraphics

final class PanoContext {

    static let shared = PanoContext()

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let ciContext: CIContext
    /// Dedicated CIContext for HDR display + export, in HLG space.
    ///
    /// HLG (BT.2100 hybrid log-gamma) chosen over PQ (ST.2084) because PQ's
    /// near-exponential high-end transfer curve causes CoreAnimation's
    /// resampler to overshoot dramatically when zooming a tagged CGImage —
    /// HDR highlights blow out into oversaturation. HLG's gamma-then-log
    /// curve is gentle enough that linear interpolation in encoded space
    /// produces visually correct results. HLG is also display-relative
    /// rather than absolute-nits, which is the right model for photography
    /// display (and the SDR fallback is graceful by design).
    let hdrCIContext: CIContext

    static let linearColorSpace  = ImageExportColorSpace.linearSRGB
    static let displayColorSpace = CGColorSpace(name: CGColorSpace.displayP3)!
    static let hdrColorSpace     = ImageExportColorSpace.hlg
    /// PQ (ST.2084) variant of BT.2100 — used as an alternative export color
    /// space alongside HLG. Display is still routed through `hdrColorSpace`
    /// (HLG) to avoid the PQ resampler-overshoot issue described above.
    static let pqColorSpace      = ImageExportColorSpace.pq

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available")
        }
        device = dev

        guard let queue = dev.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        commandQueue = queue

        ciContext = CIContext(
            mtlDevice: dev,
            options: [
                .workingColorSpace: PanoContext.linearColorSpace,
                .workingFormat:     CIFormat.RGBAh,
                .outputColorSpace:  NSNull(),
            ]
        )

        hdrCIContext = CIContext(
            mtlDevice: dev,
            options: [
                .workingColorSpace: PanoContext.hdrColorSpace,
                .outputColorSpace:  PanoContext.hdrColorSpace,
            ]
        )
    }

    // MARK: - Pipeline-state cache

    /// Look up a compute kernel by name. Returns nil + dbg-log on failure.
    /// Intentionally not cached — PSOs are loaded once per pipeline run.
    func loadPSO(_ name: String) -> MTLComputePipelineState? {
        guard let lib = device.makeDefaultLibrary(),
              let fn  = lib.makeFunction(name: name) else {
            dbg("[PanoContext] kernel '\(name)' missing from default library")
            return nil
        }
        return try? device.makeComputePipelineState(function: fn)
    }
}
