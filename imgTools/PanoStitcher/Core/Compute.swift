//
//  Compute.swift
//  panoDev
//
//  Thin helpers around MTLCommandBuffer / MTLComputeCommandEncoder so callers
//  don't have to open-code 8–12 lines of dispatch boilerplate per kernel.
//

import Metal

enum Compute {

    /// 16×16 threadgroup — same shape every kernel uses. Grid is rounded up.
    private static let tg = MTLSize(width: 16, height: 16, depth: 1)

    private static func gridFor(_ w: Int, _ h: Int) -> MTLSize {
        MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)
    }

    // MARK: - Encode

    /// Encode a single dispatch into an existing command buffer.
    ///
    /// Buffers are bound at indices `0 ..< buffers.count`.
    /// `dims` are bound *as inline bytes* at indices `buffers.count ..< +dims.count`
    /// — used for the `BufDims` constants every buffer-based kernel takes.
    /// `gridW × gridH` is the kernel's logical output size.
    static func encode(_ cb: MTLCommandBuffer,
                        _ pso: MTLComputePipelineState,
                        buffers: [MTLBuffer],
                        dims: [SIMD2<UInt32>] = [],
                        textures: [MTLTexture] = [],
                        bytes: [(UnsafeRawPointer, Int)] = [],
                        gridW: Int, gridH: Int)
    {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        for (i, b) in buffers.enumerated()  { enc.setBuffer(b,  offset: 0, index: i) }
        for (i, t) in textures.enumerated() { enc.setTexture(t,            index: i) }

        var bufIdx = buffers.count
        for var d in dims {
            enc.setBytes(&d, length: MemoryLayout<SIMD2<UInt32>>.stride, index: bufIdx)
            bufIdx += 1
        }
        for (ptr, len) in bytes {
            enc.setBytes(ptr, length: len, index: bufIdx)
            bufIdx += 1
        }

        enc.dispatchThreadgroups(gridFor(gridW, gridH), threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // MARK: - Run

    /// Build a one-shot command buffer, run `body` to populate it, then commit
    /// and wait. Returns whether anything was scheduled.
    @discardableResult
    static func run(_ body: (MTLCommandBuffer) -> Void) -> Bool {
        guard let cb = PanoContext.shared.commandQueue.makeCommandBuffer() else {
            return false
        }
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        return true
    }
}
