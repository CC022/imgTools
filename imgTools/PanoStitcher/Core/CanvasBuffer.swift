//
//  CanvasBuffer.swift
//  panoDev
//
//  Canvas-sized storage backed by an MTLBuffer rather than an MTLTexture
//  (no 16384-px-per-side Metal 2D-texture limit). Holds either rgba16f
//  (8 B/px) or rgba32f (16 B/px) layouts; row-major, tightly packed.
//
//  Indexing in shaders: `pixel = buf[y * width + x]`.
//

import Metal
import Foundation
import CoreImage
import CoreGraphics

nonisolated struct CanvasBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
    let width: Int
    let height: Int
    /// 8 (`rgba16f`) or 16 (`rgba32f`).
    let bytesPerPixel: Int

    var bytesPerRow: Int { width * bytesPerPixel }
    var byteCount: Int   { width * height * bytesPerPixel }

    /// Pack the dimensions for shader `BufDims` slots.
    var dimsPacked: SIMD2<UInt32> { SIMD2(UInt32(width), UInt32(height)) }
}

nonisolated extension PanoContext {
    /// rgba16f canvas buffer (8 B/pixel — pipeline's primary format).
    func makeHalfBuffer(width: Int, height: Int,
                         storage: MTLStorageMode = .private) -> CanvasBuffer? {
        makeRawBuffer(width: width, height: height, bytesPerPixel: 8, storage: storage)
    }

    /// rgba32f canvas buffer (16 B/pixel — used for accumulators / collapse temps).
    func makeFloatBuffer(width: Int, height: Int,
                          storage: MTLStorageMode = .private) -> CanvasBuffer? {
        makeRawBuffer(width: width, height: height, bytesPerPixel: 16, storage: storage)
    }

    private func makeRawBuffer(width: Int, height: Int,
                                bytesPerPixel: Int,
                                storage: MTLStorageMode) -> CanvasBuffer? {
        let opts: MTLResourceOptions = (storage == .shared)
            ? .storageModeShared : .storageModePrivate
        guard let buf = device.makeBuffer(length: width * height * bytesPerPixel,
                                          options: opts) else { return nil }
        return CanvasBuffer(buffer: buf, width: width, height: height,
                            bytesPerPixel: bytesPerPixel)
    }
}

// MARK: - Gaussian mip chain

/// Gaussian mip chain over a CanvasBuffer (`levels[0]` is the base buffer).
/// Built with the blend pipeline's `pyramidReduce` kernel. Used by the
/// display path for minification filtering (no nearest-neighbour aliasing
/// when zoomed out) and by the edit math's low-frequency luma zone
/// weighting (local highlights/shadows).
nonisolated struct CanvasMips: @unchecked Sendable {
    let levels: [CanvasBuffer]

    static func build(for base: CanvasBuffer, maxLevels: Int = 8) -> CanvasMips {
        let ctx = PanoContext.shared
        var levels = [base]
        guard let pso = ctx.loadPSO("pyramidReduce") else { return CanvasMips(levels: levels) }
        let zero = SIMD2<UInt32>(repeating: 0)
        while levels.count < maxLevels {
            let cur = levels[levels.count - 1]
            let lw = cur.width / 2, lh = cur.height / 2
            guard lw >= 16, lh >= 16,
                  let nxt = ctx.makeHalfBuffer(width: lw, height: lh) else { break }
            Compute.run { cb in
                Compute.encode(cb, pso,
                    buffers: [cur.buffer, nxt.buffer],
                    dims:    [cur.dimsPacked, nxt.dimsPacked, zero, nxt.dimsPacked],
                    gridW: lw, gridH: lh)
            }
            levels.append(nxt)
        }
        return CanvasMips(levels: levels)
    }
}

// MARK: - Display / export bridges

nonisolated extension CanvasBuffer {
    /// Build a CGImage from this buffer's bytes (only valid when storage is shared).
    /// Assumes `rgba16f` (the only display-bound format we currently produce).
    func cgImage(colorSpace: CGColorSpace) -> CGImage? {
        precondition(bytesPerPixel == 8, "cgImage() expects rgba16f buffer")
        let data = Data(bytes: buffer.contents(), count: byteCount)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        let bitmapInfo = CGBitmapInfo(rawValue:
            CGBitmapInfo.byteOrder16Little.rawValue |
            CGBitmapInfo.floatComponents.rawValue |
            CGImageAlphaInfo.premultipliedLast.rawValue)

        return CGImage(
            width: width, height: height,
            bitsPerComponent: 16, bitsPerPixel: 64,
            bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo,
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        )
    }

    /// Wrap as a scene-linear CIImage tagged `extendedLinearSRGB`. Bypasses
    /// any Metal-texture size limit on the way to CoreImage.
    func linearCIImage() -> CIImage? {
        cgImage(colorSpace: PanoContext.linearColorSpace).map(CIImage.init(cgImage:))
    }
}
