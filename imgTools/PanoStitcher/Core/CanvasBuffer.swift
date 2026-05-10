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

struct CanvasBuffer: @unchecked Sendable {
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

extension PanoContext {
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

// MARK: - Display / export bridges

extension CanvasBuffer {
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
