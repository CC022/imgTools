//
//  TextureScaler.swift
//  panoDev
//
//  Lanczos-downsample an existing MTLTexture into a smaller one. Used to
//  derive the detection-resolution image from the already-loaded full-res
//  texture, avoiding a second disk decode + colorspace pass per file.
//

import Metal
import MetalPerformanceShaders

enum TextureScaler {

    /// Returns a new texture whose long edge is at most `maxLongEdge`,
    /// downsampled from `src` via MPSImageLanczosScale. Returns `src` itself
    /// if it's already small enough. Same `pixelFormat` as `src`.
    static func scaledDown(_ src: MTLTexture, maxLongEdge: Int) -> MTLTexture? {
        let srcLong = max(src.width, src.height)
        guard srcLong > maxLongEdge else { return src }

        let scale = Double(maxLongEdge) / Double(srcLong)
        let dstW  = max(1, Int((Double(src.width)  * scale).rounded()))
        let dstH  = max(1, Int((Double(src.height) * scale).rounded()))

        let ctx = PanoContext.shared
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: src.pixelFormat,
            width:  dstW, height: dstH,
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let dst = ctx.device.makeTexture(descriptor: desc) else { return nil }

        let scaler = MPSImageLanczosScale(device: ctx.device)
        Compute.run { cb in
            scaler.encode(commandBuffer: cb,
                          sourceTexture: src,
                          destinationTexture: dst)
        }
        return dst
    }
}
