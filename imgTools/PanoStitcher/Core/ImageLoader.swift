//
//  ImageLoader.swift
//  panoDev
//
//  Loads a source image into a scene-linear rgba16Float MTLTexture, returning
//  an `ImageTexture` (PanoImage metadata + GPU texture, owned by the caller).
//
//  Color-space strategy
//  ────────────────────
//  • The CIImage is loaded without a forced target space so that CoreImage reads
//    the file's embedded profile (sRGB, Display P3, HLG, etc.) correctly.
//  • ciContext.render(…colorSpace: linearColorSpace) then converts whatever that
//    profile is into scene-linear extendedLinearSRGB before writing to the texture.
//    CoreImage handles every transfer function (gamma 2.2, sRGB piecewise,
//    PQ, HLG) transparently.
//  • The *original* color space is stored in PanoImage.colorSpace and is used
//    only at output time (display / export) to convert back from linear.
//
//  Blending therefore happens in linear light for all input types without any
//  shader-side color conversion.
//

import CoreImage
import Metal
import CoreGraphics

/// Decode `url` and upload to a fresh rgba16Float MTLTexture, returning the
/// loaded `ImageTexture`. The caller owns the result; release the texture by
/// dropping the returned value.
///
/// - Parameter maxLongEdge: if non-nil, the image is Lanczos-downscaled so
///   its long edge is at most this many pixels (detection / low-res warps).
///   The returned `PanoImage` metadata always records the *original* pixel
///   dimensions; consumers read decoded sizes from the texture.
nonisolated func loadImageTexture(url: URL, maxLongEdge: Int? = nil) throws -> ImageTexture {
    guard url.isFileURL,
          FileManager.default.fileExists(atPath: url.path) else {
        throw PanoError.fileNotFound(url)
    }

    let ctx = PanoContext.shared

    // Load without forcing a destination space — preserves the source profile.
    guard let ci = CIImage(contentsOf: url, options: [.expandToHDR: true, .applyOrientationProperty: true]) else {
        throw PanoError.decodeFailed(url)
    }

    // Remember the original profile for output; fall back to sRGB if untagged.
    let originalColorSpace = ci.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!

    let srcW = ci.extent.width
    let srcH = ci.extent.height
    let finalCI: CIImage
    if let cap = maxLongEdge,
       case let scale = CGFloat(cap) / max(srcW, srcH), scale < 1.0 {
        finalCI = ci.applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey:       scale,
            kCIInputAspectRatioKey: 1.0,
        ])
        dbg("[Loader] \(url.lastPathComponent): \(Int(srcW))×\(Int(srcH)) → \(Int(srcW*scale))×\(Int(srcH*scale)) (scale \(String(format:"%.3f", scale)))")
    } else {
        finalCI = ci
    }

    let width  = Int(finalCI.extent.width.rounded())
    let height = Int(finalCI.extent.height.rounded())

    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float,
        width: width, height: height, mipmapped: false
    )
    desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
    desc.storageMode = .private

    guard let texture = ctx.device.makeTexture(descriptor: desc) else {
        throw PanoError.textureFailed
    }

    // CIImage origin is bottom-left; Metal is top-left.
    let flipped = finalCI.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
        .translatedBy(x: 0, y: -finalCI.extent.height))

    // Convert source → scene-linear extendedLinearSRGB.
    // CoreImage applies the correct inverse transfer function for any profile.
    ctx.ciContext.render(
        flipped,
        to: texture,
        commandBuffer: nil,
        bounds: CGRect(x: 0, y: 0, width: width, height: height),
        colorSpace: PanoContext.linearColorSpace
    )

    // Metadata carries the original dimensions even when the texture is a
    // downscaled decode — intrinsics and canvas geometry are derived from
    // these, and per-texture consumers rescale by texture/metadata ratio.
    let image = PanoImage(width: Int(srcW.rounded()), height: Int(srcH.rounded()),
                          sourceURL: url, colorSpace: originalColorSpace)
    return ImageTexture(image: image, texture: texture)
}
