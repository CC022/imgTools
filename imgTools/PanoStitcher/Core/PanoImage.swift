//
//  PanoImage.swift
//  panoDev
//
//  Two values, two responsibilities:
//
//    • `PanoImage`    — pure metadata (dimensions, source URL, original colour
//                       space). Cheap, immutable, sharable; safe to keep for
//                       the entire pipeline lifetime.
//    • `ImageTexture` — a loaded GPU texture (full-res or a downscaled
//                       decode). Lifetime is the value's lifetime — drop the
//                       `ImageTexture` to free GPU memory.
//
//  Colour-space contract
//  ─────────────────────
//  • `ImageTexture.texture` is rgba16Float in scene-linear extendedLinearSRGB.
//    All GPU math (warp, blend) operates on these values.
//  • `PanoImage.colorSpace` is the *original* profile of the source file
//    (sRGB, Display P3, HLG, …). Not the texture's encoding; used only at
//    output time to convert linear → original on the way out.
//

import Metal
import CoreImage
import CoreGraphics

// MARK: - PanoImage (metadata)

nonisolated struct PanoImage: Sendable {
    let width: Int
    let height: Int
    let sourceURL: URL
    /// Original colour space of the source file — used for display and export.
    let colorSpace: CGColorSpace

    /// Decode `sourceURL` and upload a fresh rgba16Float MTLTexture, capped
    /// to `maxLongEdge` when given. Each call produces an independent
    /// texture; the caller owns the returned `ImageTexture` and releases the
    /// texture by dropping the value.
    func loadTexture(maxLongEdge: Int? = nil) throws -> ImageTexture {
        try loadImageTexture(url: sourceURL, maxLongEdge: maxLongEdge)
    }
}

// MARK: - ImageTexture (loaded GPU resource)

nonisolated struct ImageTexture: @unchecked Sendable {
    let image: PanoImage
    /// rgba16Float, scene-linear extendedLinearSRGB, top-left origin.
    let texture: MTLTexture

    /// Texture-side dimensions — these are *not* always equal to
    /// `image.width/height`. A detection-size `ImageTexture` carries the
    /// original `PanoImage` metadata (so URL/colour space are preserved)
    /// but a downsampled `texture`; consumers must read pixel sizes from
    /// here, not from the metadata.
    var width:  Int { texture.width }
    var height: Int { texture.height }

    /// Render the texture back to a CGImage in HDR display space.
    /// Lives on `ImageTexture` because it requires the GPU resource —
    /// a bare `PanoImage` would have to hide a load + GPU upload.
    func toCGImage() -> CGImage? {
        let ctx = PanoContext.shared
        // Tag the texture as linear — that's what's actually stored.
        guard let ciImg = CIImage(mtlTexture: texture, options: [
            .colorSpace: PanoContext.linearColorSpace
        ]) else { return nil }

        let flipped = ciImg.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
            .translatedBy(x: 0, y: -ciImg.extent.height))

        // CoreImage converts linear → HLG for HDR display (.RGBAh preserves > 1.0).
        return ctx.hdrCIContext.createCGImage(
            flipped,
            from: flipped.extent,
            format: .RGBAh,
            colorSpace: PanoContext.hdrColorSpace
        )
    }
}
