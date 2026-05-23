import CoreImage
import CoreGraphics
import UniformTypeIdentifiers
import Foundation

/// Unified set of image output formats available across every tool in the
/// app. Each case knows its file extension, UTType, and a human-readable
/// label suitable for a Picker.
enum ImageExportFormat: String, CaseIterable, Sendable {
    case heifHLG
    case heifPQ
    case tiff16
    case exr

    var fileExtension: String {
        switch self {
        case .heifHLG, .heifPQ: return "heic"
        case .tiff16:           return "tif"
        case .exr:              return "exr"
        }
    }
    var utType: UTType {
        switch self {
        case .heifHLG, .heifPQ: return .heic
        case .tiff16:           return .tiff
        case .exr:              return UTType(filenameExtension: "exr") ?? .data
        }
    }
    var displayName: String {
        switch self {
        case .heifHLG: return "HEIF 10-bit (HLG HDR)"
        case .heifPQ:  return "HEIF 10-bit (PQ HDR)"
        case .tiff16:  return "TIFF 16-bit (HLG HDR)"
        case .exr:     return "OpenEXR (linear float)"
        }
    }
}

enum ImageExportColorSpace {
    static let hlg = CGColorSpace(name: CGColorSpace.itur_2100_HLG)!
    static let pq  = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!
    static let linearSRGB = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
}

/// Encode a CIImage to disk in the chosen format. The pixel pipeline is
/// identical across formats; only the final encoder call and color-space
/// tag differ.
///
/// Caller supplies the CIContext — for HDR exports it should be configured
/// for HDR (e.g. workingColorSpace = HLG or extendedLinearSRGB). For most
/// callers `defaultImageExportCIContext()` is a fine choice.
nonisolated func saveCIImage(_ image: CIImage,
                             to url: URL,
                             format: ImageExportFormat,
                             ciContext: CIContext,
                             quality: Double = 0.95) throws {
    switch format {
    case .heifHLG:
        try ciContext.writeHEIF10Representation(
            of: image, to: url, colorSpace: ImageExportColorSpace.hlg,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality]
        )
    case .heifPQ:
        try ciContext.writeHEIF10Representation(
            of: image, to: url, colorSpace: ImageExportColorSpace.pq,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality]
        )
    case .tiff16:
        try ciContext.writeTIFFRepresentation(
            of: image, to: url,
            format: .RGBA16,
            colorSpace: ImageExportColorSpace.hlg,
            options: [:]
        )
    case .exr:
        try ciContext.writeOpenEXRRepresentation(of: image, to: url, options: [:])
    }
}

/// A reasonable default CIContext for image export — scene-linear working
/// space, half-float intermediate precision, HLG output tagging. Works for
/// all four `ImageExportFormat` cases (HEIF PQ overrides the color space
/// at write time; EXR ignores it).
nonisolated func defaultImageExportCIContext() -> CIContext {
    CIContext(options: [
        .workingColorSpace: ImageExportColorSpace.linearSRGB,
        .workingFormat:     CIFormat.RGBAh,
        .outputColorSpace:  ImageExportColorSpace.hlg,
    ])
}
