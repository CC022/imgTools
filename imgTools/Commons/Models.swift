import Foundation

enum ImageOperation: String, CaseIterable {
    case convert = "Convert Images"
    case hdrBoost = "HDR Boost"
    case slicer = "Slicer"
    case video = "Images to Video"
    case panoStitch = "Pano Stitcher"
    case longExposure = "Long Exposure"
    case hdmiCapture = "Video Capture"

    var systemImage: String {
        switch self {
        case .convert:       return "arrow.triangle.2.circlepath"
        case .hdrBoost:      return "sparkles"
        case .slicer:        return "scissors"
        case .video:         return "film"
        case .panoStitch:    return "pano"
        case .longExposure:  return "camera.aperture"
        case .hdmiCapture:   return "tv"
        }
    }
}

struct ImageItem: Identifiable {
    let id = UUID()
    let url: URL
    var status: ProcessingStatus = .pending
}

enum ProcessingStatus {
    case pending
    case processing
    case success([URL])
    case failed(String)
}

enum ActiveImporter: Hashable {
    case images
    case hdrInput
    case hdrMask
    case outputFolder
}

enum PendingOutputAction {
    case processAll
    case hdrBoost(inputURL: URL, maskURL: URL)
}

enum ImageToolsError: LocalizedError {
    case invalidImage
    case invalidMaskImage
    case unsupportedFormat
    case slicingFailed
    case processingFailed
    case noImages
    case videoCreationFailed
    case previewFailed
    case alignmentFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:        return "Unable to load image"
        case .invalidMaskImage:    return "Unable to load mask image"
        case .unsupportedFormat:   return "Unsupported image format"
        case .slicingFailed:       return "Failed to create sliced images"
        case .processingFailed:    return "Failed to process image"
        case .noImages:            return "No images to process"
        case .videoCreationFailed: return "Failed to create video"
        case .previewFailed:       return "Failed to render preview"
        case .alignmentFailed:     return "Frame alignment failed"
        }
    }
}
