import Foundation
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

enum SpatialPhotoError: LocalizedError {
    case couldNotOpenURLAsImageSource(URL)
    case couldNotCopyImageProperties
    case unableToReadImageSize
    case leftAndRightImageSizesDoNotMatch
    case unableToCreateImageDestination
    case unableToFinalizeImageDestination

    var errorDescription: String? {
        switch self {
        case .couldNotOpenURLAsImageSource(let url):
            return "Could not open \(url.lastPathComponent) as an image."
        case .couldNotCopyImageProperties:
            return "Could not read image properties."
        case .unableToReadImageSize:
            return "Could not determine image dimensions."
        case .leftAndRightImageSizesDoNotMatch:
            return "Left and right image dimensions do not match."
        case .unableToCreateImageDestination:
            return "Could not create the output HEIC file."
        case .unableToFinalizeImageDestination:
            return "Could not finalize the output HEIC file."
        }
    }
}

private let identityRotation: [Double] = [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
]

private struct StereoPairImage {
    let source: CGImageSource
    let primaryImageIndex: Int
    let width: Int
    let height: Int

    init(url: URL) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw SpatialPhotoError.couldNotOpenURLAsImageSource(url)
        }
        self.source = source
        self.primaryImageIndex = CGImageSourceGetPrimaryImageIndex(source)

        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, primaryImageIndex, nil) as? [CFString: Any] else {
            throw SpatialPhotoError.couldNotCopyImageProperties
        }
        guard let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw SpatialPhotoError.unableToReadImageSize
        }
        self.width = width
        self.height = height
    }

    func intrinsics(horizontalFOV: Double) -> [Double] {
        let w = Double(width)
        let h = Double(height)
        let fovRad = horizontalFOV / 180.0 * .pi
        let fx = (w * 0.5) / tan(fovRad * 0.5)
        let fy = fx
        let cx = 0.5 * w
        let cy = 0.5 * h
        return [
            fx, 0,  cx,
            0,  fy, cy,
            0,  0,  1
        ]
    }
}

private func spatialPropertiesDictionary(
    isLeft: Bool,
    encodedDisparityAdjustment: Int,
    position: [Double],
    intrinsics: [Double]
) -> [CFString: Any] {
    return [
        kCGImagePropertyGroups: [
            kCGImagePropertyGroupIndex: 0,
            kCGImagePropertyGroupType: kCGImagePropertyGroupTypeStereoPair,
            (isLeft ? kCGImagePropertyGroupImageIsLeftImage : kCGImagePropertyGroupImageIsRightImage): true,
            kCGImagePropertyGroupImageDisparityAdjustment: encodedDisparityAdjustment
        ],
        kCGImagePropertyHEIFDictionary: [
            kIIOMetadata_CameraExtrinsicsKey: [
                kIIOCameraExtrinsics_Position: position,
                kIIOCameraExtrinsics_Rotation: identityRotation
            ],
            kIIOMetadata_CameraModelKey: [
                kIIOCameraModel_Intrinsics: intrinsics,
                kIIOCameraModel_ModelType: kIIOCameraModelType_SimplifiedPinhole
            ]
        ],
        kCGImagePropertyHasAlpha: false
    ]
}

nonisolated func writeSpatialPhoto(
    leftURL: URL,
    rightURL: URL,
    outputURL: URL,
    baselineMM: Double,
    horizontalFOV: Double,
    disparityAdjustment: Double
) throws {
    let leftImage = try StereoPairImage(url: leftURL)
    let rightImage = try StereoPairImage(url: rightURL)

    guard leftImage.width == rightImage.width, leftImage.height == rightImage.height else {
        throw SpatialPhotoError.leftAndRightImageSizesDoNotMatch
    }

    let baselineMeters = baselineMM / 1000.0
    let leftPosition: [Double] = [0, 0, 0]
    let rightPosition: [Double] = [baselineMeters, 0, 0]
    let intrinsics = leftImage.intrinsics(horizontalFOV: horizontalFOV)
    let encodedDisparityAdjustment = Int(disparityAdjustment * 1e4)

    let leftProperties = spatialPropertiesDictionary(
        isLeft: true,
        encodedDisparityAdjustment: encodedDisparityAdjustment,
        position: leftPosition,
        intrinsics: intrinsics
    )
    let rightProperties = spatialPropertiesDictionary(
        isLeft: false,
        encodedDisparityAdjustment: encodedDisparityAdjustment,
        position: rightPosition,
        intrinsics: intrinsics
    )

    if FileManager.default.fileExists(atPath: outputURL.path) {
        try FileManager.default.removeItem(at: outputURL)
    }

    let destinationProperties: [CFString: Any] = [kCGImagePropertyPrimaryImage: 0]
    guard let destination = CGImageDestinationCreateWithURL(
        outputURL as CFURL,
        UTType.heic.identifier as CFString,
        2,
        destinationProperties as CFDictionary
    ) else {
        throw SpatialPhotoError.unableToCreateImageDestination
    }

    CGImageDestinationAddImageFromSource(
        destination,
        leftImage.source,
        leftImage.primaryImageIndex,
        leftProperties as CFDictionary
    )
    CGImageDestinationAddImageFromSource(
        destination,
        rightImage.source,
        rightImage.primaryImageIndex,
        rightProperties as CFDictionary
    )

    guard CGImageDestinationFinalize(destination) else {
        throw SpatialPhotoError.unableToFinalizeImageDestination
    }
}

nonisolated func performSpatialPhotoExport(
    leftURL: URL,
    rightURL: URL,
    baselineMM: Double,
    horizontalFOV: Double,
    disparityAdjustment: Double,
    outputFolder: URL?
) async throws -> [URL] {
    try await Task.detached {
        let folder = outputFolder ?? leftURL.deletingLastPathComponent()
        let baseName = leftURL.deletingPathExtension().lastPathComponent
        let outputURL = folder
            .appendingPathComponent("\(baseName)_spatial")
            .appendingPathExtension("heic")

        try writeSpatialPhoto(
            leftURL: leftURL,
            rightURL: rightURL,
            outputURL: outputURL,
            baselineMM: baselineMM,
            horizontalFOV: horizontalFOV,
            disparityAdjustment: disparityAdjustment
        )
        return [outputURL]
    }.value
}

/// Estimate horizontal field of view (in degrees) from a 35-mm equivalent focal length.
/// Uses the 36 mm full-frame sensor width as reference.
nonisolated func inferHorizontalFOV(from url: URL) -> Double? {
    guard let focalMM35 = EXIF.focalLengthMM35(for: url) else { return nil }
    let f = Double(focalMM35)
    guard f > 0 else { return nil }
    let fovRad = 2.0 * atan(36.0 / (2.0 * f))
    return fovRad * 180.0 / .pi
}
