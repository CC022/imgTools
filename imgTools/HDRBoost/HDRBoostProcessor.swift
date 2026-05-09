import CoreImage
import Foundation

nonisolated func fit(maskImage: CIImage, to targetExtent: CGRect) -> CIImage {
    let translated = maskImage.transformed(
        by: CGAffineTransform(translationX: -maskImage.extent.origin.x, y: -maskImage.extent.origin.y)
    )
    guard maskImage.extent.size != targetExtent.size else {
        return translated.cropped(to: CGRect(origin: .zero, size: targetExtent.size))
    }
    let scaleX = targetExtent.width / maskImage.extent.width
    let scaleY = targetExtent.height / maskImage.extent.height
    return translated
        .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        .cropped(to: CGRect(origin: .zero, size: targetExtent.size))
}

nonisolated func makeHDRBoostImage(inputURL: URL, maskURL: URL, boost: Double) throws -> CIImage {
    guard let inputImage = CIImage(contentsOf: inputURL, options: [.applyOrientationProperty: true]) else {
        throw ImageToolsError.invalidImage
    }
    guard let rawMaskImage = CIImage(contentsOf: maskURL, options: [.applyOrientationProperty: true]) else {
        throw ImageToolsError.invalidMaskImage
    }
    let maskImage = fit(maskImage: rawMaskImage, to: inputImage.extent)
    let boosted = inputImage.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: boost])
    return boosted.applyingFilter("CIBlendWithAlphaMask", parameters: [
        kCIInputBackgroundImageKey: inputImage,
        kCIInputMaskImageKey: maskImage
    ])
}

nonisolated func fittedPreviewImage(_ image: CIImage, maxDimension: CGFloat) -> CIImage {
    let extent = image.extent.integral
    let largestSide = max(extent.width, extent.height)
    guard largestSide > maxDimension else { return image }
    let scale = maxDimension / largestSide
    return image
        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        .cropped(to: CGRect(origin: .zero, size: CGSize(width: extent.width * scale, height: extent.height * scale)))
}

nonisolated func performHDRBoost(inputURL: URL, maskURL: URL, boost: Double, outputFolder: URL?) async throws -> [URL] {
    try await Task.detached {
        let context = CIContext()
        let folder = outputFolder ?? inputURL.deletingLastPathComponent()
        let blended = try makeHDRBoostImage(inputURL: inputURL, maskURL: maskURL, boost: boost)
        let outputURL = folder
            .appendingPathComponent("\(inputURL.deletingPathExtension().lastPathComponent)_hdrBoost")
            .appendingPathExtension("heif")
        let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_HLG)!
        try context.writeHEIF10Representation(
            of: blended,
            to: outputURL,
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
        )
        return [outputURL]
    }.value
}

nonisolated func makeHDRPreview(inputURL: URL, maskURL: URL, boost: Double) async throws -> CGImage {
    try await Task.detached {
        let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_HLG)!
        let context = CIContext(options: [
            .workingColorSpace: colorSpace,
            .outputColorSpace: colorSpace
        ])
        let outputImage = try makeHDRBoostImage(inputURL: inputURL, maskURL: maskURL, boost: boost)
        let fitted = fittedPreviewImage(outputImage, maxDimension: 900)
        guard let cgImage = context.createCGImage(fitted, from: fitted.extent, format: .RGBAh, colorSpace: colorSpace) else {
            throw ImageToolsError.previewFailed
        }
        return cgImage
    }.value
}
