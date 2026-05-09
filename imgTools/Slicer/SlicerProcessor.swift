import CoreImage

nonisolated func sliceImage(ciImage: CIImage, sourceURL: URL, outputFolder: URL, context: CIContext) throws -> [URL] {
    let width = Int(ciImage.extent.width)
    let height = Int(ciImage.extent.height)
    let sliceWidth = width / 3
    let baseName = sourceURL.deletingPathExtension().lastPathComponent
    var outputURLs: [URL] = []
    let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!

    for i in 0..<3 {
        let x = i * sliceWidth
        let w = (i == 2) ? (width - x) : sliceWidth
        let cropped = ciImage.cropped(to: CGRect(x: x, y: 0, width: w, height: height))
        let outputURL = outputFolder.appendingPathComponent("\(baseName)_slice\(i+1).heif")
        try context.writeHEIF10Representation(
            of: cropped, to: outputURL, colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
        )
        outputURLs.append(outputURL)
    }

    let targetSize = CGSize(width: sliceWidth, height: height)
    let scale = min(targetSize.width / CGFloat(width), targetSize.height / CGFloat(height))
    let scaledSize = CGSize(width: CGFloat(width) * scale, height: CGFloat(height) * scale)
    let xOffset = (targetSize.width - scaledSize.width) / 2
    let yOffset = (targetSize.height - scaledSize.height) / 2

    let scaledImage = ciImage
        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        .transformed(by: CGAffineTransform(translationX: xOffset, y: yOffset))

    let whiteBG = CIImage(color: .white).cropped(to: CGRect(origin: .zero, size: targetSize))
    let composed = scaledImage.composited(over: whiteBG)
    let outURL = outputFolder.appendingPathComponent("\(baseName)_centered.heif")
    try context.writeHEIF10Representation(
        of: composed, to: outURL, colorSpace: colorSpace,
        options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
    )
    outputURLs.append(outURL)

    if outputURLs.isEmpty { throw ImageToolsError.slicingFailed }
    return outputURLs
}
