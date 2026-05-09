import AVFoundation
import CoreImage
import Foundation

nonisolated func createVideoFromImages(imageURLs: [URL], outputFolder: URL?) async throws -> URL {
    try await Task.detached {
        let FPS = 30
        let ciContext = CIContext()

        guard !imageURLs.isEmpty else { throw ImageToolsError.noImages }
        guard let firstCIImage = CIImage(contentsOf: imageURLs[0]) else { throw ImageToolsError.invalidImage }

        let scale = CGFloat(0.5)
        let width = Int(firstCIImage.extent.width * scale)
        let height = Int(firstCIImage.extent.height * scale)

        let folder = outputFolder ?? imageURLs[0].deletingLastPathComponent()
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
            .replacingOccurrences(of: ":", with: "-")
        let outputURL = folder.appendingPathComponent("video_\(timestamp).mov")
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_2100_HLG,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
            ],
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: writerInput)
        writer.add(writerInput)
        guard writer.startWriting() else { throw ImageToolsError.videoCreationFailed }
        writer.startSession(atSourceTime: .zero)

        var frameIndex: Int64 = 0
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA,
            [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary,
            &pixelBuffer
        )

        for imageURL in imageURLs {
            guard let ciImage = CIImage(contentsOf: imageURL, options: [:])?
                .oriented(.up)
                .transformed(by: CGAffineTransform(scaleX: scale, y: scale)) else { continue }
            ciContext.render(ciImage, to: pixelBuffer!)
            let frameTime = CMTime(value: frameIndex, timescale: Int32(FPS))
            if writerInput.isReadyForMoreMediaData {
                adaptor.append(pixelBuffer!, withPresentationTime: frameTime)
            }
            frameIndex += 1
        }

        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status == .completed { return outputURL }
        throw ImageToolsError.videoCreationFailed
    }.value
}
