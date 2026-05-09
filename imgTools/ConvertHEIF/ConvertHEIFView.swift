import SwiftUI
import CoreImage

struct ConvertHEIFView: View {
    var body: some View {
        ImageBatchProcessView { url, outputFolder in
            try await Task.detached {
                guard let ciImage = CIImage(contentsOf: url) else {
                    throw ImageToolsError.invalidImage
                }
                let context = CIContext()
                let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!
                let folder = outputFolder ?? url.deletingLastPathComponent()
                let outputURL = folder
                    .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("heif")
                try context.writeHEIF10Representation(
                    of: ciImage,
                    to: outputURL,
                    colorSpace: colorSpace,
                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
                )
                return [outputURL]
            }.value
        }
    }
}
