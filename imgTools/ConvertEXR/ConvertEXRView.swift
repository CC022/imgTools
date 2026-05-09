import SwiftUI
import CoreImage

struct ConvertEXRView: View {
    var body: some View {
        ImageBatchProcessView { url, outputFolder in
            try await Task.detached {
                guard let ciImage = CIImage(contentsOf: url) else {
                    throw ImageToolsError.invalidImage
                }
                let context = CIContext()
                let folder = outputFolder ?? url.deletingLastPathComponent()
                let outputURL = folder
                    .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("exr")
                try context.writeOpenEXRRepresentation(of: ciImage, to: outputURL)
                return [outputURL]
            }.value
        }
    }
}
