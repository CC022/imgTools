import SwiftUI
import CoreImage

struct SlicerView: View {
    var body: some View {
        ImageBatchProcessView { url, outputFolder in
            try await Task.detached {
                guard let ciImage = CIImage(contentsOf: url) else {
                    throw ImageToolsError.invalidImage
                }
                let context = CIContext()
                let folder = outputFolder ?? url.deletingLastPathComponent()
                return try sliceImage(ciImage: ciImage, sourceURL: url, outputFolder: folder, context: context)
            }.value
        }
    }
}
