import SwiftUI
import CoreImage

struct ConvertImagesView: View {
    @State private var outputFormat: ImageExportFormat = .heifHLG

    var body: some View {
        let format = outputFormat
        VStack(spacing: 0) {
            HStack {
                Picker("Format", selection: $outputFormat) {
                    ForEach(ImageExportFormat.allCases, id: \.self) { f in
                        Text(f.displayName).tag(f)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 260)
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 12)

            ImageBatchProcessView { url, outputFolder in
                try await convertImage(url: url, outputFolder: outputFolder, format: format)
            }
        }
    }
}

/// Load the source image with CoreImage and re-encode it via the shared
/// exporter. The pipeline is intentionally trivial: any per-format
/// behavior (color space tagging, codec, file extension) is decided by
/// `saveCIImage(...)` in [ImageExport.swift](../Commons/ImageExport.swift).
nonisolated func convertImage(url: URL,
                              outputFolder: URL?,
                              format: ImageExportFormat) async throws -> [URL] {
    try await Task.detached {
        guard let ciImage = CIImage(contentsOf: url, options: [
            .applyOrientationProperty: true
        ]) else {
            throw ImageToolsError.invalidImage
        }
        let folder = outputFolder ?? url.deletingLastPathComponent()
        let outputURL = folder
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
            .appendingPathExtension(format.fileExtension)

        try saveCIImage(ciImage, to: outputURL, format: format,
                        ciContext: defaultImageExportCIContext())
        return [outputURL]
    }.value
}
