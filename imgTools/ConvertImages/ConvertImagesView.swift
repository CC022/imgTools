import SwiftUI
import CoreImage

enum ImageOutputFormat: String, CaseIterable, Sendable {
    case heif = "HEIF"
    case exr = "EXR"

    var fileExtension: String {
        switch self {
        case .heif: return "heif"
        case .exr:  return "exr"
        }
    }
}

struct ConvertImagesView: View {
    @State private var outputFormat: ImageOutputFormat = .heif

    var body: some View {
        let format = outputFormat
        VStack(spacing: 0) {
            HStack {
                Picker("Format", selection: $outputFormat) {
                    ForEach(ImageOutputFormat.allCases, id: \.self) { format in
                        Text(format.rawValue).tag(format)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 160)
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

func convertImage(url: URL, outputFolder: URL?, format: ImageOutputFormat) async throws -> [URL] {
    let fileExt = format.fileExtension
    return try await Task.detached {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw ImageToolsError.invalidImage
        }
        let context = CIContext()
        let folder = outputFolder ?? url.deletingLastPathComponent()
        let outputURL = folder
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
            .appendingPathExtension(fileExt)

        switch fileExt {
        case "exr":
            try context.writeOpenEXRRepresentation(of: ciImage, to: outputURL)
        case "heif":
            let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_HLG)!
            try context.writeHEIF10Representation(
                of: ciImage,
                to: outputURL,
                colorSpace: colorSpace,
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
            )
        default:
            throw ImageToolsError.unsupportedFormat
        }
        return [outputURL]
    }.value
}
