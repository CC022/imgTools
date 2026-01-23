import SwiftUI
import CoreImage
import UniformTypeIdentifiers
import ImageIO
import AVFoundation

@main
struct ImageToolsApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

enum ImageOperation: String, CaseIterable {
    case exr = "Convert to EXR"
    case heif = "Convert to HEIF"
    case slicer = "Slicer"
    case video = "Images to Video"
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

struct ContentView: View {
    @State private var images: [ImageItem] = []
    @State private var selectedOperation: ImageOperation = .heif
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    
    var body: some View {
        VStack() {
            Picker("", selection: $selectedOperation) {
                ForEach(ImageOperation.allCases, id: \.self) { operation in
                    Text(operation.rawValue).tag(operation)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            .disabled(isProcessing)
            
            if images.isEmpty {
                VStack(spacing: 15) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    
                    Text("Drop images here or click to add")
                        .foregroundColor(.secondary)
                    
                    Button("Add Images") {
                        showFilePicker()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(12)
                .padding()
            } else {
                ScrollView {
                    VStack(spacing: 10) {
                        ForEach(images) { item in
                            ImageRow(item: item) {
                                if let index = images.firstIndex(where: { $0.id == item.id }) {
                                    images.remove(at: index)
                                }
                            }
                        }
                    }
                    .padding()
                }
                
                HStack(spacing: 15) {
                    Button("Add More") {
                        showFilePicker()
                    }
                    .disabled(isProcessing)
                    
                    Button("Clear All") {
                        images.removeAll()
                    }
                    .disabled(isProcessing)
                    
                    Spacer()
                    
                    if let folder = outputFolder {
                        Text("Output: \(folder.lastPathComponent)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Button(isProcessing ? "Processing..." : "Process All") {
                        processAll()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing)
                }
                .padding()
            }
        }
        .frame(minWidth: 600, minHeight: 400)
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleDrop(providers: providers)
            return true
        }
    }
    
    func showFilePicker() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image]
        
        panel.begin { response in
            if response == .OK {
                let newImages = panel.urls.map { ImageItem(url: $0) }
                images.append(contentsOf: newImages)
            }
        }
    }
    
    func handleDrop(providers: [NSItemProvider]) {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
                if let data = item as? Data,
                   let url = URL(dataRepresentation: data, relativeTo: nil) {
                    DispatchQueue.main.async {
                        images.append(ImageItem(url: url))
                    }
                }
            }
        }
    }
    
    func processAll() {
        // If no output folder is set, ask the user to choose one
        if outputFolder == nil {
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.canCreateDirectories = true
            panel.prompt = "Choose Output Folder"
            panel.message = "Select where to save processed images"
            
            panel.begin { response in
                if response == .OK, let url = panel.url {
                    outputFolder = url
                    startProcessing()
                }
            }
        } else {
            startProcessing()
        }
    }
    
    func startProcessing() {
        isProcessing = true
        
        Task {
            if selectedOperation == .video {
                // For video, process all images together
                await processImagesToVideo()
            } else {
                // For other operations, process individually
                for i in images.indices {
                    await processImage(at: i)
                }
            }
            
            await MainActor.run {
                isProcessing = false
            }
        }
    }
    
    func processImagesToVideo() async {
        await MainActor.run {
            for i in images.indices {
                images[i].status = .processing
            }
        }
        
        do {
            let outputURL = try await createVideoFromImages(
                imageURLs: images.map { $0.url }
            )
            
            await MainActor.run {
                for i in images.indices {
                    images[i].status = .success([outputURL])
                }
            }
        } catch {
            await MainActor.run {
                for i in images.indices {
                    images[i].status = .failed(error.localizedDescription)
                }
            }
        }
    }
    
    func processImage(at index: Int) async {
        await MainActor.run {
            images[index].status = .processing
        }
        
        let url = images[index].url
        
        do {
            let outputURLs = try await performOperation(url: url, operation: selectedOperation)
            await MainActor.run {
                images[index].status = .success(outputURLs)
            }
        } catch {
            await MainActor.run {
                images[index].status = .failed(error.localizedDescription)
            }
        }
    }
    
    func performOperation(url: URL, operation: ImageOperation) async throws -> [URL] {
        return try await Task.detached { [outputFolder] in
            guard let ciImage = CIImage(contentsOf: url) else {
                throw ImageToolsError.invalidImage
            }
            
            let context = CIContext()
            let folder = outputFolder ?? url.deletingLastPathComponent()
            
            switch operation {
            case .exr:
                let outputURL = folder.appendingPathComponent(url.deletingPathExtension().lastPathComponent).appendingPathExtension("exr")
                try context.writeOpenEXRRepresentation(of: ciImage, to: outputURL)
                return [outputURL]
                
            case .heif:
                let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!
                let outputURL = folder.appendingPathComponent(url.deletingPathExtension().lastPathComponent).appendingPathExtension("heif")
                try context.writeHEIF10Representation(of: ciImage, to: outputURL, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95])
                return [outputURL]
                
            case .slicer:
                return try await sliceImage(ciImage: ciImage, sourceURL: url, outputFolder: folder, context: context)
                
            case .video:
                // Video is handled separately
                return []
            }
        }.value
    }
    
    func createVideoFromImages(imageURLs: [URL]) async throws -> URL {
        return try await Task.detached { [outputFolder] in
            let FPS = 30
            let ciContext = CIContext()
            
            guard !imageURLs.isEmpty else {
                throw ImageToolsError.noImages
            }
            
            // Load first image to determine dimensions and HDR status
            guard let firstCIImage = CIImage(contentsOf: imageURLs[0]) else {
                throw ImageToolsError.invalidImage
            }
            
            let scale = CGFloat(0.5)
            let width = Int(firstCIImage.extent.width * scale)
            let height = Int(firstCIImage.extent.height * scale)
            
            let folder = outputFolder ?? imageURLs[0].deletingLastPathComponent()
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
                .replacingOccurrences(of: ":", with: "-")
            let outputURL = folder.appendingPathComponent("video_\(timestamp).mov")
            
            // Remove existing file if it exists
            try? FileManager.default.removeItem(at: outputURL)
            
            // Create asset writer
            let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
            
            // Configure video settings based on HDR status
            var videoSettings: [String: Any]
            
            videoSettings = [
                AVVideoCodecKey : AVVideoCodecType.hevc,
                AVVideoColorPropertiesKey : [AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                                           AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_2100_HLG,
                                                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020],
                AVVideoWidthKey : width,
                AVVideoHeightKey: height,
            ]
            
            let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: writerInput)
            writer.add(writerInput)
            guard writer.startWriting() else {
                throw ImageToolsError.videoCreationFailed
            }
            
            writer.startSession(atSourceTime: .zero)
            var frameIndex: Int64 = 0
            
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault,
                                width,
                                height,
                                kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                         kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary,
                                &pixelBuffer)
            
            // Process each image
            for imageURL in imageURLs {
                guard let ciImage = CIImage(contentsOf: imageURL,
                                            options: [:])?
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
            
            if writer.status == .completed {
                return outputURL
            } else {
                throw ImageToolsError.videoCreationFailed
            }
        }.value
    }
    
    func sliceImage(ciImage: CIImage, sourceURL: URL, outputFolder: URL, context: CIContext) async throws -> [URL] {
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
            try context.writeHEIF10Representation(of: cropped, to: outputURL, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95])
            outputURLs.append(outputURL)
        }
        
        let targetSize = CGSize(width: sliceWidth, height: height)
        let scale = min(targetSize.width / CGFloat(width),targetSize.height / CGFloat(height))
        let scaledSize = CGSize(width: CGFloat(width) * scale, height: CGFloat(height) * scale)
        
        let xOffset = (targetSize.width - scaledSize.width) / 2
        let yOffset = (targetSize.height - scaledSize.height) / 2
        
        let scaledImage = ciImage
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: xOffset, y: yOffset))
        
        let whiteBG = CIImage(color: .white)
            .cropped(to: CGRect(origin: .zero, size: targetSize))
        
        let composed = scaledImage.composited(over: whiteBG)
        let outURL = outputFolder.appendingPathComponent("\(baseName)_centered.heif")

        try context.writeHEIF10Representation(of: composed, to: outURL, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95])
        outputURLs.append(outURL)

        if outputURLs.isEmpty {
            throw ImageToolsError.slicingFailed
        }
        
        return outputURLs
    }
    
}

struct ImageRow: View {
    let item: ImageItem
    let onDelete: () -> Void
    
    var body: some View {
        HStack {
            Image(systemName: "photo")
                .foregroundColor(.blue)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(item.url.lastPathComponent)
                    .lineLimit(1)
                
                statusView
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            statusIcon
            
            Button(action: onDelete) {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .help("Remove image")
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
    }
    
    @ViewBuilder
    var statusView: some View {
        switch item.status {
        case .pending:
            Text("Ready")
        case .processing:
            Text("Processing...")
        case .success(let urls):
            if urls.count == 1 {
                Text("Saved to: \(urls[0].lastPathComponent)")
            } else {
                Text("Created \(urls.count) files: \(urls.map { $0.lastPathComponent }.joined(separator: ", "))")
            }
        case .failed(let error):
            Text("Error: \(error)")
                .foregroundColor(.red)
        }
    }
    
    @ViewBuilder
    var statusIcon: some View {
        switch item.status {
        case .pending:
            Image(systemName: "clock")
        case .processing:
            ProgressView()
                .scaleEffect(0.5)
        case .success:
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .foregroundColor(.red)
        }
    }
}

enum ImageToolsError: LocalizedError {
    case invalidImage
    case unsupportedFormat
    case slicingFailed
    case processingFailed
    case noImages
    case videoCreationFailed
    
    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Unable to load image"
        case .unsupportedFormat:
            return "Unsupported image format"
        case .slicingFailed:
            return "Failed to create sliced images"
        case .processingFailed:
            return "Failed to process image"
        case .noImages:
            return "No images to process"
        case .videoCreationFailed:
            return "Failed to create video"
        }
    }
}
