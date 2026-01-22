import SwiftUI
import CoreImage
import UniformTypeIdentifiers
import ImageIO

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
    case slicer = "Slice into 3 Parts"
    case fitCenter = "Fit & Center"
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
        VStack(spacing: 20) {
            Text("Image Tools")
                .font(.title)
                .padding(.top)
            
            Picker("Operation:", selection: $selectedOperation) {
                ForEach(ImageOperation.allCases, id: \.self) { operation in
                    Text(operation.rawValue).tag(operation)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
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
            for i in images.indices {
                await processImage(at: i)
            }
            
            await MainActor.run {
                isProcessing = false
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
                
            case .fitCenter:
                return try await fitAndCenterImage(ciImage: ciImage, sourceURL: url, outputFolder: folder, context: context)
            }
        }.value
    }
    
    func sliceImage(ciImage: CIImage, sourceURL: URL, outputFolder: URL, context: CIContext) throws -> [URL] {
        let width = Int(ciImage.extent.width)
        let height = Int(ciImage.extent.height)
        let sliceWidth = width / 3
        
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let ext = sourceURL.pathExtension
        let baseName = sourceURL.deletingPathExtension().lastPathComponent
        
        guard let imageType = UTType(filenameExtension: ext) else {
            throw ImageToolsError.unsupportedFormat
        }
        
        var outputURLs: [URL] = []
        
        for i in 0..<3 {
            let x = i * sliceWidth
            let w = (i == 2) ? (width - x) : sliceWidth
            
            let cropped = ciImage.cropped(
                to: CGRect(x: x, y: 0, width: w, height: height)
            )
            
            guard let outCG = context.createCGImage(
                cropped,
                from: cropped.extent,
                format: .RGBA8,
                colorSpace: colorSpace
            ) else { continue }
            
            let outURL = outputFolder.appendingPathComponent("\(baseName)_slice\(i+1).\(ext)")
            guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, imageType.identifier as CFString, 1, nil) else { continue }
            
            CGImageDestinationAddImage(dest, outCG, nil)
            CGImageDestinationFinalize(dest)
            
            outputURLs.append(outURL)
        }
        
        if outputURLs.isEmpty {
            throw ImageToolsError.slicingFailed
        }
        
        return outputURLs
    }
    
    func fitAndCenterImage(ciImage: CIImage, sourceURL: URL, outputFolder: URL, context: CIContext) throws -> [URL] {
        let width = Int(ciImage.extent.width)
        let height = Int(ciImage.extent.height)
        let sliceWidth = width / 3
        
        let targetSize = CGSize(width: sliceWidth, height: height)
        
        let scale = min(
            targetSize.width / CGFloat(width),
            targetSize.height / CGFloat(height)
        )
        
        let scaledSize = CGSize(
            width: CGFloat(width) * scale,
            height: CGFloat(height) * scale
        )
        
        let xOffset = (targetSize.width - scaledSize.width) / 2
        let yOffset = (targetSize.height - scaledSize.height) / 2
        
        let scaledImage = ciImage
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: xOffset, y: yOffset))
        
        let whiteBG = CIImage(color: .white)
            .cropped(to: CGRect(origin: .zero, size: targetSize))
        
        let composed = scaledImage.composited(over: whiteBG)
        
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let ext = sourceURL.pathExtension
        let baseName = sourceURL.deletingPathExtension().lastPathComponent
        
        guard let imageType = UTType(filenameExtension: ext) else {
            throw ImageToolsError.unsupportedFormat
        }
        
        guard let outCG = context.createCGImage(
            composed,
            from: CGRect(origin: .zero, size: targetSize),
            format: .RGBA8,
            colorSpace: colorSpace
        ) else {
            throw ImageToolsError.processingFailed
        }
        
        let outURL = outputFolder.appendingPathComponent("\(baseName)_centered.\(ext)")
        guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, imageType.identifier as CFString, 1, nil) else {
            throw ImageToolsError.processingFailed
        }
        
        CGImageDestinationAddImage(dest, outCG, nil)
        CGImageDestinationFinalize(dest)
        
        return [outURL]
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
        }
    }
}
