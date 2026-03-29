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
    case hdrBoost = "HDR Boost"
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
    @State private var hdrInputURL: URL?
    @State private var hdrMaskURL: URL?
    @State private var hdrBoost: Double = 2.0
    @State private var hdrStatus: ProcessingStatus = .pending
    @State private var hdrPreview: CGImage?
    @State private var hdrPreviewError: String?
    @State private var isRenderingHDRPreview = false
    @State private var selectedOperation: ImageOperation = .heif
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    @State private var activeImporter: ActiveImporter?
    @State private var pendingOutputAction: PendingOutputAction?
    
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
            
            if selectedOperation == .hdrBoost {
                hdrBoostView
            } else if images.isEmpty {
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
            if selectedOperation == .hdrBoost {
                handleHDRDrop(providers: providers)
            } else {
                handleDrop(providers: providers)
            }
            return true
        }
        .fileImporter(
            isPresented: Binding(
                get: { activeImporter == .images },
                set: { if !$0, activeImporter == .images { activeImporter = nil } }
            ),
            allowedContentTypes: [.image],
            allowsMultipleSelection: true,
            onCompletion: handleImageImport
        )
        .fileImporter(
            isPresented: Binding(
                get: { activeImporter == .hdrInput },
                set: { if !$0, activeImporter == .hdrInput { activeImporter = nil } }
            ),
            allowedContentTypes: [.image],
            onCompletion: { handleHDRImport($0, forMask: false) }
        )
        .fileImporter(
            isPresented: Binding(
                get: { activeImporter == .hdrMask },
                set: { if !$0, activeImporter == .hdrMask { activeImporter = nil } }
            ),
            allowedContentTypes: [.image],
            onCompletion: { handleHDRImport($0, forMask: true) }
        )
        .fileImporter(
            isPresented: Binding(
                get: { activeImporter == .outputFolder },
                set: { if !$0, activeImporter == .outputFolder { activeImporter = nil } }
            ),
            allowedContentTypes: [.folder],
            onCompletion: handleOutputFolderImport
        )
    }

    var hdrBoostView: some View {
        VStack(spacing: 16) {
            HStack(spacing: 12) {
                HDRSelectionCard(
                    title: "Input SDR Image",
                    url: hdrInputURL,
                    placeholder: "Choose the base SDR image",
                    isProcessing: isProcessing,
                    onPick: { showHDRPicker(forMask: false) },
                    onClear: {
                        hdrInputURL = nil
                        hdrStatus = .pending
                    }
                )

                HDRSelectionCard(
                    title: "Mask with Alpha",
                    url: hdrMaskURL,
                    placeholder: "Choose the mask image",
                    isProcessing: isProcessing,
                    onPick: { showHDRPicker(forMask: true) },
                    onClear: {
                        hdrMaskURL = nil
                        hdrStatus = .pending
                    }
                )
            }

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Boost")
                            .font(.headline)
                        Text("Increase exposure before applying the alpha mask.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Text(hdrBoost, format: .number.precision(.fractionLength(1)))
                        .font(.system(.title3, design: .rounded, weight: .semibold))
                        .monospacedDigit()
                }

                Slider(value: $hdrBoost, in: 0...8, step: 0.1)
                    .disabled(isProcessing)
                    .onChange(of: hdrBoost) {
                        hdrStatus = .pending
                    }
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)

            hdrPreviewView

            HDRStatusView(status: hdrStatus)

            HStack(spacing: 15) {
                if let folder = outputFolder {
                    Text("Output: \(folder.lastPathComponent)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Button(isProcessing ? "Processing..." : "Create HDR HEIF") {
                    processHDRBoost()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || hdrInputURL == nil || hdrMaskURL == nil)
            }
        }
        .padding()
        .task(id: HDRPreviewRequest(inputURL: hdrInputURL, maskURL: hdrMaskURL, boost: hdrBoost)) {
            await loadHDRPreview()
        }
    }

    @ViewBuilder
    var hdrPreviewView: some View {
        if hdrInputURL == nil || hdrMaskURL == nil {
            EmptyView()
        } else if isRenderingHDRPreview {
            HStack {
                ProgressView()
                Text("Rendering preview...")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        } else if let hdrPreview {
            VStack(alignment: .leading, spacing: 10) {
                Text("Preview")
                    .font(.headline)

                Image(decorative: hdrPreview, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: 260)
                    .background(Color.black.opacity(0.08))
                    .cornerRadius(10)
                    .allowedDynamicRange(.high)
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        } else if let hdrPreviewError {
            HStack {
                Label(hdrPreviewError, systemImage: "exclamationmark.triangle.fill")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        }
    }
    
    func showFilePicker() {
        activeImporter = .images
    }

    func showHDRPicker(forMask: Bool) {
        activeImporter = forMask ? .hdrMask : .hdrInput
    }
    
    func handleDrop(providers: [NSItemProvider]) {
        for provider in providers {
            Task {
                guard let url = try? await provider.loadFileURL() else { return }
                await MainActor.run {
                    images.append(ImageItem(url: url))
                }
            }
        }
    }

    func handleHDRDrop(providers: [NSItemProvider]) {
        for provider in providers {
            Task {
                guard let url = try? await provider.loadFileURL() else { return }
                await MainActor.run {
                    if hdrInputURL == nil {
                        hdrInputURL = url
                    } else {
                        hdrMaskURL = url
                    }
                    hdrStatus = .pending
                }
            }
        }
    }

    func loadHDRPreview() async {
        guard let inputURL = hdrInputURL, let maskURL = hdrMaskURL else {
            hdrPreview = nil
            hdrPreviewError = nil
            isRenderingHDRPreview = false
            return
        }

        await MainActor.run {
            isRenderingHDRPreview = true
            hdrPreview = nil
            hdrPreviewError = nil
        }

        do {
            let image = try await makeHDRPreview(inputURL: inputURL, maskURL: maskURL, boost: hdrBoost)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                hdrPreview = image
                isRenderingHDRPreview = false
            }
        } catch {
            guard !Task.isCancelled else { return }
            await MainActor.run {
                hdrPreviewError = "Preview unavailable"
                isRenderingHDRPreview = false
            }
        }
    }

    func processAll() {
        if outputFolder == nil {
            pendingOutputAction = .processAll
            activeImporter = .outputFolder
        } else {
            startProcessing()
        }
    }

    func processHDRBoost() {
        guard let inputURL = hdrInputURL, let maskURL = hdrMaskURL else { return }

        if outputFolder == nil {
            pendingOutputAction = .hdrBoost(inputURL: inputURL, maskURL: maskURL)
            activeImporter = .outputFolder
        } else {
            startHDRBoost(inputURL: inputURL, maskURL: maskURL)
        }
    }

    func startHDRBoost(inputURL: URL, maskURL: URL) {
        isProcessing = true
        hdrStatus = .processing

        Task {
            do {
                let outputURLs = try await performHDRBoost(inputURL: inputURL, maskURL: maskURL, boost: hdrBoost)
                await MainActor.run {
                    hdrStatus = .success(outputURLs)
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    hdrStatus = .failed(error.localizedDescription)
                    isProcessing = false
                }
            }
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

            case .hdrBoost:
                throw ImageToolsError.processingFailed
                
            case .slicer:
                return try await sliceImage(ciImage: ciImage, sourceURL: url, outputFolder: folder, context: context)
                
            case .video:
                // Video is handled separately
                return []
            }
        }.value
    }

    func performHDRBoost(inputURL: URL, maskURL: URL, boost: Double) async throws -> [URL] {
        try await Task.detached { [outputFolder] in
            let context = CIContext()
            let folder = outputFolder ?? inputURL.deletingLastPathComponent()
            let blended = try makeHDRBoostImage(inputURL: inputURL, maskURL: maskURL, boost: boost)

            let outputURL = folder
                .appendingPathComponent("\(inputURL.deletingPathExtension().lastPathComponent)_hdrBoost")
                .appendingPathExtension("heif")
            let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!

            try context.writeHEIF10Representation(
                of: blended,
                to: outputURL,
                colorSpace: colorSpace,
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
            )
            return [outputURL]
        }.value
    }

    func makeHDRPreview(inputURL: URL, maskURL: URL, boost: Double) async throws -> CGImage {
        try await Task.detached {
            let colorSpace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)!
            let context = CIContext(options: [
                .workingColorSpace: colorSpace,
                .outputColorSpace: colorSpace
            ])
            let outputImage = try makeHDRBoostImage(inputURL: inputURL, maskURL: maskURL, boost: boost)
            let fitted = fittedPreviewImage(outputImage, maxDimension: 900)

            guard let cgImage = context.createCGImage(
                fitted,
                from: fitted.extent,
                format: .RGBAh,
                colorSpace: colorSpace
            ) else {
                throw ImageToolsError.previewFailed
            }

            return cgImage
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
    
    func handleImageImport(_ result: Result<[URL], Error>) {
        activeImporter = nil

        guard case .success(let urls) = result else { return }
        images.append(contentsOf: urls.map { ImageItem(url: $0) })
    }

    func handleHDRImport(_ result: Result<URL, Error>, forMask: Bool) {
        activeImporter = nil

        guard case .success(let url) = result else { return }

        if forMask {
            hdrMaskURL = url
        } else {
            hdrInputURL = url
        }
        hdrStatus = .pending
    }

    func handleOutputFolderImport(_ result: Result<URL, Error>) {
        activeImporter = nil

        guard case .success(let url) = result else {
            pendingOutputAction = nil
            return
        }

        outputFolder = url

        switch pendingOutputAction {
        case .processAll:
            startProcessing()
        case .hdrBoost(let inputURL, let maskURL):
            startHDRBoost(inputURL: inputURL, maskURL: maskURL)
        case nil:
            break
        }

        pendingOutputAction = nil
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

func fit(maskImage: CIImage, to targetExtent: CGRect) -> CIImage {
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

func makeHDRBoostImage(inputURL: URL, maskURL: URL, boost: Double) throws -> CIImage {
    guard let inputImage = CIImage(contentsOf: inputURL, options: [.applyOrientationProperty: true]) else {
        throw ImageToolsError.invalidImage
    }
    guard let rawMaskImage = CIImage(contentsOf: maskURL, options: [.applyOrientationProperty: true]) else {
        throw ImageToolsError.invalidMaskImage
    }

    let maskImage = fit(maskImage: rawMaskImage, to: inputImage.extent)
    let boosted = inputImage.applyingFilter(
        "CIExposureAdjust",
        parameters: [kCIInputEVKey: boost]
    )

    return boosted.applyingFilter(
        "CIBlendWithAlphaMask",
        parameters: [
            kCIInputBackgroundImageKey: inputImage,
            kCIInputMaskImageKey: maskImage
        ]
    )
}

func fittedPreviewImage(_ image: CIImage, maxDimension: CGFloat) -> CIImage {
    let extent = image.extent.integral
    let largestSide = max(extent.width, extent.height)
    guard largestSide > maxDimension else { return image }

    let scale = maxDimension / largestSide
    return image
        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        .cropped(to: CGRect(origin: .zero, size: CGSize(width: extent.width * scale, height: extent.height * scale)))
}

enum ActiveImporter: Hashable {
    case images
    case hdrInput
    case hdrMask
    case outputFolder
}

enum PendingOutputAction {
    case processAll
    case hdrBoost(inputURL: URL, maskURL: URL)
}

extension NSItemProvider {
    func loadFileURL() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let data = item as? Data,
                          let url = URL(dataRepresentation: data, relativeTo: nil) {
                    continuation.resume(returning: url)
                } else {
                    continuation.resume(throwing: ImageToolsError.processingFailed)
                }
            }
        }
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

struct HDRSelectionCard: View {
    let title: String
    let url: URL?
    let placeholder: String
    let isProcessing: Bool
    let onPick: () -> Void
    let onClear: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Button(url == nil ? "Choose" : "Replace", action: onPick)
                    .disabled(isProcessing)
            }

            Text(url?.lastPathComponent ?? placeholder)
                .foregroundColor(url == nil ? .secondary : .primary)
                .lineLimit(2)

            if url != nil {
                Button("Clear", role: .destructive, action: onClear)
                    .disabled(isProcessing)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
}

struct HDRStatusView: View {
    let status: ProcessingStatus

    var body: some View {
        HStack {
            switch status {
            case .pending:
                Label("Ready to create an HDR HEIF image", systemImage: "sparkles")
                    .foregroundColor(.secondary)
            case .processing:
                ProgressView("Creating HDR HEIF...")
            case .success(let urls):
                Label("Saved to \(urls.first?.lastPathComponent ?? "output.heif")", systemImage: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .failed(let error):
                Label(error, systemImage: "xmark.circle.fill")
                    .foregroundColor(.red)
            }

            Spacer()
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
}

struct HDRPreviewRequest: Equatable {
    let inputURL: URL?
    let maskURL: URL?
    let boost: Double
}

enum ImageToolsError: LocalizedError {
    case invalidImage
    case invalidMaskImage
    case unsupportedFormat
    case slicingFailed
    case processingFailed
    case noImages
    case videoCreationFailed
    case previewFailed
    
    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Unable to load image"
        case .invalidMaskImage:
            return "Unable to load mask image"
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
        case .previewFailed:
            return "Failed to render preview"
        }
    }
}
