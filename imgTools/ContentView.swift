import SwiftUI
import CoreImage
import UniformTypeIdentifiers

enum ConversionType: String, CaseIterable {
    case toEXR = "Convert to EXR"
    case toHEIF = "Convert to HEIF"
}

struct ConversionResult {
    let successCount: Int
    let failCount: Int
}

struct ContentView: View {
    @State private var selectedImages: [URL] = []
    @State private var conversionType: ConversionType = .toEXR
    @State private var isProcessing = false
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var showFilePicker = false
    @State private var progress: Double = 0.0
    @State private var currentProcessingFile = ""
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Image Format Converter")
                .font(.title)
                .padding(.top)
            
            // Conversion type picker
            Picker("Conversion Type", selection: $conversionType) {
                ForEach(ConversionType.allCases, id: \.self) { type in
                    Text(type.rawValue).tag(type)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .disabled(isProcessing)
            
            // Progress indicator
            if isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: progress, total: Double(selectedImages.count))
                        .padding(.horizontal)
                    Text("Processing: \(currentProcessingFile)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Image list
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if selectedImages.isEmpty {
                        Text("No images selected")
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity)
                            .padding()
                    } else {
                        ForEach(selectedImages, id: \.self) { url in
                            HStack {
                                Image(systemName: "photo")
                                Text(url.lastPathComponent)
                                    .lineLimit(1)
                                Spacer()
                                Button(action: {
                                    removeImage(url)
                                }) {
                                    Image(systemName: "xmark.circle.fill")
                                        .foregroundColor(.red)
                                }
                                .buttonStyle(.plain)
                                .disabled(isProcessing)
                            }
                            .padding(.horizontal)
                        }
                    }
                }
            }
            .frame(minHeight: 200)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            .padding(.horizontal)
            
            // Buttons
            HStack(spacing: 20) {
                Button(action: { showFilePicker = true }) {
                    Label("Add Images", systemImage: "plus.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing)
                
                Button(action: { Task { await convertImages() } }) {
                    Label(isProcessing ? "Converting..." : "Convert", systemImage: "arrow.triangle.2.circlepath")
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedImages.isEmpty || isProcessing)
            }
            .padding(.bottom)
        }
        .frame(minWidth: 500, minHeight: 400)
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                selectedImages.append(contentsOf: urls)
            case .failure(let error):
                alertMessage = "Failed to select files: \(error.localizedDescription)"
                showAlert = true
            }
        }
        .alert("Conversion Result", isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }
    
    func removeImage(_ url: URL) {
        selectedImages.removeAll { $0 == url }
    }
    
    @MainActor
    func convertImages() async {
        isProcessing = true
        progress = 0.0
        
        let result = await convertAllImages()
        
        isProcessing = false
        progress = 0.0
        currentProcessingFile = ""
        alertMessage = "Conversion complete!\nSuccess: \(result.successCount)\nFailed: \(result.failCount)"
        showAlert = true
    }
    
    private func convertAllImages() async -> ConversionResult {
        var successCount = 0
        var failCount = 0
        
        for (index, imageURL) in selectedImages.enumerated() {
            await MainActor.run {
                currentProcessingFile = imageURL.lastPathComponent
            }
            
            let success = await convertImage(url: imageURL)
            
            if success {
                successCount += 1
            } else {
                failCount += 1
            }
            
            await MainActor.run {
                progress = Double(index + 1)
            }
        }
        
        return ConversionResult(successCount: successCount, failCount: failCount)
    }
    
    private func convertImage(url: URL) async -> Bool {
        return await Task.detached(priority: .userInitiated) {
            guard url.startAccessingSecurityScopedResource() else {
                return false
            }
            defer { url.stopAccessingSecurityScopedResource() }
            
            guard let ciImage = CIImage(contentsOf: url) else {
                return false
            }
            
            let context = CIContext()
            let outputURL = getOutputURL(for: url)
            
            let success: Bool
            switch await MainActor.run(body: { conversionType }) {
            case .toEXR:
                success = convertToEXR(ciImage: ciImage, context: context, outputURL: outputURL)
            case .toHEIF:
                success = convertToHEIF(ciImage: ciImage, context: context, outputURL: outputURL)
            }
            
            return success
        }.value
    }
    
    private func convertToEXR(ciImage: CIImage, context: CIContext, outputURL: URL) -> Bool {
        do {
            let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) ?? CGColorSpaceCreateDeviceRGB()
            try context.writeOpenEXRRepresentation(
                of: ciImage,
                to: outputURL,
                options: [:]
            )
            return true
        } catch {
            print("EXR conversion failed: \(error)")
            return false
        }
    }
    
    private func convertToHEIF(ciImage: CIImage, context: CIContext, outputURL: URL) -> Bool {
        do {
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            try context.writeHEIFRepresentation(
                of: ciImage,
                to: outputURL,
                format: .RGBA8,
                colorSpace: colorSpace,
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9]
            )
            return true
        } catch {
            print("HEIF conversion failed: \(error)")
            return false
        }
    }
    
    private func getOutputURL(for inputURL: URL) -> URL {
        let directory = inputURL.deletingLastPathComponent()
        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let ext = conversionType == .toEXR ? "exr" : "heif"
        let outputName = "\(baseName)_converted.\(ext)"
        return directory.appendingPathComponent(outputName)
    }
    
}

#Preview {
    ContentView()
}
