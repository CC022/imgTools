import SwiftUI
import UniformTypeIdentifiers

struct ImageBatchProcessView: View {
    let processImage: @Sendable (_ url: URL, _ outputFolder: URL?) async throws -> [URL]

    @State private var images: [ImageItem] = []
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    @State private var showImagePicker = false
    @State private var showFolderPicker = false
    @State private var pendingProcessAfterFolder = false

    var body: some View {
        Group {
            if images.isEmpty {
                emptyState
            } else {
                imageList
            }
        }
        .dropDestination(for: URL.self) { items, _ in
            for url in items {
                images.append(ImageItem(url: url))
            }
            return true
        }
    }

    var emptyState: some View {
        VStack(spacing: 15) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            Text("Drop images here or click to add")
                .foregroundColor(.secondary)
            Button("Add Images") { showImagePicker = true }
                .buttonStyle(.borderedProminent)
                .fileImporter(
                    isPresented: $showImagePicker,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true
                ) { result in
                    if case .success(let urls) = result {
                        images.append(contentsOf: urls.map { ImageItem(url: $0) })
                    }
                }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
        .padding()
    }

    var imageList: some View {
        VStack {
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(images) { item in
                        ImageRow(item: item) {
                            images.removeAll { $0.id == item.id }
                        }
                    }
                }
                .padding()
            }

            HStack(spacing: 15) {
                Button("Add More") { showImagePicker = true }.disabled(isProcessing)
                    .fileImporter(
                        isPresented: $showImagePicker,
                        allowedContentTypes: [.image],
                        allowsMultipleSelection: true
                    ) { result in
                        if case .success(let urls) = result {
                            images.append(contentsOf: urls.map { ImageItem(url: $0) })
                        }
                    }
                Button("Clear All") { images.removeAll() }.disabled(isProcessing)
                Spacer()
                if let folder = outputFolder {
                    Text("Output: \(folder.lastPathComponent)")
                        .font(.caption).foregroundColor(.secondary)
                }
                Button(isProcessing ? "Processing..." : "Process All") { processAll() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing)
                    .fileImporter(
                        isPresented: $showFolderPicker,
                        allowedContentTypes: [.folder]
                    ) { result in
                        if case .success(let url) = result {
                            outputFolder = url
                            if pendingProcessAfterFolder {
                                pendingProcessAfterFolder = false
                                startProcessing()
                            }
                        } else {
                            pendingProcessAfterFolder = false
                        }
                    }
            }
            .padding()
        }
    }

    func processAll() {
        if outputFolder == nil {
            pendingProcessAfterFolder = true
            showFolderPicker = true
        } else {
            startProcessing()
        }
    }

    func startProcessing() {
        isProcessing = true
        Task {
            for i in images.indices {
                await processItem(at: i)
            }
            isProcessing = false
        }
    }

    func processItem(at index: Int) async {
        images[index].status = .processing
        let url = images[index].url
        do {
            let outputURLs = try await processImage(url, outputFolder)
            images[index].status = .success(outputURLs)
        } catch {
            images[index].status = .failed(error.localizedDescription)
        }
    }
}
