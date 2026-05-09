import SwiftUI
import UniformTypeIdentifiers

struct ImagesToVideoView: View {
    @State private var images: [ImageItem] = []
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    @State private var showImagePicker = false
    @State private var showFolderPicker = false
    @State private var pendingProcessAfterFolder = false
    @State private var videoStatus: ProcessingStatus = .pending

    var body: some View {
        VStack {
            if images.isEmpty {
                emptyState
            } else {
                imageList
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for provider in providers {
                Task {
                    guard let url = try? await provider.loadFileURL() else { return }
                    images.append(ImageItem(url: url))
                    videoStatus = .pending
                }
            }
            return true
        }
        .fileImporter(
            isPresented: $showImagePicker,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                images.append(contentsOf: urls.map { ImageItem(url: $0) })
                videoStatus = .pending
            }
        }
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

    var emptyState: some View {
        VStack(spacing: 15) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            Text("Drop images here or click to add")
                .foregroundColor(.secondary)
            Button("Add Images") { showImagePicker = true }
                .buttonStyle(.borderedProminent)
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
                            videoStatus = .pending
                        }
                    }
                }
                .padding()
            }

            VideoStatusView(status: videoStatus)
                .padding(.horizontal)

            HStack(spacing: 15) {
                Button("Add More") { showImagePicker = true }.disabled(isProcessing)
                Button("Clear All") { images.removeAll(); videoStatus = .pending }.disabled(isProcessing)
                Spacer()
                if let folder = outputFolder {
                    Text("Output: \(folder.lastPathComponent)")
                        .font(.caption).foregroundColor(.secondary)
                }
                Button(isProcessing ? "Processing..." : "Create Video") { processAll() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || images.isEmpty)
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
        videoStatus = .processing
        for i in images.indices { images[i].status = .processing }
        Task {
            do {
                let outputURL = try await createVideoFromImages(imageURLs: images.map { $0.url }, outputFolder: outputFolder)
                for i in images.indices { images[i].status = .success([outputURL]) }
                videoStatus = .success([outputURL])
                isProcessing = false
            } catch {
                for i in images.indices { images[i].status = .failed(error.localizedDescription) }
                videoStatus = .failed(error.localizedDescription)
                isProcessing = false
            }
        }
    }
}

private struct VideoStatusView: View {
    let status: ProcessingStatus

    var body: some View {
        HStack {
            switch status {
            case .pending:
                Label("Ready to create video", systemImage: "film")
                    .foregroundColor(.secondary)
            case .processing:
                ProgressView("Creating video...")
            case .success(let urls):
                Label("Saved to \(urls.first?.lastPathComponent ?? "output.mov")", systemImage: "checkmark.circle.fill")
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
