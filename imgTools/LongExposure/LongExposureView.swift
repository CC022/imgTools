import SwiftUI
import CoreImage
import UniformTypeIdentifiers

struct LongExposureView: View {
    @State private var images: [ImageItem] = []

    @State private var shouldAlign: Bool = false
    @State private var mergeMethod: MergeMethod = .max
    @State private var exportFormat: ImageExportFormat = .heifHLG

    @State private var cachedLoaded: [CIImage] = []
    @State private var cachedAligned: [CIImage]? = nil

    @State private var previewImage: CGImage?
    @State private var previewMessage: String?
    @State private var isLoading: Bool = false
    @State private var isAligning: Bool = false
    @State private var isMerging: Bool = false

    @State private var processingStatus: ProcessingStatus = .pending
    @State private var isProcessing: Bool = false
    @State private var processingProgress: String = ""
    @State private var outputFolder: URL?

    @State private var showFilePicker: Bool = false
    @State private var showFolderPicker: Bool = false
    @State private var pendingProcessAfterFolder: Bool = false

    var body: some View {
        Group {
            if images.isEmpty {
                emptyState
            } else {
                editor
            }
        }
        .padding()
        .dropDestination(for: URL.self) { items, _ in
            for url in items { images.append(ImageItem(url: url)) }
            return true
        }
        .task(id: ImageListKey(urls: images.map(\.url))) {
            await loadImagesDebounced()
        }
        .task(id: PreviewKey(count: cachedLoaded.count, align: shouldAlign, method: mergeMethod)) {
            await updatePreview()
        }
    }

    /// Single file-picker handler shared by both Add buttons.
    private func handleImagePick(_ result: Result<[URL], Error>) {
        if case .success(let urls) = result {
            images.append(contentsOf: urls.map { ImageItem(url: $0) })
        }
    }

    private func handleFolderPick(_ result: Result<URL, Error>) {
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

    // MARK: Empty state

    private var emptyState: some View {
        VStack(spacing: 15) {
            Image(systemName: "camera.aperture")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            Text("Drop images here or click to add")
                .foregroundColor(.secondary)
            Button("Add Images") { showFilePicker = true }
                .buttonStyle(.borderedProminent)
                .fileImporter(
                    isPresented: $showFilePicker,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true,
                    onCompletion: handleImagePick
                )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: Editor

    private var editor: some View {
        VStack(spacing: 14) {
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(images) { item in
                        ImageRow(item: item) {
                            images.removeAll { $0.id == item.id }
                        }
                    }
                }
            }
            .frame(maxHeight: 220)

            optionsSection
            previewSection
            statusBar
            actionRow
        }
    }

    // MARK: Options

    private var optionsSection: some View {
        HStack(spacing: 16) {
            Toggle("Align frames", isOn: $shouldAlign)
                .disabled(isProcessing)
                .help("Use SuperPoint + LightGlue feature matching to align frames before merging. Slower but much higher quality on handheld shots.")

            Divider().frame(height: 22)

            Picker("Merge", selection: $mergeMethod) {
                ForEach(MergeMethod.allCases, id: \.self) { m in
                    Text(m.displayName).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 260)
            .disabled(isProcessing)

            Spacer()

            Picker("Format", selection: $exportFormat) {
                ForEach(ImageExportFormat.allCases, id: \.self) { f in
                    Text(f.displayName).tag(f)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 220)
            .disabled(isProcessing)

            Text("\(images.count) image\(images.count == 1 ? "" : "s")")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: Preview

    @ViewBuilder
    private var previewSection: some View {
        if isLoading || isAligning || isMerging {
            HStack {
                ProgressView()
                Text(previewBusyText).foregroundColor(.secondary)
                Spacer()
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        } else if let previewImage {
            VStack(alignment: .leading, spacing: 8) {
                Text("Preview").font(.headline)
                Image(decorative: previewImage, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: 360)
                    .background(Color.black.opacity(0.08))
                    .cornerRadius(8)
                    .allowedDynamicRange(.high)
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        } else if let previewMessage {
            HStack {
                Label(previewMessage, systemImage: "info.circle")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        }
    }

    private var previewBusyText: String {
        if isLoading  { return "Loading images…" }
        if isAligning { return "Aligning frames…" }
        return "Merging…"
    }

    // MARK: Status

    private var statusBar: some View {
        HStack {
            switch processingStatus {
            case .pending:
                EmptyView()
            case .processing:
                ProgressView()
                Text(processingProgress.isEmpty ? "Rendering full-resolution image…" : processingProgress)
                    .foregroundColor(.secondary)
            case .success(let urls):
                Label("Saved \(urls.first?.lastPathComponent ?? "")", systemImage: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .failed(let msg):
                Label(msg, systemImage: "xmark.circle.fill")
                    .foregroundColor(.red)
            }
            Spacer()
        }
        .padding(.horizontal, 4)
    }

    // MARK: Actions

    private var actionRow: some View {
        HStack(spacing: 12) {
            Button("Add More") { showFilePicker = true }
                .disabled(isProcessing)
                .fileImporter(
                    isPresented: $showFilePicker,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true,
                    onCompletion: handleImagePick
                )
            Button("Clear All") {
                images.removeAll()
                cachedLoaded = []
                cachedAligned = nil
                previewImage = nil
                processingStatus = .pending
            }
            .disabled(isProcessing)
            Spacer()
            if let folder = outputFolder {
                Text("Output: \(folder.lastPathComponent)")
                    .font(.caption).foregroundColor(.secondary)
            }
            Button(isProcessing ? "Processing…" : "Merge & Export") { processFinal() }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || images.isEmpty)
                .fileImporter(
                    isPresented: $showFolderPicker,
                    allowedContentTypes: [.folder],
                    onCompletion: handleFolderPick
                )
        }
    }

    // MARK: - Load images (debounced)

    private func loadImagesDebounced() async {
        cachedAligned = nil
        if images.isEmpty {
            cachedLoaded = []
            previewImage = nil
            previewMessage = nil
            return
        }

        // Small debounce so quickly dropping many files batches into one load.
        do { try await Task.sleep(nanoseconds: 250_000_000) } catch { return }
        if Task.isCancelled { return }

        isLoading = true
        previewMessage = nil
        let urls = images.map(\.url)
        do {
            let loaded = try await loadLongExposureImages(urls, maxDimension: 1500)
            if Task.isCancelled { return }
            cachedLoaded = loaded
            isLoading = false
        } catch {
            if Task.isCancelled { return }
            isLoading = false
            cachedLoaded = []
            previewMessage = error.localizedDescription
        }
    }

    // MARK: - Preview (live on toggle change)

    private func updatePreview() async {
        if cachedLoaded.isEmpty {
            previewImage = nil
            return
        }
        do {
            let source: [CIImage]
            if shouldAlign {
                if let cached = cachedAligned {
                    source = cached
                } else {
                    isAligning = true
                    let frames = cachedLoaded
                    let aligned = await alignImagesPanoMatcher(frames)
                    if Task.isCancelled { isAligning = false; return }
                    cachedAligned = aligned
                    isAligning = false
                    source = aligned
                }
            } else {
                source = cachedLoaded
            }
            isMerging = true
            let mergeSource = source
            let method = mergeMethod
            let merged = await Task.detached(priority: .userInitiated) {
                mergeImages(mergeSource, method: method)
            }.value
            if Task.isCancelled { isMerging = false; return }
            guard let merged else { isMerging = false; return }
            let cg = try await renderLongExposurePreview(merged)
            if Task.isCancelled { isMerging = false; return }
            previewImage = cg
            isMerging = false
        } catch {
            isAligning = false
            isMerging = false
            previewMessage = error.localizedDescription
        }
    }

    // MARK: - Final processing

    private func processFinal() {
        guard !images.isEmpty else { return }
        if outputFolder == nil {
            pendingProcessAfterFolder = true
            showFolderPicker = true
            return
        }
        startProcessing()
    }

    private func startProcessing() {
        guard !images.isEmpty else { return }
        isProcessing = true
        processingStatus = .processing
        processingProgress = "Loading images…"
        let urls = images.map(\.url)
        let align = shouldAlign
        let method = mergeMethod
        let format = exportFormat
        let folder = outputFolder
        let progressHandler: @Sendable (String) -> Void = { message in
            Task { @MainActor in processingProgress = message }
        }
        Task {
            do {
                let url = try await performLongExposureMerge(
                    urls: urls,
                    align: align,
                    method: method,
                    format: format,
                    outputFolder: folder,
                    progress: progressHandler
                )
                await MainActor.run {
                    processingStatus = .success([url])
                    isProcessing = false
                    processingProgress = ""
                }
            } catch {
                await MainActor.run {
                    processingStatus = .failed(error.localizedDescription)
                    isProcessing = false
                    processingProgress = ""
                }
            }
        }
    }
}

// MARK: - Task keys

private struct ImageListKey: Equatable {
    let urls: [URL]
}

private struct PreviewKey: Equatable {
    let count: Int
    let align: Bool
    let method: MergeMethod
}
