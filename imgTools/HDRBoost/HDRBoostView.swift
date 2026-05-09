import SwiftUI
import UniformTypeIdentifiers

struct HDRBoostView: View {
    @State private var hdrInputURL: URL?
    @State private var hdrMaskURL: URL?
    @State private var hdrBoost: Double = 2.0
    @State private var hdrStatus: ProcessingStatus = .pending
    @State private var hdrPreview: CGImage?
    @State private var hdrPreviewError: String?
    @State private var isRenderingHDRPreview = false
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    @State private var activeImporter: ActiveImporter?
    @State private var pendingOutputAction: PendingOutputAction?

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 12) {
                HDRSelectionCard(
                    title: "Input SDR Image",
                    url: hdrInputURL,
                    placeholder: "Choose the base SDR image",
                    isProcessing: isProcessing,
                    onPick: { activeImporter = .hdrInput },
                    onClear: { hdrInputURL = nil; hdrStatus = .pending }
                )
                .fileImporter(
                    isPresented: Binding(
                        get: { activeImporter == .hdrInput },
                        set: { if !$0, activeImporter == .hdrInput { activeImporter = nil } }
                    ),
                    allowedContentTypes: [.image]
                ) { handleHDRImport($0, forMask: false) }
                HDRSelectionCard(
                    title: "Mask with Alpha",
                    url: hdrMaskURL,
                    placeholder: "Choose the mask image",
                    isProcessing: isProcessing,
                    onPick: { activeImporter = .hdrMask },
                    onClear: { hdrMaskURL = nil; hdrStatus = .pending }
                )
                .fileImporter(
                    isPresented: Binding(
                        get: { activeImporter == .hdrMask },
                        set: { if !$0, activeImporter == .hdrMask { activeImporter = nil } }
                    ),
                    allowedContentTypes: [.image]
                ) { handleHDRImport($0, forMask: true) }
            }

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Boost").font(.headline)
                        Text("Increase exposure before applying the alpha mask.")
                            .font(.caption).foregroundColor(.secondary)
                    }
                    Spacer()
                    Text(hdrBoost, format: .number.precision(.fractionLength(1)))
                        .font(.system(.title3, design: .rounded, weight: .semibold))
                        .monospacedDigit()
                }
                Slider(value: $hdrBoost, in: 0...8, step: 0.1)
                    .disabled(isProcessing)
                    .onChange(of: hdrBoost) { hdrStatus = .pending }
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)

            hdrPreviewView
            HDRStatusView(status: hdrStatus)

            HStack(spacing: 15) {
                if let folder = outputFolder {
                    Text("Output: \(folder.lastPathComponent)")
                        .font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                Button(isProcessing ? "Processing..." : "Create HDR HEIF") {
                    processHDRBoost()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || hdrInputURL == nil || hdrMaskURL == nil)
                .fileImporter(
                    isPresented: Binding(
                        get: { activeImporter == .outputFolder },
                        set: { if !$0, activeImporter == .outputFolder { activeImporter = nil } }
                    ),
                    allowedContentTypes: [.folder]
                ) { handleOutputFolderImport($0) }
            }
        }
        .padding()
        .dropDestination(for: URL.self) { items, _ in
            for url in items {
                if hdrInputURL == nil { hdrInputURL = url } else { hdrMaskURL = url }
                hdrStatus = .pending
            }
            return true
        }
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
                Text("Rendering preview...").foregroundColor(.secondary)
                Spacer()
            }
            .padding()
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(12)
        } else if let hdrPreview {
            VStack(alignment: .leading, spacing: 10) {
                Text("Preview").font(.headline)
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

    func loadHDRPreview() async {
        guard let inputURL = hdrInputURL, let maskURL = hdrMaskURL else {
            hdrPreview = nil; hdrPreviewError = nil; isRenderingHDRPreview = false
            return
        }
        isRenderingHDRPreview = true; hdrPreview = nil; hdrPreviewError = nil
        do {
            let image = try await makeHDRPreview(inputURL: inputURL, maskURL: maskURL, boost: hdrBoost)
            guard !Task.isCancelled else { return }
            hdrPreview = image; isRenderingHDRPreview = false
        } catch {
            guard !Task.isCancelled else { return }
            hdrPreviewError = "Preview unavailable"; isRenderingHDRPreview = false
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
        isProcessing = true; hdrStatus = .processing
        Task {
            do {
                let outputURLs = try await performHDRBoost(inputURL: inputURL, maskURL: maskURL, boost: hdrBoost, outputFolder: outputFolder)
                hdrStatus = .success(outputURLs); isProcessing = false
            } catch {
                hdrStatus = .failed(error.localizedDescription); isProcessing = false
            }
        }
    }

    func handleHDRImport(_ result: Result<URL, Error>, forMask: Bool) {
        activeImporter = nil
        guard case .success(let url) = result else { return }
        if forMask { hdrMaskURL = url } else { hdrInputURL = url }
        hdrStatus = .pending
    }

    func handleOutputFolderImport(_ result: Result<URL, Error>) {
        activeImporter = nil
        guard case .success(let url) = result else { pendingOutputAction = nil; return }
        outputFolder = url
        if case .hdrBoost(let inputURL, let maskURL) = pendingOutputAction {
            startHDRBoost(inputURL: inputURL, maskURL: maskURL)
        }
        pendingOutputAction = nil
    }
}
