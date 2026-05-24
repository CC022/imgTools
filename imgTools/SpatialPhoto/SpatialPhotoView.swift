import SwiftUI
import UniformTypeIdentifiers

struct SpatialPhotoView: View {
    @State private var leftURL: URL?
    @State private var rightURL: URL?

    @State private var baselineMM: Double = 64.0
    @State private var horizontalFOV: Double = 80.0
    @State private var disparityAdjustment: Double = 0.0
    @State private var fovAutoFilled: Bool = false

    @State private var status: ProcessingStatus = .pending
    @State private var isProcessing = false
    @State private var outputFolder: URL?
    @State private var activeImporter: ActiveImporter?
    @State private var pendingOutputAction: PendingOutputAction?

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                HStack(spacing: 12) {
                    SpatialSelectionCard(
                        title: "Left Eye",
                        subtitle: "Image shown to the left eye.",
                        url: leftURL,
                        placeholder: "Choose or drop the left image",
                        isProcessing: isProcessing,
                        onPick: { activeImporter = .spatialLeft },
                        onClear: { leftURL = nil; status = .pending; fovAutoFilled = false }
                    )
                    .fileImporter(
                        isPresented: Binding(
                            get: { activeImporter == .spatialLeft },
                            set: { if !$0, activeImporter == .spatialLeft { activeImporter = nil } }
                        ),
                        allowedContentTypes: [.image]
                    ) { handleImport($0, slot: .left) }

                    SpatialSelectionCard(
                        title: "Right Eye",
                        subtitle: "Image shown to the right eye.",
                        url: rightURL,
                        placeholder: "Choose or drop the right image",
                        isProcessing: isProcessing,
                        onPick: { activeImporter = .spatialRight },
                        onClear: { rightURL = nil; status = .pending }
                    )
                    .fileImporter(
                        isPresented: Binding(
                            get: { activeImporter == .spatialRight },
                            set: { if !$0, activeImporter == .spatialRight { activeImporter = nil } }
                        ),
                        allowedContentTypes: [.image]
                    ) { handleImport($0, slot: .right) }
                }

                parameterCard

                SpatialStatusView(status: status)

                HStack(spacing: 15) {
                    if let folder = outputFolder {
                        Label("Output: \(folder.lastPathComponent)", systemImage: "folder")
                            .font(.caption).foregroundColor(.secondary)
                    }
                    Spacer()
                    Button(isProcessing ? "Processing..." : "Create Spatial Photo") {
                        processExport()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || leftURL == nil || rightURL == nil)
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
        }
        .dropDestination(for: URL.self) { items, _ in
            for url in items {
                if leftURL == nil {
                    leftURL = url
                } else if rightURL == nil {
                    rightURL = url
                } else {
                    rightURL = url
                }
                status = .pending
            }
            return true
        }
        .task(id: leftURL) {
            await autoFillFOV()
        }
    }

    var parameterCard: some View {
        VStack(spacing: 16) {
            SpatialParameterSlider(
                title: "Baseline",
                caption: "Distance between the two cameras, in millimeters.",
                valueText: String(format: "%.1f mm", baselineMM),
                value: $baselineMM,
                range: 10...200,
                step: 0.5,
                isDisabled: isProcessing,
                onChange: { status = .pending }
            )
            Divider()
            SpatialParameterSlider(
                title: "Horizontal FOV",
                caption: fovAutoFilled
                    ? "Auto-filled from EXIF. Drag to override."
                    : "Field of view of each lens, in degrees.",
                valueText: String(format: "%.1f°", horizontalFOV),
                value: $horizontalFOV,
                range: 30...140,
                step: 0.5,
                isDisabled: isProcessing,
                onChange: { fovAutoFilled = false; status = .pending }
            )
            Divider()
            SpatialParameterSlider(
                title: "Disparity Adjustment",
                caption: "Horizontal presentation shift, as a fraction of image width.",
                valueText: String(format: "%+.2f", disparityAdjustment),
                value: $disparityAdjustment,
                range: -1.0...1.0,
                step: 0.01,
                isDisabled: isProcessing,
                onChange: { status = .pending }
            )
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }

    private enum Slot { case left, right }

    private func handleImport(_ result: Result<URL, Error>, slot: Slot) {
        activeImporter = nil
        guard case .success(let url) = result else { return }
        switch slot {
        case .left:
            leftURL = url
            fovAutoFilled = false
        case .right:
            rightURL = url
        }
        status = .pending
    }

    private func handleOutputFolderImport(_ result: Result<URL, Error>) {
        activeImporter = nil
        guard case .success(let url) = result else { pendingOutputAction = nil; return }
        outputFolder = url
        if case .spatial(let leftURL, let rightURL) = pendingOutputAction {
            startExport(leftURL: leftURL, rightURL: rightURL)
        }
        pendingOutputAction = nil
    }

    private func processExport() {
        guard let leftURL, let rightURL else { return }
        if outputFolder == nil {
            pendingOutputAction = .spatial(leftURL: leftURL, rightURL: rightURL)
            activeImporter = .outputFolder
        } else {
            startExport(leftURL: leftURL, rightURL: rightURL)
        }
    }

    private func startExport(leftURL: URL, rightURL: URL) {
        isProcessing = true
        status = .processing
        let baselineMM = baselineMM
        let horizontalFOV = horizontalFOV
        let disparityAdjustment = disparityAdjustment
        let outputFolder = outputFolder
        Task {
            do {
                let urls = try await performSpatialPhotoExport(
                    leftURL: leftURL,
                    rightURL: rightURL,
                    baselineMM: baselineMM,
                    horizontalFOV: horizontalFOV,
                    disparityAdjustment: disparityAdjustment,
                    outputFolder: outputFolder
                )
                status = .success(urls)
            } catch {
                status = .failed(error.localizedDescription)
            }
            isProcessing = false
        }
    }

    private func autoFillFOV() async {
        guard let leftURL, !fovAutoFilled else { return }
        let url = leftURL
        let inferred = await Task.detached { inferHorizontalFOV(from: url) }.value
        guard !Task.isCancelled, let inferred else { return }
        let clamped = min(max(inferred, 30.0), 140.0)
        await MainActor.run {
            horizontalFOV = clamped
            fovAutoFilled = true
        }
    }
}
