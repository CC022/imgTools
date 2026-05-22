//
//  PanoStitcherView.swift
//  imgTools
//
//  N-image panorama UI ported from panoDev.
//  Shows the cylindrical canvas produced by PanoPipeline directly after stitching.
//

import SwiftUI
import AppKit
import CoreImage
import CoreGraphics
import UniformTypeIdentifiers
import simd

// MARK: - Image Store

@Observable @MainActor
final class ImageStore: @unchecked Sendable {
    var result: PanoramaResult?
    var displayBuffer: CanvasBuffer?
    var displayBufferURL: URL?
    var editParams: EditParams = .identity
    var cropAspect: CropAspect = .none
    var cropRegion: CGRect? = nil
    var canvasTransform: MetalCanvasTransform = .identity
    var focalLengthMM35: Float = 24.0
    var statusText: String = "Ready"
    var isLoading: Bool = false
    var pipelineProgress: PipelineProgress?
    var importedURLs: [URL]?

    func loadAndProcess() async {
        isLoading = true
        statusText = "Loading…"
        pipelineProgress = PipelineProgress(completedUnits: 0,
                                            totalUnits: 1,
                                            message: "Loading…")

        guard let urls = importedURLs else {
            statusText = "⚠️ No images — use Import to select photos"
            isLoading = false
            pipelineProgress = nil
            return
        }

        let fmm = focalLengthMM35
        let store = self
        let progressHandler: @Sendable (PipelineProgress) -> Void = { progress in
            Task { @MainActor in
                store.pipelineProgress = progress
                store.statusText = progress.message
            }
        }
        let pipelineResult = await Task.detached(priority: .userInitiated) {
            await PanoPipeline.process(urls: urls,
                                        focalLengthMM35: fmm,
                                        progress: progressHandler)
        }.value

        guard let r = pipelineResult else {
            statusText = "⚠️ Pipeline failed"
            isLoading = false
            pipelineProgress = nil
            return
        }

        result            = r
        displayBuffer     = r.blendedBuffer
        displayBufferURL  = nil
        statusText        = formatStatus(r, suffix: r.graph.edges.isEmpty ? "no edges" : nil)
        isLoading = false
        pipelineProgress = nil
    }

    func openStandaloneImage(from url: URL) async {
        isLoading = true
        statusText = "Loading \(url.lastPathComponent)…"
        pipelineProgress = nil

        let buf: CanvasBuffer? = await Task.detached(priority: .userInitiated) {
            guard let ci = CIImage(contentsOf: url, options: [
                .expandToHDR: true,
                .applyOrientationProperty: true
            ]) else { return nil }

            let w = Int(ci.extent.width.rounded())
            let h = Int(ci.extent.height.rounded())
            guard w > 0, h > 0 else { return nil }

            let ctx = PanoContext.shared
            guard let dst = ctx.makeHalfBuffer(width: w, height: h, storage: .shared)
            else { return nil }

            ctx.ciContext.render(
                ci,
                toBitmap: dst.buffer.contents(),
                rowBytes: dst.bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: w, height: h),
                format: .RGBAh,
                colorSpace: PanoContext.linearColorSpace
            )
            return dst
        }.value

        if let buf {
            result            = nil
            displayBuffer     = buf
            displayBufferURL  = url
            editParams        = .identity
            cropAspect        = .none
            cropRegion        = nil
            statusText        = "\(url.lastPathComponent) · \(buf.width)×\(buf.height)"
        } else {
            statusText = "⚠️ Could not open \(url.lastPathComponent)"
        }
        isLoading = false
    }

    private func formatStatus(_ r: PanoramaResult, suffix: String? = nil) -> String {
        let nodes = r.graph.nodes.count
        let edges = r.graph.edges.count
        let inliers = r.graph.edges.reduce(0) { $0 + $1.inlierCount }
        let f = r.graph.nodes[r.graph.anchorIndex()].pose?.intrinsics.focal ?? 0
        let core = "nodes \(nodes) · edges \(edges) · inliers \(inliers) · f \(Int(f)) px · canvas \(r.canvas.size.x)×\(r.canvas.size.y)"
        return suffix.map { "\(core) · \($0)" } ?? core
    }
}

// MARK: - Main view

struct PanoStitcherView: View {
    @State private var store = ImageStore()
    @State private var showImporter = false
    @State private var showEditPanel = false
    @State private var showCropPanel = false
    @State private var showOriginal = false
    @State private var pendingCropTransform: MetalZoomImageView.PendingTransform? = nil
    @AppStorage("focalLengthMM35Text") private var focalText: String = "24"

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                canvasView
            }
            .backgroundExtensionEffect()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea(edges: .all)
            .overlay {
                VStack {
                    Spacer()
                    HStack {
                        HStack {
                            if store.isLoading {
                                pipelineProgressBar
                            }
                            Text(store.statusText)
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        .padding(5)
                        .glassEffect()
                        Spacer()
                    }
                    .padding()
                }
            }
        }
        .toolbar {
            ToolbarSpacer()
            ToolbarItem {
                HStack(spacing: 3) {
                    TextField("24", text: $focalText)
                        .textFieldStyle(.plain)
                        .multilineTextAlignment(.trailing)
                        .monospacedDigit()
                        .frame(width: 28)
                        .onSubmit { commitFocal() }
                    Text("mm")
                        .foregroundStyle(.secondary)
                }
                .controlSize(.small)
                .help("Focal length (35 mm equivalent)")
            }
            ToolbarItem {
                Button {
                    showImporter = true
                } label: {
                    Image(systemName: "photo.badge.plus")
                }
                .disabled(store.isLoading)
                .help("Import images")
                .fileImporter(
                    isPresented: $showImporter,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true
                ) { result in
                    guard case .success(let urls) = result, !urls.isEmpty else { return }
                    let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
                    // Single image → skip the stitcher pipeline and just open
                    // it for cropping/editing; multi → kick off the stitch.
                    if sorted.count == 1, let url = sorted.first {
                        Task { await store.openStandaloneImage(from: url) }
                    } else {
                        store.importedURLs = sorted
                        store.statusText = "\(urls.count) images selected"
                        autoFillFocal(from: sorted.first)
                        commitFocal()
                        Task { await store.loadAndProcess() }
                    }
                }
            }
            ToolbarItem {
                Button {
                    if showCropPanel { setCropMode(false) }
                    showEditPanel.toggle()
                    if !showEditPanel { showOriginal = false }
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .symbolVariant(showEditPanel ? .fill : .none)
                }
                .disabled(store.displayBuffer == nil)
                .help("Edit")
            }
            ToolbarItem {
                Button {
                    if showEditPanel { showEditPanel = false }
                    setCropMode(!showCropPanel)
                } label: {
                    Image(systemName: "crop")
                        .symbolVariant(showCropPanel ? .fill : .none)
                }
                .disabled(store.displayBuffer == nil)
                .help("Crop")
            }
            ToolbarItem {
                Menu {
                    ForEach(PanoPipeline.ExportFormat.allCases, id: \.self) { fmt in
                        Button(fmt.displayName) {
                            Task { await saveDisplayedImage(format: fmt) }
                        }
                    }
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(store.displayBuffer == nil || store.isLoading)
                .help("Export")
            }
        }
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        .task {
            commitFocal()
            if let urls = store.importedURLs {
                autoFillFocal(from: urls.first)
                await store.loadAndProcess()
            } else {
                store.statusText = "Import images to begin"
            }
        }
    }

    @ViewBuilder
    private var pipelineProgressBar: some View {
        let fraction = store.pipelineProgress?.fraction ?? 0
        ProgressView(value: fraction)
            .progressViewStyle(.linear)
            .controlSize(.small)
            .frame(width: 180)
        Text(fraction.formatted(.percent.precision(.fractionLength(0))))
            .font(.system(.caption, design: .monospaced).monospacedDigit())
            .foregroundStyle(.secondary)
            .frame(width: 38, alignment: .trailing)
    }

    @MainActor
    private func saveDisplayedImage(format: PanoPipeline.ExportFormat) async {
        guard let buffer = store.displayBuffer else { return }

        let isStitched = store.result != nil
        let ext = format.fileExtension
        let defaultName = isStitched ? "panorama.\(ext)" : suggestedName(for: store.displayBufferURL, ext: ext)
        let kind = isStitched ? "Panorama" : "Image"
        let title = "Save \(kind) — \(format.displayName)"

        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.utType]
        panel.nameFieldStringValue = defaultName
        panel.title = title
        guard let window = NSApp.keyWindow,
              await panel.beginSheetModal(for: window) == .OK,
              let url = panel.url else { return }

        let bufferSize = CGSize(width: buffer.width, height: buffer.height)
        let cropRect: CGRect?
        if let r = store.cropRegion {
            cropRect = CGRect(x: r.origin.x * bufferSize.width,
                              y: r.origin.y * bufferSize.height,
                              width:  r.size.width  * bufferSize.width,
                              height: r.size.height * bufferSize.height)
        } else {
            cropRect = canvasCropRect(aspect: store.cropAspect,
                                       transform: store.canvasTransform,
                                       canvasSize: bufferSize)
        }

        do {
            if let result = store.result {
                try PanoPipeline.save(result: result, to: url,
                                       edits: store.editParams,
                                       crop: cropRect,
                                       format: format)
            } else {
                try PanoPipeline.save(buffer: buffer, to: url,
                                       edits: store.editParams,
                                       crop: cropRect,
                                       exifSource: store.displayBufferURL,
                                       writeGPano: false,
                                       format: format)
            }
        } catch {
            let alert = NSAlert(error: error)
            alert.runModal()
        }
    }

    private func suggestedName(for source: URL?, ext: String) -> String {
        guard let source else { return "image.\(ext)" }
        return source.deletingPathExtension().lastPathComponent + ".\(ext)"
    }

    private func commitFocal() {
        if let v = Float(focalText), v > 0 {
            store.focalLengthMM35 = v
        } else {
            focalText = String(format: "%.4g", store.focalLengthMM35)
        }
    }

    private func autoFillFocal(from url: URL?) {
        guard let url, let f = EXIF.focalLengthMM35(for: url) else { return }
        focalText = String(format: "%.4g", f)
        store.focalLengthMM35 = f
        dbg("[PanoStitcherView] auto-filled focal: \(f) mm (from \(url.lastPathComponent))")
    }

    // MARK: Canvas view

    @ViewBuilder
    private var canvasView: some View {
        if let buf = store.displayBuffer {
            let activeCrop: CGRect? = showCropPanel ? nil : store.cropRegion

            MetalZoomImageView(
                buffer: buf,
                editParams: showOriginal ? .identity : store.editParams,
                cropRegion: activeCrop,
                onTransformChange: { t in
                    Task { @MainActor in store.canvasTransform = t }
                },
                pendingTransform: pendingCropTransform
            )
            .overlay {
                if showCropPanel && store.cropAspect != .none {
                    CropOverlay(aspect: store.cropAspect)
                        .allowsHitTesting(false)
                        .transition(.opacity)
                }
            }
            .overlay(alignment: .trailing) {
                if showEditPanel {
                    EditPanel(params: Bindable(store).editParams,
                              showOriginal: $showOriginal)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                } else if showCropPanel {
                    CropPanel(aspect: Bindable(store).cropAspect)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .animation(.easeInOut(duration: 0.18), value: showEditPanel)
            .animation(.easeInOut(duration: 0.18), value: showCropPanel)
            .animation(.easeInOut(duration: 0.18), value: store.cropAspect)
        } else {
            Text(store.isLoading ? "Building canvas…" : "No canvas")
                .foregroundStyle(.secondary)
        }
    }

    @MainActor
    private func setCropMode(_ on: Bool) {
        guard on != showCropPanel else { return }
        if on {
            if let region = store.cropRegion,
               let buf = store.displayBuffer,
               let restore = transformForNormalizedRect(
                    region,
                    viewSize: store.canvasTransform.viewSize,
                    canvasSize: CGSize(width: buf.width, height: buf.height),
                    aspect: store.cropAspect)
            {
                pendingCropTransform = .init(id: UUID(),
                                              offset: restore.offset,
                                              scale:  restore.scale)
            }
            showCropPanel = true
        } else {
            if store.cropAspect != .none, let buf = store.displayBuffer {
                let canvasSize = CGSize(width: buf.width, height: buf.height)
                store.cropRegion = normalizedCropRect(
                    aspect: store.cropAspect,
                    transform: store.canvasTransform,
                    canvasSize: canvasSize)
            } else {
                store.cropRegion = nil
            }
            pendingCropTransform = nil
            showCropPanel = false
        }
    }
}
