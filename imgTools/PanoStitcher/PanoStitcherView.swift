//
//  PanoStitcherView.swift
//  imgTools
//
//  N-image panorama UI ported from panoDev. Two views switched by a
//  toolbar toggle:
//    • Pairs   — per-edge match/inlier inspection
//    • Canvas  — cylindrical preview produced by PanoPipeline
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
    var nodeImages: [CGImage] = []
    var nodePixelSizes: [CGSize] = []
    var displayBuffer: CanvasBuffer?
    var displayBufferURL: URL?
    var editParams: EditParams = .identity
    var cropAspect: CropAspect = .none
    var cropRegion: CGRect? = nil
    var canvasTransform: MetalCanvasTransform = .identity
    var selectedEdge: Int = 0
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

        let images: [CGImage] = await Task.detached {
            r.graph.nodes.compactMap { (try? $0.image.loadTexture())?.toCGImage() }
        }.value
        nodeImages = images
        nodePixelSizes = r.graph.nodes.map {
            CGSize(width: $0.image.width, height: $0.image.height)
        }

        result            = r
        displayBuffer     = r.blendedBuffer
        displayBufferURL  = nil
        if r.graph.edges.isEmpty {
            statusText = formatStatus(r, suffix: "no edges")
        } else {
            selectedEdge = 0
            statusText = formatStatus(r)
        }
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
            nodeImages        = []
            nodePixelSizes    = []
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

enum ViewMode: String, CaseIterable, Identifiable {
    case pairs  = "Pairs"
    case canvas = "Canvas"
    var id: String { rawValue }
}

struct PanoStitcherView: View {
    @State private var store = ImageStore()
    @State private var mode: ViewMode = .pairs
    @State private var showImporter = false
    @State private var showOpener   = false
    @State private var showEditPanel = false
    @State private var showCropPanel = false
    @State private var showOriginal = false
    @State private var pendingCropTransform: MetalZoomImageView.PendingTransform? = nil
    @AppStorage("focalLengthMM35Text") private var focalText: String = "24"

    var body: some View {
        VStack(spacing: 0) {
            if mode == .pairs, let r = store.result, !r.graph.edges.isEmpty {
                HStack(spacing: 12) {
                    Picker("Edge", selection: $store.selectedEdge) {
                        ForEach(r.graph.edges.indices, id: \.self) { idx in
                            let e = r.graph.edges[idx]
                            Text("\(e.src)→\(e.dst)  (\(e.inlierCount)/\(e.matches.count))")
                                .tag(idx)
                        }
                    }
                    .pickerStyle(.segmented)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.top, 8)
            }

            ZStack {
                Color.black
                switch mode {
                case .pairs:  pairsView
                case .canvas: canvasView
                }
            }
            .backgroundExtensionEffect()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
//            .ignoresSafeArea(edges: .top)
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
            ToolbarItem {
                Button {
                    mode = (mode == .pairs) ? .canvas : .pairs
                } label: {
                    Image(systemName: mode == .pairs ? "pano" : "rectangle.split.2x1")
                }
                .help(mode == .pairs ? "Show canvas" : "Show pairs")
            }
            ToolbarItem {
                HStack {
                    Text("f:")
                        .foregroundStyle(.secondary)
                    TextField("mm", text: $focalText)
                        .frame(width: 36)
                        .multilineTextAlignment(.trailing)
                        .onSubmit { commitFocal() }
                    Text("mm")
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            }
            ToolbarSpacer()
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
                    store.importedURLs = sorted
                    store.statusText = "\(urls.count) image\(urls.count == 1 ? "" : "s") selected"
                    autoFillFocal(from: sorted.first)
                    commitFocal()
                    Task { await store.loadAndProcess() }
                }
            }
            ToolbarItem {
                Button {
                    commitFocal()
                    Task { await store.loadAndProcess() }
                } label: {
                    Image(systemName: "play.fill")
                }
                .disabled(store.isLoading)
                .help("Run pipeline")
            }
            ToolbarItem {
                Button {
                    showOpener = true
                } label: {
                    Image(systemName: "arrow.up.forward.app")
                }
                .disabled(store.isLoading)
                .help("Open single image")
                .fileImporter(
                    isPresented: $showOpener,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: false
                ) { result in
                    guard case .success(let urls) = result, let url = urls.first else { return }
                    mode = .canvas
                    Task { await store.openStandaloneImage(from: url) }
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
                Button {
                    Task { await saveDisplayedImage() }
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(store.displayBuffer == nil || store.isLoading)
                .help("Save")
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
    private func saveDisplayedImage() async {
        guard let buffer = store.displayBuffer else { return }

        let isStitched = store.result != nil
        let defaultName = isStitched ? "panorama.heic" : suggestedName(for: store.displayBufferURL)
        let title = isStitched ? "Save Panorama (HEIF10 HDR)" : "Save Image (HEIF10 HDR)"

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.heic]
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
                                       crop: cropRect)
            } else {
                try PanoPipeline.save(buffer: buffer, to: url,
                                       edits: store.editParams,
                                       crop: cropRect,
                                       exifSource: store.displayBufferURL,
                                       writeGPano: false)
            }
        } catch {
            let alert = NSAlert(error: error)
            alert.runModal()
        }
    }

    private func suggestedName(for source: URL?) -> String {
        guard let source else { return "image.heic" }
        return source.deletingPathExtension().lastPathComponent + ".heic"
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

    // MARK: Pairs view

    @ViewBuilder
    private var pairsView: some View {
        if let r = store.result,
           !r.graph.edges.isEmpty,
           store.selectedEdge < r.graph.edges.count
        {
            let edge = r.graph.edges[store.selectedEdge]
            GeometryReader { geo in
                let panel = CGSize(width: geo.size.width / 2, height: geo.size.height)
                ZoomDragImage {
                    ZStack(alignment: .topLeading) {
                        HStack(spacing: 0) {
                            imagePanel(at: edge.src, panelSize: panel)
                            imagePanel(at: edge.dst, panelSize: panel)
                        }
                        edgeOverlay(edge: edge, panelSize: panel)
                    }
                }
            }
        } else {
            Text("No matched pairs to display")
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func imagePanel(at index: Int, panelSize: CGSize) -> some View {
        ZStack {
            Color.black
            if index < store.nodeImages.count {
                Image(decorative: store.nodeImages[index], scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .allowedDynamicRange(.high)
            }
        }
        .frame(width: panelSize.width, height: panelSize.height)
        .clipped()
    }

    @ViewBuilder
    private func edgeOverlay(edge: PanoEdge, panelSize: CGSize) -> some View {
        Canvas { ctx, _ in
            guard edge.src < store.nodePixelSizes.count,
                  edge.dst < store.nodePixelSizes.count else { return }

            let frameA = imageRect(in: panelSize,
                                    pixelSize: store.nodePixelSizes[edge.src],
                                    panelOriginX: 0)
            let frameB = imageRect(in: panelSize,
                                    pixelSize: store.nodePixelSizes[edge.dst],
                                    panelOriginX: panelSize.width)

            guard let r = store.result,
                  edge.src < r.graph.nodes.count,
                  edge.dst < r.graph.nodes.count else { return }
            let kpA = r.graph.nodes[edge.src].keypoints
            let kpB = r.graph.nodes[edge.dst].keypoints

            let inlierSet = Set(edge.inliers)
            let drawOrder = edge.matches.indices.sorted {
                let aIn = inlierSet.contains($0)
                let bIn = inlierSet.contains($1)
                if aIn != bIn { return !aIn }
                return edge.matches[$0].confidence < edge.matches[$1].confidence
            }
            for idx in drawOrder {
                let m = edge.matches[idx]
                guard m.indexA < kpA.count, m.indexB < kpB.count else { continue }
                let pA = mapToScreen(x: CGFloat(kpA[m.indexA].x),
                                     y: CGFloat(kpA[m.indexA].y),
                                     frame: frameA,
                                     pixelSize: store.nodePixelSizes[edge.src])
                let pB = mapToScreen(x: CGFloat(kpB[m.indexB].x),
                                     y: CGFloat(kpB[m.indexB].y),
                                     frame: frameB,
                                     pixelSize: store.nodePixelSizes[edge.dst])
                let isIn = inlierSet.contains(idx)
                let colour = isIn ? Color.green.opacity(0.85) : Color.red.opacity(0.25)
                let lw: CGFloat = isIn ? 0.9 : 0.6
                var path = Path()
                path.move(to: pA); path.addLine(to: pB)
                ctx.stroke(path, with: .color(colour), lineWidth: lw)
            }
        }
        .frame(width: panelSize.width * 2, height: panelSize.height)
        .allowsHitTesting(false)
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

    // MARK: Geometry helpers

    private func mapToScreen(x: CGFloat, y: CGFloat,
                              frame: CGRect, pixelSize: CGSize) -> CGPoint {
        guard pixelSize.width > 0, pixelSize.height > 0 else { return .zero }
        return CGPoint(
            x: frame.origin.x + x * (frame.size.width  / pixelSize.width),
            y: frame.origin.y + y * (frame.size.height / pixelSize.height)
        )
    }

    private func imageRect(in panelSize: CGSize,
                            pixelSize: CGSize,
                            panelOriginX: CGFloat) -> CGRect {
        guard pixelSize.width > 0, pixelSize.height > 0 else {
            return CGRect(x: panelOriginX, y: 0, width: 0, height: 0)
        }
        let scale = min(panelSize.width  / pixelSize.width,
                        panelSize.height / pixelSize.height)
        let w = pixelSize.width * scale
        let h = pixelSize.height * scale
        return CGRect(
            x: panelOriginX + (panelSize.width  - w) / 2,
            y:                 (panelSize.height - h) / 2,
            width: w, height: h
        )
    }
}
