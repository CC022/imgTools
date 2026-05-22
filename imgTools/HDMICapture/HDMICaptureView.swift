import AVFoundation
import AppKit
import CoreMedia
import SwiftUI
import UniformTypeIdentifiers

struct HDMICaptureView: View {
    @Environment(HDMICaptureSession.self) private var session
    @Environment(\.presentFullscreenPreview) private var presentFullscreen

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            PreviewRepresentable(layer: session.previewLayer, onDoubleClick: enterFullscreen)
                .background(Color.black)
        }
        .task { await session.start() }
        .alert("Capture", isPresented: errorBinding) {
            if isPermissionError {
                Button("Open System Settings") { openCameraSettings() }
                Button("Cancel", role: .cancel) { session.errorMessage = nil }
            } else {
                Button("OK") { session.errorMessage = nil }
            }
        } message: {
            Text(session.errorMessage ?? "")
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 10) {
            Picker("Video", selection: videoBinding) {
                Text("—").tag(AVCaptureDevice?.none)
                ForEach(session.videoDevices, id: \.uniqueID) { device in
                    Text(device.localizedName).tag(AVCaptureDevice?.some(device))
                }
            }
            .frame(maxWidth: 220)

            Picker("Audio", selection: audioBinding) {
                Text("None").tag(AVCaptureDevice?.none)
                ForEach(session.audioDevices, id: \.uniqueID) { device in
                    Text(device.localizedName).tag(AVCaptureDevice?.some(device))
                }
            }
            .frame(maxWidth: 200)

            Text(session.currentFormat.map(formatLabel) ?? "—")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)

            Toggle(isOn: muteBinding) {
                Image(systemName: session.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
            }
            .toggleStyle(.button)
            .help(session.isMuted ? "Unmute audio passthrough" : "Mute audio passthrough")

            Spacer()

            Button {
                snapshot()
            } label: {
                Label("Snapshot", systemImage: "camera")
            }
            .disabled(!session.isRunning)

            Button {
                session.isRecording ? session.stopRecording() : startRecording()
            } label: {
                Label(session.isRecording ? "Stop" : "Record",
                      systemImage: session.isRecording ? "stop.circle.fill" : "record.circle")
            }
            .tint(session.isRecording ? .red : .accentColor)
            .disabled(!session.isRunning)
        }
        .padding(10)
    }

    // MARK: - Bindings

    private var videoBinding: Binding<AVCaptureDevice?> {
        .init(
            get: { session.selectedVideoDevice },
            set: { device in
                guard let device else { return }
                Task { await session.selectVideoDevice(device) }
            }
        )
    }

    private var audioBinding: Binding<AVCaptureDevice?> {
        .init(
            get: { session.selectedAudioDevice },
            set: { device in Task { await session.selectAudioDevice(device) } }
        )
    }

    private var muteBinding: Binding<Bool> {
        .init(get: { session.isMuted }, set: { session.isMuted = $0 })
    }

    private var errorBinding: Binding<Bool> {
        .init(
            get: { session.errorMessage != nil },
            set: { if !$0 { session.errorMessage = nil } }
        )
    }

    // MARK: - Actions

    private func snapshot() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "snapshot_\(timestamp()).png"
        panel.title = "Save Snapshot"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try session.snapshot(to: url) }
        catch { session.errorMessage = error.localizedDescription }
    }

    private func startRecording() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.quickTimeMovie]
        panel.nameFieldStringValue = "recording_\(timestamp()).mov"
        panel.title = "Start Recording"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        session.startRecording(to: url)
    }

    // MARK: - Helpers

    private func timestamp() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HHmmss"
        return fmt.string(from: Date())
    }

    private var isPermissionError: Bool {
        guard let msg = session.errorMessage else { return false }
        return msg.contains("Privacy & Security") || msg.contains("restricted")
    }

    private func openCameraSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera") {
            NSWorkspace.shared.open(url)
        }
        session.errorMessage = nil
    }

    private func enterFullscreen() {
        presentFullscreen(session.makePreviewLayer())
    }

    private func formatLabel(_ format: AVCaptureDevice.Format) -> String {
        let dim = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let fps = format.videoSupportedFrameRateRanges.first.map { Int($0.maxFrameRate.rounded()) } ?? 0
        return "\(dim.width)×\(dim.height) @ \(fps)fps"
    }
}

// MARK: - Preview wrapper

private struct PreviewRepresentable: NSViewRepresentable {
    let layer: AVCaptureVideoPreviewLayer
    var onDoubleClick: (() -> Void)?

    func makeNSView(context: Context) -> PreviewLayerView {
        let view = PreviewLayerView(previewLayer: layer)
        view.onDoubleClick = onDoubleClick
        return view
    }

    func updateNSView(_ nsView: PreviewLayerView, context: Context) {
        nsView.onDoubleClick = onDoubleClick
    }
}

private final class PreviewLayerView: NSView {
    private let previewLayer: AVCaptureVideoPreviewLayer
    var onDoubleClick: (() -> Void)?

    init(previewLayer: AVCaptureVideoPreviewLayer) {
        self.previewLayer = previewLayer
        super.init(frame: .zero)
        wantsLayer = true
        layer = CALayer()
        layer?.backgroundColor = NSColor.black.cgColor
        layer?.addSublayer(previewLayer)
        layerContentsRedrawPolicy = .duringViewResize
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layout() {
        super.layout()
        previewLayer.frame = bounds
    }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            onDoubleClick?()
        } else {
            super.mouseDown(with: event)
        }
    }
}

// MARK: - Fullscreen overlay

struct FullscreenPreviewOverlay: View {
    let layer: AVCaptureVideoPreviewLayer
    let onDismiss: () -> Void

    @State private var escMonitor: Any?

    var body: some View {
        PreviewRepresentable(layer: layer, onDoubleClick: onDismiss)
            .background(Color.black)
            .ignoresSafeArea()
            .onAppear {
                escMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                    if event.keyCode == 53 { // Escape
                        onDismiss()
                        return nil
                    }
                    return event
                }
            }
            .onDisappear {
                if let m = escMonitor { NSEvent.removeMonitor(m) }
                escMonitor = nil
            }
    }
}
