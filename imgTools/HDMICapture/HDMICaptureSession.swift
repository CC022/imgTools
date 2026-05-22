@preconcurrency import AVFoundation
import AppKit
import CoreImage
import CoreMedia
import Foundation
import Synchronization

@Observable
final class HDMICaptureSession {

    // MARK: - Observable state
    private(set) var videoDevices: [AVCaptureDevice] = []
    private(set) var audioDevices: [AVCaptureDevice] = []
    private(set) var selectedVideoDevice: AVCaptureDevice?
    private(set) var selectedAudioDevice: AVCaptureDevice?
    private(set) var currentFormat: AVCaptureDevice.Format?
    private(set) var isRunning = false
    private(set) var isRecording = false
    var errorMessage: String?

    var isMuted: Bool = false {
        didSet { audioPreview.volume = isMuted ? 0 : 1 }
    }

    // MARK: - AV stack
    let previewLayer: AVCaptureVideoPreviewLayer

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.kojima.imgTools.hdmi.session")
    private let videoDataQueue = DispatchQueue(label: "com.kojima.imgTools.hdmi.video")

    private var videoInput: AVCaptureDeviceInput?
    private var audioInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let movieOutput = AVCaptureMovieFileOutput()
    private let audioPreview = AVCaptureAudioPreviewOutput()
    private let sampleDelegate = SampleBufferRelay()
    private var recordingRelay: RecordingRelay?

    init() {
        self.previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer.videoGravity = .resizeAspect
    }

    func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspect
        return layer
    }

    // MARK: - Lifecycle

    func start() async {
        guard !isRunning else { return }

        guard await ensureAuthorized(for: .video) else {
            errorMessage = unauthorizedMessage(for: .video)
            return
        }
        _ = await ensureAuthorized(for: .audio)

        discoverDevices()
        if selectedVideoDevice == nil {
            selectedVideoDevice = videoDevices.first
        }
        if selectedAudioDevice == nil {
            selectedAudioDevice = preferredAudioDevice(matching: selectedVideoDevice)
        }
        currentFormat = selectedVideoDevice?.activeFormat

        await configure()

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [captureSession] in
                if !captureSession.isRunning { captureSession.startRunning() }
                cont.resume()
            }
        }
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        if isRecording { movieOutput.stopRecording() }
        let session = captureSession
        sessionQueue.async {
            if session.isRunning { session.stopRunning() }
        }
        isRunning = false
    }

    // MARK: - Selection

    func selectVideoDevice(_ device: AVCaptureDevice) async {
        selectedVideoDevice = device
        currentFormat = device.activeFormat
        if let match = audioDevices.first(where: { $0.localizedName == device.localizedName }) {
            selectedAudioDevice = match
        }
        await configure()
    }

    func selectAudioDevice(_ device: AVCaptureDevice?) async {
        selectedAudioDevice = device
        await configure()
    }

    // MARK: - Snapshot

    func snapshot(to url: URL) throws {
        guard let sample = sampleDelegate.latest(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
            throw CaptureError.noFrame
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw CaptureError.encodeFailed
        }
        let rep = NSBitmapImageRep(cgImage: cgImage)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw CaptureError.encodeFailed
        }
        try data.write(to: url)
    }

    // MARK: - Recording

    func startRecording(to url: URL) {
        guard !isRecording, captureSession.outputs.contains(movieOutput) else { return }
        try? FileManager.default.removeItem(at: url)
        let relay = RecordingRelay { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                self.isRecording = false
                self.recordingRelay = nil
                if let error { self.errorMessage = error.localizedDescription }
            }
        }
        recordingRelay = relay
        movieOutput.startRecording(to: url, recordingDelegate: relay)
        isRecording = true
    }

    func stopRecording() {
        guard isRecording else { return }
        movieOutput.stopRecording()
    }

    // MARK: - Authorization

    private func ensureAuthorized(for mediaType: AVMediaType) async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: mediaType)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func unauthorizedMessage(for mediaType: AVMediaType) -> String {
        let kind = (mediaType == .video) ? "Camera" : "Microphone"
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .denied:
            return "\(kind) access was denied. Enable it in System Settings → Privacy & Security → \(kind), then quit and relaunch imgTools."
        case .restricted:
            return "\(kind) access is restricted on this Mac (e.g. by parental controls or an MDM profile)."
        default:
            return "\(kind) access is unavailable."
        }
    }

    // MARK: - Internals

    private func discoverDevices() {
        // Per Apple's DeviceType taxonomy on macOS 14+:
        //   .external         — USB capture cards / UVC webcams (what we want)
        //   .continuityCamera — iPhone/iPad acting as a camera (we omit this)
        //   .microphone       — audio inputs incl. HDMI capture-card audio interfaces
        videoDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external],
            mediaType: .video,
            position: .unspecified).devices

        audioDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .microphone],
            mediaType: .audio,
            position: .unspecified).devices
    }

    private func preferredAudioDevice(matching video: AVCaptureDevice?) -> AVCaptureDevice? {
        guard let video else { return nil }
        return audioDevices.first { $0.localizedName == video.localizedName }
    }

    private func configure() async {
        let video = selectedVideoDevice
        let audio = selectedAudioDevice
        let mute = isMuted

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [self] in
                captureSession.beginConfiguration()
                // No sessionPreset on macOS: for .external devices the session honors
                // the device's activeFormat directly, which is the lowest-latency path.

                if let existing = videoInput { captureSession.removeInput(existing) }
                videoInput = nil
                if let device = video,
                   let input = try? AVCaptureDeviceInput(device: device),
                   captureSession.canAddInput(input) {
                    captureSession.addInput(input)
                    videoInput = input
                    lockToMaxFrameRate(device)
                }

                if let existing = audioInput { captureSession.removeInput(existing) }
                audioInput = nil
                if let device = audio,
                   let input = try? AVCaptureDeviceInput(device: device),
                   captureSession.canAddInput(input) {
                    captureSession.addInput(input)
                    audioInput = input
                }

                if !captureSession.outputs.contains(videoDataOutput) {
                    videoDataOutput.setSampleBufferDelegate(sampleDelegate, queue: videoDataQueue)
                    videoDataOutput.alwaysDiscardsLateVideoFrames = true
                    if captureSession.canAddOutput(videoDataOutput) {
                        captureSession.addOutput(videoDataOutput)
                    }
                }
                if !captureSession.outputs.contains(movieOutput),
                   captureSession.canAddOutput(movieOutput) {
                    captureSession.addOutput(movieOutput)
                }
                if !captureSession.outputs.contains(audioPreview),
                   captureSession.canAddOutput(audioPreview) {
                    captureSession.addOutput(audioPreview)
                }
                audioPreview.volume = mute ? 0 : 1

                captureSession.commitConfiguration()
                cont.resume()
            }
        }
    }

    private func lockToMaxFrameRate(_ device: AVCaptureDevice) {
        guard let range = device.activeFormat.videoSupportedFrameRateRanges.first else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeVideoMinFrameDuration = range.minFrameDuration
            device.activeVideoMaxFrameDuration = range.minFrameDuration
        } catch {
            // Some external devices refuse lockForConfiguration; ignore.
        }
    }

    enum CaptureError: LocalizedError {
        case noFrame
        case encodeFailed
        var errorDescription: String? {
            switch self {
            case .noFrame:      return "No video frame available yet."
            case .encodeFailed: return "Failed to encode the captured frame."
            }
        }
    }
}

// MARK: - Sample buffer relay

private final class SampleBufferRelay: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let buffer = Mutex<CMSampleBuffer?>(nil)

    func latest() -> CMSampleBuffer? {
        buffer.withLock { $0 }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                    didOutput sampleBuffer: CMSampleBuffer,
                                    from connection: AVCaptureConnection) {
        buffer.withLock { $0 = sampleBuffer }
    }
}

// MARK: - Recording relay

private final class RecordingRelay: NSObject, AVCaptureFileOutputRecordingDelegate, @unchecked Sendable {
    nonisolated(unsafe) private let onFinish: (Error?) -> Void
    init(onFinish: @escaping (Error?) -> Void) { self.onFinish = onFinish }

    nonisolated func fileOutput(_ output: AVCaptureFileOutput,
                                  didFinishRecordingTo outputFileURL: URL,
                                  from connections: [AVCaptureConnection],
                                  error: Error?) {
        onFinish(error)
    }
}
