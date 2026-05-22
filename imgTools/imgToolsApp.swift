import AVFoundation
import SwiftUI

@main
struct ImageToolsApp: App {
    @State private var hdmiSession = HDMICaptureSession()
    @State private var fullscreenLayer: AVCaptureVideoPreviewLayer?

    var body: some Scene {
        WindowGroup {
            if let layer = fullscreenLayer {
                FullscreenPreviewOverlay(layer: layer) {
                    fullscreenLayer = nil
                }
            } else {
                ContentView()
                    .environment(hdmiSession)
                    .environment(\.presentFullscreenPreview) { layer in
                        fullscreenLayer = layer
                    }
            }
        }
        .windowStyle(.hiddenTitleBar)
    }
}

extension EnvironmentValues {
    @Entry var presentFullscreenPreview: (AVCaptureVideoPreviewLayer) -> Void = { _ in }
}
