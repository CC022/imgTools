import SwiftUI

struct PanoStitcherView: View {
    var body: some View {
        ImageBatchProcessView { url, outputFolder in
            // TODO: Implement panorama stitching
            throw ImageToolsError.processingFailed
        }
    }
}
