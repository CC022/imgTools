import SwiftUI

struct ContentView: View {
    @Binding var selection: ImageOperation?

    var body: some View {
        NavigationSplitView {
            List(ImageOperation.allCases, id: \.self, selection: $selection) { operation in
                Label(operation.rawValue, systemImage: operation.systemImage)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 200)
        } detail: {
            switch selection {
            case .convert:
                ConvertImagesView()
            case .hdrBoost:
                HDRBoostView()
            case .slicer:
                SlicerView()
            case .video:
                ImagesToVideoView()
            case .panoStitch:
                PanoStitcherView()
            case .hdmiCapture:
                HDMICaptureView()
            case nil:
                Text("Select a tool from the sidebar")
                    .foregroundColor(.secondary)
            }
        }
        .frame(minWidth: 700, minHeight: 450)
    }
}
