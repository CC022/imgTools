import SwiftUI

struct ContentView: View {
    @State private var selection: ImageOperation? = .panoStitch

    var body: some View {
        NavigationSplitView {
            List(ImageOperation.allCases, id: \.self, selection: $selection) { operation in
                Label(operation.rawValue, systemImage: operation.systemImage)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 200)
        } detail: {
            switch selection {
            case .exr:
                ConvertEXRView()
            case .heif:
                ConvertHEIFView()
            case .hdrBoost:
                HDRBoostView()
            case .slicer:
                SlicerView()
            case .video:
                ImagesToVideoView()
            case .panoStitch:
                PanoStitcherView()
            case nil:
                Text("Select a tool from the sidebar")
                    .foregroundColor(.secondary)
            }
        }
        .frame(minWidth: 700, minHeight: 450)
    }
}
