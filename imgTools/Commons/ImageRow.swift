import SwiftUI

struct ImageRow: View {
    let item: ImageItem
    let onDelete: () -> Void

    var body: some View {
        HStack {
            Image(systemName: "photo")
                .foregroundColor(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text(item.url.lastPathComponent)
                    .lineLimit(1)
                statusView
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
            statusIcon

            Button(action: onDelete) {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .help("Remove image")
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
    }

    @ViewBuilder
    var statusView: some View {
        switch item.status {
        case .pending:
            Text("Ready")
        case .processing:
            Text("Processing...")
        case .success(let urls):
            if urls.count == 1 {
                Text("Saved to: \(urls[0].lastPathComponent)")
            } else {
                Text("Created \(urls.count) files: \(urls.map { $0.lastPathComponent }.joined(separator: ", "))")
            }
        case .failed(let error):
            Text("Error: \(error)").foregroundColor(.red)
        }
    }

    @ViewBuilder
    var statusIcon: some View {
        switch item.status {
        case .pending:
            Image(systemName: "clock")
        case .processing:
            ProgressView().scaleEffect(0.5)
        case .success:
            Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundColor(.red)
        }
    }
}
