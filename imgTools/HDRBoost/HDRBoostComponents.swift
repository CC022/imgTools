import SwiftUI

struct HDRSelectionCard: View {
    let title: String
    let url: URL?
    let placeholder: String
    let isProcessing: Bool
    let onPick: () -> Void
    let onClear: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                Button(url == nil ? "Choose" : "Replace", action: onPick)
                    .disabled(isProcessing)
            }
            Text(url?.lastPathComponent ?? placeholder)
                .foregroundColor(url == nil ? .secondary : .primary)
                .lineLimit(2)
            if url != nil {
                Button("Clear", role: .destructive, action: onClear)
                    .disabled(isProcessing)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
}

struct HDRStatusView: View {
    let status: ProcessingStatus

    var body: some View {
        HStack {
            switch status {
            case .pending:
                Label("Ready to create an HDR HEIF image", systemImage: "sparkles")
                    .foregroundColor(.secondary)
            case .processing:
                ProgressView("Creating HDR HEIF...")
            case .success(let urls):
                Label("Saved to \(urls.first?.lastPathComponent ?? "output.heif")", systemImage: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .failed(let error):
                Label(error, systemImage: "xmark.circle.fill")
                    .foregroundColor(.red)
            }
            Spacer()
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
}

struct HDRPreviewRequest: Equatable {
    let inputURL: URL?
    let maskURL: URL?
    let boost: Double
}
