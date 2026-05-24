import SwiftUI

struct SpatialSelectionCard: View {
    let title: String
    let subtitle: String
    let url: URL?
    let placeholder: String
    let isProcessing: Bool
    let onPick: () -> Void
    let onClear: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(subtitle).font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                Button(url == nil ? "Choose" : "Replace", action: onPick)
                    .disabled(isProcessing)
            }
            HStack(spacing: 10) {
                Image(systemName: url == nil ? "photo" : "photo.fill")
                    .font(.title2)
                    .foregroundColor(url == nil ? .secondary : .accentColor)
                Text(url?.lastPathComponent ?? placeholder)
                    .foregroundColor(url == nil ? .secondary : .primary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
            }
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

struct SpatialParameterSlider: View {
    let title: String
    let caption: String
    let valueText: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let isDisabled: Bool
    let onChange: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(caption).font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                Text(valueText)
                    .font(.system(.title3, design: .rounded, weight: .semibold))
                    .monospacedDigit()
            }
            Slider(value: $value, in: range, step: step)
                .disabled(isDisabled)
                .onChange(of: value) { _, _ in onChange() }
        }
    }
}

struct SpatialStatusView: View {
    let status: ProcessingStatus

    var body: some View {
        HStack {
            switch status {
            case .pending:
                Label("Ready to create a spatial photo", systemImage: "cube.transparent")
                    .foregroundColor(.secondary)
            case .processing:
                ProgressView("Writing spatial HEIC...")
            case .success(let urls):
                Label("Saved \(urls.first?.lastPathComponent ?? "spatial.heic")", systemImage: "checkmark.circle.fill")
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
