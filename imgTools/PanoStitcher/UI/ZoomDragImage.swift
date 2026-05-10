//
//  ZoomDragImage.swift
//  panoDev
//
//  A SwiftUI view that wraps any content and adds pinch/scroll-to-zoom and drag-to-pan.
//  Double-tap resets to the identity transform.
//
//  Usage:
//      ZoomDragImage { Image(nsImage: myImage).resizable().scaledToFit() }
//      ZoomDragImage { anyOtherView }   // works for any content
//

import AppKit
import SwiftUI

struct ZoomDragImage<Content: View>: View {
    private let content: Content

    init(@ViewBuilder _ content: () -> Content) {
        self.content = content()
    }

    @State private var scale:  CGFloat = 1
    @State private var offset: CGSize  = .zero

    private let scaleRange: ClosedRange<CGFloat> = 1...20

    // Live (in-flight) deltas applied while a gesture is active.
    @GestureState private var liveScale: CGFloat = 1
    @GestureState private var liveDrag:  CGSize  = .zero

    var body: some View {
        GeometryReader { geo in
            ZStack {
                content
                    .frame(width: geo.size.width, height: geo.size.height)
                    .scaleEffect(scale * liveScale)
                    // Scale the stored offset by liveScale so the viewport-center
                    // content point stays fixed while the pinch is in progress.
                    .offset(x: offset.width  * liveScale + liveDrag.width,
                            y: offset.height * liveScale + liveDrag.height)
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipped()
            .contentShape(Rectangle())
            .gesture(pinchGesture)
            .simultaneousGesture(dragGesture)
            .overlay {
                ScrollWheelZoomReader { delta, location in
                    zoomByScroll(delta: delta, around: location, in: geo.size)
                }
                .allowsHitTesting(false)
            }
            .onTapGesture(count: 2, perform: reset)
        }
        .clipped()
        .contentShape(Rectangle())
    }

    // MARK: - Gestures

    private var pinchGesture: some Gesture {
        MagnificationGesture()
            .updating($liveScale) { value, state, _ in
                state = value
            }
            .onEnded { value in
                let newScale = (scale * value).clamped(to: scaleRange)
                // Use the actual (post-clamp) factor so offset stays consistent.
                let factor   = newScale / scale
                offset = CGSize(width:  offset.width  * factor,
                                height: offset.height * factor)
                scale = newScale
                if scale <= 1 { offset = .zero }
            }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .updating($liveDrag) { value, state, _ in
                state = value.translation
            }
            .onEnded { value in
                offset.width  += value.translation.width
                offset.height += value.translation.height
            }
    }

    private func reset() {
        withAnimation(.spring(duration: 0.3)) {
            scale  = 1
            offset = .zero
        }
    }

    private func zoomByScroll(delta: CGFloat, around location: CGPoint, in size: CGSize) {
        guard size.width > 0, size.height > 0, delta != 0 else { return }

        let oldScale = scale
        let factor = exp(delta).clamped(to: 0.75...1.35)
        let newScale = (oldScale * factor).clamped(to: scaleRange)
        guard newScale != oldScale else { return }

        let appliedFactor = newScale / oldScale
        let anchor = CGPoint(x: location.x - size.width / 2,
                             y: location.y - size.height / 2)

        offset = CGSize(
            width: offset.width * appliedFactor + anchor.x * (1 - appliedFactor),
            height: offset.height * appliedFactor + anchor.y * (1 - appliedFactor)
        )
        scale = newScale

        if scale <= scaleRange.lowerBound {
            scale = scaleRange.lowerBound
            offset = .zero
        }
    }
}

// MARK: - Helpers

private struct ScrollWheelZoomReader: NSViewRepresentable {
    var onScroll: (CGFloat, CGPoint) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onScroll: onScroll)
    }

    func makeNSView(context: Context) -> ScrollWheelMonitorView {
        let view = ScrollWheelMonitorView()
        context.coordinator.view = view
        context.coordinator.installMonitor()
        return view
    }

    func updateNSView(_ nsView: ScrollWheelMonitorView, context: Context) {
        context.coordinator.onScroll = onScroll
        context.coordinator.view = nsView
        context.coordinator.installMonitor()
    }

    static func dismantleNSView(_ nsView: ScrollWheelMonitorView, coordinator: Coordinator) {
        coordinator.removeMonitor()
    }

    final class Coordinator {
        var onScroll: (CGFloat, CGPoint) -> Void
        weak var view: ScrollWheelMonitorView?

        private var monitor: Any?

        init(onScroll: @escaping (CGFloat, CGPoint) -> Void) {
            self.onScroll = onScroll
        }

        func installMonitor() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self,
                      let view = self.view,
                      event.window === view.window else {
                    return event
                }

                let location = view.convert(event.locationInWindow, from: nil)
                guard view.bounds.contains(location) else { return event }

                let sensitivity: CGFloat = event.hasPreciseScrollingDeltas ? 0.006 : 0.12
                let delta = event.scrollingDeltaY * sensitivity
                guard delta != 0 else { return event }

                self.onScroll(delta, location)
                return nil
            }
        }

        func removeMonitor() {
            if let monitor {
                NSEvent.removeMonitor(monitor)
                self.monitor = nil
            }
        }

        deinit {
            removeMonitor()
        }
    }
}

final class ScrollWheelMonitorView: NSView {
    override var isFlipped: Bool { true }

    override func hitTest(_ point: NSPoint) -> NSView? {
        nil
    }
}

private extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
