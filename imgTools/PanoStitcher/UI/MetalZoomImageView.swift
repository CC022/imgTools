//
//  MetalZoomImageView.swift
//  panoDev
//
//  MTKView-backed HDR image viewer. It samples the panorama's rgba16Float
//  CanvasBuffer directly, avoiding Metal's 2D texture size limits.
//
//  The view publishes its current pan/zoom + viewport size back via the
//  `onTransformChange` callback so callers (e.g. the Crop tool) can map
//  screen-space rectangles back to canvas pixels.
//

import AppKit
import Metal
import MetalKit
import SwiftUI

struct MetalZoomImageView: NSViewRepresentable {
    let buffer: CanvasBuffer
    var editParams: EditParams = .identity
    /// Normalized (0…1) sub-rectangle of `buffer` to render. nil = full image.
    /// When non-nil, pan/zoom operate over the sub-rect (which fits the view).
    var cropRegion: CGRect? = nil
    /// Fired whenever the viewport size, pan offset, or zoom scale change.
    /// Values are in *points* (matching the metal view's internal state).
    var onTransformChange: ((MetalCanvasTransform) -> Void)? = nil
    /// One-shot transform restore (e.g. when re-entering crop mode). The
    /// metal view applies it once per identity change.
    var pendingTransform: PendingTransform? = nil

    struct PendingTransform: Equatable {
        let id: UUID
        let offset: CGSize
        let scale: CGFloat
    }

    func makeNSView(context: Context) -> HDRMetalImageView {
        let view = HDRMetalImageView()
        view.setBuffer(buffer)
        view.setEditParams(editParams)
        view.setCropRegion(cropRegion)
        view.onTransformChange = onTransformChange
        if let pending = pendingTransform {
            view.setTransform(offset: pending.offset, scale: pending.scale)
            context.coordinator.lastAppliedTransformID = pending.id
        }
        return view
    }

    func updateNSView(_ nsView: HDRMetalImageView, context: Context) {
        nsView.setBuffer(buffer)
        nsView.setEditParams(editParams)
        nsView.setCropRegion(cropRegion)
        nsView.onTransformChange = onTransformChange
        if let pending = pendingTransform,
           context.coordinator.lastAppliedTransformID != pending.id {
            nsView.setTransform(offset: pending.offset, scale: pending.scale)
            context.coordinator.lastAppliedTransformID = pending.id
        }
        // Re-emit the current transform on every SwiftUI update so the
        // store stays consistent if e.g. the canvas buffer changes size
        // without the view itself being recreated.
        nsView.publishTransform()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var lastAppliedTransformID: UUID?
    }
}

final class HDRMetalImageView: MTKView {
    private var canvasBuffer: CanvasBuffer?
    private var pipeline: MTLRenderPipelineState?

    private var zoomScale: CGFloat = 1
    private var offset: CGSize = .zero
    private var lastDragLocation: CGPoint?
    private var editParams: EditParams = .identity
    /// Normalized (0…1) sub-rect of `canvasBuffer` to display. nil = full image.
    private var cropRegion: CGRect?

    var onTransformChange: ((MetalCanvasTransform) -> Void)?

    private let scaleRange: ClosedRange<CGFloat> = 0.8...20

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }

    init() {
        let ctx = PanoContext.shared
        super.init(frame: .zero, device: ctx.device)

        delegate = self
        framebufferOnly = true
        isPaused = true
        enableSetNeedsDisplay = true
        colorPixelFormat = .rgba16Float
        clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        sampleCount = 1

        if let metalLayer = layer as? CAMetalLayer {
            metalLayer.pixelFormat = .rgba16Float
            metalLayer.colorspace = PanoContext.linearColorSpace
            metalLayer.wantsExtendedDynamicRangeContent = true
        }

        buildPipeline(device: ctx.device)
    }

    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func setBuffer(_ buffer: CanvasBuffer) {
        guard canvasBuffer?.buffer !== buffer.buffer else { return }
        canvasBuffer = buffer
        reset(redraw: false)
        needsDisplay = true
        publishTransform()
    }

    func setEditParams(_ p: EditParams) {
        guard editParams != p else { return }
        editParams = p
        needsDisplay = true
    }

    func setCropRegion(_ r: CGRect?) {
        guard cropRegion != r else { return }
        cropRegion = r
        // Reset pan/zoom so the new sub-rect (or full image) fits the view
        // cleanly — otherwise an offset snapshot from the old framing would
        // shift the cropped preview off-center.
        zoomScale = 1
        offset = .zero
        needsDisplay = true
        publishTransform()
    }

    /// Programmatically restore pan/zoom (used when re-entering crop mode
    /// to line the frame up with the previously saved region).
    func setTransform(offset newOffset: CGSize, scale newScale: CGFloat) {
        let s = max(scaleRange.lowerBound, min(scaleRange.upperBound, newScale))
        guard zoomScale != s || offset != newOffset else { return }
        zoomScale = s
        offset = newOffset
        needsDisplay = true
        publishTransform()
    }

    /// Push the current transform to the callback. Safe to call frequently;
    /// SwiftUI dedupes via `Equatable` on the receiving end.
    func publishTransform() {
        onTransformChange?(MetalCanvasTransform(
            viewSize: bounds.size,
            offset:   offset,
            scale:    zoomScale
        ))
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        needsDisplay = true
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        needsDisplay = true
        publishTransform()
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if event.clickCount == 2 {
            reset(redraw: true)
            lastDragLocation = nil
        } else {
            lastDragLocation = convert(event.locationInWindow, from: nil)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        if let last = lastDragLocation {
            offset.width += location.x - last.x
            offset.height += location.y - last.y
            needsDisplay = true
            publishTransform()
        }
        lastDragLocation = location
    }

    override func mouseUp(with event: NSEvent) {
        lastDragLocation = nil
    }

    override func scrollWheel(with event: NSEvent) {
        let sensitivity: CGFloat = event.hasPreciseScrollingDeltas ? 0.006 : 0.12
        let delta = event.scrollingDeltaY * sensitivity
        guard delta != 0 else { return }
        let factor = exp(delta).clamped(to: 0.75...1.35)
        zoom(by: factor, around: convert(event.locationInWindow, from: nil))
    }

    override func magnify(with event: NSEvent) {
        zoom(by: (1 + event.magnification).clamped(to: 0.75...1.35),
             around: convert(event.locationInWindow, from: nil))
    }

    private func reset(redraw: Bool) {
        zoomScale = 1
        offset = .zero
        if redraw { needsDisplay = true }
        publishTransform()
    }

    private func zoom(by factor: CGFloat, around location: CGPoint) {
        guard bounds.width > 0, bounds.height > 0, factor > 0 else { return }

        let oldScale = zoomScale
        let newScale = (oldScale * factor).clamped(to: scaleRange)
        guard newScale != oldScale else { return }

        let appliedFactor = newScale / oldScale
        let anchor = CGPoint(x: location.x - bounds.width / 2,
                             y: location.y - bounds.height / 2)
        offset = CGSize(
            width: offset.width * appliedFactor + anchor.x * (1 - appliedFactor),
            height: offset.height * appliedFactor + anchor.y * (1 - appliedFactor)
        )
        zoomScale = newScale

        if zoomScale <= scaleRange.lowerBound {
            zoomScale = scaleRange.lowerBound
            offset = .zero
        }
        needsDisplay = true
        publishTransform()
    }

    private func buildPipeline(device: MTLDevice) {
        guard let library = device.makeDefaultLibrary(),
              let vertex = library.makeFunction(name: "metalImageVertex"),
              let fragment = library.makeFunction(name: "metalImageFragment") else {
            dbg("[MetalZoomImageView] display shader missing")
            return
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertex
        desc.fragmentFunction = fragment
        desc.colorAttachments[0].pixelFormat = colorPixelFormat
        pipeline = try? device.makeRenderPipelineState(descriptor: desc)
    }
}

extension HDRMetalImageView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        needsDisplay = true
    }

    func draw(in view: MTKView) {
        guard let drawable = currentDrawable,
              let renderPass = currentRenderPassDescriptor,
              let pipeline,
              let canvasBuffer,
              let cb = PanoContext.shared.commandQueue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        let drawableSize = view.drawableSize
        let pointSize = bounds.size
        let pixelScale = CGSize(
            width: pointSize.width > 0 ? drawableSize.width / pointSize.width : 1,
            height: pointSize.height > 0 ? drawableSize.height / pointSize.height : 1
        )
        let offsetPixels = CGSize(width: offset.width * pixelScale.width,
                                  height: offset.height * pixelScale.height)

        let bufW = CGFloat(canvasBuffer.width)
        let bufH = CGFloat(canvasBuffer.height)
        let cropOriginPx: CGPoint
        let cropSizePx: CGSize
        if let r = cropRegion {
            cropOriginPx = CGPoint(x: r.origin.x * bufW, y: r.origin.y * bufH)
            cropSizePx   = CGSize(width: r.size.width * bufW, height: r.size.height * bufH)
        } else {
            cropOriginPx = .zero
            cropSizePx   = CGSize(width: bufW, height: bufH)
        }

        var uniforms = MetalImageUniforms(
            viewSize: SIMD2<Float>(Float(drawableSize.width), Float(drawableSize.height)),
            imageSize: SIMD2<Float>(Float(canvasBuffer.width), Float(canvasBuffer.height)),
            offset: SIMD2<Float>(Float(offsetPixels.width), Float(offsetPixels.height)),
            scale: Float(zoomScale),
            cropOrigin: SIMD2<Float>(Float(cropOriginPx.x), Float(cropOriginPx.y)),
            cropSize:   SIMD2<Float>(Float(cropSizePx.width), Float(cropSizePx.height))
        )

        var edits = editParams
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentBuffer(canvasBuffer.buffer, offset: 0, index: 0)
        enc.setFragmentBytes(&uniforms,
                             length: MemoryLayout<MetalImageUniforms>.stride,
                             index: 1)
        enc.setFragmentBytes(&edits,
                             length: MemoryLayout<EditParams>.stride,
                             index: 2)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

private struct MetalImageUniforms {
    var viewSize: SIMD2<Float>
    var imageSize: SIMD2<Float>
    var offset: SIMD2<Float>
    var scale: Float
    var cropOrigin: SIMD2<Float>
    var cropSize: SIMD2<Float>
}

private extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
