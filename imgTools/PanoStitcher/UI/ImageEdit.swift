//
//  ImageEdit.swift
//  panoDev
//
//  Everything related to the image-adjustment feature in one file:
//
//    • `EditParams`      — Swift mirror of the Metal struct.
//    • `CropAspect`      — fixed aspect-ratio crop choices for panoramas.
//    • `ImageEditor`     — runs `applyEditsKernel` over a CanvasBuffer for
//                          export, and trims to a crop rectangle if given.
//    • `EditPanel`       — right-side liquid-glass UI: Crop section + sliders
//                          + three-zone Color Balance wheels.
//    • `ColorWheelView`  — circular RGB tint control used by Color Balance.
//
//  The adjustment math itself lives once, in `Metal/MetalImageDisplay.metal`,
//  and is shared by the display fragment shader and the export compute
//  kernel — preview and exported pixels are bit-identical for any given
//  EditParams. Crop is applied separately at save time (CPU memcpy on a
//  shared rgba16f buffer); the screen-space dim/frame in the canvas view is
//  pure SwiftUI overlay.
//

import Metal
import SwiftUI
import Foundation
import simd

// MARK: - EditParams (memory layout matches the Metal `EditParams` struct)
//
// 10 scalar floats followed by three SIMD3<Float> colour-balance tints.
// Swift inserts 8 bytes of pad after the scalars so the first SIMD3 is
// 16-byte aligned, matching what the Metal compiler does for the
// trailing `float3` fields. Total stride: 96 bytes both sides.

struct EditParams: Equatable, Sendable {
    var exposure:    Float = 0   // stops
    var highlights:  Float = 0
    var shadows:     Float = 0
    var brightness:  Float = 0
    var contrast:    Float = 0
    var blackPoint:  Float = 0
    var saturation:  Float = 0
    var vibrance:    Float = 0
    var temperature: Float = 0
    var tint:        Float = 0

    /// Per-region linear-RGB tints. Each component is a *shift* added to
    /// the pixel after weighting by region (shadows for low luma,
    /// midtones for a mid bell, highlights for high luma). Typical
    /// range after wheel mapping: about ±0.1 per channel.
    var shadowsTint:    SIMD3<Float> = .zero
    var midtonesTint:   SIMD3<Float> = .zero
    var highlightsTint: SIMD3<Float> = .zero

    /// Per-hue HSL biases. Layout mirrors the Metal `SelectiveColorParams`
    /// struct (8 × float3 + float, with 12 bytes of trailing pad).
    var selective: SelectiveColorParams = .init()

    static let identity = EditParams()
    var isIdentity: Bool { self == .identity }
}

// MARK: - SelectiveColorParams
//
// Eight hue bands centered at the standard photo-app hues. For each band
// the user controls (hue shift, saturation, luminance), packed as a
// SIMD3<Float>. A single `range` slider scales the bandwidth of all
// bands uniformly — narrow ranges target a single colour, wide ranges
// blend across neighbours.

struct SelectiveColorParams: Equatable, Sendable {
    var red:     SIMD3<Float> = .zero
    var orange:  SIMD3<Float> = .zero
    var yellow:  SIMD3<Float> = .zero
    var green:   SIMD3<Float> = .zero
    var aqua:    SIMD3<Float> = .zero
    var blue:    SIMD3<Float> = .zero
    var purple:  SIMD3<Float> = .zero
    var magenta: SIMD3<Float> = .zero
    var range:   Float        = 1.0

    static let count = 8

    /// Indexed access into the eight bands, in the same order as the
    /// shader's `kSelHueCenters` array. 0 = red, 7 = magenta.
    subscript(i: Int) -> SIMD3<Float> {
        get {
            switch i {
            case 0: return red
            case 1: return orange
            case 2: return yellow
            case 3: return green
            case 4: return aqua
            case 5: return blue
            case 6: return purple
            default: return magenta
            }
        }
        set {
            switch i {
            case 0: red = newValue
            case 1: orange = newValue
            case 2: yellow = newValue
            case 3: green = newValue
            case 4: aqua = newValue
            case 5: blue = newValue
            case 6: purple = newValue
            default: magenta = newValue
            }
        }
    }

    var isIdentity: Bool { self == SelectiveColorParams() }
}

// MARK: - CropAspect

/// Fixed aspect-ratio choices for panorama cropping. Held outside
/// `EditParams` because the crop is not a per-pixel kernel parameter —
/// it's a buffer trim applied at save time and a screen-space dim
/// applied at preview time.
enum CropAspect: String, Equatable, Sendable, CaseIterable, Identifiable {
    case none, threeToOne, fiveToOne, fourToThree, threeToFour

    var id: String { rawValue }

    /// `width / height`. Nil for `.none`.
    var ratio: CGFloat? {
        switch self {
        case .none:        return nil
        case .threeToOne:  return 3
        case .fiveToOne:   return 5
        case .fourToThree: return 4.0 / 3.0
        case .threeToFour: return 3.0 / 4.0
        }
    }

    var label: String {
        switch self {
        case .none:        return "None"
        case .threeToOne:  return "3:1"
        case .fiveToOne:   return "5:1"
        case .fourToThree: return "4:3"
        case .threeToFour: return "3:4"
        }
    }
}

// MARK: - ImageEditor (export-side dispatch)

enum ImageEditor {

    /// Bake `params` into a freshly allocated CanvasBuffer of the same
    /// dimensions as `src`. Identity params skip the kernel and return `src`.
    static func apply(_ params: EditParams,
                      to src: CanvasBuffer,
                      storage: MTLStorageMode = .private) -> CanvasBuffer? {
        if params.isIdentity { return src }

        let ctx = PanoContext.shared
        guard let pso = ctx.loadPSO("applyEditsKernel"),
              let dst = ctx.makeHalfBuffer(width:  src.width,
                                            height: src.height,
                                            storage: storage)
        else { return nil }

        var p = params
        Compute.run { cb in
            withUnsafeBytes(of: &p) { raw in
                Compute.encode(cb, pso,
                    buffers: [src.buffer, dst.buffer],
                    dims:    [src.dimsPacked],
                    bytes:   [(raw.baseAddress!, MemoryLayout<EditParams>.stride)],
                    gridW:   src.width,
                    gridH:   src.height)
            }
        }
        return dst
    }

    /// Copy the integer-rounded sub-rectangle `rect` (canvas-pixel coords)
    /// of `src` into a fresh shared rgba16f buffer. Returns `src` if the
    /// rect equals the full buffer (or is empty / out of range).
    static func crop(_ src: CanvasBuffer, to rect: CGRect) -> CanvasBuffer? {
        let x = max(0, Int(rect.origin.x.rounded()))
        let y = max(0, Int(rect.origin.y.rounded()))
        let w = min(src.width  - x, Int(rect.size.width.rounded()))
        let h = min(src.height - y, Int(rect.size.height.rounded()))
        guard w > 0, h > 0 else { return src }
        if x == 0, y == 0, w == src.width, h == src.height { return src }
        precondition(src.bytesPerPixel == 8, "crop() expects rgba16f buffer")

        guard let dst = PanoContext.shared.makeHalfBuffer(width: w, height: h,
                                                          storage: .shared)
        else { return nil }

        // CPU memcpy — buffer is .shared; one row at a time. For the
        // panorama-typical 16384-px-wide canvas this is < 1 ms even at
        // export resolution; no need to spin up a Metal kernel.
        let srcBase = src.buffer.contents()
        let dstBase = dst.buffer.contents()
        let rowBytes = w * src.bytesPerPixel
        for row in 0 ..< h {
            let srcOffset = ((y + row) * src.width + x) * src.bytesPerPixel
            let dstOffset = row * rowBytes
            memcpy(dstBase.advanced(by: dstOffset),
                   srcBase.advanced(by: srcOffset),
                   rowBytes)
        }
        return dst
    }
}

// MARK: - EditPanel (right-side overlay UI)

struct EditPanel: View {

    @Binding var params: EditParams
    /// While true, the canvas previews `EditParams.identity` instead of
    /// `params`, so the user can A/B against the unedited image. Crop is
    /// unaffected — only the per-pixel adjustments are bypassed.
    @Binding var showOriginal: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Adjustments").font(.headline)
                Spacer()
                Button {
                    showOriginal.toggle()
                } label: {
                    Image(systemName: "rectangle.lefthalf.filled")
                        .symbolVariant(showOriginal ? .fill : .none)
                        .foregroundStyle(showOriginal ? Color.accentColor : .primary)
                }
                .buttonStyle(.borderless)
                .help("Show original (compare)")
                .disabled(params.isIdentity)
                .padding(.horizontal)
                Button("Reset") {
                    params = .identity
                }
                .buttonStyle(.borderless)
                .disabled(params.isIdentity)
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

                VStack(alignment: .leading, spacing: 14) {
                    slidersSection
                    Divider()
                    colorBalanceSection
                    Divider()
                    SelectiveColorSection(selective: $params.selective)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 12)
        }
        .frame(width: 320)
        .glassEffect(in: .rect(cornerRadius: 10))
        .padding()
    }

    // MARK: Sliders

    @ViewBuilder
    private var slidersSection: some View {
        VStack(spacing: 1) {
            SliderRow(label: "Exposure",    value: $params.exposure,    range: -6 ... 6, step: 0.05)
            SliderRow(label: "Highlights",  value: $params.highlights,  range: -1 ... 1)
            SliderRow(label: "Shadows",     value: $params.shadows,     range: -1 ... 1)
            SliderRow(label: "Brightness",  value: $params.brightness,  range: -1 ... 1)
            SliderRow(label: "Contrast",    value: $params.contrast,    range: -1 ... 1)
            SliderRow(label: "Black Point", value: $params.blackPoint,  range: -1 ... 1)
            SliderRow(label: "Saturation",  value: $params.saturation,  range: -1 ... 1)
            SliderRow(label: "Vibrance",    value: $params.vibrance,    range: -1 ... 1)
            SliderRow(label: "Temperature", value: $params.temperature, range: -1 ... 1)
            SliderRow(label: "Tint",        value: $params.tint,        range: -1 ... 1)
        }
    }

    // MARK: Color balance

    @ViewBuilder
    private var colorBalanceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Color Balance")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset") {
                    params.shadowsTint    = .zero
                    params.midtonesTint   = .zero
                    params.highlightsTint = .zero
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
                .disabled(params.shadowsTint == .zero
                          && params.midtonesTint == .zero
                          && params.highlightsTint == .zero)
            }

            HStack(spacing: 14) {
                ColorWheelView(label: "Shadows",    tint: $params.shadowsTint)
                ColorWheelView(label: "Midtones",   tint: $params.midtonesTint)
                ColorWheelView(label: "Highlights", tint: $params.highlightsTint)
            }
            .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

// MARK: - SelectiveColorSection
//
// Eight hue swatches (red→magenta) act as a single-select picker; the
// three sliders below operate on whichever band is currently selected,
// so a single pass of UI controls the whole 8×3 matrix. The Range
// slider scales the bandwidth of all bands uniformly.

struct SelectiveColorSection: View {

    @Binding var selective: SelectiveColorParams
    @State private var selected: Int = 0

    private static let labels = ["Red", "Orange", "Yellow", "Green",
                                  "Aqua", "Blue", "Purple", "Magenta"]
    /// Display swatches — match the conceptual centre hues of the bands
    /// in the shader. Pure HSB colours so the picker reads as a hue strip.
    private static let swatches: [Color] = [
        Color(hue: 0/360,   saturation: 0.95, brightness: 0.95),
        Color(hue: 30/360,  saturation: 0.95, brightness: 0.95),
        Color(hue: 60/360,  saturation: 0.95, brightness: 0.95),
        Color(hue: 120/360, saturation: 0.85, brightness: 0.80),
        Color(hue: 180/360, saturation: 0.85, brightness: 0.85),
        Color(hue: 220/360, saturation: 0.90, brightness: 0.90),
        Color(hue: 280/360, saturation: 0.85, brightness: 0.90),
        Color(hue: 320/360, saturation: 0.90, brightness: 0.90),
    ]

    private var hueBinding: Binding<Float> {
        Binding(get: { selective[selected].x },
                set: { selective[selected] = SIMD3<Float>($0, selective[selected].y, selective[selected].z) })
    }
    private var satBinding: Binding<Float> {
        Binding(get: { selective[selected].y },
                set: { selective[selected] = SIMD3<Float>(selective[selected].x, $0, selective[selected].z) })
    }
    private var lumBinding: Binding<Float> {
        Binding(get: { selective[selected].z },
                set: { selective[selected] = SIMD3<Float>(selective[selected].x, selective[selected].y, $0) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Selective Color")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset") { selective = SelectiveColorParams() }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .disabled(selective.isIdentity)
            }

            // Eight swatches in a row. Selected swatch gets a brighter ring;
            // an inner dot appears on bands that have non-zero adjustments
            // so the user can see which colours are actually in use.
            HStack(spacing: 6) {
                ForEach(0 ..< SelectiveColorParams.count, id: \.self) { i in
                    Button {
                        selected = i
                    } label: {
                        ZStack {
                            Circle()
                                .fill(Self.swatches[i])
                            Circle()
                                .strokeBorder(
                                    selected == i ? Color.white : Color.white.opacity(0.2),
                                    lineWidth: selected == i ? 2 : 1)
                            if selective[i] != .zero {
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 4, height: 4)
                            }
                        }
                        .frame(width: 22, height: 22)
                    }
                    .buttonStyle(.plain)
                    .help(Self.labels[i])
                }
            }
            .frame(maxWidth: .infinity, alignment: .center)

            VStack(spacing: 1) {
                SliderRow(label: "Hue",        value: hueBinding, range: -1 ... 1)
                SliderRow(label: "Saturation", value: satBinding, range: -1 ... 1)
                SliderRow(label: "Luminance",  value: lumBinding, range: -1 ... 1)
                SliderRow(label: "Range",      value: $selective.range, range: 0 ... 2, step: 0.05)
            }
        }
    }
}

// MARK: - SliderRow

/// Photos-style horizontal slider:
/// pill background, label on the left, current value on the right, a faint
/// vertical "zero" tick in the middle, and a brighter thumb that moves with
/// `value`. Drag anywhere on the row to scrub; double-tap to reset to the
/// midpoint (zero for symmetric ranges).
private struct SliderRow: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var step: Float = 0.01

    private var midpoint: Float { (range.lowerBound + range.upperBound) / 2 }

    var body: some View {
        GeometryReader { geo in
            let w   = geo.size.width
            let h   = geo.size.height
            let pad: CGFloat = 14
            let trackW = max(w - pad * 2, 1)
            let span   = CGFloat(range.upperBound - range.lowerBound)

            let progress = min(max(CGFloat(value - range.lowerBound) / span, 0), 1)
            let centerP  = CGFloat(midpoint - range.lowerBound) / span
            let thumbX   = pad + progress * trackW
            let centerX  = pad + centerP  * trackW

            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(0.07))

                Rectangle()
                    .fill(Color.white.opacity(0.30))
                    .frame(width: 2, height: h * 0.45)
                    .position(x: centerX, y: h / 2)

                HStack {
                    Text(label).font(.system(size: 13))
                    Spacer(minLength: 0)
                    Text(value, format: .number.precision(.fractionLength(2)))
                        .font(.system(size: 13).monospacedDigit())
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, pad)
                .allowsHitTesting(false)

                Capsule()
                    .fill(Color.white.opacity(0.85))
                    .frame(width: 3, height: h * 0.55)
                    .position(x: thumbX, y: h / 2)
            }
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .onTapGesture(count: 2) { value = midpoint }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        let p = min(max((drag.location.x - pad) / trackW, 0), 1)
                        let raw = Float(p) * (range.upperBound - range.lowerBound)
                                + range.lowerBound
                        let snapped = (raw / step).rounded() * step
                        value = min(max(snapped, range.lowerBound), range.upperBound)
                    }
            )
        }
        .frame(height: 34)
    }
}

// MARK: - ColorWheelView
//
// A DaVinci-style wheel: drag the white thumb anywhere inside the disk to
// push pixel colour toward whatever hue the thumb sits over, with strength
// proportional to its distance from the centre. Hue around the angle,
// saturation along the radius. Double-tap to reset to neutral.
//
// The disk's visual layout and the maths are kept consistent: the AngularGradient
// starts at 12 o'clock with red and goes clockwise; the tint is computed from
// the same screen-space angle and the same hue→RGB function, so what the user
// sees under the thumb is what the pixels move toward.
//
// Output `tint` is a linear-RGB shift (the wheel's hue colour minus neutral
// grey, scaled by radius and `kStrength`). Default range: about ±0.1 per
// channel at the disk's edge.

private let kWheelStrength: Float = 0.12

struct ColorWheelView: View {

    let label: String
    @Binding var tint: SIMD3<Float>

    private let diameter: CGFloat = 88
    private let thumbDiameter: CGFloat = 12

    /// Internal wheel position in normalised disk coords [-1, 1] × [-1, 1].
    /// y is screen-down (matches drag gesture coords).
    @State private var pos: CGPoint = .zero

    var body: some View {
        VStack(spacing: 6) {
            ZStack {
                wheelDisk
                Circle()
                    .strokeBorder(Color.white.opacity(0.25), lineWidth: 0.5)
                Circle()
                    .strokeBorder(Color.black.opacity(0.5), lineWidth: 0.5)
                    .padding(0.5)
                Circle()
                    .fill(Color.white.opacity(0.6))
                    .frame(width: 4, height: 4)
                Circle()
                    .fill(Color.white)
                    .overlay(Circle().strokeBorder(Color.black.opacity(0.4), lineWidth: 0.5))
                    .frame(width: thumbDiameter, height: thumbDiameter)
                    .position(thumbPosition)
                    .shadow(color: .black.opacity(0.5), radius: 1.5, y: 1)
            }
            .frame(width: diameter, height: diameter)
            .contentShape(Circle())
            .onTapGesture(count: 2) {
                pos = .zero
                tint = .zero
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        let center = CGPoint(x: diameter / 2, y: diameter / 2)
                        let radius = (diameter - thumbDiameter) / 2
                        let dx = drag.location.x - center.x
                        let dy = drag.location.y - center.y
                        let r = sqrt(dx * dx + dy * dy)
                        let s = (r > radius) ? (radius / r) : 1
                        let nx = dx * s / radius
                        let ny = dy * s / radius
                        pos = CGPoint(x: nx, y: ny)
                        tint = computeTint(nx: Float(nx), ny: Float(ny))
                    }
            )
            .onChange(of: tint) { _, new in
                // External reset (e.g. "Reset" button) → snap thumb home.
                if new == .zero { pos = .zero }
            }

            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Geometry

    private var thumbPosition: CGPoint {
        let center = CGPoint(x: diameter / 2, y: diameter / 2)
        let radius = (diameter - thumbDiameter) / 2
        return CGPoint(x: center.x + pos.x * radius,
                       y: center.y + pos.y * radius)
    }

    // MARK: Visuals

    @ViewBuilder
    private var wheelDisk: some View {
        // Hue ring (rainbow around the circumference). The startAngle = -90°
        // anchor puts the gradient's first stop at 12 o'clock so the
        // visible colour at the top is red, matching `computeTint` below.
        let hue = AngularGradient(
            gradient: Gradient(colors: [
                .red, .yellow, .green, .cyan, .blue, .purple, .red
            ]),
            center: .center,
            startAngle: .degrees(-90),
            endAngle: .degrees(270)
        )
        // Saturation falloff: white centre fading to clear at the edge,
        // composited over the hue so the disk lightens toward the middle.
        let sat = RadialGradient(
            gradient: Gradient(colors: [.white, .white.opacity(0)]),
            center: .center,
            startRadius: 0,
            endRadius: diameter * 0.52
        )

        ZStack {
            Circle().fill(hue)
            Circle().fill(sat).blendMode(.normal)
        }
    }

    // MARK: Tint maths

    /// Convert a normalised wheel position (-1…1, with y growing downward)
    /// into a linear-RGB tint that visually matches the colour under the
    /// thumb. Convention: the AngularGradient is anchored at 12 o'clock
    /// with red, going clockwise — so straight-up = red, right = yellow,
    /// down = cyan, left = blue. We compute the gradient angle from the
    /// drag offset and feed it into HSV→RGB; the tint is `(rgb - 0.5)·2`
    /// scaled by radius and `kWheelStrength`.
    private func computeTint(nx: Float, ny: Float) -> SIMD3<Float> {
        let r = min(1, sqrt(nx * nx + ny * ny))
        guard r > 1e-4 else { return .zero }

        // atan2 gives angle CCW from +x in math; on screen y grows down,
        // so the angle in screen sense is atan2(ny, nx).
        // Map to gradient-angle (0 = 12 o'clock, CW positive):
        //     gradAngle = π/2 + atan2(ny, nx)
        var gradAngle = .pi / 2 + atan2(ny, nx)
        // Normalise to [0, 2π).
        let twoPi = 2 * Float.pi
        gradAngle = (gradAngle.truncatingRemainder(dividingBy: twoPi) + twoPi)
                    .truncatingRemainder(dividingBy: twoPi)
        let hue = gradAngle / twoPi  // [0, 1)

        let rgb = hsvToRGB(h: hue, s: 1, v: 1)
        // (rgb - 0.5) gives an axis around grey; ×2 maps a fully saturated
        // primary to a unit-magnitude direction, then `r * strength` scales.
        return (rgb - SIMD3<Float>(repeating: 0.5)) * 2 * r * kWheelStrength
    }
}

/// Hue-saturation-value to linear RGB. Standard formula; `h` in [0, 1).
private func hsvToRGB(h: Float, s: Float, v: Float) -> SIMD3<Float> {
    let h6 = h * 6
    let i  = floor(h6)
    let f  = h6 - i
    let p  = v * (1 - s)
    let q  = v * (1 - s * f)
    let t  = v * (1 - s * (1 - f))
    switch Int(i) % 6 {
    case 0: return SIMD3(v, t, p)
    case 1: return SIMD3(q, v, p)
    case 2: return SIMD3(p, v, t)
    case 3: return SIMD3(p, q, v)
    case 4: return SIMD3(t, p, v)
    default: return SIMD3(v, p, q)
    }
}

// MARK: - CropOverlay
//
// Pure SwiftUI overlay placed above the MetalZoomImageView while a crop
// aspect is active. Draws a darkened mask outside a centred aspect-ratio
// rectangle plus a thin frame outline. Doesn't intercept any input — pan
// and zoom continue to work directly on the metal view underneath.

struct CropOverlay: View {

    let aspect: CropAspect

    var body: some View {
        GeometryReader { geo in
            if let frame = cropFrame(in: geo.size) {
                ZStack {
                    // Dim everything outside `frame`. We draw a full-size
                    // black overlay then punch a hole in it via blendMode,
                    // so the crop region shows the underlying metal view
                    // at full brightness.
                    Rectangle()
                        .fill(Color.black.opacity(0.45))
                        .overlay(
                            Rectangle()
                                .frame(width: frame.width, height: frame.height)
                                .position(x: frame.midX, y: frame.midY)
                                .blendMode(.destinationOut)
                        )
                        .compositingGroup()
                        .ignoresSafeArea()

                    // Frame outline + rule-of-thirds gridlines, kept thin
                    // so they don't dominate the canvas.
                    cropFrameOverlay(frame: frame)
                }
                .allowsHitTesting(false)
            }
        }
    }

    /// Centred aspect-ratio rect that fits inside `size` with a small inset.
    private func cropFrame(in size: CGSize) -> CGRect? {
        guard let ratio = aspect.ratio,
              size.width > 0, size.height > 0 else { return nil }
        let pad: CGFloat = 24
        let availW = max(1, size.width - 2 * pad)
        let availH = max(1, size.height - 2 * pad)
        let w: CGFloat
        let h: CGFloat
        if availW / availH > ratio {
            // Limited by available height.
            h = availH
            w = h * ratio
        } else {
            // Limited by available width.
            w = availW
            h = w / ratio
        }
        return CGRect(
            x: (size.width  - w) / 2,
            y: (size.height - h) / 2,
            width: w, height: h
        )
    }

    @ViewBuilder
    private func cropFrameOverlay(frame: CGRect) -> some View {
        ZStack {
            Rectangle()
                .strokeBorder(Color.white.opacity(0.85), lineWidth: 1)
                .frame(width: frame.width, height: frame.height)
                .position(x: frame.midX, y: frame.midY)

            // Rule of thirds.
            Path { p in
                let x0 = frame.minX, y0 = frame.minY
                let w = frame.width, h = frame.height
                p.move(to: CGPoint(x: x0 + w / 3, y: y0))
                p.addLine(to: CGPoint(x: x0 + w / 3, y: y0 + h))
                p.move(to: CGPoint(x: x0 + 2 * w / 3, y: y0))
                p.addLine(to: CGPoint(x: x0 + 2 * w / 3, y: y0 + h))
                p.move(to: CGPoint(x: x0,     y: y0 + h / 3))
                p.addLine(to: CGPoint(x: x0 + w, y: y0 + h / 3))
                p.move(to: CGPoint(x: x0,     y: y0 + 2 * h / 3))
                p.addLine(to: CGPoint(x: x0 + w, y: y0 + 2 * h / 3))
            }
            .stroke(Color.white.opacity(0.25), lineWidth: 0.5)
        }
    }
}

// MARK: - Canvas-space crop math

/// State the canvas view publishes back to the store: viewport size and
/// current pan/zoom transform, in *points* (matching the way the metal
/// view stores them). All-zero is the reset state.
struct MetalCanvasTransform: Equatable, Sendable {
    var viewSize: CGSize = .zero
    var offset:   CGSize = .zero
    var scale:    CGFloat = 1

    static let identity = MetalCanvasTransform()
}

/// Compute the canvas-pixel rectangle that's currently visible inside
/// the screen-space crop frame, given a transform and image dimensions.
/// Returns nil if the geometry isn't valid.
///
/// Mirrors the maths in `MetalImageDisplay.metal::metalImageFragment` so
/// the saved-region matches what the user sees during framing, modulo
/// rounding to integer pixels.
func canvasCropRect(aspect: CropAspect,
                    transform t: MetalCanvasTransform,
                    canvasSize: CGSize) -> CGRect? {
    guard let ratio = aspect.ratio,
          t.viewSize.width > 0, t.viewSize.height > 0,
          canvasSize.width > 0, canvasSize.height > 0
    else { return nil }

    // The CropOverlay positions the screen frame the same way:
    let pad: CGFloat = 24
    let availW = max(1, t.viewSize.width  - 2 * pad)
    let availH = max(1, t.viewSize.height - 2 * pad)
    let frameW: CGFloat
    let frameH: CGFloat
    if availW / availH > ratio {
        frameH = availH; frameW = frameH * ratio
    } else {
        frameW = availW; frameH = frameW / ratio
    }
    let frameMinX = (t.viewSize.width  - frameW) / 2
    let frameMinY = (t.viewSize.height - frameH) / 2

    // Mirror the fragment shader's image-position maths:
    //   fit = min(viewSize / imageSize)
    //   displayed = imageSize * fit * scale
    //   center    = viewSize/2 + offset
    //   topLeft   = center - displayed/2
    let fit = min(t.viewSize.width  / canvasSize.width,
                  t.viewSize.height / canvasSize.height)
    let displayedW = canvasSize.width  * fit * t.scale
    let displayedH = canvasSize.height * fit * t.scale
    guard displayedW > 0, displayedH > 0 else { return nil }
    let centerX = t.viewSize.width  / 2 + t.offset.width
    let centerY = t.viewSize.height / 2 + t.offset.height
    let topLeftX = centerX - displayedW / 2
    let topLeftY = centerY - displayedH / 2

    // Map screen frame corners back to canvas pixels, then clamp to the
    // canvas. (If the canvas doesn't fully cover the frame, we crop only
    // the covered region.)
    let uv0x = (frameMinX - topLeftX) / displayedW
    let uv0y = (frameMinY - topLeftY) / displayedH
    let uv1x = (frameMinX + frameW - topLeftX) / displayedW
    let uv1y = (frameMinY + frameH - topLeftY) / displayedH

    let x0 = max(0, min(canvasSize.width,  uv0x * canvasSize.width))
    let y0 = max(0, min(canvasSize.height, uv0y * canvasSize.height))
    let x1 = max(0, min(canvasSize.width,  uv1x * canvasSize.width))
    let y1 = max(0, min(canvasSize.height, uv1y * canvasSize.height))

    let w = x1 - x0
    let h = y1 - y0
    guard w > 1, h > 1 else { return nil }
    return CGRect(x: x0, y: y0, width: w, height: h)
}

/// Same as `canvasCropRect` but expressed as fractions of `canvasSize`
/// (origin and size in 0…1). Stable across viewport resizes — this is what
/// the store persists when the user exits crop mode.
func normalizedCropRect(aspect: CropAspect,
                        transform t: MetalCanvasTransform,
                        canvasSize: CGSize) -> CGRect? {
    guard let r = canvasCropRect(aspect: aspect,
                                  transform: t,
                                  canvasSize: canvasSize),
          canvasSize.width > 0, canvasSize.height > 0 else { return nil }
    return CGRect(x: r.origin.x / canvasSize.width,
                  y: r.origin.y / canvasSize.height,
                  width:  r.size.width  / canvasSize.width,
                  height: r.size.height / canvasSize.height)
}

/// Inverse of `canvasCropRect` for the framing axis: given a saved
/// normalized canvas-space rect and the current viewport, return the
/// (offset, scale) that places that canvas region inside the screen-space
/// crop frame for `aspect`. Returns nil if the geometry isn't valid.
///
/// Used when re-entering crop mode so the visible frame lines up with the
/// previously saved selection.
func transformForNormalizedRect(_ rect: CGRect,
                                viewSize: CGSize,
                                canvasSize: CGSize,
                                aspect: CropAspect) -> (offset: CGSize, scale: CGFloat)? {
    guard let ratio = aspect.ratio,
          viewSize.width > 0, viewSize.height > 0,
          canvasSize.width > 0, canvasSize.height > 0,
          rect.size.width > 0, rect.size.height > 0
    else { return nil }

    // Mirror the crop-frame layout from `CropOverlay.cropFrame(in:)`.
    let pad: CGFloat = 24
    let availW = max(1, viewSize.width  - 2 * pad)
    let availH = max(1, viewSize.height - 2 * pad)
    let frameW: CGFloat
    let frameH: CGFloat
    if availW / availH > ratio {
        frameH = availH; frameW = frameH * ratio
    } else {
        frameW = availW; frameH = frameW / ratio
    }
    let frameMinX = (viewSize.width  - frameW) / 2
    let frameMinY = (viewSize.height - frameH) / 2

    // Forward mapping (from canvasCropRect):
    //   fit         = min(viewSize / canvasSize)
    //   displayedW  = canvasSize.w * fit * scale
    //   topLeft     = viewSize/2 + offset - displayedSize/2
    //   uv0         = (frameMin - topLeft) / displayedSize
    //   x0          = uv0 * canvasSize  →  rect.origin
    //   rect.size   = (frameSize / displayedSize) * canvasSize
    //
    // Solve for scale and offset:
    let fit = min(viewSize.width  / canvasSize.width,
                  viewSize.height / canvasSize.height)
    guard fit > 0 else { return nil }

    // `rect` is normalized (0…1). In the forward mapping
    //   rect.normalized.width = frameW / displayedW
    // so displayedW = frameW / rect.width  →  scale = displayedW / (canvasW * fit).
    let displayedW = frameW / rect.size.width
    let displayedH = frameH / rect.size.height
    let scale = displayedW / (canvasSize.width * fit)

    // From uv0 * canvasSize = rect.origin:
    //   topLeft = frameMin - rect.origin / canvasSize * displayedSize
    let topLeftX = frameMinX - (rect.origin.x / canvasSize.width)  * displayedW
    let topLeftY = frameMinY - (rect.origin.y / canvasSize.height) * displayedH
    let offsetX  = topLeftX + displayedW * 0.5 - viewSize.width  * 0.5
    let offsetY  = topLeftY + displayedH * 0.5 - viewSize.height * 0.5
    return (CGSize(width: offsetX, height: offsetY), scale)
}

// MARK: - CropPanel
//
// Right-side overlay shown only while crop mode is active. Mirrors the
// EditPanel framing so the two panels feel like siblings.

struct CropPanel: View {

    @Binding var aspect: CropAspect

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Crop").font(.headline)
                Spacer()
                Button("Reset") { aspect = .none }
                    .buttonStyle(.borderless)
                    .disabled(aspect == .none)
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            VStack(alignment: .leading, spacing: 10) {
                Picker("Aspect", selection: $aspect) {
                    ForEach(CropAspect.allCases) { a in
                        Text(a.label).tag(a)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                if aspect != .none {
                    Text("Pan and zoom to frame the crop. Closing the panel applies the crop to the preview and the saved file.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Pick an aspect ratio to start cropping.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
        }
        .frame(width: 280)
        .glassEffect(in: .rect(cornerRadius: 10))
        .padding()
    }
}
