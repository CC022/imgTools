import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import Foundation
import Metal
import simd

enum MergeMethod: String, CaseIterable, Sendable {
    case mean
    case max

    var displayName: String {
        switch self {
        case .mean: return "Mean"
        case .max:  return "Max"
        }
    }
}

// MARK: - Load

/// Load each URL as a CIImage with EXIF orientation applied. If `maxDimension`
/// is non-nil each image is uniformly downscaled so its long edge does not
/// exceed that value — used for the snappy preview path.
nonisolated func loadLongExposureImages(_ urls: [URL], maxDimension: CGFloat?) async throws -> [CIImage] {
    try await Task.detached {
        var images: [CIImage] = []
        images.reserveCapacity(urls.count)
        for url in urls {
            guard var ci = CIImage(contentsOf: url, options: [
                .applyOrientationProperty: true
            ]) else {
                throw ImageToolsError.invalidImage
            }
            if let maxDimension {
                let largest = max(ci.extent.width, ci.extent.height)
                if largest > maxDimension {
                    let s = maxDimension / largest
                    ci = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
                }
            }
            // Re-origin so the extent starts at (0, 0); makes accumulation extent
            // math predictable across images of slightly different sizes.
            ci = ci.transformed(by: CGAffineTransform(
                translationX: -ci.extent.minX, y: -ci.extent.minY
            ))
            images.append(ci)
        }
        return images
    }.value
}

// MARK: - Alignment (pano matcher: SuperPoint + LightGlue + RANSAC)

/// Render a CIImage into a fresh rgba16Float Metal texture sized for
/// SuperPoint detection. Detection happens at most `detectionMaxDim`
/// pixels along the long edge — matching PanoPipeline's 2048-px detection
/// rule, since SuperPoint's max-keypoints budget gives poor coverage on
/// huge inputs (each keypoint represents far too many pixels). Dimensions
/// are snapped down to multiples of 8 for the model's pixel-shuffle step.
///
/// Returns the texture wrapper plus the scale factor `texSize / originalCISize`,
/// which the caller uses to map detected keypoint coordinates back into
/// the original CIImage coordinate space before estimating a homography.
nonisolated func longExposureDetectionTexture(
    _ ci: CIImage, detectionMaxDim: CGFloat = 2048
) -> (texture: ImageTexture, scale: CGFloat)? {
    let ctx = PanoContext.shared
    let normalized = ci.transformed(by: CGAffineTransform(
        translationX: -ci.extent.minX, y: -ci.extent.minY
    ))
    let largest = max(normalized.extent.width, normalized.extent.height)
    let scale = min(1.0, detectionMaxDim / max(largest, 1))
    let scaled: CIImage
    if scale < 1.0 {
        scaled = normalized
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(
                translationX: -normalized.extent.minX * scale,
                y: -normalized.extent.minY * scale))
    } else {
        scaled = normalized
    }

    let width  = (Int(scaled.extent.width.rounded())  / 8) * 8
    let height = (Int(scaled.extent.height.rounded()) / 8) * 8
    guard width > 0, height > 0 else { return nil }

    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float,
        width: width, height: height, mipmapped: false
    )
    desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
    desc.storageMode = .private
    guard let texture = ctx.device.makeTexture(descriptor: desc) else { return nil }

    // CIImage origin is bottom-left; Metal texture is top-left → flip on the way in.
    let flipped = scaled.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
        .translatedBy(x: 0, y: -CGFloat(height)))

    ctx.ciContext.render(
        flipped, to: texture, commandBuffer: nil,
        bounds: CGRect(x: 0, y: 0, width: width, height: height),
        colorSpace: PanoContext.linearColorSpace
    )

    let dummyURL = URL(fileURLWithPath: "/tmp/longexp_frame.bin")
    let image = PanoImage(width: width, height: height,
                          sourceURL: dummyURL,
                          colorSpace: PanoContext.linearColorSpace)
    return (ImageTexture(image: image, texture: texture), scale)
}

/// Align every frame to the first frame using SuperPoint keypoint
/// detection + LightGlue matching + RANSAC homography. Frames whose
/// match count or RANSAC fails are returned unchanged.
///
/// `progress` (if provided) is called as `(completed, total)` after each
/// frame, including the reference (which counts as 1 immediately).
nonisolated func alignImagesPanoMatcher(
    _ frames: [CIImage],
    progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
) async -> [CIImage] {
    await Task.detached(priority: .userInitiated) {
        guard let reference = frames.first else { return [] }
        let detector = SuperPointDetector()
        let matcher = LightGlueMatcher()

        guard let (refTex, refScale) = longExposureDetectionTexture(reference) else {
            return frames
        }
        let refKpsTex = detector.detect(in: refTex)
        dbg("[LongExposure] reference: \(refKpsTex.count) kps  detTex=\(refTex.width)x\(refTex.height)  ciSize=\(Int(reference.extent.width))x\(Int(reference.extent.height))  scale=\(refScale)")
        progress?(1, frames.count)
        guard refKpsTex.count >= 20 else {
            dbg("[LongExposure] reference has only \(refKpsTex.count) keypoints — skipping alignment")
            return frames
        }

        // Map ref keypoints from detection-texture space (top-left origin,
        // possibly downsampled) → original CIImage space (bottom-left origin,
        // full resolution). Estimated homographies will then live in original
        // CIImage coords and can be applied to extent corners directly.
        let refCIHeight = Float(reference.extent.height)
        let refInvScale = 1.0 / Float(refScale)
        let refKpsCI: [Keypoint] = refKpsTex.map {
            var k = $0
            k.x = $0.x * refInvScale
            k.y = refCIHeight - $0.y * refInvScale
            return k
        }

        var aligned: [CIImage] = [reference]
        aligned.reserveCapacity(frames.count)
        for i in 1..<frames.count {
            let target = frames[i]
            guard let (tgtTex, tgtScale) = longExposureDetectionTexture(target) else {
                dbg("[LongExposure] frame \(i): detection texture failed; using identity")
                aligned.append(target)
                progress?(i + 1, frames.count)
                continue
            }
            let tgtKps = detector.detect(in: tgtTex)
            let matches = matcher.match(
                kpA: tgtKps, imgWidthA: tgtTex.width, imgHeightA: tgtTex.height,
                kpB: refKpsTex, imgWidthB: refTex.width, imgHeightB: refTex.height
            )
            guard matches.count >= 20 else {
                dbg("[LongExposure] frame \(i): only \(matches.count) matches (kps tgt=\(tgtKps.count) ref=\(refKpsTex.count)), using identity")
                aligned.append(target)
                progress?(i + 1, frames.count)
                continue
            }
            let tgtCIHeight = Float(target.extent.height)
            let tgtInvScale = 1.0 / Float(tgtScale)
            let srcPts = matches.map { m -> SIMD2<Float> in
                let kp = tgtKps[m.indexA]
                return SIMD2(kp.x * tgtInvScale, tgtCIHeight - kp.y * tgtInvScale)
            }
            let dstPts = matches.map { m -> SIMD2<Float> in
                let kp = refKpsCI[m.indexB]
                return SIMD2(kp.x, kp.y)
            }
            let weights = matches.map(\.confidence)
            guard let r = HomographyEstimator.estimate(
                srcPoints: srcPts, dstPoints: dstPts, weights: weights
            ), r.inlierIndices.count >= 12 else {
                dbg("[LongExposure] frame \(i): RANSAC dropped (\(matches.count) matches)")
                aligned.append(target)
                progress?(i + 1, frames.count)
                continue
            }
            dbg("[LongExposure] frame \(i): \(r.inlierIndices.count)/\(matches.count) inliers")
            aligned.append(warpedImage(target, homography: r.homography.matrix))
            progress?(i + 1, frames.count)
        }
        return aligned
    }.value
}

private nonisolated func warpedImage(_ image: CIImage, homography H: matrix_float3x3) -> CIImage {
    let e = image.extent
    let bl = projectPoint(H, CGPoint(x: e.minX, y: e.minY))
    let br = projectPoint(H, CGPoint(x: e.maxX, y: e.minY))
    let tl = projectPoint(H, CGPoint(x: e.minX, y: e.maxY))
    let tr = projectPoint(H, CGPoint(x: e.maxX, y: e.maxY))
    return image.applyingFilter("CIPerspectiveTransform", parameters: [
        "inputTopLeft":     CIVector(cgPoint: tl),
        "inputTopRight":    CIVector(cgPoint: tr),
        "inputBottomLeft":  CIVector(cgPoint: bl),
        "inputBottomRight": CIVector(cgPoint: br),
    ])
}

private nonisolated func projectPoint(_ H: matrix_float3x3, _ p: CGPoint) -> CGPoint {
    let v = SIMD3<Float>(Float(p.x), Float(p.y), 1)
    let r = H * v
    let w = r.z == 0 ? 1 : r.z
    return CGPoint(x: CGFloat(r.x / w), y: CGFloat(r.y / w))
}

// MARK: - Merge

/// Custom CIColorKernel for straight RGB accumulation with forced alpha=1.
///
/// Using `CIAdditionCompositing` instead would accumulate alpha to N, which
/// the unpremultiplication on render then divides RGB by — making the mean
/// output N× too dark. A bespoke kernel sidesteps the entire premul dance.
private let longExposureAddKernel: CIColorKernel? = {
    CIColorKernel(source: """
        kernel vec4 longExpAdd(__sample s, __sample d) {
            return vec4(s.rgb + d.rgb, 1.0);
        }
    """)
}()

/// Merge `frames` in the caller's working color space.
/// `.mean`: per-pixel sum via custom kernel, then divide by N.
/// `.max`:  progressive `CIMaximumCompositing` (alpha behaves correctly here).
nonisolated func mergeImages(_ frames: [CIImage], method: MergeMethod) -> CIImage? {
    guard let first = frames.first else { return nil }
    switch method {
    case .max:
        var acc = first
        for i in 1..<frames.count {
            acc = frames[i].applyingFilter("CIMaximumCompositing", parameters: [
                kCIInputBackgroundImageKey: acc
            ])
        }
        return acc

    case .mean:
        guard let kernel = longExposureAddKernel else { return nil }
        var acc = first
        let extent = first.extent
        for i in 1..<frames.count {
            if let next = kernel.apply(extent: extent, arguments: [frames[i], acc]) {
                acc = next
            }
        }
        let inv = CGFloat(1) / CGFloat(frames.count)
        return acc.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: inv, y: 0,   z: 0,   w: 0),
            "inputGVector": CIVector(x: 0,   y: inv, z: 0,   w: 0),
            "inputBVector": CIVector(x: 0,   y: 0,   z: inv, w: 0),
            "inputAVector": CIVector(x: 0,   y: 0,   z: 0,   w: 1),
        ])
    }
}

// MARK: - Render / save

/// CIContext configured for linear-light accumulation in HDR-capable
/// extended-range half-float precision, with HLG output color space.
nonisolated func longExposureCIContext() -> CIContext {
    CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
        .workingFormat:     CIFormat.RGBAh,
        .outputColorSpace:  CGColorSpace(name: CGColorSpace.itur_2100_HLG)!,
    ])
}

/// Render the merged CIImage to a CGImage suitable for SwiftUI display
/// (HLG-tagged 16-bit half-float). Used by the preview pane.
nonisolated func renderLongExposurePreview(_ image: CIImage, maxDimension: CGFloat = 1200) async throws -> CGImage {
    try await Task.detached {
        let context = longExposureCIContext()
        var fitted = image
        let largest = max(fitted.extent.width, fitted.extent.height)
        if largest > maxDimension {
            let s = maxDimension / largest
            fitted = fitted.transformed(by: CGAffineTransform(scaleX: s, y: s))
        }
        let hlg = CGColorSpace(name: CGColorSpace.itur_2100_HLG)!
        guard let cg = context.createCGImage(
            fitted, from: fitted.extent,
            format: .RGBAh, colorSpace: hlg
        ) else {
            throw ImageToolsError.previewFailed
        }
        return cg
    }.value
}

// MARK: - End-to-end

/// Load (full-res) → optional align → merge → write via the shared exporter.
nonisolated func performLongExposureMerge(
    urls: [URL],
    align: Bool,
    method: MergeMethod,
    format: ImageExportFormat,
    outputFolder: URL?,
    progress: (@Sendable (_ message: String) -> Void)? = nil
) async throws -> URL {
    guard !urls.isEmpty else { throw ImageToolsError.noImages }

    progress?("Loading images…")
    let frames = try await loadLongExposureImages(urls, maxDimension: nil)
    guard !frames.isEmpty else { throw ImageToolsError.noImages }

    let prepared: [CIImage]
    if align {
        progress?("Aligning frames…")
        prepared = await alignImagesPanoMatcher(frames) { done, total in
            progress?("Aligning \(done)/\(total)…")
        }
    } else {
        prepared = frames
    }

    progress?("Merging…")
    guard let merged = mergeImages(prepared, method: method) else {
        throw ImageToolsError.processingFailed
    }

    let folder = outputFolder ?? urls[0].deletingLastPathComponent()
    let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        .replacingOccurrences(of: ":", with: "-")
    let outputURL = folder
        .appendingPathComponent("merged_long_exposure_\(timestamp)")
        .appendingPathExtension(format.fileExtension)

    progress?("Writing \(format.displayName)…")
    try saveCIImage(merged, to: outputURL, format: format,
                    ciContext: longExposureCIContext())
    progress?("Done")
    return outputURL
}
