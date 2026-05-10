//
//  SuperPointDetector.swift
//  panoDev
//
//  Runs SuperPoint (CoreML) on a PanoImage and returns detected keypoints with
//  256-dim L2-normalised descriptors.  All post-processing (softmax, NMS,
//  descriptor sampling) is done on the CPU so no extra Metal kernels are needed.
//

import CoreML
import Metal
import Foundation

// MARK: - SuperPointDetector

final class SuperPointDetector {

    // ── CoreML model ────────────────────────────────────────────────────────────
    private let model: MLModel?

    // ── Metal resources ─────────────────────────────────────────────────────────
    private let ctx = PanoContext.shared
    private let tonemapPSO: MTLComputePipelineState?

    // ── Configuration ───────────────────────────────────────────────────────────
    /// Minimum detector score to keep a keypoint (after softmax + pixel-shuffle).
    var detectorThreshold: Float = 0.015
    /// Maximum number of keypoints per image.
    var maxKeypoints: Int = 1024
    /// Radius of non-maximum suppression window (NMS uses (2r+1)² neighbourhood).
    var nmsRadius: Int = 4

    // MARK: Init

    init() {
        // Xcode compiles bundled .mlpackage → .mlmodelc at build time.
        let modelURL = Bundle.main.url(forResource: "SuperPoint", withExtension: "mlmodelc")

        dbg("[SuperPoint] Model URL: \(modelURL?.path ?? "nil")")

        if let modelURL {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            do {
                self.model = try MLModel(contentsOf: modelURL, configuration: config)
                dbg("[SuperPoint] Model loaded OK")
            } catch {
                self.model = nil
                dbg("[SuperPoint] Model load FAILED: \(error)")
            }
        } else {
            self.model = nil
            dbg("[SuperPoint] Warning: SuperPoint.mlmodelc not found in bundle.")
        }

        // Build Metal compute pipeline for grayscale+tonemap pre-processing.
        // Use a local binding so the closure doesn't capture `self` before init completes.
        let metalDevice = PanoContext.shared.device
        let lib = try? metalDevice.makeDefaultLibrary()
        dbg("[SuperPoint] Metal library: \(lib != nil ? "OK" : "MISSING")")
        let fn  = lib?.makeFunction(name: "grayscaleTonemap")
        dbg("[SuperPoint] grayscaleTonemap fn: \(fn != nil ? "OK" : "MISSING")")
        do {
            self.tonemapPSO = try fn.map { try metalDevice.makeComputePipelineState(function: $0) }
        } catch {
            self.tonemapPSO = nil
            dbg("[SuperPoint] PSO creation failed: \(error)")
        }
        dbg("[SuperPoint] tonemapPSO: \(tonemapPSO != nil ? "OK" : "MISSING")")
    }

    // MARK: Public API

    /// Detect keypoints + descriptors in the texture-backed `image`.
    /// Returns an empty array (instead of throwing) if the model is absent.
    func detect(in image: ImageTexture) -> [Keypoint] {
        dbg("[SuperPoint] detect() called — size \(image.width)×\(image.height)")

        guard let model else {
            dbg("[SuperPoint] detect: no model, returning []")
            return []
        }
        guard let pso = tonemapPSO else {
            dbg("[SuperPoint] detect: no PSO, returning []")
            return []
        }

        let W = image.width
        let H = image.height

        // ── 1. Metal: rgba16Float → sRGB float32 row-major buffer ────────────
        guard let buffer = makeGrayscaleBuffer(from: image.texture,
                                               width: W, height: H, pso: pso)
        else {
            dbg("[SuperPoint] detect: makeGrayscaleBuffer failed")
            return []
        }
        dbg("[SuperPoint] grayscale buffer OK (\(W*H*4) bytes)")

        // Quick sanity: inspect first few pixels
        let fptr = buffer.contents().bindMemory(to: Float.self, capacity: W * H)
        let sample = (0..<min(5, W*H)).map { String(format: "%.3f", fptr[$0]) }.joined(separator: ", ")
        dbg("[SuperPoint] grayscale[0..4]: \(sample)")

        // ── 2. CoreML: run SuperPoint ─────────────────────────────────────────
        guard let (detRaw, dscRaw) = runModel(model: model,
                                              buffer: buffer,
                                              width: W, height: H)
        else {
            dbg("[SuperPoint] detect: runModel failed")
            return []
        }
        dbg("[SuperPoint] CoreML inference OK")
        dbg("[SuperPoint] detRaw shape: \(detRaw.shape)")
        dbg("[SuperPoint] dscRaw shape: \(dscRaw.shape)")

        let cH = H / 8
        let cW = W / 8

        // ── 3. CPU: softmax over 65 channels + pixel-shuffle → score map ──────
        let scoreMap = detectorToScoreMap(detRaw: detRaw, cH: cH, cW: cW)
        let maxScore = scoreMap.max() ?? 0
        let aboveThresh = scoreMap.filter { $0 >= detectorThreshold }.count
        dbg("[SuperPoint] score map max=\(String(format: "%.4f", maxScore)), pixels above \(detectorThreshold): \(aboveThresh)")

        // ── 4. CPU: NMS + threshold ────────────────────────────────────────────
        var candidates = nmsPeaks(scoreMap: scoreMap, fullH: H, fullW: W)
        dbg("[SuperPoint] NMS candidates: \(candidates.count)")

        // Sort by response, keep top-N
        candidates.sort { $0.response > $1.response }
        if candidates.count > maxKeypoints { candidates = Array(candidates.prefix(maxKeypoints)) }

        // ── 5. CPU: bilinear descriptor sampling + L2-normalise ───────────────
        sampleDescriptors(into: &candidates, dscRaw: dscRaw, cH: cH, cW: cW, imgW: W, imgH: H)

        dbg("[SuperPoint] final keypoints: \(candidates.count)")
        return candidates
    }

    // MARK: - Step 1: Metal grayscale tonemap

    private func makeGrayscaleBuffer(from texture: MTLTexture,
                                     width W: Int, height H: Int,
                                     pso: MTLComputePipelineState) -> MTLBuffer? {
        let byteCount = W * H * MemoryLayout<Float>.size
        // Shared storage so the CPU can read without an explicit blit.
        guard let buf = ctx.device.makeBuffer(length: byteCount,
                                              options: .storageModeShared)
        else { return nil }

        guard let cb  = ctx.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder()
        else { return nil }

        enc.setComputePipelineState(pso)
        enc.setTexture(texture, index: 0)
        enc.setBuffer(buf, offset: 0, index: 0)
        var dims = SIMD2<UInt32>(UInt32(W), UInt32(H))
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)

        let tgSize  = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width:  (W + 15) / 16,
            height: (H + 15) / 16,
            depth: 1
        )
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return buf
    }

    // MARK: - Step 2: CoreML inference

    private func runModel(model: MLModel,
                          buffer: MTLBuffer,
                          width W: Int,
                          height H: Int) -> (MLMultiArray, MLMultiArray)? {
        // Wrap the Metal buffer data into an MLMultiArray [1, 1, H, W].
        guard let multiArray = try? MLMultiArray(shape: [1, 1,
                                                          NSNumber(value: H),
                                                          NSNumber(value: W)],
                                                 dataType: .float32)
        else {
            dbg("[SuperPoint] MLMultiArray creation failed")
            return nil
        }

        // Copy from the shared Metal buffer into the MLMultiArray storage.
        let src = buffer.contents().bindMemory(to: Float.self, capacity: W * H)
        let dst = multiArray.dataPointer.bindMemory(to: Float.self, capacity: W * H)
        dst.update(from: src, count: W * H)

        // Build feature provider and run.
        do {
            let features = try MLDictionaryFeatureProvider(dictionary: ["image": multiArray])
            let out = try model.prediction(from: features)

            guard let det = out.featureValue(for: "detector_raw")?.multiArrayValue else {
                dbg("[SuperPoint] Missing detector_raw output")
                return nil
            }
            guard let dsc = out.featureValue(for: "descriptor_map")?.multiArrayValue else {
                dbg("[SuperPoint] Missing descriptor_map output")
                return nil
            }
            return (det, dsc)
        } catch {
            dbg("[SuperPoint] model.prediction failed: \(error)")
            return nil
        }
    }

    // MARK: - Step 3: Softmax + pixel-shuffle

    /// Converts raw detector output [1, 65, cH, cW] → score map [H, W].
    private func detectorToScoreMap(detRaw: MLMultiArray,
                                    cH: Int, cW: Int) -> [Float] {
        // MLMultiArray [1, 65, cH, cW] — strides: [65*cH*cW, cH*cW, cW, 1]
        let strideC  = cH * cW
        let strideRow = cW
        let ptr = detRaw.dataPointer.bindMemory(to: Float.self,
                                                capacity: 65 * cH * cW)

        let H = cH * 8
        let W = cW * 8
        var scoreMap = [Float](repeating: 0, count: H * W)

        var logits = [Float](repeating: 0, count: 65)

        for cy in 0 ..< cH {
            for cx in 0 ..< cW {
                // Extract logits for this cell
                for c in 0 ..< 65 {
                    logits[c] = ptr[c * strideC + cy * strideRow + cx]
                }

                // Numerically stable softmax
                let maxL = logits.max()!
                var expSum: Float = 0
                for c in 0 ..< 65 {
                    logits[c] = exp(logits[c] - maxL)
                    expSum += logits[c]
                }
                // Skip dustbin (channel 64); distribute first 64 probs to H×W
                for sub in 0 ..< 64 {
                    let dy = sub / 8
                    let dx = sub % 8
                    let py = cy * 8 + dy
                    let px = cx * 8 + dx
                    scoreMap[py * W + px] = logits[sub] / expSum
                }
            }
        }
        return scoreMap
    }

    // MARK: - Step 4: NMS + threshold

    private func nmsPeaks(scoreMap: [Float], fullH H: Int, fullW W: Int) -> [Keypoint] {
        var keypoints: [Keypoint] = []
        let r = nmsRadius

        for y in 0 ..< H {
            for x in 0 ..< W {
                let score = scoreMap[y * W + x]
                guard score >= detectorThreshold else { continue }

                // Check if local maximum in (2r+1)×(2r+1) window
                var isMax = true
                outer: for dy in -r ... r {
                    for dx in -r ... r {
                        if dy == 0 && dx == 0 { continue }
                        let ny = y + dy
                        let nx = x + dx
                        guard ny >= 0 && ny < H && nx >= 0 && nx < W else { continue }
                        if scoreMap[ny * W + nx] >= score {
                            isMax = false
                            break outer
                        }
                    }
                }

                if isMax {
                    keypoints.append(Keypoint(
                        x: Float(x),
                        y: Float(y),
                        response: score
                    ))
                }
            }
        }
        return keypoints
    }

    // MARK: - Step 5: Descriptor sampling + L2-normalise

    /// Bilinearly samples the [256, cH, cW] descriptor map at each keypoint position.
    private func sampleDescriptors(into keypoints: inout [Keypoint],
                                   dscRaw: MLMultiArray,
                                   cH: Int, cW: Int,
                                   imgW W: Int, imgH H: Int) {
        // MLMultiArray layout: [1, 256, cH, cW]
        let strideC   = cH * cW
        let strideRow = cW
        let ptr = dscRaw.dataPointer.bindMemory(to: Float.self,
                                                capacity: 256 * cH * cW)

        // Coordinates in cell space: keypoint pixel → (pixel + 0.5) / 8 → fractional cell coord
        let scaleX = Float(cW) / Float(W)
        let scaleY = Float(cH) / Float(H)

        let descDim = 256

        for i in 0 ..< keypoints.count {
            let kp = keypoints[i]

            // Map to descriptor map space (cell-centre aligned)
            let cx = (kp.x + 0.5) * scaleX - 0.5
            let cy = (kp.y + 0.5) * scaleY - 0.5

            // Bilinear weights
            let x0 = Int(floor(cx))
            let y0 = Int(floor(cy))
            let x1 = x0 + 1
            let y1 = y0 + 1

            let wx1 = cx - Float(x0)
            let wy1 = cy - Float(y0)
            let wx0 = 1.0 - wx1
            let wy0 = 1.0 - wy1

            func clampX(_ v: Int) -> Int { max(0, min(cW - 1, v)) }
            func clampY(_ v: Int) -> Int { max(0, min(cH - 1, v)) }

            let xx0 = clampX(x0); let xx1 = clampX(x1)
            let yy0 = clampY(y0); let yy1 = clampY(y1)

            var desc = [Float](repeating: 0, count: descDim)
            for d in 0 ..< descDim {
                let base = d * strideC
                let v00 = ptr[base + yy0 * strideRow + xx0]
                let v01 = ptr[base + yy0 * strideRow + xx1]
                let v10 = ptr[base + yy1 * strideRow + xx0]
                let v11 = ptr[base + yy1 * strideRow + xx1]
                desc[d] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11)
            }

            // L2 normalise
            var norm: Float = 0
            for v in desc { norm += v * v }
            norm = sqrt(norm)
            if norm > 1e-8 {
                for j in 0 ..< descDim { desc[j] /= norm }
            }

            keypoints[i].descriptor = desc
        }
    }
}
