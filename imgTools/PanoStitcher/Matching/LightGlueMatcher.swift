//
//  LightGlueMatcher.swift
//  panoDev
//
//  Matches two sets of SuperPoint keypoints using LightGlue (CoreML).
//
//  The CoreML graph has fixed input shape N=1024 per image. Real keypoints
//  occupy slots [0, count); padding fills the rest with zero coords and zero
//  descriptors. The matchability head naturally drives padded slots toward
//  "no match", and we additionally filter on Swift side using the original
//  keypoint counts.
//

import CoreML
import Foundation

final class LightGlueMatcher {

    /// Must match the `N` constant in `scripts/convert_lightglue.py`.
    static let maxKeypoints = 1024

    /// Confidence threshold below which a putative match is dropped.
    var confidenceThreshold: Float = 0.10

    private let model: MLModel?

    init() {
        // Xcode compiles bundled .mlpackage → .mlmodelc at build time.
        let modelURL = Bundle.main.url(forResource: "LightGlue", withExtension: "mlmodelc")

        dbg("[LightGlue] Model URL: \(modelURL?.path ?? "nil")")

        if let modelURL {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            do {
                self.model = try MLModel(contentsOf: modelURL, configuration: config)
                dbg("[LightGlue] Model loaded OK")
            } catch {
                self.model = nil
                dbg("[LightGlue] Model load FAILED: \(error)")
            }
        } else {
            self.model = nil
            dbg("[LightGlue] Warning: LightGlue.mlmodelc not found in bundle.")
        }
    }

    // MARK: Public API

    /// Match two keypoint sets. Empty result if the model is missing.
    /// - Parameters:
    ///   - kpA / kpB:        Keypoint arrays (256-dim descriptors expected).
    ///   - imgWidthA / imgHeightA: pixel dimensions of image A
    ///   - imgWidthB / imgHeightB: pixel dimensions of image B
    func match(kpA: [Keypoint], imgWidthA: Int, imgHeightA: Int,
               kpB: [Keypoint], imgWidthB: Int, imgHeightB: Int) -> [Match] {
        guard let model else {
            dbg("[LightGlue] match(): no model")
            return []
        }
        let nA = min(kpA.count, Self.maxKeypoints)
        let nB = min(kpB.count, Self.maxKeypoints)
        guard nA > 0 && nB > 0 else { return [] }
        dbg("[LightGlue] match(): nA=\(nA) nB=\(nB)")

        do {
            // Build padded MLMultiArrays.
            let kpts0 = try multiArray(shape: [1, NSNumber(value: Self.maxKeypoints), 2])
            let kpts1 = try multiArray(shape: [1, NSNumber(value: Self.maxKeypoints), 2])
            let desc0 = try multiArray(shape: [1, NSNumber(value: Self.maxKeypoints), 256])
            let desc1 = try multiArray(shape: [1, NSNumber(value: Self.maxKeypoints), 256])
            let size0 = try multiArray(shape: [1, 2])
            let size1 = try multiArray(shape: [1, 2])

            fillKeypoints(kpA, count: nA, into: kpts0)
            fillKeypoints(kpB, count: nB, into: kpts1)
            fillDescriptors(kpA, count: nA, into: desc0)
            fillDescriptors(kpB, count: nB, into: desc1)
            fillSize(width: imgWidthA, height: imgHeightA, into: size0)
            fillSize(width: imgWidthB, height: imgHeightB, into: size1)

            let features = try MLDictionaryFeatureProvider(dictionary: [
                "kpts0": kpts0, "kpts1": kpts1,
                "desc0": desc0, "desc1": desc1,
                "size0": size0, "size1": size1,
            ])
            let out = try model.prediction(from: features)

            guard let matches0 = out.featureValue(for: "matches0")?.multiArrayValue,
                  let scores0  = out.featureValue(for: "matching_scores0")?.multiArrayValue
            else {
                dbg("[LightGlue] missing outputs")
                return []
            }

            return collectMatches(matches0: matches0, scores0: scores0,
                                  nA: nA, nB: nB)
        } catch {
            dbg("[LightGlue] inference failed: \(error)")
            return []
        }
    }

    // MARK: - Helpers

    private func multiArray(shape: [NSNumber]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: .float32)
    }

    private func fillKeypoints(_ kps: [Keypoint], count: Int, into arr: MLMultiArray) {
        // Layout [1, N, 2] — strides (N*2, 2, 1)
        let ptr = arr.dataPointer.bindMemory(to: Float.self,
                                             capacity: arr.count)
        // Zero everything first (covers padded slots).
        ptr.update(repeating: 0, count: arr.count)
        for i in 0 ..< count {
            ptr[i * 2 + 0] = kps[i].x
            ptr[i * 2 + 1] = kps[i].y
        }
    }

    private func fillDescriptors(_ kps: [Keypoint], count: Int, into arr: MLMultiArray) {
        // Layout [1, N, 256]
        let ptr = arr.dataPointer.bindMemory(to: Float.self,
                                             capacity: arr.count)
        ptr.update(repeating: 0, count: arr.count)
        for i in 0 ..< count {
            let desc = kps[i].descriptor
            // Defensive: descriptor might be empty or wrong length.
            let len = min(desc.count, 256)
            if len > 0 {
                desc.withUnsafeBufferPointer { src in
                    let dst = ptr.advanced(by: i * 256)
                    dst.update(from: src.baseAddress!, count: len)
                }
            }
        }
    }

    private func fillSize(width: Int, height: Int, into arr: MLMultiArray) {
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: 2)
        ptr[0] = Float(width)
        ptr[1] = Float(height)
    }

    private func collectMatches(matches0: MLMultiArray,
                                scores0:  MLMultiArray,
                                nA: Int, nB: Int) -> [Match] {
        // CoreML may quantise outputs to int32 / float16 even when we declared
        // float32 in conversion — the readers handle both.
        let indices = readIntArray(matches0)
        let scores  = readFloatArray(scores0)

        var out: [Match] = []
        out.reserveCapacity(min(nA, nB))
        for i in 0 ..< nA {
            let j  = indices[i]
            let cf = scores[i]
            guard j >= 0, j < nB, cf >= confidenceThreshold else { continue }
            out.append(Match(indexA: i, indexB: j, confidence: cf))
        }
        out.sort { $0.confidence > $1.confidence }
        dbg("[LightGlue] matches kept: \(out.count) / \(nA) (threshold \(confidenceThreshold))")
        return out
    }

    // MLMultiArray int32/int64 → [Int]
    private func readIntArray(_ arr: MLMultiArray) -> [Int] {
        let n = arr.count
        var out = [Int](repeating: 0, count: n)
        switch arr.dataType {
        case .int32:
            let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: n)
            for i in 0..<n { out[i] = Int(p[i]) }
        default:
            // Fallback via NSNumber accessor — works for any int type.
            for i in 0..<n { out[i] = arr[i].intValue }
        }
        return out
    }

    // MLMultiArray float32/float16 → [Float]
    private func readFloatArray(_ arr: MLMultiArray) -> [Float] {
        let n = arr.count
        var out = [Float](repeating: 0, count: n)
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { out[i] = p[i] }
        case .float16:
            // Two-byte half → Float32 via Float16 (Swift 5.3+).
            let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: n)
            for i in 0..<n { out[i] = Float(p[i]) }
        default:
            for i in 0..<n { out[i] = arr[i].floatValue }
        }
        return out
    }
}
