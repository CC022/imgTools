//
//  Keypoint.swift
//  panoDev
//

import Foundation
import simd

/// A single detected keypoint with optional 256-dim SuperPoint descriptor.
struct Keypoint: Sendable {
    /// Pixel coordinates in the original image (origin = top-left).
    var x: Float
    var y: Float
    /// Detector response score in [0, 1].
    var response: Float
    /// Scale (σ in pixels). SuperPoint uses a fixed cell size of 8 px, so σ ≈ 4.
    var scale: Float = 4.0
    /// Octave index. Always 0 for SuperPoint (single-scale detector).
    var octave: Int = 0
    /// L2-normalised 256-dim descriptor. Empty until `detect()` populates it.
    var descriptor: [Float] = []
}

/// Mirror of the Metal `Keypoint` struct for buffer hand-off.
struct GPUKeypoint {
    var x: Float
    var y: Float
    var response: Float
    var scale: Float
}
