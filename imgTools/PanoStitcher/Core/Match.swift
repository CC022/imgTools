//
//  Match.swift
//  panoDev
//
//  A pair of corresponding keypoints between two images, plus a confidence
//  score returned by the matcher (LightGlue).
//

import Foundation

struct Match: Sendable {
    /// Index into the keypoints-A array.
    let indexA: Int
    /// Index into the keypoints-B array.
    let indexB: Int
    /// Match confidence in [0, 1].  Higher = more reliable correspondence.
    let confidence: Float
}
