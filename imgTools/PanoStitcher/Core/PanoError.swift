//
//  PanoError.swift
//  panoDev
//

import Foundation

enum PanoError: Error, LocalizedError {
    case fileNotFound(URL)
    case decodeFailed(URL)
    case textureFailed
    case modelNotFound(URL)
    case modelOutputMissing(String)
    case exportFailed(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):       return "File not found: \(url.lastPathComponent)"
        case .decodeFailed(let url):       return "Failed to decode image: \(url.lastPathComponent)"
        case .textureFailed:               return "Failed to create Metal texture"
        case .modelNotFound(let url):      return "CoreML model not found: \(url.lastPathComponent)"
        case .modelOutputMissing(let key): return "Model output missing: \(key)"
        case .exportFailed(let reason):    return "Export failed: \(reason)"
        }
    }
}
