//
//  EXIF.swift
//  panoDev
//
//  Two narrow EXIF helpers:
//    • `focalLengthMM35(for:)` — best-available 35-mm-equivalent focal length,
//      used to auto-populate the toolbar field on import.
//    • `metadata(for:)` — the source's CGImageMetadata blob (EXIF + TIFF +
//      aux + XMP), used to splice basic shoot info into the output HEIF.
//

import Foundation
import ImageIO
import CoreGraphics

enum EXIF {

    /// Returns the best-available 35-mm-equivalent focal length in mm.
    /// Preference order: `FocalLenIn35mmFilm` → `FocalLength` → nil.
    static func focalLengthMM35(for url: URL) -> Float? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any]
        else { return nil }

        if let f = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber {
            return f.floatValue
        }
        if let f = exif[kCGImagePropertyExifFocalLength] as? NSNumber {
            return f.floatValue
        }
        return nil
    }

    /// Returns the source's full metadata container (EXIF/TIFF/aux/XMP).
    /// Callers pass this directly via `kCGImageDestinationMetadata`.
    static func metadata(for url: URL) -> CGImageMetadata? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCopyMetadataAtIndex(src, 0, nil)
    }
}
