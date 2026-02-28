# imgTools

A lightweight macOS SwiftUI app for batch image utilities, including format conversion, slicing, and image-sequence-to-video export.

## Features

- Convert images to **EXR**
- Convert images to **HEIF (10-bit)**
- Slice each image into **3 vertical segments** and generate an additional centered HEIF output
- Combine multiple images into a **.mov video** (HEVC)
- Drag-and-drop support plus manual file picker
- Batch processing with per-item status feedback

## Requirements

- macOS with Xcode installed
- SwiftUI / Apple platform SDKs (CoreImage, AVFoundation)

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/imgTools.git
cd imgTools
```

2. Open the project in Xcode:

```bash
open imgTools.xcodeproj
```

3. Build and run the app from Xcode.

## Usage

1. Launch the app.
2. Choose an operation from the segmented control:
   - `Convert to EXR`
   - `Convert to HEIF`
   - `Slicer`
   - `Images to Video`
3. Add images via drag-and-drop or **Add Images**.
4. Click **Process All** and choose an output folder when prompted.

## Project Structure

```text
imgTools/
├── imgTools/
│   ├── imgToolsApp.swift
│   ├── Assets.xcassets/
│   └── imgToolsIcon.icon/
└── imgTools.xcodeproj/
```

## Notes

- The app currently targets desktop workflows through the macOS UI.
- Output files are saved to the selected output folder (or source folder when applicable).