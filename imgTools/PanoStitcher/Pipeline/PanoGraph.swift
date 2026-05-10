//
//  PanoGraph.swift
//  panoDev
//
//  N-image data model: nodes are images with detected keypoints, edges are
//  pairwise correspondences (matches + homography + RANSAC inliers).
//
//  Conventions:
//      • Edge stores `H` in the canonical direction src → dst:
//          dstPixel ≈ H · srcPixel        (homogeneous divide implicit)
//      • Inlier indices are positions inside `matches`, not into the
//        underlying keypoint arrays.
//

import Foundation
import simd

// MARK: - Node

/// One source image plus everything we know about it.
struct PanoNode: @unchecked Sendable {
    let id: Int
    let image: PanoImage
    let keypoints: [Keypoint]
    /// Populated by camera recovery after RANSAC.
    var pose: CameraPose?

    init(id: Int, image: PanoImage, keypoints: [Keypoint], pose: CameraPose? = nil) {
        self.id = id
        self.image = image
        self.keypoints = keypoints
        self.pose = pose
    }
}

// MARK: - Edge

/// One pairwise relationship between two nodes.
struct PanoEdge: Sendable {
    let src: Int
    let dst: Int
    let matches: [Match]
    let homography: Homography      // src → dst
    let inliers: [Int]              // indices into `matches`

    var inlierCount: Int { inliers.count }

    /// Returns the same edge with src/dst swapped and `H` inverted.
    var reversed: PanoEdge {
        // Translate inliers' (indexA, indexB) by swapping; matches list unchanged
        // structurally but the semantic role of A/B flips.
        let swapped = matches.map {
            Match(indexA: $0.indexB, indexB: $0.indexA, confidence: $0.confidence)
        }
        return PanoEdge(
            src: dst,
            dst: src,
            matches: swapped,
            homography: homography.inverse,
            inliers: inliers
        )
    }
}

// MARK: - Graph

struct PanoGraph: @unchecked Sendable {
    var nodes: [PanoNode]
    var edges: [PanoEdge]

    init(nodes: [PanoNode] = [], edges: [PanoEdge] = []) {
        self.nodes = nodes
        self.edges = edges
    }

    /// Highest-degree node — used as the anchor (R = identity) for camera recovery.
    func anchorIndex() -> Int {
        guard !nodes.isEmpty else { return 0 }
        var degree = [Int](repeating: 0, count: nodes.count)
        for e in edges { degree[e.src] += 1; degree[e.dst] += 1 }
        return degree.indices.max(by: { degree[$0] < degree[$1] }) ?? 0
    }

    /// Maximum-spanning-tree using Kruskal + Union-Find. `weight` should return
    /// a higher value for edges we'd rather keep (e.g. inlier count).
    func maximumSpanningTree(weight: (PanoEdge) -> Float) -> [PanoEdge] {
        guard nodes.count > 1 else { return [] }
        let sorted = edges.enumerated()
            .sorted { weight($0.element) > weight($1.element) }
            .map(\.element)

        var parent = Array(nodes.indices)
        func find(_ x: Int) -> Int {
            var r = x; while parent[r] != r { r = parent[r] }
            var i = x; while parent[i] != r { let n = parent[i]; parent[i] = r; i = n }
            return r
        }

        var tree: [PanoEdge] = []
        tree.reserveCapacity(nodes.count - 1)
        for e in sorted {
            let ra = find(e.src), rb = find(e.dst)
            if ra != rb {
                parent[ra] = rb
                tree.append(e)
                if tree.count == nodes.count - 1 { break }
            }
        }
        return tree
    }

    /// Returns connected components as arrays of node indices.
    func connectedComponents() -> [[Int]] {
        var adj = [[Int]](repeating: [], count: nodes.count)
        for e in edges { adj[e.src].append(e.dst); adj[e.dst].append(e.src) }

        var visited = [Bool](repeating: false, count: nodes.count)
        var components: [[Int]] = []
        for start in nodes.indices where !visited[start] {
            var stack = [start]
            var comp: [Int] = []
            while let n = stack.popLast() {
                if visited[n] { continue }
                visited[n] = true
                comp.append(n)
                stack.append(contentsOf: adj[n].filter { !visited[$0] })
            }
            components.append(comp)
        }
        return components
    }

    /// Returns human-readable warnings; empty array means "fully connected".
    func validate() -> [String] {
        let components = connectedComponents()
        guard components.count > 1 else { return [] }
        return components.enumerated().map { i, comp in
            "Component \(i): \(comp.count) image(s) — indices \(comp)"
        }
    }
}
