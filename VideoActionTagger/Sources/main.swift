import Foundation
import AVFoundation
import Vision
import CoreML

// MARK: - CLI config
struct Config {
    var inputFolder: URL
    var outputFolder: URL
    var modelPath: URL?
    var windowSeconds: Double = 3.0
    var strideSeconds: Double = 1.0
    var threshold: Double = 0.6 // per-class minimum probability
    var minSegmentSeconds: Double = 1.5
    var writeFinderTags: Bool = true
}

func parseArgs() -> Config? {
    var input: URL?
    var output: URL?
    var cfg = Config(inputFolder: URL(fileURLWithPath: "."), outputFolder: URL(fileURLWithPath: "results"), modelPath: nil)

    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--input":
            if let path = it.next() { input = URL(fileURLWithPath: path) }
        case "--output":
            if let path = it.next() { output = URL(fileURLWithPath: path) }
        case "--model":
            if let path = it.next() { cfg.modelPath = URL(fileURLWithPath: path) }
        case "--window":
            if let s = it.next(), let v = Double(s) { cfg.windowSeconds = v }
        case "--stride":
            if let s = it.next(), let v = Double(s) { cfg.strideSeconds = v }
        case "--threshold":
            if let s = it.next(), let v = Double(s) { cfg.threshold = v }
        case "--min-segment":
            if let s = it.next(), let v = Double(s) { cfg.minSegmentSeconds = v }
        case "--no-tags":
            cfg.writeFinderTags = false
        default:
            print("Unknown arg: \(arg)")
        }
    }
    guard let input else {
    print("Usage: video-action-tagger --input /path/to/folder [--output out] [--model /path/to/ActionClassifier.mlmodel|.mlmodelc] [--window 3] [--stride 1] [--threshold 0.6] [--min-segment 1.5] [--no-tags]")
        return nil
    }
    cfg.inputFolder = input
    cfg.outputFolder = output ?? input.appendingPathComponent("results", isDirectory: true)
    return cfg
}

// MARK: - Model wrapper
// Replace with your generated model class name when you add your .mlmodel to Models/
final class ActionModel {
    let model: MLModel
    let labels: [String]
    let frameSize = CGSize(width: 224, height: 224)

    init(modelURL: URL) throws {
        // Load compiled model at runtime
        let url = modelURL
        let compiledURL: URL
        if url.pathExtension == "mlmodel" {
            compiledURL = try MLModel.compileModel(at: url)
        } else {
            compiledURL = url
        }
        model = try MLModel(contentsOf: compiledURL)
        // Try to read labels from model metadata
        if let userDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let labelsStr = userDefined["classes"] ?? userDefined["classes_names"] {
            labels = labelsStr.components(separatedBy: ",")
        } else if let output = model.modelDescription.outputDescriptionsByName.values.first,
                  let dict = output.multiArrayConstraint {
            // Fallback placeholder
            labels = (0..<(dict.shape.first?.intValue ?? 1)).map { "class_\($0)" }
        } else {
            labels = []
        }
    }

    func predict(windowFrames: [CGImage]) throws -> [String: Double] {
        // Placeholder: Many Create ML action models accept a video or a multi-array sequence.
        // Without the concrete generated class, we use Vision for convenience if possible.
        // Here we fallback to a fake softmax for compilation; replace with your generated model call.
        var out: [String: Double] = [:]
        if labels.isEmpty { return out }
        let k = min(3, labels.count)
        for i in 0..<k { out[labels[i]] = 1.0 / Double(k) }
        return out
    }
}

// MARK: - Video windowing
struct Segment { let label: String; var start: CMTime; var end: CMTime; var score: Double }

func enumerateWindows(asset: AVAsset, window: Double, stride: Double) -> [(CMTimeRange, [CMTime: CGImage])] {
    let duration = CMTimeGetSeconds(asset.duration)
    guard duration.isFinite, duration > 0 else { return [] }
    var results: [(CMTimeRange, [CMTime: CGImage])] = []

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero

    var t = 0.0
    while t + window <= duration {
        let start = CMTime(seconds: t, preferredTimescale: 600)
        let end = CMTime(seconds: t + window, preferredTimescale: 600)
        let range = CMTimeRange(start: start, end: end)
        // Sample N frames uniformly per window (e.g., 8)
        let frames = 8
        var frameMap: [CMTime: CGImage] = [:]
        for i in 0..<frames {
            let u = Double(i) / Double(frames - 1)
            let ts = CMTime(seconds: t + u * window, preferredTimescale: 600)
            do {
                let imgRef = try generator.copyCGImage(at: ts, actualTime: nil)
                frameMap[ts] = imgRef
            } catch {
                // skip
            }
        }
        results.append((range, frameMap))
        t += stride
    }
    return results
}

// MARK: - Post-processing
func mergePredictions(_ perWindow: [(CMTimeRange, [String: Double])], threshold: Double, minDur: Double) -> [Segment] {
    // Per class, merge consecutive windows whose probability passes threshold; smooth by averaging
    var byClass: [String: [Segment]] = [:]
    for (range, probs) in perWindow {
        for (label, p) in probs where p >= threshold {
            var arr = byClass[label] ?? []
            if var last = arr.last, CMTimeCompare(last.end, range.start) >= 0 || last.end == range.start {
                // extend
                let newDur = CMTimeGetSeconds(range.end) - CMTimeGetSeconds(last.start)
                last.end = range.end
                last.score = (last.score + p) / 2.0
                arr[arr.count - 1] = last
            } else {
                arr.append(Segment(label: label, start: range.start, end: range.end, score: p))
            }
            byClass[label] = arr
        }
    }
    // flatten and filter by min duration
    var all: [Segment] = byClass.values.flatMap { $0 }
    all = all.filter { CMTimeGetSeconds($0.end - $0.start) >= minDur }
    // sort by start
    all.sort { $0.start < $1.start }
    return all
}

// MARK: - Serialization
struct JSONResult: Codable {
    struct SegmentOut: Codable { let label: String; let startSeconds: Double; let endSeconds: Double; let score: Double }
    let video: String
    let durationSeconds: Double
    let segments: [SegmentOut]
}

func writeOutputs(for videoURL: URL, asset: AVAsset, segments: [Segment], outDir: URL) throws {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let base = videoURL.deletingPathExtension().lastPathComponent
    // JSON
    let jsonPath = outDir.appendingPathComponent("\(base).json")
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let j = JSONResult(
        video: videoURL.lastPathComponent,
        durationSeconds: CMTimeGetSeconds(asset.duration),
        segments: segments.map { .init(label: $0.label, startSeconds: CMTimeGetSeconds($0.start), endSeconds: CMTimeGetSeconds($0.end), score: $0.score) }
    )
    let data = try enc.encode(j)
    try data.write(to: jsonPath)

    // CSV
    let csvPath = outDir.appendingPathComponent("\(base).csv")
    var csv = "label,start_seconds,end_seconds,score\n"
    for s in segments { csv += "\(s.label),\(CMTimeGetSeconds(s.start)),\(CMTimeGetSeconds(s.end)),\(s.score)\n" }
    try csv.data(using: .utf8)!.write(to: csvPath)
}

// MARK: - Finder tags (Spotlight searchable)
func writeFinderTags(fileURL: URL, segmentLabels: [String]) {
    guard !segmentLabels.isEmpty else { return }
    let unique = Array(Set(segmentLabels)).sorted()
    do {
        try (fileURL as NSURL).setResourceValue(unique, forKey: .tagNamesKey)
    } catch {
        // ignore tagging failures
    }
}

// MARK: - Main
func main() throws {
    guard var cfg = parseArgs() else { return }
    try FileManager.default.createDirectory(at: cfg.outputFolder, withIntermediateDirectories: true)

    let fm = FileManager.default
    let mp4s = try fm.contentsOfDirectory(at: cfg.inputFolder, includingPropertiesForKeys: nil).filter { $0.pathExtension.lowercased() == "mp4" }
    if mp4s.isEmpty { print("No mp4 files in \(cfg.inputFolder.path)"); return }

    guard let modelPath = cfg.modelPath else {
        throw NSError(domain: "VideoActionTagger", code: 2, userInfo: [NSLocalizedDescriptionKey: "--model path to .mlmodel or .mlmodelc is required"])
    }
    let model = try ActionModel(modelURL: modelPath)

    for url in mp4s {
        print("Processing \(url.lastPathComponent)…")
        let asset = AVAsset(url: url)
        let windows = enumerateWindows(asset: asset, window: cfg.windowSeconds, stride: cfg.strideSeconds)

        var perWindow: [(CMTimeRange, [String: Double])] = []
        for (range, frameMap) in windows {
            // In a real use: pass frames to the model. Here we call placeholder.
            let frames = frameMap.values.map { $0 }
            let probs = try model.predict(windowFrames: frames)
            perWindow.append((range, probs))
        }

        let segments = mergePredictions(perWindow, threshold: cfg.threshold, minDur: cfg.minSegmentSeconds)
        try writeOutputs(for: url, asset: asset, segments: segments, outDir: cfg.outputFolder)
        if cfg.writeFinderTags { writeFinderTags(fileURL: url, segmentLabels: segments.map { $0.label }) }
        print("Done: \(url.lastPathComponent)")
    }
}

do { try main() } catch { fputs("Error: \(error)\n", stderr) }
