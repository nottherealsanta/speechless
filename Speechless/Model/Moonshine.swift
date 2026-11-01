import CoreML
import Foundation

/// Greedy decoding for a Moonshine encoder/decoder pair exported by `training/speechless_train`.
/// The decoder has fixed shapes (no KV cache): each step re-runs all `maxTokens` positions.
final class MoonshineRunner: @unchecked Sendable {
    struct Config: Decodable {
        var tokens: [String]
        var bos: Int
        var eos: Int
        var maxTokens: Int
        var maxMotion: Int?
        var maxAudio: Int?
    }

    let config: Config
    private let encoder: MLModel
    private let decoder: MLModel

    static var directory: URL {
        URL.applicationSupportDirectory.appending(path: "Speechless/Model", directoryHint: .isDirectory)
    }

    /// `prefix` is "Motion" or "Audio"; vocab file is `vocab.json` or `audio_vocab.json`.
    init(prefix: String, vocabFile: String, directory: URL = MoonshineRunner.directory) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        encoder = try MLModel(contentsOf: Self.compiled(directory.appending(path: "\(prefix)Encoder.mlpackage")),
                              configuration: configuration)
        decoder = try MLModel(contentsOf: Self.compiled(directory.appending(path: "\(prefix)Decoder.mlpackage")),
                              configuration: configuration)
        config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: directory.appending(path: vocabFile)))
    }

    static func exists(prefix: String, vocabFile: String, directory: URL = MoonshineRunner.directory) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: directory.appending(path: "\(prefix)Encoder.mlpackage").path)
            && fm.fileExists(atPath: directory.appending(path: "\(prefix)Decoder.mlpackage").path)
            && fm.fileExists(atPath: directory.appending(path: vocabFile).path)
    }

    /// Compiles an .mlpackage once and caches the .mlmodelc next to it, keyed by modification date.
    private static func compiled(_ package: URL) throws -> URL {
        let fm = FileManager.default
        let cache = package.deletingPathExtension().appendingPathExtension("mlmodelc")
        let stamp = cache.appending(path: ".source-date")
        let sourceDate = (try? fm.attributesOfItem(atPath: package.path)[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        if let saved = try? String(contentsOf: stamp, encoding: .utf8), saved == String(sourceDate),
           fm.fileExists(atPath: cache.path) {
            return cache
        }
        let temp = try MLModel.compileModel(at: package)
        try? fm.removeItem(at: cache)
        try fm.moveItem(at: temp, to: cache)
        try? String(sourceDate).write(to: stamp, atomically: true, encoding: .utf8)
        return cache
    }

    func transcribe(_ inputs: [String: MLMultiArray]) throws -> String {
        let encoded = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        guard let states = encoded.featureValue(for: "encoder_states")?.multiArrayValue,
              let mask = encoded.featureValue(for: "encoder_mask")?.multiArrayValue else { return "" }

        let n = config.maxTokens
        let ids = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        var tokens = [config.bos]
        let vocabSize = config.tokens.count
        while tokens.count < n {
            for i in 0..<n { ids[i] = NSNumber(value: i < tokens.count ? tokens[i] : config.eos) }
            let out = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": ids, "encoder_states": states, "encoder_mask": mask,
            ]))
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else { break }
            let row = tokens.count - 1
            let strides = logits.strides.map(\.intValue)
            var best = 0
            var bestValue = -Float.infinity
            let pointer = logits.dataPointer
            for v in 0..<vocabSize {
                let offset = row * strides[1] + v * strides[2]
                let value: Float = logits.dataType == .float16
                    ? Float(pointer.assumingMemoryBound(to: Float16.self)[offset])
                    : pointer.assumingMemoryBound(to: Float.self)[offset]
                if value > bestValue { bestValue = value; best = v }
            }
            if best == config.eos { break }
            tokens.append(best)
        }
        return Self.detokenize(tokens.dropFirst().map { config.tokens[$0] })
    }

    /// SentencePiece pieces → text, including `<0xNN>` byte-fallback pieces.
    static func detokenize(_ pieces: [String]) -> String {
        var bytes: [UInt8] = []
        for piece in pieces {
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), let b = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(b)
            } else if !(piece.hasPrefix("<") && piece.hasSuffix(">")) {
                bytes.append(contentsOf: Array(piece.replacingOccurrences(of: "\u{2581}", with: " ").utf8))
            }
        }
        return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Fixed motion clock used by training: 25 Hz, the 16 MotionSample channels, attitude relative
/// to the window start. Must match `training/speechless_train/data.py`.
enum MotionTensor {
    static let hz = 25.0
    static let channels = 16

    static func resample(_ samples: [MotionSample]) -> [[Float]] {
        guard let first = samples.first, let last = samples.last else { return [] }
        let count = max(1, Int(((last.t - first.t) * hz).rounded(.up)))
        var frames: [[Float]] = []
        frames.reserveCapacity(count)
        var j = 0
        var unwrapped = [0.0, 0.0, 0.0]
        var previous: [Double]?
        var origin: [Double]?
        for k in 0..<count {
            let t = first.t + Double(k) / hz
            while j + 1 < samples.count - 1, samples[j + 1].t < t { j += 1 }
            let a = samples[j], b = samples[min(j + 1, samples.count - 1)]
            let u = b.t > a.t ? min(max((t - a.t) / (b.t - a.t), 0), 1) : 0
            var row = zip(values(a), values(b)).map { $0 + ($1 - $0) * u }
            // Unwrap pitch/roll/yaw (indices 6…8), then make them relative to the first frame.
            let attitude = Array(row[6...8])
            if let previous {
                for c in 0..<3 {
                    var d = attitude[c] - previous[c]
                    while d > .pi { d -= 2 * .pi }
                    while d < -.pi { d += 2 * .pi }
                    unwrapped[c] += d
                }
            } else {
                unwrapped = attitude
            }
            previous = attitude
            let start = origin ?? unwrapped
            origin = start
            for c in 0..<3 { row[6 + c] = unwrapped[c] - start[c] }
            frames.append(row.map(Float.init))
        }
        return frames
    }

    private static func values(_ s: MotionSample) -> [Double] {
        [s.ax, s.ay, s.az, s.rx, s.ry, s.rz, s.pitch, s.roll, s.yaw, s.gx, s.gy, s.gz, s.qw, s.qx, s.qy, s.qz]
    }
}

/// Motion → text with the LoRA-adapted Moonshine.
final class MotionTextModel: @unchecked Sendable {
    private let runner: MoonshineRunner
    static var isInstalled: Bool { MoonshineRunner.exists(prefix: "Motion", vocabFile: "vocab.json") }

    init() throws {
        runner = try MoonshineRunner(prefix: "Motion", vocabFile: "vocab.json")
    }

    func transcribe(_ samples: [MotionSample]) throws -> String {
        let frames = MotionTensor.resample(samples)
        let maxFrames = runner.config.maxMotion ?? 150
        let motion = try MLMultiArray(shape: [1, NSNumber(value: maxFrames), NSNumber(value: MotionTensor.channels)], dataType: .float32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: maxFrames)], dataType: .float32)
        let m = motion.dataPointer.assumingMemoryBound(to: Float.self)
        let k = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<maxFrames {
            k[i] = i < frames.count ? 1 : 0
            for c in 0..<MotionTensor.channels {
                m[i * MotionTensor.channels + c] = i < frames.count ? frames[i][c] : 0
            }
        }
        return try runner.transcribe(["motion": motion, "mask": mask]).lowercased()
    }
}

/// Audio (16 kHz mono) → text with stock Moonshine Tiny. Used to label "Say it" takes.
final class AudioTextModel: @unchecked Sendable {
    private let runner: MoonshineRunner
    static var isInstalled: Bool { MoonshineRunner.exists(prefix: "Audio", vocabFile: "audio_vocab.json") }

    init() throws {
        runner = try MoonshineRunner(prefix: "Audio", vocabFile: "audio_vocab.json")
    }

    var maxSamples: Int { runner.config.maxAudio ?? 16000 * 8 }

    func transcribe(_ audio: [Float]) throws -> String {
        let n = maxSamples
        let input = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
        let a = input.dataPointer.assumingMemoryBound(to: Float.self)
        let k = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<n {
            a[i] = i < audio.count ? audio[i] : 0
            k[i] = i < audio.count ? 1 : 0
        }
        return try runner.transcribe(["audio": input, "mask": mask])
    }
}
