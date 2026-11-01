import Foundation

/// One finalized speech-recognizer word on the motion clock (seconds since boot).
struct TimedWord: Hashable, Sendable {
    var text: String
    var start: TimeInterval
    var end: TimeInterval
    /// Recognizer confidence, 0…1. Partial results report 0.
    var confidence: Double
}

/// A motion slice that speech recognition says was a particular word. Waits in the review
/// queue until accepted into the training set (or discarded).
struct LabelCandidate: Identifiable, Hashable, Sendable {
    var id = UUID()
    var label: String
    var heard: String
    var samples: [MotionSample]
    var confidence: Double
    var sessionID: String
    var condition: String
    var createdAt = Date()

    var template: Template {
        Template(label: label, samples: samples, sessionID: sessionID, condition: condition,
                 source: "stt", sttConfidence: confidence)
    }
}

struct AutoLabelSettings: Hashable, Sendable {
    var minConfidence: Double = 0.5
    /// Motion kept on each side of the word, seconds.
    var padding: TimeInterval = 0.15
    var minDuration: TimeInterval = 0.12
    var maxDuration: TimeInterval = 1.5
    /// Only keep words already in the vocabulary.
    var vocabularyOnly = true
    /// Reject windows whose attitude swings more than this (radians): that's head movement, not articulation.
    var maxHeadTurn: Double = 0.35
    /// Fraction of the window that must be covered by motion samples.
    var minCoverage: Double = 0.8
}

enum SkipReason: String, CaseIterable, Sendable {
    case lowConfidence = "Low STT confidence"
    case notInVocabulary = "Not in vocabulary"
    case badDuration = "Too short / too long"
    case missingMotion = "Missing motion"
    case headMovement = "Head movement"
}

/// Turns speech-recognizer words into labelled motion segments.
enum AutoLabeler {
    static func normalize(_ word: String) -> String {
        word.lowercased()
            .unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == "'" }
            .map(String.init).joined()
    }

    static func label(
        _ word: TimedWord, motion: [MotionSample], vocabulary: Set<String>, settings: AutoLabelSettings
    ) -> Result<(label: String, samples: [MotionSample]), SkipError> {
        let label = normalize(word.text)
        let duration = word.end - word.start
        guard !label.isEmpty, duration >= settings.minDuration, duration <= settings.maxDuration else {
            return .failure(.init(.badDuration))
        }
        guard word.confidence >= settings.minConfidence else { return .failure(.init(.lowConfidence)) }
        if settings.vocabularyOnly, !vocabulary.contains(label) { return .failure(.init(.notInVocabulary)) }

        let lo = word.start - settings.padding
        let hi = word.end + settings.padding
        let window = motion.filter { $0.t >= lo && $0.t <= hi }
        guard window.count >= 4, let first = window.first, let last = window.last,
              (last.t - first.t) >= settings.minCoverage * (hi - lo) else {
            return .failure(.init(.missingMotion))
        }
        if headTurn(window) > settings.maxHeadTurn { return .failure(.init(.headMovement)) }
        return .success((label, window))
    }

    /// Largest attitude swing (pitch/roll/yaw range) inside the window, radians.
    static func headTurn(_ samples: [MotionSample]) -> Double {
        func range(_ key: KeyPath<MotionSample, Double>) -> Double {
            let values = samples.map { $0[keyPath: key] }
            return (values.max() ?? 0) - (values.min() ?? 0)
        }
        return max(range(\.pitch), range(\.roll), min(range(\.yaw), 2 * .pi - range(\.yaw)))
    }

    struct SkipError: Error, Hashable {
        var reason: SkipReason
        init(_ reason: SkipReason) { self.reason = reason }
    }
}
