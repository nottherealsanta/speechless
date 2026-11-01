import XCTest
@testable import Speechless

final class DecodingTests: XCTestCase {
    /// A synthetic "word": a burst of rotation at `frequency` cycles over the segment.
    private func take(_ frequency: Double, axis: Int, frames: Int = 30, noise: Double = 0.01) -> [MotionSample] {
        (0..<frames).map { i in
            let t = Double(i) / 25
            let envelope = sin(Double.pi * Double(i) / Double(frames - 1))
            let v = 0.2 * envelope * sin(2 * .pi * frequency * Double(i) / Double(frames))
            let jitter = { Double.random(in: -noise...noise) }
            var rotation = (jitter(), jitter(), jitter())
            switch axis {
            case 0: rotation.0 += v
            case 1: rotation.1 += v
            default: rotation.2 += v
            }
            return MotionSample(t: t, rotation: rotation)
        }
    }

    func testDTWIdenticalIsZeroAndWarpTolerant() {
        let a = MotionFeatures.frames(take(2, axis: 0, noise: 0), set: .rotation)
        let stretched = MotionFeatures.frames(take(2, axis: 0, frames: 40, noise: 0), set: .rotation)
        let other = MotionFeatures.frames(take(2, axis: 1, noise: 0), set: .rotation)
        XCTAssertEqual(DTW.distance(a, a), 0, accuracy: 1e-12)
        XCTAssertLessThan(DTW.distance(a, stretched), DTW.distance(a, other))
    }

    func testClassifierSeparatesDistinctWords() throws {
        var templates: [Template] = []
        for _ in 0..<6 {
            templates.append(Template(label: "yes", samples: take(2, axis: 0)))
            templates.append(Template(label: "no", samples: take(3, axis: 1)))
        }
        let classifier = try XCTUnwrap(TemplateClassifier(templates: templates, featureSet: .all))
        let prediction = classifier.predict(take(3, axis: 1), maxDistance: 10, minConfidence: 0)
        XCTAssertEqual(prediction.best?.label, "no")
        XCTAssertTrue(prediction.accepted)
    }

    func testEvaluationBeatsChanceOnlyWhenSignalExists() throws {
        var real: [Template] = []
        var noise: [Template] = []
        for _ in 0..<8 {
            real.append(Template(label: "yes", samples: take(2, axis: 0)))
            real.append(Template(label: "no", samples: take(3, axis: 2)))
            noise.append(Template(label: "yes", samples: take(0, axis: 0, noise: 0.05)))
            noise.append(Template(label: "no", samples: take(0, axis: 0, noise: 0.05)))
        }
        let signal = try XCTUnwrap(Evaluation.run(templates: real, featureSet: .all, permutations: 200))
        XCTAssertGreaterThan(signal.accuracy, 0.9)
        XCTAssertLessThan(signal.pValue, 0.01)

        let null = try XCTUnwrap(Evaluation.run(templates: noise, featureSet: .all, permutations: 200))
        XCTAssertGreaterThan(null.pValue, 0.01)
    }

    func testTrimDropsQuietPadding() {
        let quiet = (0..<20).map { MotionSample(t: Double($0) / 25) }
        let burst = take(2, axis: 0, noise: 0).map { s -> MotionSample in
            var s = s; s.t += 0.8; return s
        }
        let trimmed = MotionFeatures.trim(quiet + burst + quiet.map { var s = $0; s.t += 2; return s })
        XCTAssertLessThan(trimmed.count, 40)
        XCTAssertGreaterThanOrEqual(trimmed.count, 20)
    }
}

final class AutoLabelerTests: XCTestCase {
    private let motion = (0..<100).map { MotionSample(t: 10 + Double($0) / 25) }

    func testCutsWindowAroundWord() throws {
        let word = TimedWord(text: "Yes.", start: 11, end: 11.4, confidence: 0.9)
        let result = try AutoLabeler.label(word, motion: motion, vocabulary: ["yes"], settings: .init()).get()
        XCTAssertEqual(result.label, "yes")
        XCTAssertEqual(result.samples.first!.t, 10.85, accuracy: 0.05)
        XCTAssertEqual(result.samples.last!.t, 11.55, accuracy: 0.05)
    }

    func testSkipsUncertainUnknownAndMissingMotion() {
        let settings = AutoLabelSettings()
        func reason(_ w: TimedWord) -> SkipReason? {
            if case .failure(let e) = AutoLabeler.label(w, motion: motion, vocabulary: ["yes"], settings: settings) { return e.reason }
            return nil
        }
        XCTAssertEqual(reason(.init(text: "yes", start: 11, end: 11.4, confidence: 0.2)), .lowConfidence)
        XCTAssertEqual(reason(.init(text: "maybe", start: 11, end: 11.4, confidence: 0.9)), .notInVocabulary)
        XCTAssertEqual(reason(.init(text: "yes", start: 30, end: 30.4, confidence: 0.9)), .missingMotion)
    }

    func testSessionSplitExcludesSameSessionNeighbours() throws {
        var templates: [Template] = []
        for s in 0..<4 {
            for _ in 0..<3 {
                templates.append(Template(label: "yes", samples: (0..<30).map { MotionSample(t: Double($0) / 25, rotation: (sin(Double($0) / 3) * 0.2, 0, 0)) }, sessionID: "s\(s)"))
                templates.append(Template(label: "no", samples: (0..<30).map { MotionSample(t: Double($0) / 25, rotation: (0, sin(Double($0) / 2) * 0.2, 0)) }, sessionID: "s\(s)"))
            }
        }
        let result = try XCTUnwrap(Evaluation.run(templates: templates, featureSet: .rotation, permutations: 20, splitBySession: true))
        XCTAssertTrue(result.sessionSplit)
        XCTAssertGreaterThan(result.accuracy, 0.9)
    }
}

final class HexExampleTests: XCTestCase {
    /// Hex writes `<day>/<id>/{motion.csv, meta.json}`; the app lists a piece
    /// only when both are there and the text is usable.
    func testHexExamplesSkipUnfinishedFolders() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "hex-\(UUID().uuidString)")
        let day = root.appending(path: "2026-11-01")
        let good = day.appending(path: "20261101T143205.123-7f3a-0")
        let noMotion = day.appending(path: "20261101T143230.000-7f3a-1")
        let noText = day.appending(path: "20261101T143300.000-7f3a-2")
        defer { try? FileManager.default.removeItem(at: root) }
        for folder in [good, noMotion, noText] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try "t,ax,ay,az\n0,0,0,0\n".write(to: folder.appending(path: "motion.csv"),
                                            atomically: true, encoding: .utf8)
        }
        try #"{"schema":1,"text":"send the report","durationMs":2310}"#
            .write(to: good.appending(path: "meta.json"), atomically: true, encoding: .utf8)
        try #"{"schema":1,"text":"no motion column","durationMs":500}"#
            .write(to: noMotion.appending(path: "meta.json"), atomically: true, encoding: .utf8)
        try #"{"schema":1,"text":"  ","durationMs":500}"#
            .write(to: noText.appending(path: "meta.json"), atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: noMotion.appending(path: "motion.csv"))

        let examples = Studio.hexExamples(in: root)
        XCTAssertEqual(examples.count, 1, "only the finished piece counts")
        XCTAssertEqual(examples.first?.text, "send the report")
        XCTAssertEqual(examples.first?.day, "2026-11-01")
        XCTAssertEqual(examples.first?.duration ?? 0, 2.31, accuracy: 0.001)
    }
}
