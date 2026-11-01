import Foundation

struct Template: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    var label: String
    var samples: [MotionSample]
    var createdAt = Date()
    /// Recording session the take came from. Evaluation never matches a take against
    /// neighbours from its own session when sessions are available.
    var sessionID: String?
    /// "Normal speech", "Whisper", "Silent mouthing"…
    var condition: String?
    /// "manual" (Space-held) or "stt" (auto-labelled by speech recognition).
    var source: String?
    var sttConfidence: Double?

    var duration: TimeInterval {
        guard let first = samples.first, let last = samples.last else { return 0 }
        return last.t - first.t
    }
}

struct RankedLabel: Identifiable, Hashable, Sendable {
    var label: String
    var distance: Double
    var id: String { label }
}

struct Prediction: Hashable, Sendable {
    var ranked: [RankedLabel]
    /// 0…1 margin between the best and second-best class.
    var confidence: Double
    var accepted: Bool

    var best: RankedLabel? { ranked.first }
}

/// Small-vocabulary recognizer: nearest-neighbour DTW against user-recorded templates.
/// Deliberately simple — the question this prototype answers is whether the signal is
/// there at all, not how far a big model can push it.
struct TemplateClassifier: Sendable {
    let featureSet: FeatureSet
    private let entries: [(label: String, frames: [[Double]])]
    private let scales: [Double]

    var labels: [String] { Array(Set(entries.map(\.label))).sorted() }

    init?(templates: [Template], featureSet: FeatureSet) {
        guard !featureSet.isEmpty else { return nil }
        let raw = templates.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
        guard Set(templates.map(\.label)).count >= 2 else { return nil }
        let scales = MotionFeatures.channelScales(raw)
        self.featureSet = featureSet
        self.scales = scales
        self.entries = zip(templates, raw).map { ($0.label, MotionFeatures.scaled($1, by: scales)) }
    }

    func rank(_ samples: [MotionSample]) -> [RankedLabel] {
        let query = MotionFeatures.scaled(
            MotionFeatures.frames(MotionFeatures.trim(samples), set: featureSet), by: scales)
        var best: [String: Double] = [:]
        for entry in entries {
            let d = DTW.distance(query, entry.frames)
            best[entry.label] = min(best[entry.label] ?? .infinity, d)
        }
        return best.map { RankedLabel(label: $0.key, distance: $0.value) }.sorted { $0.distance < $1.distance }
    }

    func predict(_ samples: [MotionSample], maxDistance: Double, minConfidence: Double) -> Prediction {
        let ranked = rank(samples)
        let confidence = Self.margin(ranked.map(\.distance))
        let accepted = (ranked.first?.distance ?? .infinity) <= maxDistance && confidence >= minConfidence
        return Prediction(ranked: ranked, confidence: confidence, accepted: accepted)
    }

    static func margin(_ sortedDistances: [Double]) -> Double {
        guard sortedDistances.count >= 2, sortedDistances[1] > 0 else { return 0 }
        return max(0, 1 - sortedDistances[0] / sortedDistances[1])
    }
}

/// Leave-one-out accuracy plus a label-permutation test: the honest answer to
/// "is the motion stream statistically distinguishable between words?"
struct Evaluation: Sendable {
    var labels: [String]
    /// confusion[actual][predicted]
    var confusion: [[Int]]
    var accuracy: Double
    /// Accuracy of always guessing the most common label.
    var chance: Double
    var permutationMean: Double
    var permutationP95: Double
    var pValue: Double
    var permutations: Int
    var templateCount: Int
    /// True if neighbours from the same recording session were excluded.
    var sessionSplit = false

    /// Group key used to keep a take from matching its own session. Takes without a session
    /// get a unique group, i.e. plain leave-one-out.
    static func groups(_ templates: [Template]) -> [String] {
        templates.map { $0.sessionID ?? $0.id.uuidString }
    }

    static func run(templates: [Template], featureSet: FeatureSet, permutations: Int = 500,
                    splitBySession: Bool = false) -> Evaluation? {
        let labels = Array(Set(templates.map(\.label))).sorted()
        let counts = Dictionary(grouping: templates, by: \.label).mapValues(\.count)
        guard labels.count >= 2, counts.values.allSatisfy({ $0 >= 2 }), !featureSet.isEmpty else { return nil }

        let raw = templates.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
        let scales = MotionFeatures.channelScales(raw)
        let seqs = raw.map { MotionFeatures.scaled($0, by: scales) }
        let n = seqs.count

        // Pairwise distances once; every labelling (true or permuted) reuses them.
        var distance = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<max(i + 1, n) {
                let d = DTW.distance(seqs[i], seqs[j])
                distance[i][j] = d
                distance[j][i] = d
            }
        }

        let index = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })
        let truth = templates.map { index[$0.label]! }
        let group = splitBySession ? groups(templates) : templates.map(\.id.uuidString)
        guard Set(group).count >= 2 else { return nil }

        func leaveOneOut(_ assignment: [Int]) -> [Int] {
            (0..<n).map { i in
                var best = [Double](repeating: .infinity, count: labels.count)
                for j in 0..<n where j != i && group[j] != group[i] {
                    best[assignment[j]] = min(best[assignment[j]], distance[i][j])
                }
                return best.indices.min { best[$0] < best[$1] } ?? 0
            }
        }

        func accuracy(_ predicted: [Int], _ actual: [Int]) -> Double {
            Double(zip(predicted, actual).filter { $0 == $1 }.count) / Double(n)
        }

        let predicted = leaveOneOut(truth)
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: labels.count), count: labels.count)
        for (p, a) in zip(predicted, truth) { confusion[a][p] += 1 }
        let observed = accuracy(predicted, truth)

        var generator = SystemRandomNumberGenerator()
        var nullAccuracies: [Double] = []
        nullAccuracies.reserveCapacity(permutations)
        for _ in 0..<permutations {
            let shuffled = truth.shuffled(using: &generator)
            nullAccuracies.append(accuracy(leaveOneOut(shuffled), shuffled))
        }
        nullAccuracies.sort()
        let atLeast = nullAccuracies.filter { $0 >= observed }.count

        return Evaluation(
            labels: labels,
            confusion: confusion,
            accuracy: observed,
            chance: Double(counts.values.max()!) / Double(n),
            permutationMean: nullAccuracies.reduce(0, +) / Double(max(permutations, 1)),
            permutationP95: nullAccuracies.isEmpty ? 0 : nullAccuracies[Int(Double(nullAccuracies.count - 1) * 0.95)],
            pValue: Double(atLeast + 1) / Double(permutations + 1),
            permutations: permutations,
            templateCount: n,
            sessionSplit: splitBySession)
    }
}

/// Train on one set of takes (e.g. STT-labelled spoken sessions), test on another
/// (e.g. silent mouthing). Answers "does what was learned from speech survive silence?"
struct TransferEvaluation: Sendable {
    var labels: [String]
    var confusion: [[Int]]
    var accuracy: Double
    var chance: Double
    var trainCount: Int
    var testCount: Int
    /// Accuracy with shuffled training labels, 95th percentile, and p-value.
    var permutationP95: Double
    var pValue: Double

    static func run(train: [Template], test: [Template], featureSet: FeatureSet,
                    permutations: Int = 200) -> TransferEvaluation? {
        let labels = Array(Set(train.map(\.label))).sorted()
        let test = test.filter { labels.contains($0.label) }
        guard labels.count >= 2, !test.isEmpty, !featureSet.isEmpty else { return nil }

        let trainRaw = train.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
        let scales = MotionFeatures.channelScales(trainRaw)
        let trainSeqs = trainRaw.map { MotionFeatures.scaled($0, by: scales) }
        let testSeqs = test.map {
            MotionFeatures.scaled(MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet), by: scales)
        }
        let distance = testSeqs.map { q in trainSeqs.map { DTW.distance(q, $0) } }

        let index = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })
        let trainTruth = train.map { index[$0.label]! }
        let testTruth = test.map { index[$0.label]! }

        func predict(_ assignment: [Int]) -> [Int] {
            distance.map { row in
                var best = [Double](repeating: .infinity, count: labels.count)
                for (j, d) in row.enumerated() { best[assignment[j]] = min(best[assignment[j]], d) }
                return best.indices.min { best[$0] < best[$1] } ?? 0
            }
        }
        func accuracy(_ p: [Int]) -> Double {
            Double(zip(p, testTruth).filter { $0 == $1 }.count) / Double(testTruth.count)
        }

        let predicted = predict(trainTruth)
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: labels.count), count: labels.count)
        for (p, a) in zip(predicted, testTruth) { confusion[a][p] += 1 }
        let observed = accuracy(predicted)

        var generator = SystemRandomNumberGenerator()
        var null = (0..<permutations).map { _ in accuracy(predict(trainTruth.shuffled(using: &generator))) }
        null.sort()
        let counts = Dictionary(grouping: testTruth, by: { $0 }).mapValues(\.count)
        return TransferEvaluation(
            labels: labels, confusion: confusion, accuracy: observed,
            chance: Double(counts.values.max() ?? 0) / Double(testTruth.count),
            trainCount: train.count, testCount: test.count,
            permutationP95: null.isEmpty ? 0 : null[Int(Double(null.count - 1) * 0.95)],
            pValue: Double(null.filter { $0 >= observed }.count + 1) / Double(permutations + 1))
    }
}
