import AppKit
import Observation

struct DecodedWord: Identifiable, Hashable {
    let id = UUID()
    var prediction: Prediction
    var text: String { prediction.best?.label ?? "?" }
}

@MainActor @Observable
final class AppModel {
    enum Screen: String, CaseIterable, Identifiable {
        case sensors = "Sensors"
        case train = "Train"
        case decode = "Decode"
        case autoLabel = "Auto-label"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .sensors: "waveform.path.ecg"
            case .train: "square.stack.3d.up"
            case .decode: "text.bubble"
            case .autoLabel: "captions.bubble"
            }
        }
    }

    let motion = HeadphoneMotionService()
    let audio = AudioService()
    let transcriber = MicTranscriber()
    let capture = CaptureController()
    let recorder = SessionRecorder()

    var screen: Screen = .train {
        didSet { syncCaptureMode() }
    }

    // MARK: Training

    private(set) var dataset = TemplateStore.load()
    var selectedWord: String?
    var featureSet: FeatureSet = .all {
        didSet { rebuildClassifier() }
    }
    private(set) var classifier: TemplateClassifier?
    private(set) var evaluation: Evaluation?
    private(set) var isEvaluating = false
    private(set) var lastTemplateID: Template.ID?

    // MARK: Decoding

    var decodeMode: CaptureController.Mode = .hold {
        didSet { syncCaptureMode() }
    }
    var maxDistance: Double = 1.5
    var minConfidence: Double = 0.05
    private(set) var decoded: [DecodedWord] = []
    private(set) var lastSegment: [MotionSample] = []

    // MARK: Auto-labelling (speech recognizer → motion training data)

    /// Listen with STT and turn recognized words into candidate motion takes.
    private(set) var autoLabelEnabled = false
    /// Temporarily stop collecting without stopping the mic.
    var autoLabelPaused = false
    var autoLabelSettings = AutoLabelSettings()
    /// Add candidates straight to the training set instead of the review queue.
    var autoAccept = false
    private(set) var candidates: [LabelCandidate] = []
    private(set) var skipCounts: [SkipReason: Int] = [:]
    private(set) var acceptedCount = 0
    var splitEvalBySession = true {
        didSet { evaluation = nil }
    }
    private(set) var transfer: TransferEvaluation?
    private(set) var isTransferEvaluating = false
    var retentionDays: Int = UserDefaults.standard.object(forKey: "retentionDays") as? Int ?? 14 {
        didSet { UserDefaults.standard.set(retentionDays, forKey: "retentionDays") }
    }

    static let maxPendingCandidates = 500
    /// STT finalizes several seconds after speech, so keep enough motion history to look back.
    static let motionHistorySeconds: TimeInterval = 90

    @ObservationIgnored private var recentMotion: [MotionSample] = []
    @ObservationIgnored private var autoLabelSessionID = ""
    @ObservationIgnored private let launchSessionID = "manual-" + Date().ISO8601Format()

    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var started = false

    init() {
        selectedWord = dataset.vocabulary.first
        rebuildClassifier()
    }

    func start() {
        guard !started else { return }
        started = true
        motion.onSample = { [weak self] sample in self?.ingest(sample) }
        transcriber.onWords = { [weak self] words in self?.handleWords(words) }
        SessionRecorder.purge(olderThan: retentionDays)
        capture.onSegment = { [weak self] segment in self?.handleSegment(segment) }
        motion.start()
        audio.refreshDevices()
        installKeyMonitor()
    }

    private func ingest(_ sample: MotionSample) {
        capture.ingest(sample)
        recorder.record(sample)
        recentMotion.append(sample)
        if let first = recentMotion.first, sample.t - first.t > Self.motionHistorySeconds + 5 {
            let cutoff = sample.t - Self.motionHistorySeconds
            recentMotion.removeFirst(recentMotion.firstIndex { $0.t >= cutoff } ?? 0)
        }
    }

    private var currentSessionID: String {
        recorder.sessionID ?? (autoLabelEnabled ? autoLabelSessionID : launchSessionID)
    }

    // MARK: Auto-labelling

    func setAutoLabel(_ on: Bool) async {
        if on {
            autoLabelSessionID = "stt-" + Date().ISO8601Format()
            autoLabelPaused = false
            autoLabelEnabled = true
            if !audio.isRunning { await toggleMic() }
            if !audio.isRunning { autoLabelEnabled = false }
        } else {
            autoLabelEnabled = false
        }
    }

    private func handleWords(_ words: [TimedWord]) {
        guard autoLabelEnabled, !autoLabelPaused else { return }
        let vocabulary = Set(dataset.vocabulary)
        for word in words {
            switch AutoLabeler.label(word, motion: recentMotion, vocabulary: vocabulary, settings: autoLabelSettings) {
            case .success(let result):
                let candidate = LabelCandidate(
                    label: result.label, heard: word.text, samples: result.samples,
                    confidence: word.confidence, sessionID: currentSessionID,
                    condition: recorder.condition.rawValue)
                recorder.logWord(word, status: autoAccept ? "accepted" : "queued")
                if autoAccept {
                    accept(candidate)
                } else {
                    candidates.append(candidate)
                    if candidates.count > Self.maxPendingCandidates { candidates.removeFirst() }
                }
            case .failure(let skip):
                skipCounts[skip.reason, default: 0] += 1
                recorder.logWord(word, status: "skipped: \(skip.reason.rawValue)")
            }
        }
    }

    func accept(_ candidate: LabelCandidate, as label: String? = nil) {
        var candidate = candidate
        if let label {
            let word = AutoLabeler.normalize(label)
            guard !word.isEmpty else { return }
            candidate.label = word
        }
        candidates.removeAll { $0.id == candidate.id }
        if !dataset.vocabulary.contains(candidate.label) { dataset.vocabulary.append(candidate.label) }
        dataset.templates.append(candidate.template)
        acceptedCount += 1
        persist()
    }

    func acceptAllCandidates() {
        let pending = candidates
        candidates.removeAll()
        for c in pending where !dataset.vocabulary.contains(c.label) { dataset.vocabulary.append(c.label) }
        dataset.templates.append(contentsOf: pending.map(\.template))
        acceptedCount += pending.count
        persist()
    }

    func reject(_ id: LabelCandidate.ID) {
        candidates.removeAll { $0.id == id }
    }

    func clearCandidates() {
        candidates.removeAll()
        skipCounts.removeAll()
    }

    /// Train on STT-labelled spoken/whispered takes, test on silent-mouthing takes.
    func evaluateTransfer() {
        guard !isTransferEvaluating else { return }
        let silent = SessionRecorder.Condition.silent.rawValue
        let train = dataset.templates.filter { $0.source == "stt" && $0.condition != silent }
        let test = dataset.templates.filter { $0.condition == silent }
        let featureSet = featureSet
        isTransferEvaluating = true
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                TransferEvaluation.run(train: train, test: test, featureSet: featureSet)
            }.value
            self.transfer = result
            self.isTransferEvaluating = false
        }
    }

    func deleteAutoLabelledTakes() {
        dataset.templates.removeAll { $0.source == "stt" }
        persist()
    }

    private func syncCaptureMode() {
        capture.autoEnabled = screen == .decode && decodeMode == .auto
    }

    private func handleSegment(_ segment: [MotionSample]) {
        lastSegment = segment
        switch screen {
        case .train:
            guard let word = selectedWord else { return }
            addTemplate(Template(label: word, samples: segment, sessionID: currentSessionID,
                                 condition: recorder.condition.rawValue, source: "manual"))
            recorder.logEvent(segment, kind: "train", label: word, confidence: nil)
        case .decode:
            guard let classifier else { return }
            let prediction = classifier.predict(segment, maxDistance: maxDistance, minConfidence: minConfidence)
            decoded.append(DecodedWord(prediction: prediction))
            recorder.logEvent(segment, kind: prediction.accepted ? "decode" : "reject",
                              label: prediction.best?.label ?? "", confidence: prediction.confidence)
        case .sensors, .autoLabel:
            recorder.logEvent(segment, kind: "segment", label: recorder.prompt, confidence: nil)
        }
    }

    // MARK: Space bar = hold to capture

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard event.keyCode == 49 else { return event }
            let isDown = event.type == .keyDown
            let isRepeat = isDown && event.isARepeat
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self, self.spaceCapturesMotion else { return false }
                if !isDown {
                    self.capture.end()
                } else if !isRepeat {
                    self.capture.begin()
                }
                return true
            }
            return consumed ? nil : event
        }
    }

    private var spaceCapturesMotion: Bool {
        // Let text fields keep the space bar.
        if NSApp.keyWindow?.firstResponder is NSText { return false }
        switch screen {
        case .train: return selectedWord != nil
        case .decode: return decodeMode == .hold && classifier != nil
        case .sensors, .autoLabel: return recorder.isRecording
        }
    }

    // MARK: Dataset editing

    func templates(for word: String) -> [Template] {
        dataset.templates.filter { $0.label == word }
    }

    func addWord(_ raw: String) {
        let word = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !word.isEmpty, !dataset.vocabulary.contains(word) else { return }
        dataset.vocabulary.append(word)
        selectedWord = word
        persist()
    }

    func removeWord(_ word: String) {
        dataset.vocabulary.removeAll { $0 == word }
        dataset.templates.removeAll { $0.label == word }
        if selectedWord == word { selectedWord = dataset.vocabulary.first }
        persist()
    }

    func addTemplate(_ template: Template) {
        dataset.templates.append(template)
        lastTemplateID = template.id
        persist()
    }

    func deleteTemplate(_ id: Template.ID) {
        dataset.templates.removeAll { $0.id == id }
        persist()
    }

    private func persist() {
        TemplateStore.save(dataset)
        rebuildClassifier()
    }

    private func rebuildClassifier() {
        classifier = TemplateClassifier(templates: dataset.templates, featureSet: featureSet)
        evaluation = nil
        transfer = nil
    }

    func evaluate() {
        guard !isEvaluating else { return }
        isEvaluating = true
        let templates = dataset.templates
        let featureSet = featureSet
        let split = splitEvalBySession
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Evaluation.run(templates: templates, featureSet: featureSet, splitBySession: split)
            }.value
            self.evaluation = result
            self.isEvaluating = false
        }
    }

    // MARK: Decode transcript

    var transcript: String {
        decoded.filter(\.prediction.accepted).map(\.text).joined(separator: " ")
    }

    func clearDecoded() {
        decoded.removeAll()
        lastSegment = []
    }

    // MARK: Mic

    func toggleMic() async {
        if audio.isRunning {
            autoLabelEnabled = false
            transcriber.stop()
            audio.stop()
        } else {
            await audio.start()
            if audio.isRunning { await transcriber.start(sink: audio.sink) }
        }
    }
}
