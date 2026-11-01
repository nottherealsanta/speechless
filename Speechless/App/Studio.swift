import AVFoundation
import AppKit
import Observation

/// One saved training example: the motion + audio of one utterance and its (edited) text.
/// Stored at ~/Documents/Speechless/Clips/<id>/{motion.csv, audio.wav, meta.json}; read by
/// `training/speechless_train/data.py:load_clips`.
struct DraftSegment: Identifiable, Hashable {
    var id = UUID()
    var text = ""
    var motion: [MotionSample]
    var audio: [Float]
    var duration: Double { (motion.last?.t ?? 0) - (motion.first?.t ?? 0) }
}

struct Clip: Identifiable, Codable, Hashable {
    var id: String
    var text: String
    var condition: String
    var duration: Double
    var createdAt: Date
}

@MainActor @Observable
final class Studio {
    enum Screen: String, CaseIterable, Identifiable {
        case train = "Train"
        case inference = "Inference"
        var id: String { rawValue }
        var symbol: String { self == .train ? "square.and.arrow.down.on.square" : "text.bubble" }
    }

    enum Condition: String, CaseIterable, Identifiable, Codable {
        case normal = "Normal speech"
        case whisper = "Whisper"
        case silent = "Silent mouthing"
        var id: String { rawValue }
    }

    let motion = HeadphoneMotionService()
    let audio = AudioService()

    var screen: Screen = .train
    var condition: Condition = .normal

    // Recording state (shared by both screens).
    private(set) var isRecording = false
    private(set) var recordingStarted: Date?

    // Train: a long recording is split at pauses into segments, each with editable text.
    var drafts: [DraftSegment] = []
    private(set) var draftMotion: [MotionSample] = []
    private(set) var isTranscribing = false
    private(set) var clips: [Clip] = []
    private(set) var message: String?

    // Inference
    private(set) var inferenceLines: [String] = []
    private(set) var isInferring = false
    private(set) var lastInferenceMotion: [MotionSample] = []
    private(set) var modelStatus = ""
    /// Parakeet (via transcribe.cpp) when available; otherwise Moonshine Tiny.
    private(set) var parakeet = ParakeetTranscriber.locate()
    var sttName: String { parakeet.map { "Parakeet · \($0.modelName)" } ?? "Moonshine Tiny" }

    // Training-data selection (multi-select in the clip list).
    var selectedClips: Set<Clip.ID> = []

    // Training run (python)
    private(set) var isTraining = false
    private(set) var trainingLog = ""

    /// Long recordings are allowed; they're split into model-sized pieces afterwards.
    static let maxSeconds: TimeInterval = 15 * 60
    var passageIndex = Int.random(in: 0..<ReadingPassages.all.count)
    var passage: String { ReadingPassages.all[passageIndex % ReadingPassages.all.count] }

    func nextPassage() { passageIndex = (passageIndex + 1) % ReadingPassages.all.count }
    static var clipsRoot: URL {
        URL.documentsDirectory.appending(path: "Speechless/Clips", directoryHint: .isDirectory)
    }
    /// The repo's `training/` folder, located from this source file at build time.
    static let trainingDirectory = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "training", directoryHint: .isDirectory)

    @ObservationIgnored private let audioClip = AudioClipBuffer()
    @ObservationIgnored private var audioSinkID: UUID?
    @ObservationIgnored private var recordingMotion: [MotionSample] = []
    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var audioModel: AudioTextModel?
    @ObservationIgnored private var motionModel: MotionTextModel?
    @ObservationIgnored private var started = false

    func start() {
        guard !started else { return }
        started = true
        motion.onSample = { [weak self] sample in self?.ingest(sample) }
        motion.start()
        audio.refreshDevices()
        loadClips()
        refreshModelStatus()
        installKeyMonitor()
        Task {
            await audio.start()
            if audio.isRunning, audioSinkID == nil {
                audioSinkID = audio.sink.add { [audioClip] buffer, time in audioClip.append(buffer, time) }
            }
        }
    }

    private func ingest(_ sample: MotionSample) {
        guard isRecording else { return }
        recordingMotion.append(sample)
        if let first = recordingMotion.first, sample.t - first.t > Self.maxSeconds { stopRecording() }
    }

    // MARK: Recording

    func toggleRecording() {
        isRecording ? stopRecording() : startRecording()
    }

    func startRecording() {
        guard !isRecording, !isTranscribing, !isInferring else { return }
        message = nil
        recordingMotion = []
        if screen == .train {
            audioClip.begin()
        }
        recordingStarted = Date()
        isRecording = true
    }

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        let motion = recordingMotion
        recordingMotion = []
        switch screen {
        case .train:
            let (samples, audioStart) = audioClip.end()
            draftMotion = motion
            prepareDrafts(motion: motion, audio: samples, audioStart: audioStart)
        case .inference:
            infer(motion)
        }
    }

    // MARK: Train

    /// Splits the recording into pieces. Spoken: Parakeet transcribes the whole recording and
    /// pieces follow its word timestamps (fallback: split at audio pauses + Moonshine per piece).
    /// Silent: split at motion pauses; you type the text.
    private func prepareDrafts(motion: [MotionSample], audio: [Float], audioStart: TimeInterval?) {
        drafts = []
        if condition != .silent, !audio.isEmpty, let a0 = audioStart, let parakeet {
            transcribeWithParakeet(parakeet, motion: motion, audio: audio, audioStart: a0)
            return
        }
        guard let m0 = motion.first?.t, motion.count >= 4 else {
            message = "No motion captured — are the AirPods streaming?"
            return
        }
        let useAudio = condition != .silent && !audio.isEmpty && audioStart != nil
        var windows: [(start: TimeInterval, end: TimeInterval)] = []
        if useAudio, let a0 = audioStart {
            let ranges = Segmenter.segments(Segmenter.audioEnergy(audio), .init(frameRate: 50, absoluteFloor: 0.004))
            windows = ranges.map { (a0 + Double($0.lowerBound) / 50, a0 + Double($0.upperBound + 1) / 50) }
        } else {
            let ranges = Segmenter.segments(Segmenter.motionEnergy(motion), .init(frameRate: MotionTensor.hz))
            windows = ranges.map { (motion[$0.lowerBound].t, motion[$0.upperBound].t) }
        }
        if windows.isEmpty, let last = motion.last?.t, last - m0 <= Segmenter.Options(frameRate: 1).maxLength {
            windows = [(m0, last)]
        }
        drafts = windows.compactMap { w in
            let slice = motion.filter { $0.t >= w.start && $0.t <= w.end }
            guard slice.count >= 4 else { return nil }
            var clipAudio: [Float] = []
            if useAudio, let a0 = audioStart {
                let lo = max(0, Int((w.start - a0) * 16000)), hi = min(audio.count, Int((w.end - a0) * 16000))
                if hi > lo { clipAudio = Array(audio[lo..<hi]) }
            }
            return DraftSegment(motion: slice, audio: clipAudio)
        }
        if drafts.isEmpty { message = "Couldn't find any speech or movement in that recording." }
        if useAudio { transcribeDrafts() }
    }

    private func transcribeWithParakeet(_ parakeet: ParakeetTranscriber, motion: [MotionSample],
                                        audio: [Float], audioStart a0: TimeInterval) {
        isTranscribing = true
        Task {
            let result = await Task.detached(priority: .userInitiated) { try? parakeet.transcribe(audio) }.value
            self.isTranscribing = false
            guard let result else {
                self.message = "Parakeet failed — falling back to Moonshine."
                self.parakeet = nil
                self.prepareDrafts(motion: motion, audio: audio, audioStart: a0)
                return
            }
            self.drafts = ParakeetTranscriber.utterances(result.words).compactMap { u in
                let start = a0 + u.start, end = a0 + u.end
                let slice = motion.filter { $0.t >= start && $0.t <= end }
                guard slice.count >= 4 else { return nil }
                let lo = max(0, Int(u.start * 16000)), hi = min(audio.count, Int(u.end * 16000))
                return DraftSegment(text: u.text, motion: slice, audio: hi > lo ? Array(audio[lo..<hi]) : [])
            }
            if self.drafts.isEmpty { self.message = "Parakeet didn't hear any words." }
        }
    }

    private func transcribeDrafts() {
        guard AudioTextModel.isInstalled else {
            message = "Moonshine audio model not installed. Run: cd training && uv run python -m speechless_train.export_audio --install"
            return
        }
        isTranscribing = true
        let jobs = drafts.map { ($0.id, $0.audio) }
        let cached = audioModel
        Task {
            var model = cached
            for (id, audio) in jobs where !audio.isEmpty {
                let result = await Task.detached(priority: .userInitiated) { () -> (AudioTextModel?, Result<String, Error>) in
                    do {
                        let m = try model ?? AudioTextModel()
                        return (m, .success(try m.transcribe(Array(audio.prefix(m.maxSamples)))))
                    } catch {
                        return (model, .failure(error))
                    }
                }.value
                model = result.0
                switch result.1 {
                case .success(let text):
                    if let i = self.drafts.firstIndex(where: { $0.id == id }) { self.drafts[i].text = text }
                case .failure(let error):
                    self.message = "Transcription failed: \(error.localizedDescription)"
                }
            }
            self.audioModel = model
            self.isTranscribing = false
        }
    }

    var savableCount: Int {
        isRecording ? 0 : drafts.filter { !Self.normalize($0.text).isEmpty }.count
    }

    /// Saves every segment that has text as its own clip.
    func saveDrafts() {
        var saved = 0
        for draft in drafts where !Self.normalize(draft.text).isEmpty {
            if save(draft) { saved += 1 }
        }
        drafts.removeAll { !Self.normalize($0.text).isEmpty }
        if drafts.isEmpty { draftMotion = [] }
        message = "Added \(saved) clip\(saved == 1 ? "" : "s") to training data."
    }

    private func save(_ draft: DraftSegment) -> Bool {
        guard let first = draft.motion.first, let last = draft.motion.last else { return false }
        let id = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: true))
            .replacingOccurrences(of: ":", with: "") + "-" + draft.id.uuidString.prefix(4)
        let dir = Self.clipsRoot.appending(path: id, directoryHint: .isDirectory)
        let clip = Clip(id: id, text: Self.normalize(draft.text), condition: condition.rawValue,
                        duration: last.t - first.t, createdAt: Date())
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let csv = ([MotionSample.csvHeader] + draft.motion.map { $0.csvRow(relativeTo: first.t) })
                .joined(separator: "\n") + "\n"
            try Data(csv.utf8).write(to: dir.appending(path: "motion.csv"))
            if !draft.audio.isEmpty { try AudioClipBuffer.writeWAV(draft.audio, to: dir.appending(path: "audio.wav")) }
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(clip).write(to: dir.appending(path: "meta.json"))
        } catch {
            message = "Couldn't save: \(error.localizedDescription)"
            return false
        }
        clips.insert(clip, at: 0)
        return true
    }

    func removeDraft(_ id: DraftSegment.ID) {
        drafts.removeAll { $0.id == id }
    }

    func discardDrafts() {
        drafts = []
        draftMotion = []
    }

    func deleteClip(_ clip: Clip) {
        deleteClips([clip.id])
    }

    /// Moves clips to the Trash (recoverable) and removes them from the list.
    func deleteClips(_ ids: Set<Clip.ID>) {
        for id in ids {
            let url = Self.clipsRoot.appending(path: id, directoryHint: .isDirectory)
            if (try? FileManager.default.trashItem(at: url, resultingItemURL: nil)) == nil {
                try? FileManager.default.removeItem(at: url)
            }
        }
        clips.removeAll { ids.contains($0.id) }
        selectedClips.subtract(ids)
        message = "Deleted \(ids.count) clip\(ids.count == 1 ? "" : "s") (moved to Trash)."
    }

    func revealClips() {
        try? FileManager.default.createDirectory(at: Self.clipsRoot, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([Self.clipsRoot])
    }

    private func loadClips() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let dirs = (try? FileManager.default.contentsOfDirectory(at: Self.clipsRoot, includingPropertiesForKeys: nil)) ?? []
        clips = dirs.compactMap { dir in
            (try? Data(contentsOf: dir.appending(path: "meta.json"))).flatMap { try? decoder.decode(Clip.self, from: $0) }
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    static func normalize(_ text: String) -> String {
        let kept = text.lowercased().unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0) || $0 == "'" || $0 == " "
        }
        return String(String.UnicodeScalarView(kept)).split(separator: " ").joined(separator: " ")
    }

    // MARK: Inference

    func refreshModelStatus() {
        motionModel = nil
        if MotionTextModel.isInstalled {
            let date = (try? FileManager.default.attributesOfItem(
                atPath: MoonshineRunner.directory.appending(path: "MotionEncoder.mlpackage").path)[.modificationDate] as? Date)
            modelStatus = "Motion model installed" + (date.map { " · \($0.formatted(date: .abbreviated, time: .shortened))" } ?? "")
        } else {
            modelStatus = "No motion model yet — record clips, then Train Model."
        }
    }

    private func infer(_ samples: [MotionSample]) {
        lastInferenceMotion = samples
        guard samples.count >= 4 else { return }
        guard MotionTextModel.isInstalled else {
            message = "No motion model installed yet."
            return
        }
        isInferring = true
        let cached = motionModel
        var pieces = Segmenter.segments(Segmenter.motionEnergy(samples), .init(frameRate: MotionTensor.hz))
            .map { Array(samples[$0]) }
        if pieces.isEmpty { pieces = [samples] }
        Task {
            let result = await Task.detached(priority: .userInitiated) { () -> (MotionTextModel?, Result<[String], Error>) in
                do {
                    let model = try cached ?? MotionTextModel()
                    let texts = try pieces.filter { $0.count >= 4 }.map { try model.transcribe($0) }
                    return (model, .success(texts.filter { !$0.isEmpty }))
                } catch {
                    return (cached, .failure(error))
                }
            }.value
            self.motionModel = result.0
            switch result.1 {
            case .success(let texts): self.inferenceLines.append(texts.isEmpty ? "…" : texts.joined(separator: " "))
            case .failure(let error): self.message = "Inference failed: \(error.localizedDescription)"
            }
            self.isInferring = false
        }
    }

    func clearInference() {
        inferenceLines = []
        lastInferenceMotion = []
    }

    // MARK: Training (runs training/ with uv)

    func trainModel(epochs: Int) {
        guard !isTraining else { return }
        isTraining = true
        trainingLog = ""
        let dir = Self.trainingDirectory
        let script = """
        export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"; unset VIRTUAL_ENV
        cd "\(dir.path)" && uv run python -m speechless_train.train --epochs \(epochs) \
          && uv run python -m speechless_train.export checkpoints/motion-moonshine.pt --install
        """
        let process = Process()
        process.executableURL = URL(filePath: "/bin/zsh")
        process.arguments = ["-lc", script]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let text = String(decoding: handle.availableData, as: UTF8.self)
            guard !text.isEmpty else { return }
            Task { @MainActor in self?.appendLog(text) }
        }
        process.terminationHandler = { [weak self] p in
            let status = p.terminationStatus
            Task { @MainActor in
                pipe.fileHandleForReading.readabilityHandler = nil
                self?.isTraining = false
                self?.appendLog(status == 0 ? "\n✓ Model trained and installed.\n" : "\n✗ Training failed (exit \(status)).\n")
                self?.refreshModelStatus()
            }
        }
        do {
            try process.run()
        } catch {
            isTraining = false
            appendLog("Couldn't start training: \(error.localizedDescription)\n")
        }
    }

    private func appendLog(_ text: String) {
        // Drop noisy progress-bar and warning lines.
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).filter {
            !$0.contains("it/s]") && !$0.contains("Redirects") && !$0.contains("VIRTUAL_ENV") && !$0.contains("max_length")
        }
        trainingLog += lines.joined(separator: "\n")
        if trainingLog.count > 20_000 { trainingLog = String(trainingLog.suffix(20_000)) }
    }

    // MARK: Space = hold to record

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard event.keyCode == 49 else { return event }
            let isDown = event.type == .keyDown
            let isRepeat = event.isARepeat
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self, !(NSApp.keyWindow?.firstResponder is NSText) else { return false }
                if isDown, !isRepeat { self.startRecording() } else if !isDown { self.stopRecording() }
                return true
            }
            return consumed ? nil : event
        }
    }
}

/// Collects microphone audio as 16 kHz mono floats from the audio thread.
final class AudioClipBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var active = false
    private var samples: [Float] = []
    private var startTime: TimeInterval?
    private var converter: AVAudioConverter?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!

    func begin() {
        lock.withLock { samples = []; startTime = nil; active = true }
    }

    /// Audio and the motion-clock time (seconds since boot) of its first sample.
    func end() -> ([Float], TimeInterval?) {
        lock.withLock { active = false; return (samples, startTime) }
    }

    func append(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
        lock.withLock {
            guard active else { return }
            if startTime == nil {
                startTime = time.isHostTimeValid
                    ? AVAudioTime.seconds(forHostTime: time.hostTime)
                    : ProcessInfo.processInfo.systemUptime - Double(buffer.frameLength) / buffer.format.sampleRate
            }
            if converter?.inputFormat != buffer.format {
                converter = AVAudioConverter(from: buffer.format, to: target)
            }
            guard let converter else { return }
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000 / buffer.format.sampleRate) + 32
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var fed = false
            converter.convert(to: out, error: nil) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true
                status.pointee = .haveData
                return buffer
            }
            if let data = out.floatChannelData?[0] {
                samples.append(contentsOf: UnsafeBufferPointer(start: data, count: Int(out.frameLength)))
            }
        }
    }

    static func writeWAV(_ samples: [Float], to url: URL) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
    }
}
