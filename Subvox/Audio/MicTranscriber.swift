import AVFoundation
import Observation
import Speech

/// Continuous speech-to-text from the microphone. Used as the reference ("ground truth")
/// next to the motion decoder, and to test whether whispering is enough for input.
@MainActor @Observable
final class MicTranscriber {
    private(set) var isRunning = false
    private(set) var text = ""
    private(set) var isOnDevice = false
    private(set) var error: String?
    /// Finalized words with timestamps on the motion clock (seconds since boot).
    @ObservationIgnored var onWords: (([TimedWord]) -> Void)?

    @ObservationIgnored private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    @ObservationIgnored private let box = RequestBox()
    @ObservationIgnored private var task: SFSpeechRecognitionTask?
    @ObservationIgnored private var sinkID: UUID?
    @ObservationIgnored private weak var sink: AudioSink?
    @ObservationIgnored private var finalized = ""
    @ObservationIgnored private var generation = 0

    func start(sink: AudioSink) async {
        guard !isRunning else { return }
        error = nil

        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard status == .authorized else {
            error = "Speech recognition not authorized."
            return
        }
        guard let recognizer, recognizer.isAvailable else {
            error = "Speech recognizer unavailable."
            return
        }

        self.sink = sink
        sinkID = sink.add { [box] buffer, time in box.append(buffer, time) }
        isRunning = true
        startTask()
    }

    func stop() {
        isRunning = false
        generation += 1
        if let sinkID { sink?.remove(sinkID) }
        sinkID = nil
        box.set(nil)
        task?.cancel()
        task = nil
    }

    func clear() {
        finalized = ""
        text = ""
    }

    private func startTask() {
        guard isRunning, let recognizer else { return }
        generation += 1

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = false
        isOnDevice = recognizer.supportsOnDeviceRecognition
        request.requiresOnDeviceRecognition = isOnDevice
        box.set(request)

        task?.cancel()
        task = recognizer.recognitionTask(
            with: request, resultHandler: Self.resultHandler(owner: self, generation: generation, box: box, request: request))
    }

    private func handle(generation: Int, partial: String?, isFinal: Bool, failed: Bool, words: [TimedWord]) {
        guard generation == self.generation, isRunning else { return }
        if isFinal, !words.isEmpty { onWords?(words) }
        if let partial {
            text = finalized + partial
        }
        if isFinal || failed {
            if let partial, !partial.isEmpty { finalized += partial + " " }
            text = finalized
            // Recognition tasks end on silence/timeout; keep listening with a fresh one.
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(failed ? 500 : 50))
                self?.startTask()
            }
        }
    }

    nonisolated private static func resultHandler(
        owner: MicTranscriber, generation: Int, box: RequestBox, request: SFSpeechAudioBufferRecognitionRequest
    ) -> (SFSpeechRecognitionResult?, Error?) -> Void {
        { [weak owner] result, error in
            let partial = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let failed = error != nil
            var words: [TimedWord] = []
            // Segment timestamps are relative to the first buffer appended to this request.
            if isFinal, let result, let origin = box.origin(of: request) {
                words = result.bestTranscription.segments.map {
                    TimedWord(text: $0.substring, start: origin + $0.timestamp,
                              end: origin + $0.timestamp + $0.duration, confidence: Double($0.confidence))
                }
            }
            Task { @MainActor in
                owner?.handle(generation: generation, partial: partial, isFinal: isFinal, failed: failed, words: words)
            }
        }
    }
}

/// The current recognition request, appended to from the audio thread.
private final class RequestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    /// Host time (seconds since boot) of the first buffer fed to each request.
    private var origins: [(key: ObjectIdentifier, host: TimeInterval)] = []

    func set(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.withLock {
            self.request?.endAudio()
            self.request = request
            if origins.count > 4 { origins.removeFirst(origins.count - 4) }
        }
    }

    func append(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
        lock.withLock {
            guard let request else { return }
            let key = ObjectIdentifier(request)
            if !origins.contains(where: { $0.key == key }) {
                let frames = Double(buffer.frameLength) / max(buffer.format.sampleRate, 1)
                let host = time.isHostTimeValid
                    ? AVAudioTime.seconds(forHostTime: time.hostTime)
                    : ProcessInfo.processInfo.systemUptime - frames
                origins.append((key, host))
            }
            request.append(buffer)
        }
    }

    func origin(of request: SFSpeechAudioBufferRecognitionRequest) -> TimeInterval? {
        lock.withLock { origins.last { $0.key == ObjectIdentifier(request) }?.host }
    }
}
