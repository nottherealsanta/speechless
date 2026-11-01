import SwiftUI

/// Speech recognition labels motion: speak (or whisper) normally, each recognized vocabulary
/// word becomes a candidate motion take, and you review them before they join the training set.
struct AutoLabelView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                controls
                filters
                queue
                evaluation
                privacy
            }
            .padding(20)
        }
    }

    // MARK: Controls

    private var controls: some View {
        @Bindable var model = model
        @Bindable var recorder = model.recorder
        return GroupBox("Collect") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Speak your vocabulary words. Each word the recognizer finalizes is cut from the AirPods motion stream using its timestamps and queued for review. Speech isn't silent mouthing — check transfer below before trusting it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                HStack {
                    Button {
                        Task { await model.setAutoLabel(!model.autoLabelEnabled) }
                    } label: {
                        Label(model.autoLabelEnabled ? "Stop Listening" : "Start Listening",
                              systemImage: model.autoLabelEnabled ? "stop.fill" : "mic.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(model.autoLabelEnabled ? .red : .accentColor)
                    .disabled(model.motion.status != .streaming && !model.autoLabelEnabled)

                    Toggle("Pause", isOn: $model.autoLabelPaused)
                        .toggleStyle(.button)
                        .disabled(!model.autoLabelEnabled)
                    LevelMeter(value: Double(model.audio.level))
                    Picker("Condition", selection: $recorder.condition) {
                        ForEach(SessionRecorder.Condition.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .frame(maxWidth: 260)
                    .disabled(recorder.isRecording)
                }
                if model.motion.status != .streaming {
                    Label(model.motion.status.label, systemImage: "airpodspro").font(.caption).foregroundStyle(.orange)
                }
                if model.autoLabelEnabled, !model.transcriber.isOnDevice {
                    Label("On-device recognition unavailable — audio is sent to Apple's servers.",
                          systemImage: "exclamationmark.icloud")
                        .font(.caption).foregroundStyle(.orange)
                }
                if let error = model.audio.error ?? model.transcriber.error {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                Text(model.transcriber.text.isEmpty ? "Heard: —" : "Heard: " + String(model.transcriber.text.suffix(200)))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack(spacing: 16) {
                    Text("Queued \(model.candidates.count)")
                    Text("Accepted \(model.acceptedCount)")
                    ForEach(SkipReason.allCases, id: \.self) { reason in
                        if let n = model.skipCounts[reason], n > 0 {
                            Text("\(reason.rawValue): \(n)").foregroundStyle(.secondary)
                        }
                    }
                }
                .font(.caption.monospacedDigit())
                Text("Tip: use Record Session too, so raw motion, audio and words.csv are saved for offline re-alignment.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private var filters: some View {
        @Bindable var model = model
        return GroupBox("Filters") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 24) {
                    slider("Min STT confidence", $model.autoLabelSettings.minConfidence, 0...1)
                    slider("Padding (s)", $model.autoLabelSettings.padding, 0...0.5)
                    slider("Max head turn (rad)", $model.autoLabelSettings.maxHeadTurn, 0.05...1.5)
                }
                HStack(spacing: 24) {
                    Toggle("Vocabulary words only", isOn: $model.autoLabelSettings.vocabularyOnly)
                    Toggle("Skip review (auto-accept)", isOn: $model.autoAccept)
                }
                .toggleStyle(.checkbox)
            }
            .font(.callout)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func slider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        LabeledContent(title) {
            Slider(value: value, in: range).frame(width: 110)
            Text(String(format: "%.2f", value.wrappedValue)).monospacedDigit().frame(width: 36)
        }
    }

    // MARK: Review queue

    private var queue: some View {
        GroupBox("Review queue (\(model.candidates.count))") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button("Accept All") { model.acceptAllCandidates() }
                        .disabled(model.candidates.isEmpty)
                    Button("Discard All", role: .destructive) { model.clearCandidates() }
                        .disabled(model.candidates.isEmpty)
                }
                if model.candidates.isEmpty {
                    Text("Nothing queued. Start listening and say some vocabulary words.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 50)
                } else {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 10)], spacing: 10) {
                        ForEach(model.candidates.reversed()) { CandidateTile(candidate: $0) }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    // MARK: Evaluation

    private var evaluation: some View {
        @Bindable var model = model
        let stt = model.dataset.templates.filter { $0.source == "stt" }.count
        let silent = model.dataset.templates.filter { $0.condition == SessionRecorder.Condition.silent.rawValue }.count
        return GroupBox("Does speech-labelled data transfer to silent mouthing?") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Trains on auto-labelled Normal/Whisper takes and tests on Silent mouthing takes (record those in Train with the condition set to Silent mouthing). The Train tab's Evaluate also holds out whole sessions when “split by session” is on.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("\(stt) STT takes · \(silent) silent takes").font(.callout.monospacedDigit())
                    Toggle("Train tab: split by session", isOn: $model.splitEvalBySession).toggleStyle(.checkbox)
                    Spacer()
                    Button(model.isTransferEvaluating ? "Evaluating…" : "Evaluate Transfer") { model.evaluateTransfer() }
                        .disabled(model.isTransferEvaluating)
                }
                if let t = model.transfer {
                    HStack(spacing: 24) {
                        metric("Accuracy", t.accuracy)
                        metric("Chance", t.chance)
                        metric("Shuffled 95th pct", t.permutationP95)
                        VStack(alignment: .leading) {
                            Text("p-value").font(.caption).foregroundStyle(.secondary)
                            Text(t.pValue < 0.001 ? "< 0.001" : String(format: "%.3f", t.pValue)).font(.title3.monospacedDigit())
                        }
                    }
                    Text(t.pValue < 0.05
                         ? "Speech-trained templates recognise silent takes above chance (\(t.trainCount) train / \(t.testCount) test)."
                         : "No evidence yet that speech-trained templates transfer to silent mouthing.")
                        .foregroundStyle(t.pValue < 0.05 ? .green : .orange)
                } else if !model.isTransferEvaluating {
                    Text("Needs ≥ 2 words with STT takes and some silent takes of those words.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func metric(_ title: String, _ value: Double) -> some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(String(format: "%.0f%%", value * 100)).font(.title3.monospacedDigit())
        }
    }

    // MARK: Privacy

    private var privacy: some View {
        @Bindable var model = model
        return GroupBox("Privacy & retention") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Everything stays on this Mac. Auto-label keeps only motion slices and word labels; audio is saved only while Record Session is on. Always-on listening can capture other people — get their consent and pause when needed.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Stepper("Delete recorded sessions after \(model.retentionDays) days",
                            value: $model.retentionDays, in: 1...365)
                    Button("Purge Now") {
                        SessionRecorder.purge(olderThan: model.retentionDays)
                    }
                    Spacer()
                    Button("Delete All Auto-labelled Takes", role: .destructive) { model.deleteAutoLabelledTakes() }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}

private struct CandidateTile: View {
    @Environment(AppModel.self) private var model
    var candidate: LabelCandidate
    @State private var correction = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(candidate.label).font(.headline)
                if candidate.heard.lowercased() != candidate.label {
                    Text("“\(candidate.heard)”").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text(String(format: "%.2f", candidate.confidence))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(candidate.confidence < 0.7 ? .orange : .secondary)
            }
            SignalPlot(title: "", samples: candidate.samples, channels: SignalPlot.rotation,
                       minScale: 0.05, height: 44, showsHeader: false)
            HStack(spacing: 4) {
                TextField("Correct to…", text: $correction)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .onSubmit { model.accept(candidate, as: correction) }
                Button("Accept", systemImage: "checkmark") {
                    model.accept(candidate, as: correction.isEmpty ? nil : correction)
                }
                .labelStyle(.iconOnly)
                Button("Reject", systemImage: "xmark", role: .destructive) { model.reject(candidate.id) }
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))
    }
}
