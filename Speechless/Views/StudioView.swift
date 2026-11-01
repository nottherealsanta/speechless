import SwiftUI

struct StudioView: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        VStack(spacing: 0) {
            HStack {
                Picker("", selection: $studio.screen) {
                    ForEach(Studio.Screen.allCases) { Label($0.rawValue, systemImage: $0.symbol).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 260)
                .disabled(studio.isRecording)
                Spacer()
                MotionStatusBadge(status: studio.motion.status, rate: studio.motion.sampleRate)
                MicBadge()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            Divider()
            switch studio.screen {
            case .train: TrainScreen()
            case .inference: InferenceScreen()
            }
        }
        .task { studio.start() }
    }
}

// MARK: - Shared pieces

private struct MicBadge: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: studio.audio.isRunning ? "mic.fill" : "mic.slash")
                .foregroundStyle(studio.audio.isRunning ? .green : .secondary)
            LevelMeter(value: Double(studio.audio.level)).frame(width: 60)
        }
        .help(studio.audio.error ?? studio.audio.activeDeviceName ?? "Microphone")
    }
}

/// Live AirPods motion, last 8 seconds.
private struct LiveSensors: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        let m = studio.motion
        let span = m.latest.map { ($0.t - 8)...$0.t }
        GroupBox("Live sensors") {
            VStack(spacing: 8) {
                SignalPlot(title: "Acceleration (g)", samples: m.history, channels: SignalPlot.acceleration,
                           span: span, minScale: 0.01, height: 70)
                SignalPlot(title: "Rotation rate (rad/s)", samples: m.history, channels: SignalPlot.rotation,
                           span: span, minScale: 0.05, height: 70)
                SignalPlot(title: "Attitude (rad)", samples: m.history, channels: SignalPlot.attitude,
                           span: span, minScale: 0.1, height: 70)
            }
            .padding(6)
        }
    }
}

private struct RecordButton: View {
    @Environment(Studio.self) private var studio
    var idleTitle: String

    var body: some View {
        Button(action: studio.toggleRecording) {
            HStack(spacing: 10) {
                Image(systemName: studio.isRecording ? "stop.circle.fill" : "record.circle")
                    .font(.title)
                    .symbolEffect(.pulse, isActive: studio.isRecording)
                VStack(alignment: .leading, spacing: 2) {
                    Text(studio.isRecording ? "Stop" : idleTitle).font(.headline)
                    if studio.isRecording, let started = studio.recordingStarted {
                        Text(started, style: .timer).font(.caption.monospacedDigit())
                    } else {
                        Text("Click, or hold Space").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(minWidth: 200, minHeight: 44)
        }
        .buttonStyle(.borderedProminent)
        .tint(studio.isRecording ? .red : .accentColor)
        .disabled(studio.motion.status != .streaming && !studio.isRecording)
    }
}

private struct ConditionPicker: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        Picker("Mode", selection: $studio.condition) {
            ForEach(Studio.Condition.allCases) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: 380)
        .disabled(studio.isRecording)
    }
}

// MARK: - Train

private struct TrainScreen: View {
    @Environment(Studio.self) private var studio
    @FocusState private var textFocused: Bool
    @State private var epochs = 30

    var body: some View {
        @Bindable var studio = studio
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    LiveSensors()

                    GroupBox("Record") {
                        VStack(alignment: .leading, spacing: 12) {
                            ConditionPicker()
                            Text(studio.condition == .silent
                                 ? "Record as long as you like. Mouth the words silently with short pauses between sentences, then type the text for each piece."
                                 : "Record as long as you like. \(studio.sttName) transcribes it and splits it into sentences; fix any text, then add.")
                                .font(.callout).foregroundStyle(.secondary)
                            RecordButton(idleTitle: "Record")
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                    }

                    PassageCard()

                    GroupBox("Transcript (\(studio.drafts.count) piece\(studio.drafts.count == 1 ? "" : "s"))") {
                        VStack(alignment: .leading, spacing: 10) {
                            if studio.isTranscribing {
                                ProgressView("Transcribing…").controlSize(.small)
                            }
                            if studio.drafts.isEmpty {
                                Text("Record something…").foregroundStyle(.tertiary).font(.title3)
                                    .frame(maxWidth: .infinity, minHeight: 50, alignment: .leading)
                            }
                            ForEach($studio.drafts) { $draft in
                                DraftRow(draft: $draft, focused: $textFocused)
                            }
                            if !studio.draftMotion.isEmpty {
                                SignalPlot(title: String(format: "Whole recording · %.1f s",
                                                         (studio.draftMotion.last!.t - studio.draftMotion.first!.t)),
                                           samples: studio.draftMotion, channels: SignalPlot.rotation,
                                           minScale: 0.05, height: 50)
                            }
                            HStack {
                                Button {
                                    textFocused = false
                                    studio.saveDrafts()
                                } label: {
                                    Label("Add \(studio.savableCount) to Training Data", systemImage: "plus.circle.fill")
                                }
                                .buttonStyle(.borderedProminent)
                                .keyboardShortcut(.return, modifiers: .command)
                                .disabled(studio.savableCount == 0 || studio.isTranscribing)
                                Button("Discard All", role: .destructive, action: studio.discardDrafts)
                                    .disabled(studio.drafts.isEmpty)
                                Spacer()
                                Text("⌘↩ to add · pieces with empty text are skipped").font(.caption).foregroundStyle(.secondary)
                            }
                            if let message = studio.message {
                                Text(message).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .padding(6)
                    }
                }
                .padding(20)
            }
            .frame(minWidth: 560)

            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("Training data").font(.headline)
                    Text("\(studio.exampleCount)").foregroundStyle(.secondary).monospacedDigit()
                    Spacer()
                    Button("Show in Finder", systemImage: "folder", action: studio.revealClips)
                        .labelStyle(.iconOnly)
                }
                .padding(.horizontal, 12).padding(.top, 12)
                HStack {
                    Button(studio.selectedClips.count == studio.clips.count && !studio.clips.isEmpty ? "Select None" : "Select All") {
                        studio.selectedClips = studio.selectedClips.count == studio.clips.count
                            ? [] : Set(studio.clips.map(\.id))
                    }
                    .disabled(studio.clips.isEmpty)
                    Spacer()
                    Button("Delete \(studio.selectedClips.count)", systemImage: "trash", role: .destructive) {
                        studio.deleteClips(studio.selectedClips)
                    }
                    .disabled(studio.selectedClips.isEmpty)
                }
                .controlSize(.small)
                .padding(.horizontal, 12).padding(.vertical, 6)
                List(studio.clips, selection: $studio.selectedClips) { clip in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(clip.text)
                        Text("\(clip.condition) · \(String(format: "%.1f s", clip.duration))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .tag(clip.id)
                }
                .contextMenu(forSelectionType: Clip.ID.self) { ids in
                    Button("Delete \(ids.count == 1 ? "Clip" : "\(ids.count) Clips")", role: .destructive) {
                        studio.deleteClips(ids)
                    }
                }
                .onDeleteCommand { studio.deleteClips(studio.selectedClips) }

                // Hex's dictations live in their own folder and are read-only
                // here; every training run includes them.
                HStack {
                    Text("From Hex dictations").font(.subheadline.weight(.semibold))
                    Text("\(studio.hexExamples.count)").foregroundStyle(.secondary).monospacedDigit()
                    Spacer()
                    Button("Refresh", systemImage: "arrow.clockwise", action: studio.loadHexExamples)
                        .labelStyle(.iconOnly)
                        .help("Re-read \(Studio.hexRoot.path)")
                    Button("Show in Finder", systemImage: "folder", action: studio.revealHexExamples)
                        .labelStyle(.iconOnly)
                }
                .controlSize(.small)
                .padding(.horizontal, 12).padding(.vertical, 6)
                Group {
                    if studio.hexExamples.isEmpty {
                        Text("Nothing yet. Dictate with Hex Speechless and Labs › AirPods training data on; each pasted dictation appears here.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                    } else {
                        List(studio.hexExamples) { example in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(example.text)
                                Text("\(example.day) · \(String(format: "%.1f s", example.duration)) · motion only")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .frame(height: 150)
                    }
                }
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Stepper("Epochs \(epochs)", value: $epochs, in: 5...200, step: 5)
                        Spacer()
                        Button(studio.isTraining ? "Training…" : "Train Model") { studio.trainModel(epochs: epochs) }
                            .disabled(studio.isTraining || studio.exampleCount < 4)
                    }
                    if !studio.trainingLog.isEmpty {
                        ScrollView {
                            Text(studio.trainingLog)
                                .font(.caption2.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .defaultScrollAnchor(.bottom)
                        .frame(height: 160)
                    }
                }
                .padding(12)
            }
            .frame(minWidth: 280, idealWidth: 320)
        }
    }
}

// MARK: - Inference

private struct InferenceScreen: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                LiveSensors()

                GroupBox("Motion → text") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 16) {
                            RecordButton(idleTitle: "Record & Decode")
                            if studio.isInferring { ProgressView().controlSize(.small) }
                            Spacer()
                            Label(studio.modelStatus, systemImage: "cpu")
                                .font(.caption).foregroundStyle(.secondary)
                            Button("Reload", systemImage: "arrow.clockwise", action: studio.refreshModelStatus)
                                .labelStyle(.iconOnly)
                        }
                        Text("Record as long as you like — mouth the passage above (silently or not), pausing between sentences. Motion is split at pauses and each piece is decoded. Only motion is used; the microphone is ignored.")
                            .font(.callout).foregroundStyle(.secondary)

                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(studio.inferenceLines.enumerated()), id: \.offset) { i, line in
                                Text(line)
                                    .font(i == studio.inferenceLines.count - 1 ? .title2.bold() : .title3)
                                    .foregroundStyle(i == studio.inferenceLines.count - 1 ? .primary : .secondary)
                            }
                            if studio.inferenceLines.isEmpty {
                                Text("Decoded text appears here.").foregroundStyle(.tertiary).font(.title3)
                            }
                        }
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
                        .padding(10)
                        .background(RoundedRectangle(cornerRadius: 8).fill(.background))
                        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))

                        HStack {
                            Button("Clear", action: studio.clearInference)
                            Spacer()
                            if let message = studio.message {
                                Text(message).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if !studio.lastInferenceMotion.isEmpty {
                            SignalPlot(title: "Last decoded motion", samples: studio.lastInferenceMotion,
                                       channels: SignalPlot.rotation, minScale: 0.05, height: 60)
                        }
                    }
                    .padding(6)
                }
            }
            .padding(20)
        }
    }
}

private struct DraftRow: View {
    @Environment(Studio.self) private var studio
    @Binding var draft: DraftSegment
    var focused: FocusState<Bool>.Binding

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            SignalPlot(title: "", samples: draft.motion, channels: SignalPlot.rotation,
                       minScale: 0.05, height: 34, showsHeader: false)
                .frame(width: 110)
            VStack(alignment: .leading, spacing: 2) {
                TextField("Type what you said", text: $draft.text, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .focused(focused)
                Text(String(format: "%.1f s", draft.duration)).font(.caption2).foregroundStyle(.secondary)
            }
            Button("Remove", systemImage: "xmark.circle", role: .destructive) { studio.removeDraft(draft.id) }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
        }
    }
}

/// Something to read aloud (or mouth) during long recordings.
private struct PassageCard: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Read this").font(.headline)
                    Spacer()
                    Button("Next Passage", systemImage: "arrow.right.circle", action: studio.nextPassage)
                        .disabled(studio.isRecording)
                }
                Text(studio.passage)
                    .font(.title3)
                    .lineSpacing(6)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
        }
    }
}
