import Foundation

/// NVIDIA Parakeet (unified-en-0.6b, GGUF) via transcribe.cpp's `transcribe-cli`.
/// Much more accurate than Moonshine Tiny and returns word timestamps, which are used to
/// split long recordings at sentence boundaries.
struct ParakeetTranscriber: Sendable {
    struct Word: Hashable, Sendable {
        var text: String
        var start: Double
        var end: Double
    }

    struct Result: Sendable {
        var text: String
        var words: [Word]
    }

    let cli: URL
    let model: URL

    /// Finds the CLI (bundled build in ~/src or on PATH) and the GGUF in the Hugging Face cache.
    static func locate() -> ParakeetTranscriber? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let cliCandidates = [
            ProcessInfo.processInfo.environment["SPEECHLESS_TRANSCRIBE_CLI"],
            home.appending(path: "src/transcribe.cpp/build/bin/transcribe-cli").path,
            "/opt/homebrew/bin/transcribe-cli", "/usr/local/bin/transcribe-cli",
        ].compactMap { $0 }
        guard let cli = cliCandidates.first(where: { fm.isExecutableFile(atPath: $0) }) else { return nil }

        let modelCandidates: [String] = [ProcessInfo.processInfo.environment["SPEECHLESS_PARAKEET_MODEL"]].compactMap { $0 }
            + ["parakeet-unified-en-0.6b-gguf", "parakeet-tdt-0.6b-v3-gguf"].flatMap { repo -> [String] in
                let snapshots = home.appending(path: ".cache/huggingface/hub/models--handy-computer--\(repo)/snapshots")
                let dirs = (try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? []
                return dirs.flatMap { dir in
                    ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                        .filter { $0.pathExtension == "gguf" }.map(\.path)
                }
            }
        guard let model = modelCandidates.first(where: { fm.fileExists(atPath: $0) }) else { return nil }
        return ParakeetTranscriber(cli: URL(filePath: cli), model: URL(filePath: model))
    }

    var modelName: String { model.deletingPathExtension().lastPathComponent }

    func transcribe(_ samples: [Float]) throws -> Result {
        let wav = FileManager.default.temporaryDirectory.appending(path: "speechless-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        try AudioClipBuffer.writeWAV(samples, to: wav)

        let process = Process()
        process.executableURL = cli
        process.arguments = ["-m", model.path, "-q", "--timestamps", "word", wav.path]
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "Parakeet", code: Int(process.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "transcribe-cli exited with \(process.terminationStatus)"])
        }
        return Self.parse(String(decoding: data, as: UTF8.self))
    }

    /// Parses `text: …` and the `words:` block (`[ start -> end] word`).
    static func parse(_ output: String) -> Result {
        var text = ""
        var words: [Word] = []
        var inWords = false
        for line in output.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("text:") {
                text = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("words:") {
                inWords = true
            } else if inWords, trimmed.hasPrefix("["), let close = trimmed.firstIndex(of: "]") {
                let times = trimmed[trimmed.index(after: trimmed.startIndex)..<close]
                    .components(separatedBy: "->").map { Double($0.trimmingCharacters(in: .whitespaces)) }
                let word = trimmed[trimmed.index(after: close)...].trimmingCharacters(in: .whitespaces)
                if times.count == 2, let s = times[0], let e = times[1], !word.isEmpty {
                    words.append(Word(text: word, start: s, end: e))
                }
            } else if inWords, !trimmed.isEmpty {
                inWords = false
            }
        }
        return Result(text: text, words: words)
    }

    /// Groups words into utterances: break at sentence punctuation or pauses, cap length.
    static func utterances(_ words: [Word], maxLength: Double = 5.8, pause: Double = 0.45,
                           padding: Double = 0.15) -> [(text: String, start: Double, end: Double)] {
        // 1. Sentences: break after . ? ! or at long pauses.
        var sentences: [[Word]] = []
        for w in words {
            if let last = sentences.last?.last, !".?!".contains(last.text.last ?? " "), w.start - last.end < pause {
                sentences[sentences.count - 1].append(w)
            } else {
                sentences.append([w])
            }
        }
        // 2. Split overlong sentences in two, preferring a comma, then the largest gap,
        //    near the middle; repeat until every piece fits.
        func fits(_ g: [Word]) -> Bool { g.last!.end - g.first!.start + 2 * padding <= maxLength }
        func split(_ g: [Word]) -> [[Word]] {
            guard g.count > 1, !fits(g) else { return [g] }
            let mid = (g.first!.start + g.last!.end) / 2
            let cut = (1..<g.count).max { a, b in
                func score(_ i: Int) -> Double {
                    let comma = g[i - 1].text.hasSuffix(",") || g[i - 1].text.hasSuffix(";") ? 1.0 : 0
                    let gap = g[i].start - g[i - 1].end
                    let centered = 1 - abs(g[i].start - mid) / max(g.last!.end - g.first!.start, 0.01)
                    return comma + 2 * gap + centered
                }
                return score(a) < score(b)
            }!
            return split(Array(g[..<cut])) + split(Array(g[cut...]))
        }
        let groups = sentences.flatMap(split)
        // Pad without overlapping neighbours.
        return groups.enumerated().map { i, g in
            let prevEnd = i > 0 ? groups[i - 1].last!.end : -Double.infinity
            let nextStart = i + 1 < groups.count ? groups[i + 1].first!.start : Double.infinity
            let start = max(g.first!.start - padding, (prevEnd + g.first!.start) / 2, 0)
            let end = min(g.last!.end + padding, (g.last!.end + nextStart) / 2)
            return (g.map(\.text).joined(separator: " "), start, end)
        }
    }
}
