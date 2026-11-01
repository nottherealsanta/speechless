import Foundation

/// Splits a long recording into utterance-sized pieces at pauses, so each piece fits the
/// model window (`maxLength` seconds). Works on any per-frame energy: audio RMS or motion energy.
enum Segmenter {
    struct Options {
        var frameRate: Double
        /// Pauses shorter than this don't split.
        var minGap: Double = 0.45
        var minLength: Double = 0.25
        var maxLength: Double = 5.8  // motion model window is 6 s
        var padding: Double = 0.2
        /// Energy must exceed this regardless of the recording's own range (e.g. mic noise).
        var absoluteFloor: Double = 0
    }

    /// Frame index ranges (inclusive) of active regions, padded and split to `maxLength`.
    static func segments(_ energy: [Double], _ o: Options) -> [ClosedRange<Int>] {
        guard energy.count >= 3 else { return [] }
        let sorted = energy.sorted()
        let floor = sorted[sorted.count / 5]
        let peak = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
        guard peak > max(floor * 2, o.absoluteFloor) else { return [] }
        let threshold = max(floor + 0.25 * (peak - floor), o.absoluteFloor)

        let gap = Int(o.minGap * o.frameRate)
        let maxFrames = max(2, Int(o.maxLength * o.frameRate))
        let pad = Int(o.padding * o.frameRate)

        var runs: [ClosedRange<Int>] = []
        var start: Int?
        for (i, e) in energy.enumerated() {
            if e > threshold {
                if start == nil { start = i }
            } else if let s = start {
                runs.append(s...(i - 1)); start = nil
            }
        }
        if let s = start { runs.append(s...(energy.count - 1)) }

        var merged: [ClosedRange<Int>] = []
        for run in runs {
            if let last = merged.last, run.lowerBound - last.upperBound - 1 < gap,
               run.upperBound - last.lowerBound < maxFrames - 2 * pad {
                merged[merged.count - 1] = last.lowerBound...run.upperBound
            } else {
                merged.append(run)
            }
        }

        let minFrames = Int(o.minLength * o.frameRate)
        var out: [ClosedRange<Int>] = []
        var previousEnd = -1
        for run in merged where run.count >= max(minFrames, 1) {
            // Pad only at real pauses, never past the previous piece.
            var lo = max(previousEnd + 1, run.lowerBound - pad)
            let end = min(energy.count - 1, run.upperBound + pad)
            // Split overlong runs at their quietest frame in the back half of the window.
            while end - lo + 1 > maxFrames {
                let window = (lo + maxFrames / 2)...(lo + maxFrames - 1)
                let cut = window.min { energy[$0] < energy[$1] } ?? (lo + maxFrames - 1)
                out.append(lo...cut)
                lo = cut + 1
            }
            out.append(lo...end)
            previousEnd = end
        }
        return out
    }

    /// 20 ms RMS frames of 16 kHz audio (50 frames/s).
    static func audioEnergy(_ samples: [Float]) -> [Double] {
        let hop = 320
        return stride(from: 0, to: samples.count, by: hop).map { i in
            let end = min(i + hop, samples.count)
            var sum: Float = 0
            for j in i..<end { sum += samples[j] * samples[j] }
            return Double((sum / Float(max(end - i, 1))).squareRoot())
        }
    }

    /// Per-sample motion energy, smoothed over ~0.2 s.
    static func motionEnergy(_ samples: [MotionSample]) -> [Double] {
        let raw = samples.map(MotionFeatures.energy)
        let k = 2
        return raw.indices.map { i in
            let lo = max(0, i - k), hi = min(raw.count - 1, i + k)
            return raw[lo...hi].reduce(0, +) / Double(hi - lo + 1)
        }
    }
}
