# Plan: Hex fork that collects AirPods motion training data

**Goal:** every time you dictate with Hex while wearing AirPods, save the AirPods head motion
together with the transcript. Each dictation becomes a labelled training example for the Speechless
motion → text model, with no extra effort.

**Fork of:** [anomalyco/hex](https://github.com/anomalyco/hex). This is a local‑first voice
dictation app written in Rust, with Swift bridges on macOS and Parakeet running locally through
transcribe.cpp. Studied at commit `a888312` (2026‑10‑31).

**Status:** M0–M5 built (2026‑11‑01), the HUD motion mark included; hardware is not verified
yet. This is a one-Mac experiment: no CI, no goldens, and no Linux or Windows work until a real
recording trains a model that reads silent speech (see [`AGENTS.md`](../AGENTS.md)). The next step
is the proof of concept, not the test matrix. See [Progress](#12-progress) at the end.

**Scope:**
- ✅ Hold Option → AirPods motion is recorded → Hex transcribes → motion + text are saved (sliced into pieces if long).
- ❌ No audio is saved.
- ❌ No Hex Meetings, Annotate or voice notes.
- ❌ No always-on or background recording.

---

## 1. Why Hex

The Speechless app can only collect data when you sit down and record. That gives minutes of data per
day, but the model needs hundreds of hours. Dictation already produces the pair we need, many
times a day:

| What we need | What Hex already has |
|---|---|
| Accurate text | Parakeet transcribes every dictation locally (`src/parakeet.rs`, `src/local_transcriber.rs`) |
| Word timings, for slicing | `Transcript.words` (`src/transcript.rs`), `transcribe_words()` |
| A human check of the text | Text is only final after a successful paste, and Modes corrections are applied first |
| A clear start and end | The Option press and release |
| A precise clock | On macOS, `CaptureInstant` is `mach_absolute_time`, the **same clock** as `CMDeviceMotion.timestamp` |

The only missing piece is AirPods motion.

---

## 2. What happens

```
Hold Option ──► Hex starts recording (unchanged)
     │           └─► fork: if AirPods are connected, start keeping motion samples
     │
Release Option ──► Hex transcribes with Parakeet (unchanged)
     │              └─► fork: stop keeping motion samples
     │
Text pasted ──► fork: take the final text + word timings + the motion window
     │            ├─ ≤ 5.8 s → save as one example
     │            └─ > 5.8 s → slice at sentence or pause boundaries using word timings,
     │                         save each piece
     │
Cancelled, failed, near-silent, or no AirPods ──► nothing saved
```

- **Zero extra steps.** Dictate exactly as before.
- **Off by default.** A Labs toggle, "AirPods training data" (Hex's rule: new features land in Labs, off by default).
- **Visible.** A small indicator in the dictation HUD shows when motion is being captured. The menu bar shows today's count, e.g. "Collected today: 214 examples · 18 min".
- **Undo.** "Discard last example" in the menu (for example, after dictating a password).

---

## 3. Architecture

```
 AirPods ──CoreMotion──► motion_bridge.swift ──► motion.rs
                              (new)              │ connection status
                                                 │ keeps samples only while Option is held
                                                 ▼
 Option press/release ──► dictation (existing) ──► parakeet.rs ──► paste
                                                 │                  │
                                                 │ clip start/end   │ final text + words
                                                 ▼                  ▼
                                         training_capture.rs (new)
                                           match · check · slice · save
                                                 │
                                                 ▼
                          ~/Library/Application Support/hex-speechless/Training/
                                                 │
                                                 ▼
                       speechless/training  (load_hex() → Moonshine + LoRA)
```

### 3.1 New files

| File | Language | Job |
|---|---|---|
| `src/motion_bridge.swift` | Swift | Wraps `CMHeadphoneMotionManager`: connection status, motion updates, and the permission prompt. Exposes C functions to Rust. Built by `build.rs` with `compile_swift_bridge`, like `calendar.swift`. |
| `src/motion.rs` | Rust | Safe wrapper. Motion updates are kept only while Option is held. A 0.5 s look-back buffer ensures the start of the first word isn't cut off. |
| `src/training_capture.rs` | Rust | After paste: matches the motion window to the clip, runs quality checks, slices, saves. Runs off the paste path, so it can never slow or break dictation. |
| `src/training_store.rs` | Rust | File layout, retention, size limit, delete. Same style as `history.rs`. |
| Labs row + menu items | Rust (Rune UI) | Toggle, status, today's count, Discard last, Open folder, Delete all. |
| `hex training status\|list\|delete\|purge` | Rust | Command-line access for scripts and the trainer. |
| `docs/features/training-data.md` | Markdown | Feature map entry, required by Hex's `AGENTS.md`. |

### 3.2 Hooks in existing code (three small changes)

1. **Option pressed:** when dictation capture starts, call `motion::begin(capture_id)`.
2. **Before transcription:** in `parakeet.rs`, next to `dictation_diagnostics::persist(&samples)`
   (about line 423), call `motion::end(capture_id, clip_start, clip_end)`. Use the clip's real
   audio start and end times (press time adjusted by `press_lead_ms` / `release_lead_ms` from
   `ClipTimeline` in `dictation.rs`). The audio samples are **not** copied or kept.
3. **After paste:** in `record_history(...)` in `parakeet.rs`, which runs only after a successful
   paste, call `training_capture::finish(capture_id, &completed)`. This is where the final text
   and word timings are available.

Also: when the feature is on, ask the model for word timestamps (`TimestampKind::Word`). Parakeet
provides them at no extra cost.

### 3.3 Timing

- Motion time: `CMDeviceMotion.timestamp` (seconds since boot, `mach_absolute_time`).
- Audio time: Hex's `CaptureInstant` (`mach_absolute_time` in nanoseconds).
- Word times from Parakeet are relative to the clip start, so word time + clip start = motion time.
- **Self-check (once per AirPods connection):** compare when speech starts (from word timings)
  with when jaw motion starts. Store the measured offset in `meta.json`. Skip saving if it exceeds 80 ms.

---

## 4. Slicing long dictations

The motion model reads at most 6 s at a time. Slicing uses the same rules as Speechless's
`ParakeetTranscriber.utterances`:

1. Group words into sentences: break after `.` `?` `!` or a pause ≥ 0.45 s.
2. If a sentence is longer than 5.8 s, split it at a comma or the largest pause near the middle. Repeat until every piece fits.
3. Pad each piece by 0.15 s, never overlapping its neighbour.
4. Each piece gets its own motion slice and text. Pieces from one dictation share a `dictation` ID.

Short dictations (≤ 5.8 s) are saved whole.

---

## 5. Saved data format

Compatible with the Speechless trainer. One folder per example:

```
~/Library/Application Support/hex-speechless/Training/
  2026-11-01/
    20261101T143205.123-7f3a-0/
      motion.csv     t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz   (t = seconds from piece start)
      meta.json
```

`meta.json`:

```json
{
  "schema": 1,
  "text": "send the report to maria tomorrow",
  "rawText": "Send the report to Maria tomorrow.",
  "words": [{ "text": "send", "start": 0.15, "end": 0.38, "confidence": 0.98 }],
  "dictation": "20261101T143205.123-7f3a",
  "piece": 0,
  "pieces": 1,
  "durationMs": 2310,
  "motionHz": 24.8,
  "motionCoverage": 0.99,
  "maxHeadTurnRad": 0.12,
  "clockOffsetMs": 3,
  "model": "parakeet-unified-en-0.6b-Q8_0",
  "airpods": "AirPods Pro 2",
  "session": "2026-11-01-a",
  "createdAt": "2026-11-01T14:32:05.123Z"
}
```

- `text`: lowercase, no punctuation. This is the training target. `rawText` keeps the original.
- `words`: times relative to this piece's start.
- `session`: changes every time the AirPods reconnect. The trainer uses it to split train and test data, because re-seating the AirPods changes the signal.
- No audio, app name, window title or URL is stored.

---

## 6. Quality checks (before saving)

| Check | Rule | Why |
|---|---|---|
| Motion present | ≥ 90% of the piece covered, ≥ 20 Hz | Bluetooth drops |
| Length | 0.4 s – 5.8 s per piece | No fragments; fits the model |
| Head movement | Attitude range < 0.5 rad | Large head turns hide jaw motion |
| Transcript confidence | Mean word confidence ≥ 0.6 | Wrong labels hurt more than missing ones |
| Speech present | Hex's existing near-silent check | No speech, no label |
| Timing | Self-check offset within ±80 ms | Misaligned data is useless |

Failures are counted by reason (visible via `hex training status`). Nothing is saved for them.

---

## 7. Training integration (`speechless/training`)

1. Add `load_hex(root)` to `speechless_train/data.py`. It reads the format above and uses `session` for the train/test split.
2. The audio-teacher distillation loss is skipped for Hex data (there's no audio). Text-only training already works.
3. **Silent-mouthing test data still comes from the Speechless app** (Train screen, *Silent mouthing* mode). It is always held out. Hex data teaches the model; silent clips show whether it really reads silent speech.
4. Report word error rate against the most-frequent-phrase baseline, and on a held-out split of unseen word combinations.
5. Optional nightly job: train → export to Core ML → install into Speechless.

---

## 8. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** | Fork, build, run Hex unchanged | `cargo build`, dictation works |
| **M1** | `motion_bridge.swift` + `motion.rs`: connection status, permission, capture while Option is held | `hex training status` shows "AirPods connected · 25 Hz" |
| **M2** | Three hooks; write `motion.csv` + `meta.json` | One dictation makes one folder; motion duration matches audio duration |
| **M3** | Slicing + quality checks + timing self-check | A 20 s dictation becomes ≥ 4 pieces with correct text; offset < 80 ms |
| **M4** | Labs toggle, HUD indicator, menu count, Discard last, Delete all, retention | Off by default; all controls work |
| **M5** | `load_hex()` in the trainer | `uv run python -m speechless_train.train` trains on Hex data |
| **M6** | Two weeks of normal use, then evaluate on silent clips | Report: hours collected, silent WER vs baseline. **Go/no-go point** |

**Rough size:** M1–M4 is about 1,200 lines of Rust and Swift, plus UI goldens.

---

## 9. Privacy

- **Off by default** in Labs, with a short explanation before the first enable.
- **No audio saved,** ever. Only motion and the text you already pasted.
- **Motion only while Option is held.** Nothing is kept between dictations.
- **Local only.** No upload or telemetry. Owner-only file permissions.
- **Discard last** example from the menu.
- **App exclusions:** never collect in password managers (configurable list with sensible defaults).
- **Retention:** default 90 days and 1 GB, oldest deleted first. "Delete all" removes everything immediately.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Using AirPods as the mic switches them to the Bluetooth headset profile | Lower transcript quality, possibly different motion | Recommend the Mac's built-in mic for dictation. Record which mic was used in `meta.json` |
| Motion only streams when AirPods are the active audio output | Gaps or no data | Check connection status; the coverage check rejects gaps |
| **Speaking ≠ silent mouthing** | Model learns speech-related motion that disappears when silent | Always evaluate on silent clips from Speechless. Never report spoken-only scores as success |
| 25 Hz head motion may not carry enough information | The project may not work | M6 is the decision point |
| Upstream Hex changes fast (1,000+ commits) | Merge conflicts | New files + three hooks only. Rebase weekly. Consider proposing a generic "dictation completed" hook upstream |
| Licenses | Hex is MIT; Parakeet uses the NVIDIA Open Model License | Keep notices. Personal research use is fine; check before distributing |

---

## 11. First steps

```sh
gh repo fork anomalyco/hex --clone          # into speechless/hex, gitignored by the parent repo
cd hex && git switch -c speechless/training-capture dev
cargo build && ./target/debug/hex      # M0
```

Then start M1 with `src/motion_bridge.swift`, modelled on `src/calendar.swift` and its `build.rs` entry.

---

## 12. Progress

**Fork:** https://github.com/nottherealsanta/hex, cloned to `hex/` in this checkout, branch
`speechless/training-capture` off `dev` at `5c118db` (three commits past the studied `a888312`),
pushed to the fork. It builds as `Hex Speechless.app`
(`HEX_SPEECHLESS_BUILD=1 ./scripts/build-app.sh`). Actions are off for the fork: no CI.

| # | State | Notes |
|---|---|---|
| M0 | ✅ | `cargo build` and the CLI work; a real dictation still needs Microphone, Accessibility, and a downloaded model on the Mac. |
| M1 | ✅ code | `src/motion_bridge.swift`, `src/motion.rs`, `CoreMotion` in `build.rs`, `NSMotionUsageDescription`, `hex training status` (prints the CoreMotion state and measured Hz). Not yet run against worn AirPods. |
| M2 | ✅ code | Three hooks: `motion::begin()` at the audio owner's capture start, `motion::end(clip.audio_window())` in `parakeet::submit`, `training_capture::finish(id, …)` in `record_history`. The clip's window comes from `DictationClip::audio_window()`, which applies `press_lead_ms` and the real sample count. |
| M3 | ✅ code | Sentence and long-piece slicing, padding without overlap, the six quality checks, and the once-per-session clock self-check. |
| M4 | ✅ code | Labs row, `hex settings labs.airpods_training`, menu bar Training Data submenu (today's count, Discard Last, Open Folder, Delete All), 90‑day/1 GB retention, and the HUD motion mark: a dot under Dot scope's plate while samples arrive, dark when the AirPods are connected but silent, gone at the release. |
| M5 | ✅ code | `load_hex()` and `--hex` in `training/speechless_train`, verified against a fixture folder. Silent clips still come from the Speechless app and stay held out. |

**Deliberate differences from the plan**

- Word timings come from a second pass over the clip as *background* work on the engine's
  loaded model, not from the dictation run itself. Dictation streams live, and a stream's text
  carries no words; the background pass yields to any dictation already sent and never loads a
  second model, so paste latency is unchanged. It does cost one extra inference (and some GPU
  time) per dictation while the feature is on.
- `words[].confidence` is absent: Parakeet reports none, so the confidence check only runs when
  a model does (it is skipped today).
- The clock self-check also requires a correlation peak (≥ 0.3) before its offset is trusted, so
  one unclear dictation leaves the session unmeasured instead of disabling it.
- Storage is `~/Library/Application Support/hex-speechless/Training` (the side build's own data
  root; a released Hex keeps `hex/Training`),
  one folder `<YYYY-MM-DD>/<stamp>-<dictation>-<piece>/`.
- `airpods` in `meta.json` is the literal `"AirPods"`; CoreMotion exposes no product name.
- The HUD's motion mark is a dot below the plate rather than a change to Dot scope itself, so the
  recording look stays exactly as it was; it fades in and out over 180 ms (instantly under Reduce
  Motion) and lives in `DotUniforms.field.y`, a spare slot, so the shader's layout is unchanged.

**Next**

1. Exercise M1–M3 on real AirPods at a Mac: `hex labs on airpods-training`, `hex training status`,
   dictate, then `hex training list` and compare durations.
2. Exercise the HUD motion mark on real AirPods: it should light within a sample of the press and
   stay dark when the AirPods are connected but sending nothing.
3. Two weeks of normal use, then M6's silent-clip evaluation — the go/no-go point.
