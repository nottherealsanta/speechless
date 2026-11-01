# Speechless

A macOS prototype that tests whether AirPods Pro head-motion data (`CMHeadphoneMotionManager`) carries enough information to turn **silently mouthed words into text**.

It doesn't use a language model. It's a small-vocabulary template recognizer plus a statistical test that tells you whether the signal is real.

## Build & run

```sh
brew install xcodegen   # once
xcodegen generate
open Speechless.xcodeproj   # ⌘R
```

Or from the command line:

```sh
xcodebuild -project Speechless.xcodeproj -scheme Speechless -derivedDataPath build -allowProvisioningUpdates build
open build/Build/Products/Debug/Speechless.app
xcodebuild -project Speechless.xcodeproj -scheme Speechless -derivedDataPath build test
```

Requires macOS 14+, AirPods Pro or another headphone that supports head tracking, connected as the Mac's audio output. On first launch, grant **Motion**. The mic and speech prompts only appear when you start the mic.

## Screens

| Screen | What it does |
|---|---|
| **Train** | Shows live AirPods sensor data. Press **Record** (or hold Space), say a phrase and stop. The app records the AirPods motion during that time, and Moonshine Tiny on your Mac turns your speech into text in the text box. Edit the text, then click **Add to Training Data** (⌘↩). In *Silent mouthing* mode, nothing is transcribed and you type the text yourself. **Train Model** runs `training/`, then exports and installs the motion model. |
| **Inference** | Shows live sensor data. Record while mouthing a phrase, and the LoRA-adapted Moonshine motion model decodes the motion into text. The microphone isn't used. |

Clips are saved in `~/Documents/Speechless/Clips/<id>/` as `motion.csv`, `audio.wav` and `meta.json`. Models are stored in `~/Library/Application Support/Speechless/Model/`. First-time setup:

```sh
cd training && uv sync && uv run python -m speechless_train.export_audio --install
```

## How decoding works

1. **Segmentation** (`CaptureController`). Space-hold, or an energy trigger above a calibrated noise floor with pre-roll and hangover.
2. **Features** (`MotionFeatures`). Per frame: user acceleration (3), rotation rate (3), and attitude change since the start of the segment (3). Each channel is mean-centered, and quiet lead-in and lead-out frames are trimmed.
3. **Matching** (`DTW`, `TemplateClassifier`). Each channel is z-scaled across the training set. Matching is 1-nearest-neighbour per word, using dynamic time warping with a Sakoe–Chiba band. A segment is rejected if the best distance is above *max distance* or the margin to the runner-up is below *min confidence*.
4. **Is it real?** (`Evaluation`). The leave-one-out accuracy is compared with the same procedure run on shuffled labels. If p < 0.05, the motion stream distinguishes your words better than chance.

## Experiment protocol

Use **Record Session** in the toolbar or on the Sensors screen. Recordings go to `~/Documents/Speechless/Sessions/<timestamp>-<condition>/`:

- `motion.csv`: every motion sample. `t` is seconds from session start.
- `audio.caf`: microphone audio, if the mic is running.
- `events.csv`: every Space-held segment, with its label or prediction.
- `meta.json`: condition, prompt, and the audio start offset for aligning audio with motion.

Suggested runs: the same 5-word script under **Normal speech**, **Whisper** and **Silent mouthing**, 10 reps each. Then, in Train, compare the Evaluate results across feature sets.

The training set lives at `~/Library/Application Support/Speechless/dataset.json`.

## Auto-labelling with speech recognition

The **Auto-label** screen listens with on-device STT while you speak normally or whisper. Each finalized word includes timestamps, which are mapped onto the motion clock and used to cut that window from the AirPods stream. The app skips words with low confidence, words outside the vocabulary, windows with missing motion and big head turns. Kept words go to a review queue, where you can accept, correct or reject them. With **Record Session** on, the app also writes every word to `words.csv` in the session folder.

Takes record their session and condition. Evaluate can hold out whole sessions, and **Evaluate Transfer** trains on speech-labelled takes and tests on silent-mouthing takes. A red toolbar badge shows while listening. You can pause, delete old sessions automatically after N days, or delete all auto-labelled takes.

## Moonshine Tiny + LoRA (`training/`)

The template recognizer is now the baseline. The main model reuses [Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) as a pretrained speech decoder:

```
motion (16 ch @ 25 Hz) → MotionFrontend (new convs, resampled to Moonshine's 41.7 Hz, 288-d)
  → Moonshine encoder layers (frozen + LoRA) → Moonshine decoder (frozen + LoRA) → text
```

Training changes only the frontend and LoRA adapters on q/k/v/o and the MLPs: about 0.87M of 28M parameters. If a session includes `audio.caf`, an extra loss compares the motion encoder's output with Moonshine's original audio encoder output for the same window.

```sh
cd training
uv sync
uv run python -m speechless_train.train --synthetic --epochs 15   # smoke test
uv run python -m speechless_train.train                           # your Sessions + dataset.json
uv run python -m speechless_train.train --eval-only checkpoints/motion-moonshine.pt
```

Phrases are built from consecutive words in `words.csv`. Train and test splits use whole sessions. Silent-mouthing sessions are always kept for testing, so the reported WER shows whether the model works without speaking. The output also shows a most-frequent-phrase baseline.

## Known limits

- Headphone motion is processed and low-rate (roughly 25 Hz on AirPods). The speech-detecting accelerometer and the inward mic are not exposed. Expect jaw and head movement, not vocal vibration.
- Templates are user- and session-specific. Re-seating the AirPods can shift the signal.
- Choosing AirPods as the mic input switches them to the Bluetooth headset (HFP) profile.
