# Speechless Agent Guide

Speechless is a macOS prototype that turns **silently mouthed words into text** from AirPods
head motion (`CMHeadphoneMotionManager`). Training data comes from two places: takes recorded in
this app, and (via the Hex fork) every dictation made with AirPods on.

## Scope: one Mac, no CI

This is a personal experiment on one Mac. Keep every change aimed at running it here:

- **macOS only.** Do not port, cross-compile, or "keep the build green" for Linux or Windows.
- **No CI.** Do not dispatch, wait on, or fix GitHub Actions, and do not add workflows. Do not
  regenerate goldens (the ones in the Hex fork are rendered on Linux; they are not a gate here).
- **Prove it on real data first.** New capability is judged by one real dictation or one real
  training run on this Mac, not by a test matrix. Tests stay for the code you touch, but hardware
  and data questions are settled by trying them.
- Work in the working tree and commit locally; pushing is enough. No PR flow, no review.

The Hex fork lives at `hex/` inside this checkout (gitignored, remote
`https://github.com/nottherealsanta/hex`, branch `speechless/training-capture`) and is where the
dictation capture lives. It builds as `Hex Speechless.app` with
`HEX_SPEECHLESS_BUILD=1 ./scripts/build-app.sh` from `hex/`. Its `AGENTS.md` is upstream's guide
and mostly does not apply here; this file wins.

## Layout

| Path | What it is |
|---|---|
| `Speechless/` | The macOS app (SwiftUI). Motion capture, template decoder, Moonshine runner. |
| `training/` | `uv` project `speechless_train`: PyTorch LoRA trainer, Core ML export, Hex loader. |
| `docs/HEX_FORK_PLAN.md` | What the Hex fork does, milestone by milestone. |
| `project.yml` | XcodeGen source of truth; `Speechless.xcodeproj` is generated and gitignored. |

## Running it

```sh
xcodegen generate
xcodebuild -project Speechless.xcodeproj -scheme Speechless -derivedDataPath build build
open build/Build/Products/Debug/Speechless.app
```

Signing: `project.yml` names team `K78G42H4U2`, which has no account on this Mac. Build with
`CODE_SIGNING_ALLOWED=NO` for a plain check, or set your own `DEVELOPMENT_TEAM` to run it with
motion and microphone permissions.

```sh
cd training && uv sync
uv run python -m speechless_train.train --synthetic --epochs 1   # pipeline smoke test
uv run python -m speechless_train.train                           # real data
uv run python -m speechless_train.export checkpoints/motion-moonshine.pt --install
```

Data lives at `~/Documents/Speechless/{Sessions,Clips}` and
`~/Library/Application Support/Speechless/{dataset.json,Model}`. Hex's dictations land in
`~/Library/Application Support/hex-speechless/Training`; `--hex` points the trainer at them (it is already
the default).

## Current state

The Hex side (M0–M5 in `docs/HEX_FORK_PLAN.md`) is built: motion capture around a dictation,
slicing and quality checks, the Training folder, the menu bar and `hex training` CLI, the HUD
motion mark, and `load_hex()` in the trainer. Nothing has run against worn AirPods yet.

The next step is the proof of concept on this Mac, in this order:

1. Build the Hex fork and dictate with AirPods, Labs › AirPods training data on.
2. Confirm examples appear (`hex training list`) with real Hz, coverage, and clock offset.
3. Train on them (`uv run python -m speechless_train.train --hex`), export, install.
4. Only if the model shows signal, collect for two weeks and run the silent-clip evaluation —
   that is the go/no-go (M6).
