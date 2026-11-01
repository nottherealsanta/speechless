"""Loads Speechless recordings into (motion, text[, audio]) training examples.

Sources:
  * Session folders (~/Documents/Speechless/Sessions/*): motion.csv + words.csv (STT words with
    timestamps) + meta.json. Consecutive words are grouped into short phrases.
  * dataset.json (~/Library/Application Support/Speechless): reviewed single-word takes.
  * Hex Speechless's training folder
    (~/Library/Application Support/hex-speechless/Training/<day>/<example>/):
    motion.csv + meta.json from each dictation made with Labs › AirPods training data
    on. See docs/HEX_FORK_PLAN.md.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw",
            "gx", "gy", "gz", "qw", "qx", "qy", "qz"]
MOTION_HZ = 25.0
SILENT = "Silent mouthing"

DEFAULT_SESSIONS = Path.home() / "Documents/Speechless/Sessions"
DEFAULT_CLIPS = Path.home() / "Documents/Speechless/Clips"
DEFAULT_DATASET = Path.home() / "Library/Application Support/Speechless/dataset.json"
DEFAULT_HEX = Path.home() / "Library/Application Support/hex-speechless/Training"


@dataclass
class Example:
    motion: np.ndarray            # (frames, len(CHANNELS)) resampled to MOTION_HZ
    text: str
    session: str
    condition: str
    audio: np.ndarray | None = None  # 16 kHz mono, same window, for distillation
    meta: dict = field(default_factory=dict)


def resample(t: np.ndarray, values: np.ndarray, start: float, end: float) -> np.ndarray:
    """Linear-interpolates irregular motion samples onto a fixed MOTION_HZ grid."""
    grid = np.arange(start, end, 1.0 / MOTION_HZ)
    if len(grid) == 0:
        grid = np.array([start])
    out = np.stack([np.interp(grid, t, values[:, c]) for c in range(values.shape[1])], axis=1)
    # Attitude: express relative to window start so absolute head pose doesn't leak in.
    for name in ("pitch", "roll", "yaw"):
        c = CHANNELS.index(name)
        out[:, c] = np.unwrap(out[:, c]) - out[0, c]
    return out.astype(np.float32)


def _samples_to_array(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([s["t"] for s in samples], dtype=np.float64)
    v = np.array([[s[c] for c in CHANNELS] for s in samples], dtype=np.float64)
    return t, v


def load_motion_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return np.zeros(0), np.zeros((0, len(CHANNELS)))
    data = np.atleast_1d(data)
    return data["t"].astype(np.float64), np.stack([data[c] for c in CHANNELS], axis=1)


def load_audio(path: Path, offset: float | None) -> tuple[np.ndarray, float] | None:
    """Returns (16 kHz mono audio, start time on the motion clock) or None."""
    if offset is None or not path.exists():
        return None
    try:
        import soundfile as sf
    except ImportError:
        return None
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio, float(offset)


def phrases(words: list[dict], max_gap: float, max_len: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["start"] - groups[-1][-1]["end"] <= max_gap \
                and w["end"] - groups[-1][0]["start"] <= max_len:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def load_sessions(root: Path = DEFAULT_SESSIONS, min_conf: float = 0.5, pad: float = 0.2,
                  max_gap: float = 0.6, max_len: float = 6.0, with_audio: bool = False) -> list[Example]:
    out: list[Example] = []
    for d in sorted(p for p in root.glob("*") if (p / "words.csv").exists()):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        t, v = load_motion_csv(d / "motion.csv")
        if len(t) < 8:
            continue
        with open(d / "words.csv") as f:
            words = [
                {"start": float(r["start"]), "end": float(r["end"]), "word": r["word"].strip().lower(),
                 "conf": float(r["confidence"] or 0)}
                for r in csv.DictReader(f)
                if not r["status"].startswith("skipped") or "vocabulary" in r["status"]
            ]
        words = [w for w in words if w["conf"] >= min_conf and w["word"]]
        audio = load_audio(d / "audio.caf", meta.get("audioStartOffsetSeconds")) if with_audio else None
        for group in phrases(words, max_gap, max_len):
            start, end = group[0]["start"] - pad, group[-1]["end"] + pad
            if start < t[0] or end > t[-1]:
                continue
            ex = Example(resample(t, v, start, end), " ".join(w["word"] for w in group),
                         d.name, meta.get("condition", ""))
            if audio is not None:
                a, a0 = audio
                i, j = int((start - a0) * 16000), int((end - a0) * 16000)
                if 0 <= i < j <= len(a):
                    ex.audio = a[i:j]
            out.append(ex)
    return out


def load_templates(path: Path = DEFAULT_DATASET) -> list[Example]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for tpl in data.get("templates", []):
        samples = tpl.get("samples") or []
        if len(samples) < 4:
            continue
        t, v = _samples_to_array(samples)
        out.append(Example(resample(t, v, t[0], t[-1]), tpl["label"],
                           tpl.get("sessionID") or tpl.get("id", "unknown"),
                           tpl.get("condition") or SILENT))
    return out


def load_clips(root: Path = DEFAULT_CLIPS, with_audio: bool = False) -> list[Example]:
    """Clips saved from the app's Train screen: motion.csv + meta.json (+ audio.wav).
    Clips recorded on the same day share a session, so held-out data is a different day."""
    out: list[Example] = []
    for d in sorted(p for p in root.glob("*") if (p / "meta.json").exists() and (p / "motion.csv").exists()):
        meta = json.loads((d / "meta.json").read_text())
        text = meta.get("text", "").strip().lower()
        t, v = load_motion_csv(d / "motion.csv")
        if not text or len(t) < 4:
            continue
        ex = Example(resample(t, v, t[0], t[-1]), text, d.name[:8], meta.get("condition", ""))
        if with_audio and (d / "audio.wav").exists():
            audio = load_audio(d / "audio.wav", 0.0)
            if audio is not None:
                ex.audio = audio[0]
        out.append(ex)
    return out


def load_hex(root: Path = DEFAULT_HEX, min_conf: float = 0.6) -> list[Example]:
    """Examples Hex saved while dictating with AirPods training data on.

    Each folder is one dictation piece: motion.csv (t in seconds from the piece's
    first sample) and meta.json (lowercase text, word timings, the AirPods session,
    and the quality numbers Hex checked before saving). There is no audio, so the
    audio-teacher loss must be skipped for these; `session` changes whenever the
    AirPods reconnect, which is what the train/test split uses. Spoken dictation is
    not silent mouthing, so these teach the model but never prove it works.
    """
    out: list[Example] = []
    for d in sorted(p for p in root.glob("*/*") if (p / "meta.json").exists() and (p / "motion.csv").exists()):
        meta = json.loads((d / "meta.json").read_text())
        text = meta.get("text", "").strip().lower()
        words = meta.get("words") or []
        confs = [w["confidence"] for w in words if w.get("confidence") is not None]
        if confs and sum(confs) / len(confs) < min_conf:
            continue
        t, v = load_motion_csv(d / "motion.csv")
        if not text or len(t) < 8:
            continue
        session = meta.get("session") or d.parent.name
        out.append(Example(resample(t, v, t[0], t[-1]), text, session, "",
                           meta={**meta, "source": "hex"}))
    return out


def split_by_session(examples: list[Example], test_condition: str | None = SILENT,
                     test_fraction: float = 0.2, seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Whole sessions go to one side. By default every silent-mouthing session is held out
    for testing, so the score answers "does it work when I don't make a sound?"."""
    sessions = sorted({e.session for e in examples})
    if test_condition and any(e.condition == test_condition for e in examples):
        test = {e.session for e in examples if e.condition == test_condition}
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(sessions)
        test = set(sessions[: max(1, int(len(sessions) * test_fraction))])
    train = [e for e in examples if e.session not in test]
    if not train:
        # One session (or everything silent): fall back to holding out random clips.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(examples))
        k = max(1, int(len(examples) * test_fraction)) if len(examples) > 1 else 0
        held = set(idx[:k].tolist())
        print("warning: not enough separate sessions; holding out random clips instead (optimistic score)")
        return [e for i, e in enumerate(examples) if i not in held], [e for i, e in enumerate(examples) if i in held]
    return train, [e for e in examples if e.session in test]
