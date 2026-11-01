"""Train / evaluate the motion→text Moonshine adapter.

    uv run python -m speechless_train.train                      # train on your recordings
    uv run python -m speechless_train.train --eval-only ckpt.pt  # score a checkpoint
    uv run python -m speechless_train.train --synthetic          # smoke test, no data needed
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .model import MotionMoonshine


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment(m: np.ndarray, rng: random.Random) -> np.ndarray:
    m = m * rng.uniform(0.8, 1.2) + np.random.normal(0, 0.01, m.shape).astype(np.float32)
    rate = rng.uniform(0.85, 1.15)  # speaking-rate jitter
    n = max(4, int(len(m) * rate))
    idx = np.linspace(0, len(m) - 1, n)
    return np.stack([np.interp(idx, np.arange(len(m)), m[:, c]) for c in range(m.shape[1])], 1).astype(np.float32)


BOS, EOS = 1, 2  # Moonshine: decoder starts from BOS; labels end with EOS


def encode_labels(tokenizer, texts: list[str]) -> torch.Tensor:
    ids = [[t for t in tokenizer(x).input_ids if t != BOS] + [EOS] for x in texts]
    labels = torch.full((len(ids), max(map(len, ids))), -100)
    for k, row in enumerate(ids):
        labels[k, :len(row)] = torch.tensor(row)
    return labels


def batches(examples, tokenizer, size, rng=None, shuffle=False):
    order = list(range(len(examples)))
    if shuffle:
        (rng or random).shuffle(order)
    for i in range(0, len(order), size):
        chunk = [examples[j] for j in order[i:i + size]]
        motions = [augment(e.motion, rng) if rng else e.motion for e in chunk]
        T = max(len(m) for m in motions)
        motion = torch.zeros(len(chunk), T, len(D.CHANNELS))
        mask = torch.zeros(len(chunk), T, dtype=torch.bool)
        for k, m in enumerate(motions):
            motion[k, :len(m)] = torch.from_numpy(m)
            mask[k, :len(m)] = True
        labels = encode_labels(tokenizer, [e.text for e in chunk])
        audio = None
        if all(e.audio is not None for e in chunk):
            A = max(len(e.audio) for e in chunk)
            audio = torch.zeros(len(chunk), A)
            amask = torch.zeros(len(chunk), A, dtype=torch.long)
            for k, e in enumerate(chunk):
                audio[k, :len(e.audio)] = torch.from_numpy(e.audio)
                amask[k, :len(e.audio)] = 1
            audio = (audio, amask)
        yield chunk, motion, mask, labels, audio


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def evaluate(model, examples, dev, batch_size=16, show=5) -> dict:
    model.eval()
    refs, hyps = [], []
    for chunk, motion, mask, *_ in batches(examples, model.tokenizer, batch_size):
        hyps += model.transcribe(motion.to(dev), mask.to(dev))
        refs += [e.text for e in chunk]
    model.train()
    if not refs:
        return {}
    for r, h in list(zip(refs, hyps))[:show]:
        print(f"    ref: {r!r:30}  hyp: {h!r}")
    # Chance baseline: always predict the most frequent training phrase.
    return {
        "wer": float(np.mean([wer(r, h) for r, h in zip(refs, hyps)])),
        "exact": float(np.mean([r == h for r, h in zip(refs, hyps)])),
        "n": len(refs),
    }


def baseline(train, test) -> dict:
    if not train or not test:
        return {}
    common = max(set(e.text for e in train), key=[e.text for e in train].count)
    return {"wer": float(np.mean([wer(e.text, common) for e in test])),
            "exact": float(np.mean([e.text == common for e in test]))}


def synthetic(n=200, seed=0) -> list[D.Example]:
    """Each word is a distinct motion pattern, so the pipeline must learn to read it."""
    rng = np.random.default_rng(seed)
    words = ["yes", "no", "next", "back", "stop"]
    out = []
    for i in range(n):
        w = int(rng.integers(len(words)))
        T = int(rng.integers(18, 30))
        m = rng.normal(0, 0.01, (T, len(D.CHANNELS))).astype(np.float32)
        m[:, 3 + w % 3] += 0.3 * np.sin(np.linspace(0, (w + 1) * np.pi, T))
        out.append(D.Example(m, words[w], f"s{i % 10}", "Silent mouthing" if i % 10 >= 8 else "Normal speech"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=Path, default=D.DEFAULT_SESSIONS)
    p.add_argument("--dataset", type=Path, default=D.DEFAULT_DATASET)
    p.add_argument("--clips", type=Path, default=D.DEFAULT_CLIPS)
    p.add_argument("--hex", type=Path, default=D.DEFAULT_HEX,
                   help="Hex's AirPods training folder: dictations saved with Labs › AirPods training data on")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("checkpoints/motion-moonshine.pt"))
    p.add_argument("--eval-only", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--distill", type=float, default=1.0, help="weight of the audio-teacher loss (needs soundfile + audio.caf)")
    p.add_argument("--test-condition", default=D.SILENT, help="hold out sessions with this condition; '' for random sessions")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.synthetic:
        examples = synthetic()
    else:
        examples = (D.load_clips(a.clips, with_audio=a.distill > 0)
                    + D.load_sessions(a.sessions, min_conf=a.min_conf, with_audio=a.distill > 0)
                    + D.load_hex(a.hex)
                    + D.load_templates(a.dataset))
    train, test = D.split_by_session(examples, a.test_condition or None, seed=a.seed)
    print(f"{len(examples)} examples · train {len(train)} ({len({e.session for e in train})} sessions)"
          f" · test {len(test)} ({len({e.session for e in test})} sessions)")
    if not train:
        raise SystemExit("No training data. Record sessions with Auto-label on, or use --synthetic.")

    dev = device()
    model = MotionMoonshine(lora_rank=a.lora_rank).to(dev)
    if a.eval_only:
        model.load_trainable_state(torch.load(a.eval_only, map_location=dev))
        print("test", evaluate(model, test, dev), "baseline", baseline(train, test))
        return

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,} / {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * max(1, -(-len(train) // a.batch_size))
    try:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    except ZeroDivisionError:
        # Torch divides by a zero-width warmup at some step counts (ten steps at
        # 10% is one); a run that short keeps a flat rate instead of crashing.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    rng = random.Random(a.seed)
    best = float("inf")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(a.epochs):
        total, n = 0.0, 0
        for chunk, motion, mask, labels, audio in batches(train, model.tokenizer, a.batch_size, rng, shuffle=True):
            loss, enc = model(motion.to(dev), mask.to(dev), labels.to(dev))
            if audio is not None and a.distill > 0 and not a.synthetic:
                teacher = model.teacher_encode(audio[0].to(dev), audio[1].to(dev))
                loss = loss + a.distill * model.distill_loss(enc, teacher)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(chunk); n += len(chunk)
        line = f"epoch {epoch + 1:3d}  loss {total / n:.3f}"
        if test and ((epoch + 1) % 5 == 0 or epoch + 1 == a.epochs):
            m = evaluate(model, test, dev, show=3)
            line += f"  test WER {m['wer']:.2f} exact {m['exact']:.2f}"
            if m["wer"] < best:
                best = m["wer"]
                torch.save(model.trainable_state(), a.out)
        print(line, flush=True)
    if not test:
        torch.save(model.trainable_state(), a.out)
    print(f"saved {a.out} · best test WER {best:.2f} · most-frequent-phrase baseline {baseline(train, test)}")


if __name__ == "__main__":
    main()
