"""Exports a trained checkpoint to Core ML for the Speechless app.

    uv run python -m speechless_train.export checkpoints/motion-moonshine.pt
    → checkpoints/MotionEncoder.mlpackage, MotionDecoder.mlpackage, vocab.json

Install into the app with `--install` (copies to ~/Library/Application Support/Speechless/Model).
LoRA is merged into the base weights first, so the exported models are plain Moonshine-sized
networks. The decoder has fixed shapes and no KV cache: each step re-runs MAX_TOKENS positions,
which is cheap for Moonshine Tiny and short phrases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from .data import CHANNELS, MOTION_HZ
from .model import MotionMoonshine

MAX_MOTION = int(6 * MOTION_HZ)  # 6 s window
MAX_TOKENS = 24
INSTALL_DIR = Path.home() / "Library/Application Support/Speechless/Model"


class Encoder(nn.Module):
    def __init__(self, m: MotionMoonshine):
        super().__init__()
        self.m = m

    def forward(self, motion, mask):
        enc = self.m.encode(motion, mask > 0.5)
        return enc.last_hidden_state, enc.attention_mask.float()


class Decoder(nn.Module):
    def __init__(self, m: MotionMoonshine):
        super().__init__()
        self.m = m

    def forward(self, input_ids, encoder_states, encoder_mask):
        base = self.m.base
        dec = base.model.decoder
        h = dec.embed_tokens(input_ids.long())
        n = h.shape[1]
        neg = torch.finfo(h.dtype).min
        pos = torch.arange(n)[None]
        causal = torch.triu(torch.full((n, n), neg), diagonal=1)[None, None]
        cross = ((1 - encoder_mask) * neg)[:, None, None, :]
        rope = dec.rotary_emb(h, position_ids=pos)
        for layer in dec.layers:
            h = layer(h, causal, encoder_states, encoder_attention_mask=cross,
                      position_ids=pos, position_embeddings=rope)
        return base.proj_out(dec.norm(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--install", action="store_true")
    a = p.parse_args()

    model = MotionMoonshine(attn_implementation="eager")
    model.load_trainable_state(torch.load(a.checkpoint, map_location="cpu"))
    model.model = model.model.merge_and_unload()
    model.eval()

    motion = torch.zeros(1, MAX_MOTION, len(CHANNELS))
    mask = torch.ones(1, MAX_MOTION)
    with torch.no_grad():
        states, enc_mask = Encoder(model)(motion, mask)
        ids = torch.ones(1, MAX_TOKENS, dtype=torch.int32)
        enc_traced = torch.export.export(Encoder(model), (motion, mask)).run_decompositions({})
        dec_traced = torch.export.export(Decoder(model), (ids, states, enc_mask)).run_decompositions({})

    a.out.mkdir(parents=True, exist_ok=True)
    common = dict(convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                  compute_precision=ct.precision.FLOAT32)
    enc_ml = ct.convert(
        enc_traced,
        inputs=[ct.TensorType("motion", motion.shape, np.float32), ct.TensorType("mask", mask.shape, np.float32)],
        outputs=[ct.TensorType("encoder_states"), ct.TensorType("encoder_mask")], **common)
    dec_ml = ct.convert(
        dec_traced,
        inputs=[ct.TensorType("input_ids", ids.shape, np.int32),
                ct.TensorType("encoder_states", states.shape, np.float32),
                ct.TensorType("encoder_mask", enc_mask.shape, np.float32)],
        outputs=[ct.TensorType("logits")], **common)
    enc_ml.save(str(a.out / "MotionEncoder.mlpackage"))
    dec_ml.save(str(a.out / "MotionDecoder.mlpackage"))

    tok = model.tokenizer
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    (a.out / "vocab.json").write_text(json.dumps({
        "tokens": vocab, "bos": 1, "eos": 2, "maxMotion": MAX_MOTION, "maxTokens": MAX_TOKENS,
        "motionHz": MOTION_HZ, "channels": CHANNELS,
    }))

    # Reference output for checking the Swift side.
    with torch.no_grad():
        ref = model.transcribe(motion, mask.bool())
    print("exported to", a.out, "· reference transcription of zeros:", ref)

    if a.install:
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("MotionEncoder.mlpackage", "MotionDecoder.mlpackage", "vocab.json"):
            dst = INSTALL_DIR / name
            if dst.is_dir():
                shutil.rmtree(dst)
            (shutil.copytree if (a.out / name).is_dir() else shutil.copy)(a.out / name, dst)
        print("installed to", INSTALL_DIR)


if __name__ == "__main__":
    main()
