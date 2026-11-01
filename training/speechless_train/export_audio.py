"""Exports stock Moonshine Tiny (audio → text) to Core ML. The app uses it to transcribe what
you say during "Say it" training so you can edit the text and save it as the label.

    uv run python -m speechless_train.export_audio --install
"""
import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, MoonshineForConditionalGeneration

from .export import INSTALL_DIR, MAX_TOKENS
from .model import BASE

MAX_AUDIO = 16000 * 8  # 8 s


class AudioEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, audio, mask):
        enc = self.m.model.encoder
        x = torch.tanh(enc.conv1(audio[:, None]))
        x = enc.groupnorm(x)
        x = nn.functional.gelu(enc.conv2(x))
        x = nn.functional.gelu(enc.conv3(x)).permute(0, 2, 1)
        n = x.shape[1]
        m = mask[:, ::384][:, :n]
        neg = torch.finfo(x.dtype).min
        attn = ((1 - m) * neg)[:, None, None, :]
        pos = torch.arange(n)[None]
        rope = enc.rotary_emb(x, position_ids=pos)
        for layer in enc.layers:
            x = layer(x, attention_mask=attn, position_ids=pos, position_embeddings=rope)
        return enc.layer_norm(x), m


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, encoder_states, encoder_mask):
        dec = self.m.model.decoder
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
        return self.m.proj_out(dec.norm(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--install", action="store_true")
    a = p.parse_args()

    model = MoonshineForConditionalGeneration.from_pretrained(BASE, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(BASE)
    audio = torch.zeros(1, MAX_AUDIO)
    mask = torch.ones(1, MAX_AUDIO)
    with torch.no_grad():
        states, enc_mask = AudioEncoder(model)(audio, mask)
        ids = torch.ones(1, MAX_TOKENS, dtype=torch.int32)
        enc_ep = torch.export.export(AudioEncoder(model), (audio, mask)).run_decompositions({})
        dec_ep = torch.export.export(Decoder(model), (ids, states, enc_mask)).run_decompositions({})
    common = dict(convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                  compute_precision=ct.precision.FLOAT32)
    enc_ml = ct.convert(enc_ep, inputs=[ct.TensorType("audio", audio.shape, np.float32),
                                        ct.TensorType("mask", mask.shape, np.float32)],
                        outputs=[ct.TensorType("encoder_states"), ct.TensorType("encoder_mask")], **common)
    dec_ml = ct.convert(dec_ep, inputs=[ct.TensorType("input_ids", ids.shape, np.int32),
                                        ct.TensorType("encoder_states", states.shape, np.float32),
                                        ct.TensorType("encoder_mask", enc_mask.shape, np.float32)],
                        outputs=[ct.TensorType("logits")], **common)
    a.out.mkdir(parents=True, exist_ok=True)
    enc_ml.save(str(a.out / "AudioEncoder.mlpackage"))
    dec_ml.save(str(a.out / "AudioDecoder.mlpackage"))
    (a.out / "audio_vocab.json").write_text(json.dumps({
        "tokens": tok.convert_ids_to_tokens(list(range(len(tok)))), "bos": 1, "eos": 2,
        "maxAudio": MAX_AUDIO, "maxTokens": MAX_TOKENS}))
    print("exported to", a.out)
    if a.install:
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("AudioEncoder.mlpackage", "AudioDecoder.mlpackage", "audio_vocab.json"):
            dst = INSTALL_DIR / name
            if dst.is_dir():
                shutil.rmtree(dst)
            (shutil.copytree if (a.out / name).is_dir() else shutil.copy)(a.out / name, dst)
        print("installed to", INSTALL_DIR)


if __name__ == "__main__":
    main()
