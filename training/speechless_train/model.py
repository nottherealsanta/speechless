"""Moonshine Tiny with its audio frontend replaced by a motion frontend, adapted with LoRA.

    motion (B, T, 16) @ 25 Hz
      → MotionFrontend (trainable convs, upsampled to Moonshine's ~41.7 Hz frame rate, 288-d)
      → Moonshine encoder transformer layers (frozen + LoRA)
      → Moonshine decoder (frozen + LoRA) → text

The pretrained decoder already knows English; the frontend + LoRA learn to make motion look
like what the encoder expects. If audio was recorded, `distill_loss` pulls the motion encoder
output toward Moonshine's own audio encoder output for the same window (teacher).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, MoonshineForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .data import CHANNELS, MOTION_HZ

BASE = "UsefulSensors/moonshine-tiny"
MOONSHINE_HZ = 16000 / 384  # encoder frames per second for audio input


class MotionFrontend(nn.Module):
    def __init__(self, hidden: int, in_ch: int = len(CHANNELS), width: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, hidden, 3, padding=1),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(motion).transpose(1, 2)              # (B, C, T)
        x = self.net(x)
        out_len = max(1, int(int(x.shape[-1]) * MOONSHINE_HZ / MOTION_HZ))
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        m = F.interpolate(mask[:, None].float(), size=out_len, mode="nearest")[:, 0] > 0.5
        return self.dropout(x.transpose(1, 2)), m


class MotionMoonshine(nn.Module):
    def __init__(self, base: str = BASE, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 attn_implementation: str | None = None):
        super().__init__()
        kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
        model = MoonshineForConditionalGeneration.from_pretrained(base, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        )
        self.model = get_peft_model(model, config)
        self.frontend = MotionFrontend(model.config.hidden_size)

    @property
    def base(self) -> MoonshineForConditionalGeneration:
        from peft import PeftModel
        return self.model.base_model.model if isinstance(self.model, PeftModel) else self.model

    def encode(self, motion: torch.Tensor, mask: torch.Tensor) -> BaseModelOutput:
        enc = self.base.model.encoder
        h, m = self.frontend(motion, mask)
        pos = torch.arange(h.shape[1], device=h.device)[None]
        rope = enc.rotary_emb(h, position_ids=pos)
        attn = (~m)[:, None, None, :].to(h.dtype) * torch.finfo(h.dtype).min
        for layer in enc.layers:
            h = layer(h, attention_mask=attn, position_ids=pos, position_embeddings=rope)
        out = BaseModelOutput(last_hidden_state=enc.layer_norm(h))
        out.attention_mask = m.int()
        return out

    @torch.no_grad()
    def teacher_encode(self, audio: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Moonshine's own encoder on audio, with LoRA disabled (the original model)."""
        with self.model.disable_adapter():
            return self.base.model.encoder(audio, attention_mask=audio_mask).last_hidden_state

    def forward(self, motion, mask, labels):
        enc = self.encode(motion, mask)
        out = self.base(encoder_outputs=enc, attention_mask=enc.attention_mask, labels=labels)
        return out.loss, enc

    def distill_loss(self, enc: BaseModelOutput, teacher: torch.Tensor) -> torch.Tensor:
        student = enc.last_hidden_state
        n = min(student.shape[1], teacher.shape[1])
        return F.mse_loss(student[:, :n], teacher[:, :n])

    @torch.no_grad()
    def transcribe(self, motion: torch.Tensor, mask: torch.Tensor, max_new_tokens: int = 24) -> list[str]:
        enc = self.encode(motion, mask)
        ids = self.base.generate(encoder_outputs=enc, attention_mask=enc.attention_mask,
                                 max_length=max_new_tokens, num_beams=1)
        return [s.strip().lower() for s in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def trainable_state(self) -> dict:
        """Only what training changes: LoRA weights + motion frontend (~1 MB)."""
        return {k: v for k, v in self.state_dict().items() if "lora_" in k or k.startswith("frontend.")}

    def load_trainable_state(self, state: dict) -> None:
        missing = set(state) - set(self.state_dict())
        if missing:
            raise ValueError(f"unexpected keys: {sorted(missing)[:5]}")
        self.load_state_dict(state, strict=False)
