"""Greedy-decodes synthetic examples with the exported Core ML models and compares with PyTorch."""
import json, sys
from pathlib import Path
import coremltools as ct
import numpy as np
import torch
from .model import MotionMoonshine
from .train import synthetic

d = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints")
cfg = json.loads((d / "vocab.json").read_text())
enc = ct.models.MLModel(str(d / "MotionEncoder.mlpackage"))
dec = ct.models.MLModel(str(d / "MotionDecoder.mlpackage"))
model = MotionMoonshine()
model.load_trainable_state(torch.load(d / "motion-moonshine.pt", map_location="cpu"))
model.eval()

agree = 0
examples = synthetic(20, seed=7)
for e in examples:
    T = cfg["maxMotion"]
    m = np.zeros((1, T, 16), np.float32); m[0, :len(e.motion)] = e.motion
    mask = np.zeros((1, T), np.float32); mask[0, :len(e.motion)] = 1
    o = enc.predict({"motion": m, "mask": mask})
    ids = [cfg["bos"]]
    while len(ids) < cfg["maxTokens"]:
        x = np.full((1, cfg["maxTokens"]), cfg["eos"], np.int32); x[0, :len(ids)] = ids
        logits = dec.predict({"input_ids": x, "encoder_states": o["encoder_states"], "encoder_mask": o["encoder_mask"]})["logits"]
        nxt = int(logits[0, len(ids) - 1].argmax())
        if nxt == cfg["eos"]: break
        ids.append(nxt)
    coreml = "".join(cfg["tokens"][i] for i in ids[1:]).replace("▁", " ").strip().lower()
    torch_out = model.transcribe(torch.from_numpy(m), torch.from_numpy(mask > 0))[0]
    agree += coreml == torch_out
    print(f"{e.text:6} coreml={coreml!r:10} torch={torch_out!r}")
print(f"agreement {agree}/{len(examples)}")
