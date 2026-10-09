#!/usr/bin/env python3
"""扫描全部帧：fp16 前向找 NaN 帧，输出清单。"""
import glob
import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
Q = "图中这个人是在推行还是骑行这辆车？"

device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

bad = []
files = sorted(glob.glob(FRAMES + "/*.jpg"))
for i, p in enumerate(files, 1):
    img = Image.open(p).convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "url": p},
            {"type": "text", "text": Q},
        ]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt + "骑行"], images=[img], return_tensors="pt")
    batch = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**batch).logits[0].float()
    nan_frac = torch.isnan(logits).float().mean().item()
    if nan_frac > 0:
        bad.append({"frame": p.split("/")[-1], "nan_frac": round(nan_frac, 4)})
        print(f"[BAD {i}/{len(files)}] {p.split('/')[-1]} nan_frac={nan_frac:.3f}", flush=True)
    elif i % 50 == 0:
        print(f"{i}/{len(files)} ok", flush=True)

with open("/home/lawson/vlm-active/results/nan_frames.json", "w", encoding="utf-8") as f:
    json.dump(bad, f, ensure_ascii=False, indent=1)
print(f"done. total={len(files)} bad={len(bad)} -> /home/lawson/vlm-active/results/nan_frames.json")
