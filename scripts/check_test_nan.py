#!/usr/bin/env python3
"""快速检查 48 张测试集帧里哪些是 fp16 NaN 帧。"""
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
Q = "图中这个人是在推行还是骑行这辆车？"

test = [json.loads(l) for l in open("/tmp/test_balanced.jsonl", encoding="utf-8")]
device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

bad = []
for r in test:
    p = f"{FRAMES}/{r['frame_id']}.jpg"
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
    if torch.isnan(logits).any():
        bad.append(r["frame_id"])
        print("BAD:", r["frame_id"], r["label"], flush=True)

print(f"测试集 48 张中 NaN 帧: {len(bad)} -> {bad}")
with open("/home/lawson/vlm-active/results/nan_test_frames.json", "w") as f:
    json.dump(bad, f)
