#!/usr/bin/env python3
"""只测基座模型在坏图/好图上的 NaN 表现。"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

BASE = "/mnt/e/vlm-data/models/Qwen2.5-VL-3B-Instruct"
IMG_BAD = "/mnt/d/vlm-active/frames/trafficqa/sg_03_016.jpg"
IMG_OK = "/mnt/d/vlm-active/frames/trafficqa/sg_02_019.jpg"
Q = "图中这个人是在推行还是骑行这辆车？"

device = "cuda"
processor = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.float16).to(device).eval()

for tag, p in (("基座+坏图", IMG_BAD), ("基座+好图", IMG_OK)):
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
    print(tag, "nan:", torch.isnan(logits).any().item(), flush=True)
