#!/usr/bin/env python3
"""复现 tail 扫描崩溃：加载模型 + 前向 sg_03_013 和 y_ 首帧。"""
import glob
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
Q = "图中这个人是在推行还是骑行这辆车？"

files = sorted(glob.glob(FRAMES + "/*.jpg"))
targets = [files[10170], files[10170 + 500], files[-1]]
for t in targets:
    print("target:", t.split("/")[-1], flush=True)
    im = Image.open(t)
    print("  size:", im.size, im.mode, flush=True)

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
print("model loaded", flush=True)

for t in targets:
    img = Image.open(t).convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "url": t},
            {"type": "text", "text": Q},
        ]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt + "骑行"], images=[img], return_tensors="pt")
    print("  seq len:", inputs["input_ids"].shape[1], flush=True)
    batch = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**batch).logits[0].float()
    print("  nan:", torch.isnan(logits).any().item(), flush=True)
print("ALL OK", flush=True)
