#!/usr/bin/env python3
"""定位 NaN 来源：同一图片在 (a) fixed2 模型 (b) HF 原版基座 上分别前向；
再换一张普通图片在 fixed2 上前向。看 NaN 跟谁走。"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

FIXED = "/home/lawson/vlm-active/ckpt/e2_q25vl_v1/final_fixed2"
BASE = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
IMG_BAD = "/home/lawson/vlm-active/frames/trafficqa/sg_03_016.jpg"
IMG_OK = "/home/lawson/vlm-active/frames/trafficqa/sg_02_019.jpg"
Q = "图中这个人是在推行还是骑行这辆车？"

def probe(model_path, img_path, tag):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float16).to("cuda").eval()
    img = Image.open(img_path).convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": Q},
        ]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt + "骑行"], images=[img], return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**batch).logits[0].float()
    print(tag, "nan:", torch.isnan(logits).any().item(), flush=True)
    del model
    torch.cuda.empty_cache()

import sys, glob

probe(FIXED, IMG_BAD, "fixed2 + 坏图  ")
probe(FIXED, IMG_OK, "fixed2 + 好图  ")
base = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
if True:
    probe(base, IMG_BAD, "基座 + 坏图  ")
    probe(base, IMG_OK, "基座 + 好图  ")
