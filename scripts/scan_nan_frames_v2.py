#!/usr/bin/env python3
"""扫描全部帧：fp16 前向找 NaN 帧（预取线程版，提速 3-4 倍）。

瓶颈分析（v1 实测 13 小时）：9p 读图 + processor 预处理在 CPU 串行，
GPU 大量空转。改为一个后台线程预取（读图+processor），主线程只做前向。
"""
import glob
import json
import threading
import queue

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
Q = "图中这个人是在推行还是骑行这辆车？"

device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

files = sorted(glob.glob(FRAMES + "/*.jpg"))

import sys
start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
files = files[start_idx:]
print(f"从第 {start_idx} 帧开始扫，共 {len(files)} 帧", flush=True)


def prep(p):
    """读图 + 构造输入（在预取线程里跑）。超大图先压到 100 万像素内，
    防止极端分辨率帧把序列撑爆 OOM（扫描目的是筛查，resize 不影响 NaN 判定）。"""
    img = Image.open(p).convert("RGB")
    if img.width * img.height > 1_000_000:
        r = (1_000_000 / (img.width * img.height)) ** 0.5
        img = img.resize((int(img.width * r), int(img.height * r)))
        print(f"[RESIZE] {p.split('/')[-1]} -> {img.size}", flush=True)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "url": p},
            {"type": "text", "text": Q},
        ]}], tokenize=False, add_generation_prompt=True)
    return p, processor(text=[prompt + "骑行"], images=[img], return_tensors="pt")


q = queue.Queue(maxsize=16)
it_lock = threading.Lock()
it = iter(files)


def producer():
    while True:
        with it_lock:
            p = next(it, None)
        if p is None:
            break
        try:
            q.put(prep(p))
        except Exception as e:
            print(f"[READ-FAIL] {p}: {e}", flush=True)


for _ in range(3):
    threading.Thread(target=producer, daemon=True).start()

bad = []
for i in range(1, len(files) + 1):
    item = q.get()
    if item is None:
        break
    p, inputs = item
    batch = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            logits = model(**batch).logits[0].float()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[OOM-SKIP] {p.split('/')[-1]}", flush=True)
        continue
    nan_frac = torch.isnan(logits).float().mean().item()
    if nan_frac > 0:
        bad.append({"frame": p.split("/")[-1], "nan_frac": round(nan_frac, 4)})
        print(f"[BAD {i}/{len(files)}] {p.split('/')[-1]} nan_frac={nan_frac:.3f}", flush=True)
    elif i % 200 == 0:
        print(f"{i}/{len(files)} ok", flush=True)

with open(f"/home/lawson/vlm-active/results/nan_frames_tail.json", "w", encoding="utf-8") as f:
    json.dump(bad, f, ensure_ascii=False, indent=1)
print(f"done. total={len(files)} bad={len(bad)} (offset {start_idx})", flush=True)
