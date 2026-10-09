#!/usr/bin/env python3
"""可拖动性机制探针：对比基座 vs LoRA 模型在决策 token 上的 logit 位移。

对 48 张测试帧，取「答案第一个 token 位置」的 推行/骑行 logits，
计算 (微调后 - 基座) 的位移量。看 no-op 模型（2B/7B）是没位移，
还是位移了但方向与决策无关。

模型清单（基座, 微调）:
  3B: Qwen2.5-VL-3B-Instruct vs ckpt/e2_q25vl_v1/final_fixed2（动了）
  2B: Qwen3.5-2B vs ckpt/e2_2b/final（no-op）
  7B: Qwen2.5-VL-7B-Instruct vs ckpt/e2_7b/final（no-op）
"""
import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"

PAIRS = [
    ("3B", "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct",
     "/home/lawson/vlm-active/ckpt/e2_q25vl_v1/final_fixed2"),
    ("2B", "/home/lawson/vlm-active/models/Qwen3.5-2B",
     "/home/lawson/vlm-active/ckpt/e2_2b/final"),
    ("7B", "/home/lawson/vlm-active/models/Qwen2.5-VL-7B-Instruct",
     "/home/lawson/vlm-active/ckpt/e2_7b/final"),
]

test = [json.loads(l) for l in open("/tmp/test_balanced.jsonl")]
test = [t for t in test if t["label"] in ("push", "ride")]
nan_skip = {b["frame"][:-4] for b in json.load(open("/home/lawson/vlm-active/results/nan_frames.json"))}
# 早前 48 帧检查发现的 7 张 NaN 帧全显式加入（黑名单差 1 张会毒化 3B 统计）
nan_skip |= {"sg_05_018", "b_18f4y1U73n_clip_035_f002", "sg_03_016", "j_3616_f001",
             "b_194411p7Ho_clip_019_f001", "b_13t411F7iD_clip_029_f001", "sg_10_008"}
test = [t for t in test if t["frame_id"] not in nan_skip]
print("评测帧数(去NaN):", len(test), flush=True)

def answer_logits(model, processor, img_path, device):
    img = Image.open(img_path).convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": Q},
        ]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")
    batch = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**batch).logits[0].float()
    pos = logits[-1]  # 下一个 token = 答案首 token
    ids = processor.tokenizer("推行", add_special_tokens=False)["input_ids"]
    rid = processor.tokenizer("骑行", add_special_tokens=False)["input_ids"]
    push_id, ride_id = ids[0], rid[0]
    return pos[push_id].item(), pos[ride_id].item()

device = "cuda"
results = {}
for tag, base_p, ft_p in PAIRS:
    print(f"== {tag} ==", flush=True)
    proc = AutoProcessor.from_pretrained(base_p)

    # 逐模型加载：先基座算完释放，再装微调（双模型同驻会挤爆激活显存）
    base = AutoModelForImageTextToText.from_pretrained(base_p, dtype=torch.float16).to(device).eval()
    base_out = []
    for t in test:
        p = f"{FRAMES}/{t['frame_id']}.jpg"
        base_out.append(answer_logits(base, proc, p, device))
    del base
    torch.cuda.empty_cache()

    ft = AutoModelForImageTextToText.from_pretrained(ft_p, dtype=torch.float16).to(device).eval()
    deltas = []
    for i, t in enumerate(test):
        p = f"{FRAMES}/{t['frame_id']}.jpg"
        f_push, f_ride = answer_logits(ft, proc, p, device)
        b_push, b_ride = base_out[i]
        deltas.append((f_push - b_push) - (f_ride - b_ride))
    del ft
    torch.cuda.empty_cache()

    import numpy as np
    d = np.array(deltas)
    results[tag] = {"mean": float(d.mean()), "abs_mean": float(np.abs(d).mean()),
                    "max": float(d.max()), "min": float(d.min())}
    print(f"  位移(推向推行的净增量) 均值={d.mean():.6f} 绝对均值={np.abs(d).mean():.6f} "
          f"max={d.max():.6f} min={d.min():.6f}", flush=True)
    # 每对结束立即落盘，崩了也只丢当前这对
    with open("/home/lawson/vlm-active/results/malleability_probe.json", "w") as f:
        json.dump(results, f, indent=1)

with open("/home/lawson/vlm-active/results/malleability_probe.json", "w") as f:
    json.dump(results, f, indent=1)
print("done -> results/malleability_probe.json")
