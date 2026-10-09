#!/usr/bin/env python3
"""调试 DPO 第 3 对 NaN：单独前向 sg_03_016，检查 logits 数值。"""
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "/home/lawson/vlm-active/ckpt/e2_q25vl_v1/final_fixed2"
PAIR = json.loads(open("/home/lawson/vlm-active/sft/dpo_pairs_v2.jsonl", encoding="utf-8").readlines()[2])

device = "cuda"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

img = Image.open(PAIR["image"]).convert("RGB")
print("image size:", img.size)

prompt = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image", "url": PAIR["image"]},
        {"type": "text", "text": PAIR["question"]},
    ]}], tokenize=False, add_generation_prompt=True)

for text in (PAIR["chosen"], PAIR["rejected"]):
    full = prompt + text
    inputs = processor(text=[full], images=[img], return_tensors="pt")
    ans_ids = processor(text=[text], add_special_tokens=False)["input_ids"][0]
    print(f"ans={text} ans_len={len(ans_ids)} total_len={inputs['input_ids'].shape[1]}")
    batch = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**batch).logits[0].float()
    print("  logits nan:", torch.isnan(logits).any().item(),
          "inf:", torch.isinf(logits).any().item(),
          "max:", logits.max().item())
