#!/usr/bin/env python3
"""体检 e4b_projector/final：fp32 加载，干净帧前向，查 NaN 和生成。"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

CKPT = "/mnt/d/vlm-active/ckpt/e4b_projector/final"
IMG = "/mnt/d/vlm-active/frames/trafficqa/sg_02_000.jpg"
Q = "图中这个人是在推行还是骑行这辆车？"

processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForImageTextToText.from_pretrained(CKPT, dtype=torch.float32).to("cuda").eval()
print("dtype:", next(model.parameters()).dtype, flush=True)

img = Image.open(IMG).convert("RGB")
prompt = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image", "url": IMG},
        {"type": "text", "text": Q},
    ]}], tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[img], return_tensors="pt")
batch = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    logits = model(**batch).logits[0].float()
print("logits nan:", torch.isnan(logits).any().item(), flush=True)

# merger 权重范围检查（训练过的参数）
import re
sd = model.state_dict()
for k in ["model.visual.merger.mlp.0.weight", "model.visual.merger.mlp.2.weight",
          "model.visual.merger.ln_q.weight"]:
    if k in sd:
        t = sd[k].float()
        print(k, "abs max:", t.abs().max().item(), flush=True)
emb = sd.get("model.language_model.embed_tokens.weight") or sd.get("model.embed_tokens.weight")
print("embed abs max:", emb.float().abs().max().item(), flush=True)
base_emb = torch.load  # placeholder
with torch.no_grad():
    out = model.generate(**batch, max_new_tokens=10, do_sample=False)
print("gen:", processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), flush=True)
