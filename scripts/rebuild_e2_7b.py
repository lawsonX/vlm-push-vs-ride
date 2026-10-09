#!/usr/bin/env python3
"""从 checkpoint-10 重建 e2_7b/final：装回 LoRA 适配器 → 合并 → 恢复基座嵌入。"""
import os
import shutil

import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file, save_file

CKPT = "/home/lawson/vlm-active/ckpt/e2_7b/checkpoint-10"
BASE = "/home/lawson/vlm-active/models/Qwen2.5-VL-7B-Instruct"
OUT = "/home/lawson/vlm-active/ckpt/e2_7b/final"

from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.float16)
# 用官方入口加载适配器（自动处理 transformers 5.x 的命名映射，比手拼 state_dict 稳）
model = PeftModel.from_pretrained(model, CKPT)

merged = model.merge_and_unload()

# 恢复基座词嵌入
from safetensors import safe_open
embed = None
import json as _json
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
emb_file = idx["weight_map"]["model.embed_tokens.weight"]
with safe_open(os.path.join(BASE, emb_file), framework="pt") as f:
    embed = f.get_tensor("model.embed_tokens.weight")
with torch.no_grad():
    merged.model.language_model.embed_tokens.weight.copy_(embed.to(torch.float16))
print("已恢复基座嵌入")

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "preprocessor_config.json", "chat_template.json"]:
    s = os.path.join(BASE, fn)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(OUT, fn))
for f in os.listdir(OUT):
    if f.startswith(".tmp"):
        os.remove(os.path.join(OUT, f))
print("done ->", OUT)
