#!/usr/bin/env python3
"""用完整的 checkpoint-20 重建 e4b 最终模型：
- 复制 checkpoint-20 的 model.safetensors
- 从基座恢复词嵌入（防污染，老规矩）
- 补齐 processor 文件
输出到 ckpt/e4b_projector/final2
"""
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch

CKPT20 = "/mnt/d/vlm-active/ckpt/e4b_projector/checkpoint-20"
BASE = "/mnt/d/vlm-active/models/Qwen2.5-VL-3B-Instruct"
OUT = "/mnt/d/vlm-active/ckpt/e4b_projector/final2"

os.makedirs(OUT, exist_ok=True)

# 1. 读 checkpoint-20 全部张量，嵌入换基座的
tensors = {}
with safe_open(os.path.join(CKPT20, "model.safetensors"), framework="pt") as f:
    keys = list(f.keys())
    for k in keys:
        tensors[k] = f.get_tensor(k)
print("checkpoint-20 张量数:", len(tensors))

# 2. 基座嵌入（候选键名）
embed = None
for fn in os.listdir(BASE):
    if fn.endswith(".safetensors"):
        with safe_open(os.path.join(BASE, fn), framework="pt") as f:
            for cand in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"):
                if cand in f.keys():
                    embed = f.get_tensor(cand)
                    print("基座嵌入键:", cand)
                    break
        if embed is not None:
            break

# checkpoint-20 里的嵌入键名
emb_key = next((k for k in tensors if k.endswith("embed_tokens.weight")), None)
print("ckpt 嵌入键:", emb_key, "旧嵌入 abs max:", tensors[emb_key].float().abs().max().item())
if embed is not None and emb_key is not None:
    tensors[emb_key] = embed.to(tensors[emb_key].dtype)
    print("已恢复基座嵌入")

# 3. merger 体检
mk = next((k for k in tensors if "merger.mlp.0.weight" in k), None)
print("merger.mlp.0 abs max:", tensors[mk].float().abs().max().item() if mk else "未找到")

# 4. 保存
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
for fn in os.listdir(BASE):
    if fn.endswith((".json", ".txt")) and fn != "config.json":
        shutil.copy(os.path.join(BASE, fn), os.path.join(OUT, fn))
shutil.copy(os.path.join(CKPT20, "config.json"), os.path.join(OUT, "config.json"))
shutil.copy(os.path.join(CKPT20, "generation_config.json"), os.path.join(OUT, "generation_config.json"))
print("done ->", OUT)
