#!/usr/bin/env python3
"""从 checkpoint-12 重建 e4b_clean/final：换基座嵌入 + 补 processor 文件。"""
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

CKPT = "/home/lawson/vlm-active/ckpt/e4b_clean/checkpoint-12"
BASE = "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"
OUT = "/home/lawson/vlm-active/ckpt/e4b_clean/final"

tensors = {}
with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
print("张量数:", len(tensors))

embed = None
with safe_open(os.path.join(BASE, "model.safetensors.index.json"), framework="pt") if False else open(os.path.join(BASE, "model.safetensors.index.json")) as ij:
    idx = __import__("json").load(ij)
with safe_open(os.path.join(BASE, "model-00001-of-00002.safetensors"), framework="pt") as f:
    for cand in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"):
        if cand in f.keys():
            embed = f.get_tensor(cand)
            print("基座嵌入键:", cand)
            break

emb_key = next((k for k in tensors if k.endswith("embed_tokens.weight")), None)
if embed is not None and emb_key is not None:
    tensors[emb_key] = embed.to(tensors[emb_key].dtype)
    print("已恢复基座嵌入:", emb_key)

save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
# 清掉半成品 tmp
for f in os.listdir(OUT):
    if f.startswith(".tmp"):
        os.remove(os.path.join(OUT, f))
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "preprocessor_config.json", "chat_template.json"]:
    s = os.path.join(BASE, fn)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(OUT, fn))
print("done ->", OUT, sorted(os.listdir(OUT)))
