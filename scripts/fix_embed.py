#!/usr/bin/env python3
"""修复被污染的检查点：把基座的词嵌置换回原版。

背景（排障最终结论）：tie_word_embeddings=True 的模型做 PEFT 训练时，
梯度通过 lm_head 泄漏到共享的 embed_tokens，把词嵌入冲坏了——
Qwen3.5（乱码）和 Qwen2.5-VL（逗号退化）是同一个病。
LoRA 适配器本身是好的（训练 loss 正常收敛），只需恢复词嵌入。

用法：
    python fix_embed.py --base /path/to/base --ckpt /path/to/ckpt/final
"""
import argparse

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    # 基座的词嵌入（多分片合一）
    import json, os
    idx_path = None
    for fn in os.listdir(args.base):
        if fn.endswith(".safetensors.index.json"):
            idx_path = os.path.join(args.base, fn)
            break
    embed = None
    if idx_path:
        idx = json.load(open(idx_path))
        shard = os.path.join(args.base, idx["weight_map"]["model.embed_tokens.weight"])
        embed = load_file(shard)["model.embed_tokens.weight"]
    else:
        for fn in os.listdir(args.base):
            if fn.endswith(".safetensors"):
                d = load_file(os.path.join(args.base, fn))
                if "model.embed_tokens.weight" in d:
                    embed = d["model.embed_tokens.weight"]
                    break
    assert embed is not None, "基座里找不到 embed_tokens"

    # 加载微调模型，替换词嵌入
    model = AutoModelForImageTextToText.from_pretrained(args.ckpt, torch_dtype=torch.float32)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(embed.to(torch.float32))
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight.copy_(embed.to(torch.float32))
    model.tie_weights()

    # 保存：剥离 LoRA 键，保存干净权重（from_pretrained 本来也会忽略 lora 键，但剥掉更干净）
    state = model.state_dict()
    clean = {k: v for k, v in state.items() if "lora_" not in k}
    save_file(clean, os.path.join(args.ckpt, "model.safetensors.surgery"), metadata={"format": "pt"})
    print("surgery weights written; reload 时替换 model.safetensors 即可")


if __name__ == "__main__":
    main()
