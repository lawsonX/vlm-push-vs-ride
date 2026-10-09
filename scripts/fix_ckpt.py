#!/usr/bin/env python3
"""修复被污染的检查点（完整版）：合并 LoRA + 恢复原版词嵌入。

病理（排障最终结论）：tie_word_embeddings=True 的模型做 PEFT 时，
梯度经 lm_head 泄漏进共享词嵌入，把 embed 冲坏（Qwen3.5 乱码、
Qwen2.5-VL 逗号退化的共同根因）。LoRA 适配器本身是好的。

本脚本：
1. 从微调目录读全部权重（含 lora_A/lora_B）
2. 建一个同配置 PEFT 模型，把 lora 权重装回去
3. merge_and_unload() 把 LoRA 合进基座（推理不再需要适配器）
4. 用基座原版词嵌入覆盖被污染的部分
5. 保存为干净的完整模型

用法：
    python fix_ckpt.py --base /path/to/base --ckpt /path/to/ckpt/final \
        --out /path/to/ckpt/final_fixed
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_base_embed(base_dir):
    idx_path = None
    for fn in os.listdir(base_dir):
        if fn.endswith(".safetensors.index.json"):
            idx_path = os.path.join(base_dir, fn)
            break
    if idx_path:
        idx = json.load(open(idx_path))
        shard = os.path.join(base_dir, idx["weight_map"]["model.embed_tokens.weight"])
        return load_file(shard)["model.embed_tokens.weight"]
    for fn in os.listdir(base_dir):
        if fn.endswith(".safetensors"):
            d = load_file(os.path.join(base_dir, fn))
            if "model.embed_tokens.weight" in d:
                return d["model.embed_tokens.weight"]
    raise RuntimeError("基座里找不到 embed_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("只读微调文件里的 lora 键（避免全量加载撑爆内存）…")
    from safetensors import safe_open
    ft_state = {}
    with safe_open(os.path.join(args.ckpt, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if "lora_" in k:
                ft_state[k] = f.get_tensor(k)
    print(f"  lora 张量 {len(ft_state)}")

    print("加载基座模型（fp16，内存减半）…")
    model = AutoModelForImageTextToText.from_pretrained(args.base, torch_dtype=torch.float16)

    print("装配 LoRA 适配器…")
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    model = PeftModel(model, cfg)
    # 键名对齐：训练时保存的是裸名 model.X.lora_...，PEFT 包装后要加 base_model.model. 前缀
    # transformers 5.x 的 Qwen2.5-VL 结构里多一层 language_model：
    # 训练保存的裸名 model.X.lora_... → peft 期望 base_model.model.model.language_model.X.lora_...
    mapped = {}
    for k, v in ft_state.items():
        if k.startswith("base_model"):
            nk = k
        elif k.startswith("model."):
            nk = "base_model.model.model.language_model." + k[len("model."):]
        else:
            nk = "base_model.model.model." + k
        mapped[nk] = v
    info = set_peft_model_state_dict(model, mapped)
    n_unexp = len(info.unexpected_keys) if info else 0
    print(f"  装载后 unexpected 键数: {n_unexp}（应为 0，非 0 说明 lora 没装进去）")
    hit = sum(1 for k in mapped if k not in (info.unexpected_keys if info else []))
    print(f"  成功装载 lora 张量: {hit}/{len(mapped)}")

    print("合并 LoRA → 纯模型…")
    model = model.merge_and_unload()

    # 基座是现场加载的，词嵌入本来就没被污染，无需修复
    print(f"保存到 {args.out} …")
    model.save_pretrained(args.out, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)
    print("done")


if __name__ == "__main__":
    main()
