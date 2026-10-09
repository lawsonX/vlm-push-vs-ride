#!/usr/bin/env python3
"""VLM 微调训练脚本（M6 实验矩阵用）。

四种训练模式（--mode）：
- projector_only : 只训视觉→语言的对齐层，其他全冻结（最小代价方案，E1）
- llm_lora       : 冻结视觉编码器，语言模型侧加 LoRA（E2）
- vit_lora       : 视觉编码器加 LoRA，语言模型冻结（E3）
- full           : 全量微调（E4，仅 0.8B 用，22GB 显存只够它）

技术要点（针对 2080 Ti）：
- 统一 fp16（Turing 架构没有 bf16 硬件加速）
- gradient_checkpointing 省显存
- 8bit AdamW（bitsandbytes）进一步省显存
- 只算答案部分的 loss（prompt 部分 mask 掉），避免学废话

用法（等 GPU 空闲后）：
    python train_sft.py --model /mnt/e/vlm-data/models/Qwen3.5-0.8B \
        --data /mnt/d/vlm-active/sft/train_direct.jsonl --mode llm_lora \
        --out /mnt/d/vlm-active/ckpt/e2_llm_lora
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

MODES = ["projector_only", "llm_lora", "vit_lora", "full"]


def apply_mode(model, mode, logger=print):
    """按模式冻结/加 LoRA，返回可训练参数名的大概描述。"""
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model, "全量微调"

    if mode == "projector_only":
        # 只留名字里带 projector / merger 的参数
        # （Qwen2.5-VL 的视觉→语言连接层叫 visual.merger，不叫 projector；
        #   transformers 5.x 命名）
        hit = 0
        for name, p in model.named_parameters():
            train = ("projector" in name.lower()) or ("merger" in name.lower())
            p.requires_grad = train
            hit += train
        logger(f"projector 参数个数: {hit}")
        if hit == 0:
            logger("警告：没找到名字带 projector 的参数，请打印模型结构确认！")
        return model, f"projector_only ({hit} 个参数张量)"

    target = []
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    # 保守默认：Qwen3.5 是混合线性注意力架构，层数 LoRA 拉满会训崩（v1/v2 教训）
    lora_cfg.target_modules = ["q_proj", "v_proj"]
    if mode == "llm_lora":
        # 语言侧：常见注意力层名字
        # 冻结视觉
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in ("visual", "vision", "projector")):
                p.requires_grad = False
    elif mode == "vit_lora":
        # 视觉侧：Qwen2.5-VL 的 ViT 注意力是融合 qkv（一个 Linear），
        # 名字里叫 qkv；LLM 侧没有 qkv，所以只加 qkv 不会误伤语言侧。
        # 注意不能用 "proj"——它会按后缀匹配到 LLM 的 o_proj/gate_proj 等。
        lora_cfg.target_modules = ["qkv"]
        lora_cfg.r = 32  # ViT 表达瓶颈更紧，松一点（E3 首次跑可对照调）
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, mode


class SFTDataset(Dataset):
    def __init__(self, data_path, processor, max_len=512):
        self.rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        messages = [
            {"role": "user", "content": [
                {"type": "image", "url": row["image"]},
                {"type": "text", "text": row["question"]},
            ]},
            {"role": "assistant", "content": row["answer"]},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True)
        # 完整对话用官方模板渲染（自带 <|im_end|> 等结束符），
        # 手工拼接会让模型学不到「说完就停」（E2-v2 的教训）
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        inputs = self.processor(
            text=[full_text],
            images=[ImageHolder(row["image"]).pil()],
            return_tensors="pt",
        )
        # prompt 长度必须用「带图」的 tokenization 来算（图文模式和纯文本的
        # 图像占位 token 数不同），mask 才精确。同一张图，patch 数一致。
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[ImageHolder(row["image"]).pil()],
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        item = {k: v[0] for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        item["labels"] = labels[0]
        # 视觉张量不能 squeeze 第 0 维：pixel_values 是 [N_patch, dim]，grid_thw 是 [1, 3]
        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            item["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]
        return item


class ImageHolder:
    def __init__(self, path):
        self.path = path

    def pil(self):
        from PIL import Image
        return Image.open(self.path).convert("RGB")


def collate(batch, pad_id):
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    out = {}
    seq_keys = ["input_ids", "attention_mask", "labels"]
    if "mm_token_type_ids" in batch[0]:
        seq_keys.append("mm_token_type_ids")
    for key in seq_keys:
        vals = []
        for b in batch:
            v = b[key]
            pad = maxlen - v.shape[0]
            if key == "labels":
                vals.append(torch.cat([v, torch.full((pad,), -100)]))
            elif key == "attention_mask":
                vals.append(torch.cat([v, torch.zeros(pad, dtype=v.dtype)]))
            else:
                fill = 0 if key == "mm_token_type_ids" else pad_id
                vals.append(torch.cat([v, torch.full((pad,), fill, dtype=v.dtype)]))
        out[key] = torch.stack(vals)
    pixel = [b["pixel_values"] for b in batch if "pixel_values" in b]
    if pixel:
        out["pixel_values"] = torch.cat(pixel)
    if "image_grid_thw" in batch[0]:
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    # CPU 上必须用 fp32 训练：纯 fp16 的 CPU 反向传播数值上不可靠（第一轮预实验教训）
    # 直接训练原始参数的模式（projector_only/full）必须 fp32：
    # fp16 原生参数的梯度过不了 GradScaler（"Attempting to unscale FP16 gradients"）。
    # LoRA 模式无此问题（适配器本身是 fp32）。fp32 3B 仅 projector_only 装得下。
    if args.mode in ("projector_only", "full"):
        dtype = torch.float32
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    model, desc = apply_mode(model, args.mode)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # 冻结层在前时梯度检查点需要这个才能生效

    ds = SFTDataset(args.data, processor)
    print(f"训练样本: {len(ds)}，模式: {desc}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        fp16=(torch.cuda.is_available() and args.mode not in ("projector_only", "full")),
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, pad_id))
    trainer.train()

    # LoRA 模式：先合并适配器再保存（否则 from_pretrained 会丢掉 lora 键，
    # 评测加载到的其实是基座——E2 时期踩过的坑）
    if args.mode in ("llm_lora", "vit_lora") and hasattr(model, "merge_and_unload"):
        print("合并 LoRA → 保存纯模型…")
        model = model.merge_and_unload()
        trainer.model = model

    # 防污染：训练期词嵌入可能被意外改动（排障战役结论，机制待查），
    # 保存前从基座原版恢复 embed / lm_head，LoRA 学到的投影层不受影响
    print("从基座恢复词嵌入…")
    from safetensors import safe_open as _sf
    import os as _os, json as _json
    _embed = None
    _idx = None
    for fn in _os.listdir(args.model):
        if fn.endswith(".safetensors.index.json"):
            _idx = _json.load(open(_os.path.join(args.model, fn)))
            break
    _candidates = ["model.embed_tokens.weight", "model.language_model.embed_tokens.weight",
                   "language_model.embed_tokens.weight", "model.model.embed_tokens.weight"]
    if _idx:
        _key = next((k for k in _candidates if k in _idx["weight_map"]), None)
        if _key:
            _shard = _os.path.join(args.model, _idx["weight_map"][_key])
            with _sf(_shard, framework="pt") as _f:
                _embed = _f.get_tensor(_key)
    else:
        for fn in _os.listdir(args.model):
            if fn.endswith(".safetensors"):
                with _sf(_os.path.join(args.model, fn), framework="pt") as _f:
                    for _k in _candidates:
                        if _k in _f.keys():
                            _embed = _f.get_tensor(_k)
                            break
                if _embed is not None:
                    break
    if _embed is not None:
        # 架构无关地找词嵌入：按名字搜，而不是写死属性路径
        # （transformers 5.x 的 Qwen2.5-VL 把嵌入藏在 model.language_model 里）
        import re
        with torch.no_grad():
            seen_ptrs = set()
            for n, pm in model.named_parameters():
                if re.search(r"(^|\.)embed_tokens\.weight$", n):
                    if pm.data_ptr() in seen_ptrs:
                        continue
                    seen_ptrs.add(pm.data_ptr())
                    pm.copy_(_embed.to(pm.dtype))
                    print(f"  恢复词嵌入: {n}")

    trainer.save_model(args.out + "/final")
    processor.save_pretrained(args.out + "/final")
    print("done ->", args.out + "/final")

    # 训后自检：用第 1 条训练样本生成一次，确认输出不是乱码
    model.eval()
    row = ds[0]
    gen = model.generate(
        input_ids=row["input_ids"].unsqueeze(0).to(model.device),
        attention_mask=row["attention_mask"].unsqueeze(0).to(model.device),
        pixel_values=row["pixel_values"].to(model.device),
        image_grid_thw=row["image_grid_thw"].to(model.device),
        max_new_tokens=20, do_sample=False)
    text = processor.decode(gen[0][row["input_ids"].shape[0]:], skip_special_tokens=True)
    print("SANITY-GEN:", text[:80])


if __name__ == "__main__":
    main()
