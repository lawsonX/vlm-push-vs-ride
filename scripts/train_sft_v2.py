#!/usr/bin/env python3
"""VLM 微调训练脚本 v2：吃 delivery_v02 交付包的 messages 多轮格式，按 docs/13 配方训练。

与 train_sft.py（旧单行格式）的区别：
- 数据格式：delivery_v02 的 messages 格式（<图片占位> 标记、direct/probe/caption/chain 单轮 + guide 两轮）
- LoRA 配方（docs/13，依据《LoRA Without Regret》）：
  语言侧全部线性层 q/k/v/o_proj + gate/up/down_proj，rank 32 / alpha 64，
  学习率 2e-4 恒定，有效批次 32，视觉编码器和 merger 冻结
- 多轮 loss：每个 assistant 轮的 token 都算 loss，其余（用户问题、图片、模板符）mask 掉
- bf16（A100 支持）；旧 2080Ti 机器才需要 fp16

用法：
    python train_sft_v2.py --model models/Qwen2.5-VL-3B-Instruct \
        --data data/sft_v2/train.jsonl \
        --out ckpt/a1_baseline
    # 增强版（A3 消融）：--data data/sft_v2/train.jsonl --data data/sft_v2/train_aug.jsonl
"""

import argparse
import hashlib
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

# docs/13 第 4 节配方：语言侧全部线性层（含 MLP）。
# ViT 侧是融合 qkv，不存在这些名字，不会误伤视觉。
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
# 视觉侧（Qwen2.5-VL 的 ViT 注意力是融合 qkv，一个 Linear 叫 qkv；
# 不能用 "proj" 后缀匹配，会误伤 LLM 的 o_proj/gate_proj）
VIT_LORA_TARGETS = ["qkv"]

MODES = ["llm_lora", "llm_vit_lora", "vit_lora", "projector_only"]

PLACEHOLDER = "<图片占位>"


def build_blocks(messages):
    """把 jsonl 里的 {role, content(str)} 列表转成 processor 吃的多模态 content 块。

    图片只插到第一条 user 消息的占位符处；占位符可能出现在句首或句中。
    其余消息原样是纯文本。
    """
    blocks = []
    image_placed = False
    for m in messages:
        content = m["content"]
        if (not image_placed) and m["role"] == "user" and PLACEHOLDER in content:
            text = content.replace(PLACEHOLDER, "", 1).strip()
            blocks.append({
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            })
            image_placed = True
        else:
            assert PLACEHOLDER not in content, "占位符出现在第一条 user 消息之外"
            blocks.append({"role": m["role"], "content": content})
    return blocks


class SFTDatasetV2(Dataset):
    """messages 多轮数据；tokenize 一次并缓存到磁盘（预处理要跑 4~5 遍 processor，不缓存每个 epoch 都要重跑）。"""

    def __init__(self, data_paths, processor, cache_dir, max_len=2048):
        self.items = []
        rows = []
        for p in data_paths:
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        tag = hashlib.md5(
            ("enc2" + json.dumps([str(p) for p in data_paths]) + str(len(rows))).encode()
        ).hexdigest()[:12]
        cache = Path(cache_dir) / f"tok_{tag}.pt"

        if cache.exists():
            self.items = torch.load(cache, weights_only=False)
            print(f"tokenize 缓存命中: {cache}")
        else:
            from PIL import Image
            n_skip = 0
            for r in rows:
                img = Image.open(r["image"]).convert("RGB")
                item = self._encode(r, img, processor)
                if item["input_ids"].shape[0] > max_len:
                    n_skip += 1
                    continue
                self.items.append(item)
            print(f"tokenize 完成: {len(self.items)} 条, 超长丢弃 {n_skip} 条")
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.items, cache)
        self._report_labels(rows)

    @staticmethod
    def _report_labels(rows):
        from collections import Counter
        styles = Counter(r.get("style", "?") for r in rows)
        labels = Counter(r.get("label", "?") for r in rows)
        print(f"样本问法分布: {dict(styles)}")
        print(f"样本标签分布: {dict(labels)}")

    def _encode(self, row, img, processor):
        blocks = build_blocks(row["messages"])
        full_text = processor.apply_chat_template(
            blocks, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[full_text], images=[img], return_tensors="pt")
        ids_img = inputs["input_ids"][0]
        n = ids_img.shape[0]

        # 用「金标答案在全文里的字符区间」定位 loss 范围，对任何聊天模板成立。
        # （增量前缀法在 Qwen3.5 上会破：其模板给 assistant 加 <think> 包裹层，
        #   带生成头的前缀渲染与全文渲染在前缀处对不齐。）
        tok = processor.tokenizer(full_text, add_special_tokens=False,
                                  return_offsets_mapping=True)
        text_ids = tok["input_ids"]
        offsets = tok["offset_mapping"]
        # 图片占位 token 在带图序列里被展开成 N 个 token，其后所有下标平移 (N-1)。
        # N-1 = 带图总长 - 纯文本总长（一张图， empirically 计算，与模型/合并率无关）
        pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        pad_pos = text_ids.index(pad_id)
        shift = n - len(text_ids)

        labels = torch.full((n,), -100)
        assistant_spans = []
        cursor = 0
        for m in blocks:
            if m["role"] != "assistant":
                continue
            gold = m["content"]
            # 从 cursor 起找 gold 的最后一次出现：直接问句里也含「推行/骑行」字样，
            # 最后一次出现才是答案本身
            idx = full_text.rfind(gold, cursor)
            assert idx >= cursor, f"{row['frame_id']} 在渲染文本中找不到答案"
            s, e = idx, idx + len(gold)
            ts = next(i for i, (a, b) in enumerate(offsets) if b > s)
            te = next(i for i in range(ts, len(offsets)) if offsets[i][0] >= e)
            ts_img = ts + (shift if ts > pad_pos else 0)
            te_img = te + (shift if te > pad_pos else 0)
            decoded = processor.decode(ids_img[ts_img:te_img], skip_special_tokens=True)
            assert decoded.strip() == gold.strip(), \
                f"{row['frame_id']} loss区与金标不符: {decoded[:40]!r} vs {gold[:40]!r}"
            assistant_spans.append((ts_img, te_img))
            labels[ts_img:te_img] = ids_img[ts_img:te_img]
            cursor = e

        item = {k: v[0] for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")}
        item["labels"] = labels
        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            item["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]
        item["meta"] = {"frame_id": row["frame_id"], "style": row.get("style"),
                        "n_loss_tokens": int((labels != -100).sum()),
                        "spans": assistant_spans}
        return item

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return {k: v for k, v in item.items() if k != "meta"}


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
    ap.add_argument("--data", action="append", required=True,
                    help="jsonl 数据文件，可传多个（如 train + train_aug）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4, help="每卡批次（4×8=32 有效批次）")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="关掉梯度检查点（显存够时更快；默认开，3B+2048长度+A100 40GB 实测 bs4 不开会 OOM）")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--mode", choices=MODES, default="llm_lora",
                    help="llm_lora=只训语言侧(默认,A1配方); llm_vit_lora=语言+视觉都加LoRA; "
                         "vit_lora=只训视觉; projector_only=只训merger")
    ap.add_argument("--alpha", type=int, default=None, help="默认 2×rank")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟测试用）")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    if args.mode == "projector_only":
        # 只训视觉→语言连接层（visual.merger），其余全冻结，不加 LoRA
        hit = 0
        for name, p in model.named_parameters():
            train = "merger" in name.lower() or "projector" in name.lower()
            p.requires_grad = train
            hit += train
        print(f"projector_only: {hit} 个参数张量可训练")
    else:
        # 冻结策略：llm_lora 冻视觉；vit_lora 冻语言；llm_vit_lora 全训 LoRA 都不冻
        if args.mode in ("llm_lora", "llm_vit_lora"):
            for name, p in model.named_parameters():
                if "visual" in name:
                    p.requires_grad = False
        if args.mode == "vit_lora":
            for name, p in model.named_parameters():
                if "visual" not in name:
                    p.requires_grad = False
        targets = {"llm_lora": LORA_TARGETS,
                   "vit_lora": VIT_LORA_TARGETS,
                   "llm_vit_lora": LORA_TARGETS + VIT_LORA_TARGETS}[args.mode]
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.alpha or 2 * args.rank,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    if not args.no_grad_ckpt:
        model.enable_input_require_grads()  # 冻结层在前时梯度检查点需要这个

    ds = SFTDatasetV2(args.data, processor, cache_dir=Path(args.out) / "cache",
                      max_len=args.max_len)
    if args.limit:
        ds.items = ds.items[: args.limit]
        print(f"冒烟模式: 只用前 {len(ds)} 条")
    n_loss = sum(it["meta"]["n_loss_tokens"] for it in ds.items)
    print(f"训练样本: {len(ds)}, 参与 loss 的 token 总数: {n_loss}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not args.no_grad_ckpt,
        bf16=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=4,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, pad_id))
    trainer.train()

    # 合并 LoRA 再保存（from_pretrained 直接加载 lora 键会静默丢适配器，E2 时期踩过的坑）
    if hasattr(model, "merge_and_unload"):
        print("合并 LoRA → 保存纯模型…")
        model = model.merge_and_unload()
        trainer.model = model
    trainer.save_model(args.out + "/final")
    processor.save_pretrained(args.out + "/final")

    # 训练摘要：配置 + 数据 + 最终 loss，进 results/ 做实验管理
    summary = {
        "out": args.out, "model": args.model, "data": args.data, "mode": args.mode,
        "epochs": args.epochs, "lr": args.lr, "effective_batch": args.bs * args.grad_accum,
        "lora_rank": args.rank, "lora_alpha": args.alpha or 2 * args.rank,
        "lora_targets": {"llm_lora": LORA_TARGETS, "vit_lora": VIT_LORA_TARGETS,
                       "llm_vit_lora": LORA_TARGETS + VIT_LORA_TARGETS}.get(args.mode, []),
        "train_samples": len(ds),
        "loss_tokens": n_loss,
        "final_train_loss": trainer.state.log_history[-1].get("loss"),
    }
    Path(args.out, "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 训后自检：第 1 条样本生成一次，确认不是乱码
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
