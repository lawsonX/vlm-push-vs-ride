#!/usr/bin/env python3
"""A5：GRPO 强化学习进阶实验（docs/13 第 6 节）。

从 A1（SFT 后）出发，语言侧加 rank 16 LoRA，用可验证奖励做 GRPO：
direct/probe_seat/probe_feet 三种客观问法，答案判对给 1 分、判错 0 分。
caption/chain/guide 不参与（开放描述奖励不可靠、多轮 GRPO 不支持）。

用法：
    python train_grpo_v2.py --model ckpt/a1_baseline/final \
        --data data/sft_v2/train.jsonl --out ckpt/a5_grpo --max-steps 100
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent))

RL_STYLES = ("direct", "probe_seat", "probe_feet")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def build_dataset(data_path, processor, limit=0):
    from PIL import Image
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["style"] in RL_STYLES]
    # 去重：同一图同问法只留一条（数据本就是一图一条，保险起见）
    seen, recs = set(), []
    for r in rows:
        key = (r["frame_id"], r["style"])
        if key in seen:
            continue
        seen.add(key)
        user_msg = next(m for m in r["messages"] if m["role"] == "user")
        text = user_msg["content"].replace("<图片占位>", "", 1).strip()
        gold = next(m for m in r["messages"] if m["role"] == "assistant")["content"]
        img = Image.open(r["image"]).convert("RGB")
        recs.append({
            "prompt": [{"role": "user", "content": [
                {"type": "image"},  # 占位，TRL 会从 "images" 列填充
                {"type": "text", "text": text},
            ]}],
            "images": [img],
            "answer": gold, "style": r["style"], "frame_id": r["frame_id"],
        })
    if limit:
        recs = recs[:limit]
    print(f"GRPO 训练提示: {len(recs)} 条 (样式: {sorted(set(x['style'] for x in recs))})")
    return Dataset.from_list(recs)


def reward_correct(completions, style, answer, **kwargs):
    """可验证奖励：用与评测一致的判分逻辑，对=1.0，错/解析不出=0.0。"""
    rewards = []
    for comp, st, gold in zip(completions, style, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        _, ok = _judge(st, text, gold)
        rewards.append(1.0 if ok else 0.0)
    return rewards


def _judge(style, text, gold_word):
    """按问法判分。gold_word 是金标答案词本身（推行/骑行/有/没有/地上/踏板）。"""
    from eval_v2 import parse_direct, parse_probe_seat, parse_probe_feet
    if style == "direct":
        p = parse_direct(text)
        return p, p == gold_word
    if style == "probe_seat":
        p = parse_probe_seat(text)
        exp = {"有": "yes", "没有": "no"}.get(gold_word)
        return p, exp is not None and p == exp
    if style == "probe_feet":
        p = parse_probe_feet(text)
        exp = {"踏板": "pedal", "地上": "ground"}.get(gold_word)
        return p, exp is not None and p == exp
    return "none", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="起点模型（A1 final）")
    ap.add_argument("--data", default="data/sft_v2/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="每卡生成批次（须被 num_generations 整除关系见 TRL 文档）")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条提示（冒烟）")
    ap.add_argument("--save-steps", type=int, default=50)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    for name, p in model.named_parameters():  # 视觉冻结，省显存（A6 结论：不碰视觉）
        if "visual" in name:
            p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.print_trainable_parameters()

    ds = build_dataset(args.data, processor, args.limit)

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=16,
        temperature=1.0,
        beta=0.0,                # 无参考模型 KL（省一份前向；LoRA 模式下 TRL 用适配器禁用技巧）
        scale_rewards=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_correct,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
    )
    trainer.train()
    model = model.merge_and_unload()
    trainer.model = model
    model.save_pretrained(args.out + "/final")
    processor.save_pretrained(args.out + "/final")
    print("done ->", args.out + "/final")


if __name__ == "__main__":
    main()
