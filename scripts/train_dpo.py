#!/usr/bin/env python3
"""E8：难例 DPO 训练（在 SFT 检查点之上，压「编造证据」倾向）。

数据：make_dpo_data.py 产的偏好对（chosen=人工正确短答，rejected=模型自己的错误回答）。
做法（来自调研，OPA-DPO / HardVQA-DPO）：被拒样本是模型自己生成的（on-policy），
对这种「语言合理但视觉无依据」的回答给负奖励，专治编造证据。

技术要点同 train_sft.py：CPU 强制 fp32、保守 LoRA、保存前恢复词嵌入。

用法：
    python train_dpo.py --model /path/to/sft_ckpt/final \
        --pairs /path/dpo_pairs.jsonl --out /path/ckpt/e8_dpo \
        --beta 0.1 --lr 5e-6 --epochs 1
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig, DPOTrainer

QUESTION = "图中这个人是在推行还是骑行这辆车？"


def build_dataset(pairs_path, processor):
    rows = [__import__("json").loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]

    def render(img, text):
        return processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "url": img},
                {"type": "text", "text": QUESTION},
            ]}], tokenize=False, add_generation_prompt=True) + text

    data = {
        "prompt": [render(r["image"], "") for r in rows],
        "chosen": [r["chosen"] + "<|im_end|>" for r in rows],
        "rejected": [r["rejected"][:300] + "<|im_end|>" for r in rows],
        "images": [r["image"] for r in rows],
    }
    return Dataset.from_dict(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SFT 后的检查点")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    ref = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    ds = build_dataset(args.pairs, processor)
    print(f"偏好对: {len(ds)}")

    cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        fp16=False,  # 模型权重本身已是 fp16，不能再开混合精度缩放器
        optim="adamw_torch",
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataset_num_proc=1,
        # 注意：trl 的 precompute_ref_log_probs 不支持视觉模型，CPU 上双模型放不下。
        # CPU 模式仅适合 ≤1B 的模型；3B DPO 等 GPU（22GB 显存放双模型轻轻松松）。
        gradient_checkpointing=True,
    )
    trainer = DPOTrainer(model=model, ref_model=ref, args=cfg,
                         train_dataset=ds, processing_class=processor)
    trainer.train()

    # 防污染：恢复词嵌入（同 train_sft.py 的教训）
    from safetensors import safe_open
    import os, json as _json
    embed = None
    for fn in os.listdir(args.model):
        if fn.endswith(".safetensors.index.json"):
            idx = _json.load(open(os.path.join(args.model, fn)))
            with safe_open(os.path.join(args.model, idx["weight_map"]["model.embed_tokens.weight"]), framework="pt") as f:
                embed = f.get_tensor("model.embed_tokens.weight")
            break
    if embed is not None:
        with torch.no_grad():
            tgt = trainer.model.model.embed_tokens.weight
            tgt.copy_(embed.to(tgt.dtype))

    trainer.save_model(args.out + "/final")
    processor.save_pretrained(args.out + "/final")
    print("done ->", args.out + "/final")


if __name__ == "__main__":
    main()
