#!/usr/bin/env python3
"""E8-v2 手写版 DPO：支持多 epoch（v1 每对只过 1 次）。

与 v1 的差异：
- 新增 --epochs，数据对循环跑 N 轮，观察 margin 是否逐轮增大
- 其余逻辑（LoRA q/v、beta、恢复基座词嵌入）与 v1 完全一致，保证可比
"""
import argparse
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "图中这个人是在推行还是骑行这辆车？"


def seq_logprob(model, batch, answer_len, device):
    out = model(**batch)
    logits = out.logits[0]  # [T, V]
    labels = batch["input_ids"][0]
    tgt = labels[-answer_len:].to(device)
    pred = logits[-answer_len - 1:-1]
    lp = torch.log_softmax(pred.float(), dim=-1)
    return lp.gather(-1, tgt.unsqueeze(-1)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16).to(device)
    ref = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    print(f"偏好对: {len(pairs)}  x {args.epochs} epoch = {len(pairs)*args.epochs} 步", flush=True)

    from PIL import Image

    # 预扫：fp16 下 forward 出 NaN 的图像整对剔除（fp16 NaN 帧已知问题，
    # 训练中会把梯度污染成 NaN）
    print("预扫 NaN 帧…", flush=True)
    bad_imgs = set()
    uniq = sorted({r["image"] for r in pairs})
    ref.eval()
    for p in uniq:
        img = Image.open(p).convert("RGB")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "url": p},
                {"type": "text", "text": QUESTION},
            ]}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt")
        batch = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = ref(**batch).logits[0].float()
        if torch.isnan(logits).any():
            bad_imgs.add(p)
            print(f"  NaN 帧剔除: {p.split('/')[-1]}", flush=True)
    before = len(pairs)
    pairs = [r for r in pairs if r["image"] not in bad_imgs]
    print(f"剔除 {before - len(pairs)} 对，剩 {len(pairs)} 对", flush=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        for r in pairs:
            step += 1
            img = Image.open(r["image"]).convert("RGB")
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "url": r["image"]},
                    {"type": "text", "text": QUESTION},
                ]}], tokenize=False, add_generation_prompt=True)

            stats = []
            for text in (r["chosen"], r["rejected"]):
                full = prompt + text
                inputs = processor(text=[full], images=[img], return_tensors="pt")
                ans_ids = processor(text=[text], add_special_tokens=False)["input_ids"][0]
                batch = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    ref_lp = seq_logprob(ref, batch, len(ans_ids), device)
                pol_lp = seq_logprob(model, batch, len(ans_ids), device)
                stats.append((pol_lp, ref_lp))

            (pol_c, ref_c), (pol_r, ref_r) = stats
            margin = (pol_c - ref_c) - (pol_r - ref_r)
            loss = -F.logsigmoid(args.beta * margin)
            if torch.isnan(loss):
                print(f"ep{epoch} step {step} NaN 跳过该步", flush=True)
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            print(f"ep{epoch} step {step}/{len(pairs)*args.epochs} "
                  f"loss={float(loss):.4f} margin={float(margin):.3f}", flush=True)

    # 保存：合并 LoRA + 恢复基座词嵌入
    print("合并并保存…", flush=True)
    from safetensors import safe_open
    import os
    embed = None
    for fn in os.listdir(args.model):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(args.model, fn), framework="pt") as f:
                if "model.embed_tokens.weight" in f.keys():
                    embed = f.get_tensor("model.embed_tokens.weight")
                    break
    merged = model.merge_and_unload()
    if embed is not None:
        with torch.no_grad():
            for n, p in merged.named_parameters():
                if n.endswith("embed_tokens.weight"):
                    p.copy_(embed.to(p.dtype))
    merged.save_pretrained(args.out + "/final", safe_serialization=True)
    processor.save_pretrained(args.out + "/final")
    print("done ->", args.out + "/final", flush=True)


if __name__ == "__main__":
    main()
