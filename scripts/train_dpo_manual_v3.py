#!/usr/bin/env python3
"""E8-v3 手写版 DPO：fp32 policy + 方向配平 + NaN 帧走 CPU ref。

v2 的教训：
1. 偏好对 40 里 35:5 严重偏向「选骑行」，NaN 预扫又恰好剔掉全部 5 条
   反方向对，训完模型变成 100% 骑行复读机（push 17%）。
2. 所以 v3：把少数方向 5 对全部保留，多数方向随机配平到 15 对（共 20 对）。
3. NaN 帧不再剔除——policy 用 fp32（不再 NaN），ref 对 NaN 帧临时挪到
   CPU 做 fp32 前向（no-grad，慢但只 5 对）。

用法：
    python train_dpo_manual_v3.py --model ... --pairs dpo_pairs_v2.jsonl \
        --out ... --beta 0.1 --lr 1e-5 --epochs 2
"""
import argparse
import json
import random

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
    ap.add_argument("--maj-cap", type=int, default=15,
                    help="多数方向最多保留几对（配平用）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    # policy 用 fp32：彻底解决 policy 侧 NaN；显存 12GB + ref fp16 6GB ≈ 18GB
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32).to(device)
    ref = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16).to(device)
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

    # 方向配平：少数方向全保留，多数方向随机采样到 maj_cap
    dir_key = lambda r: (r["chosen"], r["rejected"])
    from collections import Counter
    cnt = Counter(dir_key(r) for r in pairs)
    min_dir = min(cnt, key=cnt.get)  # 数量少的方向
    min_pairs = [r for r in pairs if dir_key(r) == min_dir]
    maj_pairs = [r for r in pairs if dir_key(r) != min_dir]
    rng = random.Random(42)
    rng.shuffle(maj_pairs)
    maj_pairs = maj_pairs[:args.maj_cap]
    pairs = min_pairs + maj_pairs
    rng.shuffle(pairs)
    print(f"少数方向 {len(min_pairs)} 对（{min_dir[0]} > {min_dir[1]}），"
          f"多数方向采样 {len(maj_pairs)} 对，共 {len(pairs)} 对 x {args.epochs} epoch", flush=True)

    from PIL import Image

    # 预扫 NaN 帧（ref fp16 前向为准；NaN 帧的 ref 计算走 CPU fp32）
    print("预扫 NaN 帧…", flush=True)
    bad_imgs = set()
    uniq = sorted({r["image"] for r in pairs})
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
            print(f"  NaN 帧(走CPU ref): {p.split('/')[-1]}", flush=True)

    step = 0
    total = len(pairs) * args.epochs
    for epoch in range(1, args.epochs + 1):
        for r in pairs:
            step += 1
            img = Image.open(r["image"]).convert("RGB")
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "url": r["image"]},
                    {"type": "text", "text": QUESTION},
                ]}], tokenize=False, add_generation_prompt=True)

            use_cpu_ref = r["image"] in bad_imgs
            stats = []
            for text in (r["chosen"], r["rejected"]):
                full = prompt + text
                inputs = processor(text=[full], images=[img], return_tensors="pt")
                ans_ids = processor(text=[text], add_special_tokens=False)["input_ids"][0]
                batch = {k: v.to(device) for k, v in inputs.items()}
                if use_cpu_ref:
                    ref.to("cpu")
                    ref32 = ref.to(torch.float32)
                    cb = {k: v.cpu() for k, v in batch.items()}
                    with torch.no_grad():
                        ref_lp = seq_logprob(ref32, cb, len(ans_ids), "cpu")
                    ref32.to(torch.float16).to(device)
                else:
                    with torch.no_grad():
                        ref_lp = seq_logprob(ref, batch, len(ans_ids), device)
                pol_lp = seq_logprob(model, batch, len(ans_ids), device)
                stats.append((pol_lp, ref_lp))

            (pol_c, ref_c), (pol_r, ref_r) = stats
            margin = (pol_c - ref_c) - (pol_r - ref_r)
            loss = -F.logsigmoid(args.beta * margin)
            if torch.isnan(loss):
                print(f"ep{epoch} step {step} NaN 跳过", flush=True)
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            print(f"ep{epoch} step {step}/{total} loss={float(loss):.4f} "
                  f"margin={float(margin):.3f} cpu_ref={use_cpu_ref}", flush=True)

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
