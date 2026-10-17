#!/usr/bin/env python3
"""探针不确定性扫描：对训练集的探针样本用当前最佳模型多次采样，
找出「模型自己都不确定」的图——这些就是该标「看不清」或剔除的候选。

原理：v0.6 训练集把探针覆盖到 136/136 张图，但「看不清」只有 1 条，
模糊图被强行标了确定性答案（A7 实验证明这会毒害训练）。
本扫描用采样分歧率近似「图本身看不清」的程度，输出回标候选清单给数据侧。

用法：
    python scripts/scan_probe_uncertainty.py --model ckpt/a1_baseline/final \
        --data data/sft_v6_abs/train.jsonl --out data/probe_uncertainty_v6.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from eval_v2 import judge

PROBE_STYLES = ("probe_seat", "probe_feet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/a1_baseline/final")
    ap.add_argument("--data", default="data/sft_v6_abs/train.jsonl")
    ap.add_argument("--out", default="data/probe_uncertainty_v6.jsonl")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa",
    ).to(device).eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["style"] in PROBE_STYLES]
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    from PIL import Image
    fout = open(args.out, "w", encoding="utf-8")
    n_ambiguous = 0
    for i, r in enumerate(rows, 1):
        img = Image.open(r["image"]).convert("RGB")
        user_msg = next(m for m in r["messages"] if m["role"] == "user")
        text = user_msg["content"].replace("<图片占位>", "", 1).strip()
        gold = next(m for m in r["messages"] if m["role"] == "assistant")["content"]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[prompt_text], images=[img], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16, do_sample=True,
                                 temperature=args.temperature,
                                 num_return_sequences=args.samples,
                                 pad_token_id=pad_id)
        pl = enc["input_ids"].shape[1]
        preds = []
        for g in gen:
            t = processor.decode(g[pl:], skip_special_tokens=True)
            p, _ = judge(r["style"], t, gold)
            preds.append(p)
        from collections import Counter
        dist = Counter(preds)
        top, top_n = dist.most_common(1)[0]
        agree = top_n / args.samples          # 模型答案一致率
        _, gold_ok = judge(r["style"], top if top != "none" else "", gold)
        ambiguous = agree < 0.75               # 一致性<75% 视为图看不清的候选
        n_ambiguous += ambiguous
        fout.write(json.dumps({
            "frame_id": r["frame_id"], "style": r["style"], "label": r["label"],
            "gold": gold, "dist": dict(dist), "agree": round(agree, 3),
            "model_top": top, "top_matches_gold": gold_ok, "ambiguous": ambiguous,
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  候选累计 {n_ambiguous}", flush=True)
    fout.close()
    print(f"完成: {len(rows)} 条探针样本, 不确定性候选 {n_ambiguous} 条 -> {args.out}")


if __name__ == "__main__":
    main()
