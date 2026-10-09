#!/usr/bin/env python3
"""微调后模型的本地评测：加载检查点，跑测试集，算推行/骑行召回率。

对应云端 eval_accuracy.py 的本地版。M6 每个实验训完，用它出主指标。

用法（CPU 也行，GPU 更快）：
    python eval_local.py --ckpt /path/to/ckpt/final \
        --annotations /path/annotations.jsonl \
        --frames-dir /path/frames --out /path/preds.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()

    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    anns = [a for a in anns if a.get("label") in ("push", "ride")]
    if args.limit > 0:
        anns = anns[: args.limit]

    fout = open(args.out, "w", encoding="utf-8")
    for i, a in enumerate(anns, 1):
        img_path = Path(args.frames_dir) / f"{a['frame_id']}.jpg"
        if not img_path.exists():
            continue
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "url": str(img_path)},
            {"type": "text", "text": QUESTION},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_label(text)
        fout.write(json.dumps({
            "frame_id": a["frame_id"], "model": Path(args.ckpt).name,
            "style": "direct", "label_pred": pred, "answer": text[:100],
            "truth": a["label"],
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(anns)}", flush=True)

    fout.close()

    # 汇总
    recs = [json.loads(l) for l in open(args.out, encoding="utf-8")]
    stat = defaultdict(Counter)
    for r in recs:
        correct = r["label_pred"] == r["truth"]
        stat[r["truth"]]["hit" if correct else "miss"] += 1
    name = Path(args.ckpt).name
    for t in ("push", "ride"):
        c = stat[t]
        n = c["hit"] + c["miss"]
        if n:
            print("%s %s 召回: %d/%d = %.0f%%" % (name, t, c["hit"], n, 100 * c["hit"] / n))
    n = sum(c["hit"] + c["miss"] for c in stat.values())
    hit = sum(c["hit"] for c in stat.values())
    print("%s 总准确率: %d/%d = %.0f%%" % (name, hit, n, 100 * hit / max(n, 1)))


if __name__ == "__main__":
    main()
