#!/usr/bin/env python3
"""CLIP 零样本对照实验：纯视觉模型能不能分清「推行 vs 骑行」。

背景：这是「眼睛 vs 脑子」的分水岭实验。
- CLIP 只有视觉 + 文本匹配，没有语言模型推理。
- 如果 CLIP 能分清而 VLM 分不清 → 问题出在语言侧（对齐/决策），不是视觉编码。
- 如果 CLIP 也分不清 → 视觉线索本身在画面里就不好提取。

做法：每张图分别喂「整帧」和「人+车区域裁剪」给 CLIP，
对 3 组文本描述打分（推行 vs 骑行），记录分数和胜负。

用法：
    python clip_push_ride.py --pool /mnt/d/vlm-active/frames/trafficqa/pool_manifest.jsonl \
        --frames-dir /mnt/d/vlm-active/frames/trafficqa \
        --out /mnt/d/vlm-active/results/clip_scores.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROMPT_PAIRS = [
    ("a person pushing a bicycle", "a person riding a bicycle"),
    ("a person walking next to an electric scooter", "a person riding an electric scooter"),
    ("a person standing beside a bike", "a person sitting on a bike"),
]
PUSH_IDX, RIDE_IDX = 0, 1


def union_crop_box(row, ratio=0.5):
    """把人框和车框并起来，外扩一点，作为裁剪区域。"""
    boxes = [p["box"] for p in row.get("persons", [])] + [b["box"] for b in row.get("bikes", [])]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    return (
        max(0, x1 - w * ratio),
        max(0, y1 - h * ratio),
        x2 + w * ratio,
        y2 + h * ratio,
    )


@torch.no_grad()
def score(model, processor, img, prompts):
    inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True)
    out = model(**inputs)
    return out.logits_per_image[0]  # [len(prompts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        pool = pool[: args.limit]
    frames_dir = Path(args.frames_dir)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    prompts = [p for pair in PROMPT_PAIRS for p in pair]
    fout = open(args.out, "w", encoding="utf-8")

    for i, row in enumerate(pool, 1):
        img_path = frames_dir / f"{row['frame_id']}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        crop = img.crop(union_crop_box(row)) if union_crop_box(row) else img

        rec = {"frame_id": row["frame_id"]}
        for view, im in (("full", img), ("crop", crop)):
            logits = score(model, processor, im, prompts)
            for pi, pair in enumerate(PROMPT_PAIRS):
                push_s = float(logits[pi * 2 + PUSH_IDX])
                ride_s = float(logits[pi * 2 + RIDE_IDX])
                rec[f"{view}_pair{pi}"] = {"push": round(push_s, 2), "ride": round(ride_s, 2),
                                           "winner": "push" if push_s > ride_s else "ride"}
        # 综合：3 组里哪边赢得多
        for view in ("full", "crop"):
            wins = [rec[f"{view}_pair{pi}"]["winner"] for pi in range(len(PROMPT_PAIRS))]
            rec[f"{view}_overall"] = "push" if wins.count("push") >= 2 else "ride"
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(f"  进度 {i}/{len(pool)}", flush=True)

    fout.close()
    print(f"完成 -> {args.out}")


if __name__ == "__main__":
    main()
