#!/usr/bin/env python3
"""评测 v2：加载训练产物，在冻结测试集（delivery_v02/test.jsonl）上逐样本生成。

- 测试集永远用原图 48 张、161 条样本，四种问法（direct/probe_seat/probe_feet/caption）
- 每种问法用自己的方式判对错：
  direct:      答案里「推行/骑行」最后一次出现为准
  probe_seat:  期望 push→没有 / ride→有
  probe_feet:  期望 push→地上 / ride→踏板
  caption:     关键词启发式（gold=push 却不该说「骑行」，gold=ride 同理），原始文本全量存档供人工复核
- fp32 全程跑（A100 40GB 放得下 3B fp32，避开 fp16 NaN 帧乱码问题）
- 每条测试样本可能有多轮（guide 风格）：逐轮生成，只把最后一轮计入指标

用法：
    python eval_v2.py --ckpt ckpt/a1_baseline/final \
        --test data/sft_v2/test.jsonl --out results/a1_baseline_preds.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_sft_v2 import build_blocks


def parse_direct(text):
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def parse_probe_seat(text):
    # 「没有」要优先于「有」判断（避免"没有坐在车座上"被 "有" 命中）
    if "没有" in text or "未" in text or "不在" in text:
        return "no"
    if "有" in text or "坐在" in text:
        return "yes"
    return "none"


def parse_probe_feet(text):
    if "踏板" in text or "踩着" in text:
        return "pedal"
    if "地上" in text or "地面" in text or "走路" in text:
        return "ground"
    return "none"


def parse_caption(text, gold):
    """开放描述：检查有没有把推行说成骑行（或反过来）。

    返回 (判分, 细节)。push 图说了「骑行/骑在/骑着走」→ 错（幻觉）；
    ride 图说了「推行/推车/步行推」→ 错。两头都没提 → 记 none，不算对也不算错。
    """
    says_ride = any(k in text for k in ("骑行", "骑在", "骑着", "骑车走", "骑在车上"))
    says_push = any(k in text for k in ("推行", "推车", "步行推", "走着推"))
    if gold == "push":
        if says_ride:
            return "wrong", f"push图说成骑行: {text[:60]}"
        if says_push:
            return "ok", ""
        return "none", "未提及"
    else:
        if says_push:
            return "wrong", f"ride图说成推行: {text[:60]}"
        if says_ride:
            return "ok", ""
        return "none", "未提及"


def judge(style, text, gold):
    if style == "direct":
        return parse_direct(text), parse_direct(text) == gold
    if style == "probe_seat":
        p = parse_probe_seat(text)
        exp = "yes" if gold == "ride" else "no"
        return p, p == exp
    if style == "probe_feet":
        p = parse_probe_feet(text)
        exp = "pedal" if gold == "ride" else "ground"
        return p, p == exp
    if style == "caption":
        s, _ = parse_caption(text, gold)
        return s, s == "ok"
    return "none", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", default="data/sft_v2/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=torch.float32, attn_implementation="sdpa",
    ).to(device).eval()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    stat = defaultdict(Counter)      # style -> Counter
    confusion = defaultdict(Counter)  # direct 专用: truth -> pred

    from PIL import Image
    for i, r in enumerate(rows, 1):
        img = Image.open(r["image"]).convert("RGB")
        blocks = build_blocks(r["messages"])
        turns = []  # 逐轮生成记录
        for j, m in enumerate(blocks):
            if m["role"] != "assistant":
                continue
            prompt = processor.apply_chat_template(
                blocks[: j + 1], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            turns.append(text.strip())
            # 多轮时把生成结果塞回上下文（模拟真实对话），最后一条 user 之后停下
            if any(b["role"] == "user" for b in blocks[j + 1:]):
                blocks[j]["content"] = turns[-1]
        final = turns[-1] if turns else ""
        pred, correct = judge(r["style"], final, r["label"])
        stat[r["style"]]["hit" if correct else "miss"] += 1
        if r["style"] == "direct":
            confusion[r["label"]][pred] += 1
        fout.write(json.dumps({
            "frame_id": r["frame_id"], "style": r["style"], "label": r["label"],
            "pred": pred, "correct": correct, "answer_final": final[:300],
            "turns": turns,
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    fout.close()

    print("\n=== 分问法准确率 ===")
    for style in sorted(stat):
        c = stat[style]
        n = c["hit"] + c["miss"]
        print(f"{style:12s}: {c['hit']}/{n} = {100*c['hit']/max(n,1):.0f}%")
    allc = Counter()
    for c in stat.values():
        allc.update(c)
    n = allc["hit"] + allc["miss"]
    print(f"{'overall':12s}: {allc['hit']}/{n} = {100*allc['hit']/max(n,1):.0f}%")

    print("\n=== direct 混淆矩阵 (行=真值 列=预测) ===")
    for t in ("push", "ride"):
        row = confusion[t]
        total = sum(row.values())
        print(f"{t:5s}: push={row.get('push',0)} ride={row.get('ride',0)} none={row.get('none',0)}  (n={total})")

    summary = {
        "ckpt": args.ckpt, "test": args.test, "samples": len(rows),
        "per_style": {s: dict(c) for s, c in stat.items()},
        "overall": dict(allc),
        "confusion_direct": {t: dict(confusion[t]) for t in ("push", "ride")},
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写: {summary_path}")


if __name__ == "__main__":
    main()
