#!/usr/bin/env python3
"""评测 v2：NaN 帧自动 fp32 重试。

背景：部分帧在 fp16 下会让 Qwen-VL 系列输出 NaN logits，
生成结果变成 "!!!!!!!!!!" 等乱码，过去评测里这些帧一律算错。
且 48 张测试集里 7 张 NaN 帧全是 ride 类，严重拉低 ride 召回。

改动（相对 eval_local.py）：
- fp16 生成后，若答案解析不出 推行/骑行（label_pred == "none"），
  懒加载一个 fp32 副本模型重试该帧，记录里标 "dtype": "fp32"
- 其余口径与 eval_local.py 完全一致（保证可比）
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
    ap.add_argument("--all-fp32", action="store_true",
                    help="全程 fp32（NaN 帧已知问题，48 张规模下更稳妥）")
    ap.add_argument("--nan-list", default="",
                    help="NaN 帧黑名单 json（scan 产物）：名单内帧跳过不计分，"
                         "用于 fp32 放不下的超大模型（如 7B）")
    args = ap.parse_args()

    nan_set = set()
    if args.nan_list:
        nan_set = {b["frame"][:-4] for b in json.load(open(args.nan_list))}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.all_fp32 else (torch.float16 if device == "cuda" else torch.float32)

    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    fp32_model = None  # 懒加载

    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    anns = [a for a in anns if a.get("label") in ("push", "ride")]
    if args.limit > 0:
        anns = anns[: args.limit]

    def run_one(m, img):
        messages = [{"role": "user", "content": [
            {"type": "image", "url": str(img_path)},
            {"type": "text", "text": QUESTION},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=10, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    fout = open(args.out, "w", encoding="utf-8")
    n_fp32 = n_skip = 0
    for i, a in enumerate(anns, 1):
        if a["frame_id"] in nan_set:
            n_skip += 1
            continue
        img_path = Path(args.frames_dir) / f"{a['frame_id']}.jpg"
        if not img_path.exists():
            continue
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        text = run_one(model, img)
        pred = parse_label(text)
        used = "fp16"
        if pred == "none" and device == "cuda" and not args.all_fp32:
            if fp32_model is None:
                print("  检测到乱码帧，加载 fp32 副本…", flush=True)
                fp32_model = AutoModelForImageTextToText.from_pretrained(
                    args.ckpt, torch_dtype=torch.float32, attn_implementation="sdpa",
                ).to(device).eval()
            text = run_one(fp32_model, img)
            pred = parse_label(text)
            used = "fp32"
            n_fp32 += 1
        rec = {
            "frame_id": a["frame_id"], "model": Path(args.ckpt).name,
            "style": "direct", "label_pred": pred, "answer": text[:100],
            "truth": a["label"],
        }
        if used == "fp32":
            rec["dtype"] = "fp32"
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(anns)} (fp32 重试 {n_fp32} 次)", flush=True)

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
    print("%s 总准确率: %d/%d = %.0f%% (fp32 重试 %d 帧, 黑名单跳过 %d 帧)" % (name, hit, n, 100 * hit / max(n, 1), n_fp32, n_skip))


if __name__ == "__main__":
    main()
