#!/usr/bin/env python3
"""合成证据链训练样本（E7 用）。

lawson 标注时没写 evidence 字段，但 run_matrix 的 guide 问法在两类图上
都产出了高质量观察文本（推行图上诚实、骑行图上也忠实）。
对「直接问答对了」的帧，把对应观察文本拼成「证据：...结论：...」格式。

注意：证据是模型生成的（flash 的观察），不是人写的。质量抽查过，
但训练数据出处要如实记录（auto_evidence=true）。

用法：
    python make_chain_from_obs.py --annotations a.jsonl --styles r1.jsonl r2.jsonl \
        --frames-dir /path --out chain.jsonl
"""

import argparse
import json
from pathlib import Path

QUESTION = "图中这个人是在推行还是骑行这辆车？"
ANSWER = {"push": "推行", "ride": "骑行"}
OBS_MODEL = "qwen3-vl-flash"  # 观察最诚实的模型


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--styles", nargs="+", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    truth = {}
    for l in open(args.annotations, encoding="utf-8"):
        a = json.loads(l)
        if a.get("label") in ANSWER:
            truth[a["frame_id"]] = a["label"]

    obs = {}
    direct = {}
    for path in args.styles:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["model"] != OBS_MODEL:
                continue
            fid = r["frame_id"]
            if r["style"] == "guide" and r.get("observation"):
                obs[fid] = r["observation"]
            if r["style"] == "direct":
                direct[fid] = r.get("label_pred")

    frames_dir = Path(args.frames_dir)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fid, t in truth.items():
            # 只收：直接问答对了 + 有诚实观察 的帧
            if direct.get(fid) != t or fid not in obs:
                continue
            img = frames_dir / f"{fid}.jpg"
            if not img.exists():
                continue
            ev = " ".join(obs[fid].split())[:300]
            f.write(json.dumps({
                "frame_id": fid,
                "image": str(img),
                "question": QUESTION,
                "answer": f"证据：{ev}。结论：{ANSWER[t]}。",
                "auto_evidence": True,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"证据链样本: {n} -> {args.out}")


if __name__ == "__main__":
    main()
