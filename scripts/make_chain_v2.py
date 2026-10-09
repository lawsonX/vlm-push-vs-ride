#!/usr/bin/env python3
"""E7-v2 证据链训练数据：push 帧用人工证据，ride 帧用过滤后的自动观察。

E7-v1 的失败原因：推行帧的证据是模型自己生成的，在推行图上它不诚实
（描述成骑行特征），偏置被焊进证据。E7-v2 的核心改动：
- **push 帧：证据必须来自 lawson 手写**（annotations.jsonl 的 evidence 字段），
  没写证据的 push 帧不进训练集
- **ride 帧：沿用 v1 的过滤自动观察**（guide 问答对 + 直接问答对才收），
  因为 flash 在骑行图上的观察是忠实的
- 排除 NaN 帧（若 scan 产物存在）
- 两类格式完全一致：「证据：...结论：...」，防止模型拿「有没有证据」当分类特征

用法：
    python make_chain_v2.py --annotations a.jsonl --styles r1.jsonl r2.jsonl \
        --frames-dir /path --out chain_v2.jsonl [--nan-list nan_frames.json]
"""
import argparse
import json
from pathlib import Path

QUESTION = "图中这个人是在推行还是骑行这辆车？"
ANSWER = {"push": "推行", "ride": "骑行"}
OBS_MODEL = "qwen3-vl-flash"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--styles", nargs="+", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nan-list", default="")
    args = ap.parse_args()

    truth, evidence_human = {}, {}
    for l in open(args.annotations, encoding="utf-8"):
        a = json.loads(l)
        if a.get("label") in ANSWER:
            truth[a["frame_id"]] = a["label"]
            evidence_human[a["frame_id"]] = a.get("evidence", "").strip()

    nan_set = set()
    if args.nan_list:
        nan_set = {b["frame"] for b in json.load(open(args.nan_list))}

    obs, direct = {}, {}
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
    n_push = n_ride = skip = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fid, t in truth.items():
            img = frames_dir / f"{fid}.jpg"
            if not img.exists():
                continue
            if f"{fid}.jpg" in nan_set:
                skip += 1
                continue
            if t == "push":
                ev = evidence_human.get(fid, "")
                if not ev:
                    skip += 1  # 没人工证据的 push 帧不收（E7-v2 的灵魂）
                    continue
                answer = f"证据：{ev}\n结论：推行"
                n_push += 1
            else:  # ride
                if direct.get(fid) != "ride" or fid not in obs:
                    skip += 1
                    continue
                answer = f"证据：{obs[fid]}\n结论：骑行"
                n_ride += 1
            f.write(json.dumps({
                "frame_id": fid, "image": str(img),
                "question": QUESTION, "answer": answer,
                "label": t, "evidence_source": "human" if t == "push" else "auto",
            }, ensure_ascii=False) + "\n")

    print(f"push(人工证据) {n_push} 条，ride(自动观察) {n_ride} 条，跳过 {skip} 条")
    print("⚠️ 如果 push 是 0 条，说明 lawson 还没写 evidence 字段，先别训练")


if __name__ == "__main__":
    main()
