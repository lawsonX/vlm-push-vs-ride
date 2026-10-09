#!/usr/bin/env python3
"""把人工标注结果转成微调训练数据。

输入：
- annotations.jsonl（标注页面产出，frame_id -> label/note/evidence）
- 帧目录（图就在里面）

输出两种训练样本（可开关）：
1. 直接答（E1~E6 用）：图片 + 直接问 → 简短答案「推行/骑行」
2. 证据链（E7 用）：图片 + 直接问 → 「证据：... 结论：...」（证据用标注人写的 evidence 字段）

标注为「不确定」的图不进训练集（训练数据不能带脏标签）。
没有 evidence 文字的图不进证据链格式。

用法：
    python make_sft_data.py --annotations /path/annotations.jsonl \
        --frames-dir /path/frames --out-direct train_direct.jsonl --out-chain train_chain.jsonl
"""

import argparse
import json
from pathlib import Path

QUESTION = "图中这个人是在推行还是骑行这辆车？"
ANSWER = {"push": "推行", "ride": "骑行"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-direct", required=True)
    ap.add_argument("--out-chain", default=None)
    ap.add_argument("--balance", action="store_true",
                    help="两类等量抽样（推行是少数类，骑行随机配平），防训练分布失真")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    import random
    rng = random.Random(args.seed)
    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    if args.balance:
        push = [a for a in anns if a.get("label") == "push"]
        ride = [a for a in anns if a.get("label") == "ride"]
        rng.shuffle(ride)
        n = min(len(push), len(ride))
        anns = push + ride[:n]
        print(f"配平后: push={len(push)}, ride={min(len(ride), len(push))}")

    n_direct = n_chain = n_skip = 0
    fdir = open(args.out_direct, "w", encoding="utf-8")
    fchain = open(args.out_chain, "w", encoding="utf-8") if args.out_chain else None

    for a in anns:
        label = a.get("label", "")
        if label not in ANSWER:
            n_skip += 1  # 不确定或未标的，不进训练集
            continue
        img = frames_dir / f"{a['frame_id']}.jpg"
        if not img.exists():
            n_skip += 1
            continue

        fdir.write(json.dumps({
            "frame_id": a["frame_id"],
            "image": str(img),
            "question": QUESTION,
            "answer": ANSWER[label],
        }, ensure_ascii=False) + "\n")
        n_direct += 1

        ev = (a.get("evidence") or "").strip()
        if fchain and ev:
            fchain.write(json.dumps({
                "frame_id": a["frame_id"],
                "image": str(img),
                "question": QUESTION,
                "answer": f"证据：{ev}。结论：{ANSWER[label]}。",
            }, ensure_ascii=False) + "\n")
            n_chain += 1

    fdir.close()
    if fchain:
        fchain.close()
    print(f"直接答样本: {n_direct}，证据链样本: {n_chain}，跳过(不确定/缺图): {n_skip}")


if __name__ == "__main__":
    main()
