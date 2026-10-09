#!/usr/bin/env python3
"""组装 DPO 偏好对（E8 实验数据）。

思路（来自调研，OPA-DPO / HardVQA-DPO 的做法）：
- 被选回答（chosen）：人工标注的正确短答
- 被拒回答（rejected）：模型自己在这张图上的错误回答（on-policy，最有教学价值）
- 「语言合理但视觉无依据」的错误回答正是我们要压制的「编造证据」行为

来源文件：run_matrix 的直接问结果 + 人工标注。
只对「模型答错」的帧建对；答对的帧不需要。

用法：
    python make_dpo_data.py --annotations a.jsonl \
        --preds web_direct_5models.jsonl web2_direct_5models.jsonl \
        --frames-dir /path --out dpo_pairs.jsonl --model qwen3.5-omni-flash
"""

import argparse
import json
from pathlib import Path

ANSWER = {"push": "推行", "ride": "骑行"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-omni-flash",
                    help="用哪个模型的错误回答当被拒样本（默认最差的 qwen3.5-omni）")
    args = ap.parse_args()

    truth = {}
    for l in open(args.annotations, encoding="utf-8"):
        a = json.loads(l)
        if a.get("label") in ANSWER:
            truth[a["frame_id"]] = a["label"]

    wrong = {}
    for path in args.preds:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["model"] != args.model or r["style"] != "direct":
                continue
            fid = r["frame_id"]
            t = truth.get(fid)
            if t and r["label_pred"] in ("push", "ride") and r["label_pred"] != t:
                wrong[fid] = (t, r["answer"][:200])

    frames_dir = Path(args.frames_dir)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fid, (t, bad) in wrong.items():
            img = frames_dir / f"{fid}.jpg"
            if not img.exists():
                continue
            f.write(json.dumps({
                "frame_id": fid,
                "image": str(img),
                "question": "图中这个人是在推行还是骑行这辆车？",
                "chosen": ANSWER[t],
                "rejected": bad,
                "rejected_model": args.model,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"DPO 偏好对: {n}（模型 {args.model} 答错的帧）-> {args.out}")


if __name__ == "__main__":
    main()
