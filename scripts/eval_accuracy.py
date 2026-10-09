#!/usr/bin/env python3
"""把标注真值和模型预测接起来，算准确率。

输入：
- annotations.jsonl（人工标注，label 字段是 push/ride/unsure）
- 一个或多个预测结果文件（run_matrix.py / clip_push_ride.py 的产出）

输出（打印到屏幕）：
1. 每个模型/问法的总准确率（只在标注为 push/ride 的图上算，unsure 不计）
2. 分标签看：推行图里答对多少（这对「推行被误判骑行」最敏感）
3. CLIP 分数文件的准确率（如果有）
4. 混淆情况

用法：
    python eval_accuracy.py --annotations /path/annotations.jsonl \
        --preds /path/direct_5models.jsonl /path/clip_scores.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    args = ap.parse_args()

    truth = {}
    for a in load_jsonl(args.annotations):
        if a.get("label") in ("push", "ride"):
            truth[a["frame_id"]] = a["label"]

    print(f"有效真值（push/ride）: {len(truth)} 张\n")

    for path in args.preds:
        recs = load_jsonl(path)
        # 判断结果类型：模型预测（有 model 字段）还是 CLIP（有 full_overall）
        by_key = defaultdict(dict)
        for r in recs:
            fid = r["frame_id"]
            if "model" in r:
                by_key[(r["model"], r.get("style", "?"))][fid] = r.get("label_pred", "none")
            elif "full_overall" in r:
                by_key[("CLIP", "full")][fid] = r["full_overall"]
                by_key[("CLIP", "crop")][fid] = r["crop_overall"]

        print(f"=== {path} ===")
        for key in sorted(by_key):
            preds = by_key[key]
            hit_p = hit_r = n_p = n_r = 0
            for fid, t in truth.items():
                if fid not in preds:
                    continue
                p = preds[fid]
                if t == "push":
                    n_p += 1
                    hit_p += (p == "push")
                else:
                    n_r += 1
                    hit_r += (p == "ride")
            n = n_p + n_r
            if n == 0:
                continue
            acc = (hit_p + hit_r) / n
            recall_push = hit_p / max(n_p, 1)
            recall_ride = hit_r / max(n_r, 1)
            print(f"{key[0]:22s} {key[1]:10s} 总准确率={acc:5.1%} (n={n})  "
                  f"推行召回={recall_push:5.1%} (n={n_p})  骑行召回={recall_ride:5.1%} (n={n_r})")
        print()


if __name__ == "__main__":
    main()
