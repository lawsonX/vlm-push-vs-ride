#!/usr/bin/env python3
"""汇总分析导出：真值 × 预测 × 格子 → 一张大表 + 分组统计。

输出两个文件：
- frames_analysis.csv：每帧一行（真值、各模型各问法预测、格子维度、来源）
- summary.txt：按模型/问法/格子维度的召回率统计（重点：推行召回 × 距离）

用法：
    python export_analysis.py --annotations /path/annotations.jsonl \
        --preds a.jsonl b.jsonl --grid grid.jsonl --out-prefix /path/report
"""

import argparse
import csv
import json
from collections import Counter, defaultdict


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    anns = {a["frame_id"]: a for a in load_jsonl(args.annotations)}
    grid = {}
    if args.grid:
        for g in load_jsonl(args.grid):
            grid[g["frame_id"]] = g

    preds = {}  # (model, style, frame_id) -> label_pred
    for path in args.preds:
        for r in load_jsonl(path):
            preds[(r["model"], r["style"], r["frame_id"])] = r.get("label_pred", "none")

    models_styles = sorted({(m, s) for (m, s, _) in preds})

    # 大表
    with open(args.out_prefix + "_frames.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["frame_id", "truth", "dist", "light", "vehicle", "note"]
        for m, s in models_styles:
            header.append(f"{m}|{s}")
        w.writerow(header)
        for fid, a in sorted(anns.items()):
            g = grid.get(fid, {})
            row = [fid, a.get("label", ""), g.get("dist", ""), g.get("light", ""),
                   g.get("vehicle", ""), (a.get("note") or "")[:40]]
            for m, s in models_styles:
                row.append(preds.get((m, s, fid), ""))
            w.writerow(row)

    # 分组统计：只在 push/ride 真值上算，按距离维度切
    stat = defaultdict(Counter)
    totals = defaultdict(Counter)
    for (m, s, fid), p in preds.items():
        t = anns.get(fid, {}).get("label")
        if t not in ("push", "ride"):
            continue
        d = grid.get(fid, {}).get("dist", "?")
        correct = (p == t)
        if t == "push":
            stat[(m, s, d)]["push_hit"] += correct
            stat[(m, s, d)]["push_n"] += 1
        totals[(m, s, d)]["hit"] += correct
        totals[(m, s, d)]["n"] += 1

    with open(args.out_prefix + "_summary.txt", "w", encoding="utf-8") as f:
        f.write("按模型/问法/距离 的推行召回与总准确率（真值仅 push/ride）\n\n")
        for (m, s, d) in sorted(stat):
            push_n = stat[(m, s, d)]["push_n"]
            push_hit = stat[(m, s, d)]["push_hit"]
            n = totals[(m, s, d)]["n"]
            hit = totals[(m, s, d)]["hit"]
            f.write("%-20s %-9s dist=%-5s 推行召回 %d/%d=%.0f%%  总准确率 %d/%d=%.0f%%\n" % (
                m, s, d, push_hit, push_n, 100 * push_hit / max(push_n, 1),
                hit, n, 100 * hit / max(n, 1)))

    print("done:", args.out_prefix + "_frames.csv", args.out_prefix + "_summary.txt")


if __name__ == "__main__":
    main()
