#!/usr/bin/env python3
"""汇总 local2_*.jsonl（NaN 安全版重评结果），输出对比表。"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

rows = []
for f in sorted(glob.glob("/home/lawson/vlm-active/results/local2_*.jsonl")):
    tag = Path(f).stem.replace("local2_", "")
    stat = defaultdict(Counter)
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        stat[r["truth"]]["hit" if r["label_pred"] == r["truth"] else "miss"] += 1
    p, r = stat["push"], stat["ride"]
    pn, rn = p["hit"] + p["miss"], r["hit"] + r["miss"]
    if pn + rn == 0:
        print(f"{tag:<16} (空结果，评测可能失败)")
        continue
    tot_hit = p["hit"] + r["hit"]
    tot = pn + rn
    rows.append((tag, p["hit"], pn, r["hit"], rn, tot_hit, tot))

print(f"{'模型':<16} {'push':>10} {'ride':>10} {'overall':>10}")
for tag, ph, pn, rh, rn, th, tot in rows:
    print(f"{tag:<16} {ph}/{pn}={100*ph/pn:4.0f}%  {rh}/{rn}={100*rh/rn:4.0f}%  {th}/{tot}={100*th/tot:4.0f}%")
