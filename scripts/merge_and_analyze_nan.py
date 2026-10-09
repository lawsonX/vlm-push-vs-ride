#!/usr/bin/env python3
"""合并 NaN 扫描结果：主日志(前10170帧) + 尾部续扫结果 → nan_frames.json，然后跑分布分析。"""
import json
import re

# 1. 从主日志提取 BAD 帧（顺序扫描，覆盖 0~10169）
bad = {}
for line in open("/home/lawson/vlm-active/logs/scan_nan4.log", encoding="utf-8", errors="replace"):
    m = re.search(r"\[BAD \d+/\d+\] (\S+) nan_frac=([\d.]+)", line)
    if m:
        bad[m.group(1)] = {"frame": m.group(1), "nan_frac": float(m.group(2))}
print(f"主日志 BAD: {len(bad)}")

# 2. 尾部续扫结果
tail = json.load(open("/home/lawson/vlm-active/results/nan_frames_tail.json"))
for b in tail:
    bad[b["frame"]] = b
print(f"合并后 BAD: {len(bad)}")

out = sorted(bad.values(), key=lambda x: x["frame"])
json.dump(out, open("/home/lawson/vlm-active/results/nan_frames.json", "w"),
          ensure_ascii=False, indent=1)
print("-> /home/lawson/vlm-active/results/nan_frames.json")

# 3. 分布分析
import glob
from collections import Counter, defaultdict

def source(fn):
    if fn.startswith("sg_"):
        return "trafficqa"
    if fn.startswith(("b_", "BV")):
        return "b站"
    if fn.startswith("j_"):
        return "搜狗新闻"
    if fn.startswith("y_"):
        return "y_来源"
    if fn.startswith("c_"):
        return "c_来源(影视?)"
    return "其他"

all_frames = [p.split("/")[-1] for p in glob.glob("/home/lawson/vlm-active/frames/trafficqa/*.jpg")]
stat = defaultdict(Counter)
for fn in all_frames:
    stat[source(fn)]["total"] += 1
    if fn in bad:
        stat[source(fn)]["nan"] += 1

print("\n== 按来源 NaN 率 ==")
for s, c in sorted(stat.items(), key=lambda kv: -kv[1]["nan"] / max(kv[1]["total"], 1)):
    t = c["total"]
    print(f"{s:14s} {c['nan']:5d}/{t:5d} = {100*c['nan']/max(t,1):5.1f}%")

anns = {}
try:
    for l in open("/home/lawson/vlm-active/frames/trafficqa/annotations.jsonl"):
        a = json.loads(l)
        if a.get("label") in ("push", "ride"):
            anns[a["frame_id"]] = a["label"]
except FileNotFoundError:
    pass

cross = Counter(anns.get(fn[:-4], "未标注") for fn in bad)
print("\n== NaN 帧的标注类别 ==")
for lab, n in cross.most_common():
    print(f"{lab:6s} {n}")
all_lab = Counter(anns.values())
print("== 标注帧整体类别(对照) ==")
for lab, n in all_lab.most_common():
    print(f"{lab:6s} {n}")

json.dump({"by_source": {s: dict(c) for s, c in stat.items()},
           "bad_by_label": dict(cross)},
          open("/home/lawson/vlm-active/results/nan_analysis.json", "w"),
          ensure_ascii=False, indent=1)
print("-> /home/lawson/vlm-active/results/nan_analysis.json")
