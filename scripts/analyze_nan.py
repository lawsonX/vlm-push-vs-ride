#!/usr/bin/env python3
"""NaN 帧分布分析：按数据来源、标注类别统计 NaN 率。

输入：/home/lawson/vlm-active/results/nan_frames.json（扫描产物）
输出：打印统计 + 写 /home/lawson/vlm-active/results/nan_analysis.json
"""
import json
import re
from collections import Counter, defaultdict

bad = json.load(open("/home/lawson/vlm-active/results/nan_frames.json"))
bad_set = {b["frame"] for b in bad}

# 来源分类：按文件名前缀
def source(fn):
    if fn.startswith("sg_"):
        return "trafficqa视频"
    if fn.startswith(("b_", "BV")):
        return "b站"
    if fn.startswith("j_"):
        return "搜狗/新闻"
    if fn.startswith("y_"):
        return "y_来源待确认"
    if fn.startswith("c_"):
        return "c_来源待确认"
    return "其他"

import glob
all_frames = [p.split("/")[-1] for p in glob.glob("/home/lawson/vlm-active/frames/trafficqa/*.jpg")]

stat = defaultdict(lambda: Counter())
for fn in all_frames:
    s = source(fn)
    stat[s]["total"] += 1
    if fn in bad_set:
        stat[s]["nan"] += 1

print("== 按数据来源的 NaN 率 ==")
for s, c in stat.items():
    print(f"{s:12s} {c['nan']:5d}/{c['total']:5d} = {100*c['nan']/c['total']:.1f}%")

# 和标注的交叉：NaN 帧里 push/ride 各多少
try:
    anns = {json.loads(l)["frame_id"]: json.loads(l)["label"]
            for l in open("/home/lawson/vlm-active/frames/trafficqa/annotations.jsonl")}
    cross = Counter()
    for fn in bad_set:
        lab = anns.get(fn[:-4], "未标注")
        cross[lab] += 1
    print("\n== NaN 帧的标注类别分布 ==")
    for lab, n in cross.most_common():
        print(f"{lab:8s} {n}")
    # 对照：标注帧整体的类别分布
    all_lab = Counter(anns.values())
    print("\n== 全部标注帧类别分布（对照）==")
    for lab, n in all_lab.most_common():
        print(f"{lab:8s} {n}")
except FileNotFoundError:
    print("annotations.jsonl 不存在，跳过交叉分析")

json.dump({"by_source": {s: dict(c) for s, c in stat.items()},
           "bad_by_label": dict(cross)},
          open("/home/lawson/vlm-active/results/nan_analysis.json", "w"),
          ensure_ascii=False, indent=1)
print("\n-> /home/lawson/vlm-active/results/nan_analysis.json")
