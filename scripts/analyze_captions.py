#!/usr/bin/env python3
"""统计公开 VLM 训练数据里「骑行 vs 推行」的描述占比（验证 H1 的数据侧）。

数据：LLaVA 预训练用的 558k BLIP 图文标注（liuhaotian/LLaVA-Pretrain）。
数法：
1. 先筛出提到自行车/电瓶车的标注
2. 在这些标注里数动作词：骑（ride/riding） vs 推（push/pushing/walk...next to）
3. 输出数量和占比 + 每类随机样例，人工复核数得对不对

用法：
    python analyze_captions.py --json /mnt/e/vlm-data/raw/LLaVA/blip_laion_cc_sbu_558k.json
"""

import argparse
import json
import random
import re
from collections import Counter

BIKE_RE = re.compile(
    r"bicycle|bike|motorbike|motorcycle|scooter|e-?bike|moped|"
    r"自行车|电瓶车|电动车|摩托车|单车",
    re.IGNORECASE,
)
RIDE_RE = re.compile(r"rid(e|ing|es|den)|骑", re.IGNORECASE)
PUSH_RE = re.compile(
    r"push(ing|es)?|walk(ing)? (next to|alongside|beside|with)|"
    r"推(着|车|自行车)|步行",
    re.IGNORECASE,
)
PARK_RE = re.compile(r"park(ed|ing)?|停(着|放|靠)", re.IGNORECASE)


def extract_caption(item):
    """兼容两种格式：顶层 caption 字段，或 conversations 里的 gpt 回答。"""
    if "caption" in item:
        return item["caption"]
    for c in item.get("conversations", []):
        if c.get("from") in ("gpt", "assistant"):
            return c.get("value", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--sample", type=int, default=15, help="每类打印多少条样例")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    print(f"总标注数: {len(data)}")

    bike_caps = []
    for item in data:
        cap = extract_caption(item)
        if cap and BIKE_RE.search(cap):
            bike_caps.append(cap)

    n = len(bike_caps)
    print(f"提到车的标注: {n} ({100*n/len(data):.2f}%)")

    cats = {"ride": [], "push": [], "park": [], "other": []}
    for cap in bike_caps:
        if RIDE_RE.search(cap):
            cats["ride"].append(cap)
        elif PUSH_RE.search(cap):
            cats["push"].append(cap)
        elif PARK_RE.search(cap):
            cats["park"].append(cap)
        else:
            cats["other"].append(cap)

    print("\n=== 动作分布（在有车的标注里）===")
    for k in ("ride", "push", "park", "other"):
        c = len(cats[k])
        print(f"{k:6s}: {c:6d} ({100*c/max(n,1):.1f}%)")

    if cats["ride"] and cats["push"]:
        print(f"\n骑行:推行 比例 = {len(cats['ride'])/max(len(cats['push']),1):.1f} : 1")

    random.seed(42)
    for k in ("ride", "push"):
        print(f"\n=== {k} 随机样例 ===")
        for s in random.sample(cats[k], min(args.sample, len(cats[k]))):
            print(" -", s[:150])


if __name__ == "__main__":
    main()
