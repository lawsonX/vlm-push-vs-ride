#!/usr/bin/env python3
"""给每张池子里的图打「格子」标签（测试集切片用）。

格子的三个维度，全部用客观信号算，不靠人：
- 距离：人框面积占整图比例 → 近(>3%) / 中(1~3%) / 远(<1%)
- 光照：转灰度算平均亮度 → 白天(>100) / 夜晚(<60) / 黄昏(60~100)
- 车种：直接用 YOLO 预筛结果里的 bicycle / motorcycle 类型

输出 grid.jsonl：frame_id + 各维度值。标注完用这些格子切片看
「模型在哪类格子上错得多」，定位失效条件。

用法：
    python assign_grid.py --pool /path/pool_manifest.jsonl \
        --frames-dir /path/frames --out /path/grid.jsonl
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    frames_dir = Path(args.frames_dir)

    fout = open(args.out, "w", encoding="utf-8")
    for row in pool:
        fid = row["frame_id"]
        img_path = frames_dir / f"{fid}.jpg"
        if not img_path.exists():
            continue

        # 距离：最大人框面积占比
        areas = [(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for b in row.get("persons", [])]
        img_area = row["width"] * row["height"]
        person_ratio = max(areas) / img_area if areas else 0
        dist = "near" if person_ratio > 0.03 else ("mid" if person_ratio > 0.01 else "far")

        # 光照
        img = cv2.imread(str(img_path))
        brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) if img is not None else 128
        light = "day" if brightness > 100 else ("night" if brightness < 60 else "dusk")

        # 车种
        types = {b["type"] for b in row.get("bikes", [])}
        vtype = "ebike" if "motorcycle" in types else "bicycle"

        fout.write(json.dumps({
            "frame_id": fid,
            "dist": dist,
            "light": light,
            "vehicle": vtype,
            "person_ratio": round(person_ratio, 4),
            "brightness": round(brightness, 1),
        }, ensure_ascii=False) + "\n")

    fout.close()
    print(f"完成 -> {args.out}")


if __name__ == "__main__":
    main()
