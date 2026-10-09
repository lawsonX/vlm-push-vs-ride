#!/usr/bin/env python3
"""重建标注池：bili + web 三源 + trafficqa 排序池，去重，写 pool_dedup.jsonl。"""
import json
import shutil
from pathlib import Path

frames = Path("/mnt/d/vlm-active/frames/trafficqa")
combined = []

# 1) bili 新帧
for l in open("/mnt/d/vlm-active/frames/bili/pool_dedup.jsonl", encoding="utf-8"):
    r = json.loads(l)
    img = Path("/mnt/d/vlm-active/frames/bili") / (r["frame_id"] + ".jpg")
    if img.exists():
        dst = frames / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        combined.append(r)

# 2) web 三源
for src in ["/mnt/d/vlm-active/frames/web_sogou2", "/mnt/d/vlm-active/frames/web_sogou",
            "/mnt/d/vlm-active/frames/web_news"]:
    mf = Path(src) / "pool_manifest.jsonl"
    if not mf.exists():
        continue
    for l in open(mf, encoding="utf-8"):
        r = json.loads(l)
        img = Path(src) / (r["frame_id"] + ".jpg")
        if img.exists() and not (frames / img.name).exists():
            shutil.copy2(img, frames / img.name)
        combined.append(r)

# 3) trafficqa 排序池
for l in open(frames / "pool_ranked.jsonl", encoding="utf-8"):
    combined.append(json.loads(l))

seen = set()
out = []
for r in combined:
    if r["frame_id"] not in seen:
        seen.add(r["frame_id"])
        out.append(r)

open(frames / "pool_combined.jsonl", "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print("combined:", len(out))
