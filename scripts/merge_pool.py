import json
import shutil
from pathlib import Path

frames = Path("/mnt/d/vlm-active/frames/trafficqa")
combined = []

# web 命中放最前（推行高发场景），三个来源目录
for src_dir in ["/mnt/d/vlm-active/frames/web_sogou2",
                "/mnt/d/vlm-active/frames/web_sogou",
                "/mnt/d/vlm-active/frames/web_news"]:
    d = Path(src_dir)
    mf = d / "pool_manifest.jsonl"
    if not mf.exists():
        continue
    for line in open(mf, encoding="utf-8"):
        row = json.loads(line)
        img = d / (row["frame_id"] + ".jpg")
        if img.exists():
            dst = frames / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            combined.append(row)

# 原有 trafficqa 池（保持 push_suspect 排序在后）
for line in open(frames / "pool_ranked.jsonl", encoding="utf-8"):
    combined.append(json.loads(line))

seen = set()
out = []
for r in combined:
    if r["frame_id"] not in seen:
        seen.add(r["frame_id"])
        out.append(r)

with open(frames / "pool_combined.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("combined:", len(out))
