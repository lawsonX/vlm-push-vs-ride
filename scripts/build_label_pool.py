# -*- coding: utf-8 -*-
"""合并 new_pool + new_pool3 为统一标注池 label_pool（平铺图片 + manifest）。

标注 UI 用法：
  python annotate_app.py --frames-dir /mnt/e/vlm-data/label_pool \
      --manifest /mnt/e/vlm-data/label_pool/manifest.jsonl --port 8321
标注结果存在 /mnt/e/vlm-data/label_pool/annotations.jsonl
"""
import json, shutil
from pathlib import Path

OUT = Path("/mnt/e/vlm-data/label_pool")
OUT.mkdir(exist_ok=True)

rows = []
for pool in ["new_pool", "new_pool3"]:
    src = Path(f"/mnt/e/vlm-data/{pool}")
    for line in (src / "pool_items_ranked.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = src / r["file"]
        if not img.exists():
            continue
        stem = img.stem
        dst = OUT / f"{stem}.jpg"
        if not dst.exists():
            shutil.copy2(img, dst)
        rows.append({
            "frame_id": stem,
            "pool": pool,
            "source": r["source"],
            "activity_qc": r.get("activity_qc"),
            "clarity": r.get("clarity"),
            "view": r.get("view"),
            "facing": r.get("facing"),
            "source_video": r["item_id"].split("__i")[0],
        })

# 排序：push 优先，再 ride，最后 stand；同级保持原 rank 顺序（清晰度已排过）
rank = {"push": 0, "ride": 1, "stand": 2}
rows.sort(key=lambda r: rank.get(r["activity_qc"], 9))

with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("总数", len(rows), Counter(r["activity_qc"] for r in rows))
print("池子构成", Counter(r["pool"] for r in rows))
