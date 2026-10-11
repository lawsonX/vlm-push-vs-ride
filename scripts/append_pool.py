# -*- coding: utf-8 -*-
"""把 new_pool4 的 item 追加进 label_pool（图 + manifest），不动已有标注。"""
import json, shutil
from pathlib import Path

POOL = Path("/mnt/e/vlm-data/new_pool4")
LABEL = Path("/mnt/e/vlm-data/label_pool")

existing = {json.loads(l)["frame_id"] for l in (LABEL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
n_add = 0
with open(LABEL / "manifest.jsonl", "a", encoding="utf-8") as f:
    for line in (POOL / "pool_items_ranked.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = POOL / r["file"]
        stem = img.stem
        if stem in existing:
            continue
        shutil.copy2(img, LABEL / f"{stem}.jpg")
        row = {"frame_id": stem, "pool": "new_pool4", "source": r["source"],
               "activity_qc": r.get("activity_qc"), "clarity": r.get("clarity"),
               "view": r.get("view"), "facing": r.get("facing"),
               "source_video": r["item_id"].rsplit("_f", 1)[0]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_add += 1
print(f"追加 {n_add} 张")
