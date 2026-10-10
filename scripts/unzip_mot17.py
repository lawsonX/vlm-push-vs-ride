# -*- coding: utf-8 -*-
"""解压 MOT17.zip 到 /mnt/e/vlm-data/raw/MOT17/（WSL 无 unzip，用 zipfile）"""
import zipfile
from pathlib import Path

zip_path = "/mnt/e/vlm-data/raw/MOT17.zip"
out = Path("/mnt/e/vlm-data/raw/MOT17")
out.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    z.extractall(out)
print("done")
for p in sorted(out.rglob("img1"))[:20]:
    n = len(list(p.glob("*.jpg")))
    print(p.parent.name, n, "帧")
