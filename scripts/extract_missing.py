# -*- coding: utf-8 -*-
"""从残留的 SDD zip 里选择性解压缺失场景的视频，逐成员容错。"""
import zipfile
from pathlib import Path

ZIP = "/mnt/e/vlm-data/raw/sdd/OpenDataLab___Stanford_Drone/raw/stanford_campus_dataset.zip"
OUT = Path("/mnt/e/vlm-data/raw/sdd/extracted")
WANT = ("videos/deathCircle/", "videos/gates/", "videos/little/", "videos/nexus/")

z = zipfile.ZipFile(ZIP)
names = [n for n in z.namelist() if n.startswith(WANT) and not n.endswith("/")]
print(f"目标成员 {len(names)} 个")
ok, bad = 0, []
for n in names:
    # 已存在且大小一致则跳过
    dst = OUT / n
    info = z.getinfo(n)
    if dst.exists() and dst.stat().st_size == info.file_size:
        ok += 1
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with z.open(info) as src, open(dst, "wb") as f:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                f.write(chunk)
        ok += 1
    except Exception as e:
        bad.append((n, str(e)[:60]))
        dst.unlink(missing_ok=True)
        print(f"[bad] {n}: {str(e)[:60]}")
print(f"成功 {ok}，损坏 {len(bad)}")
for n, e in bad[:10]:
    print("  ", n, e)
