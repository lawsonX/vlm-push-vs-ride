#!/usr/bin/env python3
"""标注池感知去重：dHash 64 位，汉明距离 ≤ 阈值判为重复。

用途：
1. build：给 label_pool 全部图建哈希缓存（增量，可复用）
2. clean：检查指定 pool 批次（如 mine1）与全池已标图的重复，把重复的从 manifest 移除
   （不删文件，可回溯；被移除的图标注价值低，因为同图已有标签）

用法：
    python dedup_pool.py build
    python dedup_pool.py clean --pool mine1 --max-dist 6
"""

import argparse
import json
from pathlib import Path

from PIL import Image

LABEL_POOL = Path("/mnt/e/vlm-data/label_pool")
CACHE = LABEL_POOL / "phash_cache.json"


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            bits = (bits << 1) | (1 if px[row * (hash_size + 1) + col] > px[row * (hash_size + 1) + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_cache():
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    man = [json.loads(l) for l in open(LABEL_POOL / "manifest.jsonl", encoding="utf-8") if l.strip()]
    n_new = 0
    for m in man:
        fid = m["frame_id"]
        if fid in cache:
            continue
        p = LABEL_POOL / f"{fid}.jpg"
        if not p.exists():
            p = LABEL_POOL / f"{fid}.png"
        if not p.exists():
            continue
        try:
            with Image.open(p) as im:
                cache[fid] = dhash(im)
                n_new += 1
        except Exception:
            continue
    CACHE.write_text(json.dumps(cache))
    print(f"缓存 {len(cache)} 条（新增 {n_new}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "clean"])
    ap.add_argument("--pool", default="mine1")
    ap.add_argument("--max-dist", type=int, default=6)
    args = ap.parse_args()

    if args.cmd == "build":
        build_cache()
        return

    cache = json.loads(CACHE.read_text())
    man = [json.loads(l) for l in open(LABEL_POOL / "manifest.jsonl", encoding="utf-8") if l.strip()]
    ann = {json.loads(l)["frame_id"] for l in open(LABEL_POOL / "annotations.jsonl", encoding="utf-8") if l.strip()}

    batch = [m for m in man if m["pool"] == args.pool]
    labeled = {fid: h for fid, h in cache.items() if fid in ann}
    # 批内也要互查
    batch_hashes = {}
    for m in batch:
        if m["frame_id"] in cache:
            batch_hashes[m["frame_id"]] = cache[m["frame_id"]]

    dup = set()
    for fid, h in batch_hashes.items():
        # 与已标图比
        for fid2, h2 in labeled.items():
            if fid == fid2:
                continue
            if hamming(h, h2) <= args.max_dist:
                dup.add((fid, fid2))
                break
        else:
            # 批内互比
            for fid2, h2 in batch_hashes.items():
                if fid2 <= fid:
                    continue
                if hamming(h, h2) <= args.max_dist:
                    dup.add((fid2, fid))

    dup_new = {d[0] for d in dup}
    if not dup_new:
        print("未发现重复")
        return
    keep = [m for m in man if m["frame_id"] not in dup_new]
    with open(LABEL_POOL / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in keep:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    # 缓存也删掉这些，防止后续轮次拿它们当"已存在"参照出错（其实保留也行，但图已不在池里）
    for fid in dup_new:
        cache.pop(fid, None)
    CACHE.write_text(json.dumps(cache))
    print(f"移除重复 {len(dup_new)} 张（与已标图或批内互重），剩 {len(keep)} 张在池")
    for d in list(dup)[:10]:
        print("  dup:", d[0], "<->", d[1])


if __name__ == "__main__":
    main()
