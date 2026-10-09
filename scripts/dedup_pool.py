#!/usr/bin/env python3
"""标注池重复图检测：感知哈希分组，量化重复率。

lawson 标注时发现 web 抓图有很多重复（同一张图被多个关键词搜到）。
这个脚本给池子里所有图算 16x16 平均哈希，哈希距离 < 10 的归为一组，
输出每组代表帧和组大小，并写一份 dedup 后的池清单（每组只留代表帧）。

用法：
    python dedup_pool.py --pool pool_combined.jsonl --frames-dir /path \
        --out pool_dedup.jsonl
"""
import argparse
import json
from pathlib import Path

import cv2


def ahash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    val = 0
    for b in (small > mean).flatten():
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    frames = Path(args.frames_dir)

    hashes = {}
    for r in rows:
        p = frames / f"{r['frame_id']}.jpg"
        if p.exists():
            h = ahash(p)
            if h is not None:
                hashes[r["frame_id"]] = h

    # 贪心分组
    clusters = []  # list of (rep_id, [member_ids])
    for fid, h in hashes.items():
        placed = False
        for rep, members in clusters:
            if hamming(h, hashes[rep]) < args.threshold:
                members.append(fid)
                placed = True
                break
        if not placed:
            clusters.append((fid, [fid]))

    dup = sum(len(m) - 1 for _, m in clusters)
    print(f"总图数(可哈希): {len(hashes)}，独立组: {len(clusters)}，重复: {dup}（{100*dup/max(len(hashes),1):.1f}%）")
    big = [m for _, m in clusters if len(m) >= 3]
    print(f"3 张以上的重复组: {len(big)} 组")
    for m in sorted(big, key=len, reverse=True)[:10]:
        print("  组大小", len(m), ":", m[:6])

    keep = {rep for rep, _ in clusters}
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            if r["frame_id"] in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"去重后池子 -> {args.out}")


if __name__ == "__main__":
    main()
