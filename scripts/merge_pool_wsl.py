# -*- coding: utf-8 -*-
"""合并多个 item 池，过滤、去重、排序，输出标注用新池。

用法:
  python merge_pool.py --out /mnt/e/vlm-data/new_pool

输入池（硬编码，路径不存在则跳过）:
  /mnt/e/vlm-data/items_bili   (老 bili 批次, 103)
  /mnt/e/vlm-data/items_bili3  (新 bili 批次, ~1032)
  /mnt/e/vlm-data/items_visdrone (341)

过滤规则: type=good, twowheel=yes, clarity != low, activity in {push,ride,stand}
去重: 对 crop 图算 8x8 ahash，贪心 hamming<=6 视为重复丢弃
排序: push > ride > stand；同级 clarity high 优先，再按人物占比降序
输出: <out>/items/<源>__<item_id>.jpg + pool_items_ranked.jsonl + README.md
"""
import argparse, json, shutil, sys
from pathlib import Path

POOLS = [
    ("bili", "/mnt/e/vlm-data/items_bili"),
    ("bili3", "/mnt/e/vlm-data/items_bili3"),
    ("bili4", "/mnt/e/vlm-data/items_bili4"),
    ("visdrone", "/mnt/e/vlm-data/items_visdrone"),
]

import numpy as np
from PIL import Image


def ahash(img_path, size=8):
    img = Image.open(img_path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    return (a > a.mean()).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dup", type=int, default=6, help="ahash 去重阈值")
    ap.add_argument("--sources", default=None, help="只合并这些来源，逗号分隔，如 bili4 或 bili,bili4")
    args = ap.parse_args()

    pools = POOLS
    if args.sources:
        wanted = set(args.sources.split(","))
        pools = [p for p in POOLS if p[0] in wanted]
        print(f"[filter] 只合并来源: {args.sources}")

    out_dir = Path(args.out)
    img_out = out_dir / "items"
    img_out.mkdir(parents=True, exist_ok=True)

    ACT_OK = {"push", "ride", "stand"}
    records = []  # (source, qc_dict, item_dict)
    for source, pool in pools:
        pool = Path(pool)
        qc_f = pool / "qc.jsonl"
        items_f = pool / "items.jsonl"
        if not qc_f.exists():
            print(f"[skip] {pool} 缺 qc.jsonl")
            continue
        items = {}
        if items_f.exists():
            for line in items_f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    items[d["item_id"]] = d
        n_ok = 0
        for line in qc_f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qc = json.loads(line)
            fname = qc.get("frame") or (qc.get("item_id") + ".jpg")
            item_id = fname[:-4] if fname.endswith(".jpg") else qc.get("item_id")
            if qc.get("type") != "good":
                continue
            if str(qc.get("twowheel", "")).lower() != "yes":
                continue
            if qc.get("activity") not in ACT_OK:
                continue
            if qc.get("clarity") == "low":
                continue
            img = pool / fname
            if not img.exists():
                continue
            records.append({"source": source, "item_id": item_id, "qc": qc,
                            "item": items.get(item_id, {}), "img": str(img)})
            n_ok += 1
        print(f"[pool] {source}: 合格 {n_ok} / 质检 {sum(1 for _ in open(qc_f, encoding='utf-8'))}")

    # 排序: push > ride > stand, clarity high 优先, person_pct 降序
    act_rank = {"push": 0, "ride": 1, "stand": 2}
    clar_rank = {"high": 0, "mid": 1}
    def sort_key(r):
        q = r["qc"]
        pct = q.get("person_pct") or 0
        return (act_rank.get(q["activity"], 9), clar_rank.get(q.get("clarity"), 9), -pct)
    records.sort(key=sort_key)

    # 贪心去重
    kept, hashes = [], []
    dup = 0
    for r in records:
        try:
            h = ahash(r["img"])
        except Exception as e:
            print(f"[warn] 读图失败 {r['img']}: {e}")
            continue
        if any(hamming(h, k) <= args.max_dup for k in hashes):
            dup += 1
            continue
        hashes.append(h)
        kept.append(r)
    print(f"[dedup] 去重 {dup} 张，保留 {len(kept)} 张")

    # 写出
    manifest = []
    for rank, r in enumerate(kept, 1):
        new_name = f"{r['source']}__{r['item_id']}.jpg"
        shutil.copy2(r["img"], img_out / new_name)
        q = r["qc"]
        manifest.append({
            "rank": rank,
            "file": f"items/{new_name}",
            "source": r["source"],
            "item_id": r["item_id"],
            "activity_qc": q["activity"],
            "clarity": q.get("clarity"),
            "person_pct": q.get("person_pct"),
            "crop_box": r["item"].get("crop_box"),
            "person_box": r["item"].get("person_box"),
            "bike_box": r["item"].get("bike_box"),
            "bike_cls": r["item"].get("bike_cls"),
            "qc_raw": q.get("raw"),
        })
    with open(out_dir / "pool_items_ranked.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    cnt = Counter((m["source"], m["activity_qc"]) for m in manifest)
    print("\n[最终分布] 源 x 活动:")
    for k in sorted(cnt):
        print(f"  {k[0]:10s} {k[1]:6s} {cnt[k]}")
    print(f"\n共 {len(manifest)} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
