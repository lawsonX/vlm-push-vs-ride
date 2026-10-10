#!/usr/bin/env python3
"""VisDrone2019-DET → item 级抠图：用数据集自带 GT 标注做人车配对。

标注格式（每行）：x,y,w,h,score,class,truncation,occlusion
  class: 1 pedestrian / 2 people / 3 bicycle / 6 tricycle / 7 awning-tricycle / 9 motor
  score=0 表示 ignored region，跳过

配对：每个人找最近的两轮车（水平距离 < 2.5 车宽 且 垂直距离 < 0.9 人高），
一人一车出一个 item；裁剪 = 人车并集 + 30% 边距，人高 >= 80px 才要。

输出：
- <out-dir>/vd_<image_id>__i<k>.jpg
- <out>.jsonl（字段同 pair_items.py，多 source 字段）

用法：python visdrone_items.py --det-root DIR --out-dir DIR --out items.jsonl [--limit N]
"""
import argparse
import glob
import json
from pathlib import Path

import cv2

TWO_WHEEL = {3: "bicycle", 6: "tricycle", 7: "awning-tricycle", 9: "motorcycle"}
PERSON = {1, 2}


def load_ann(path):
    persons, bikes = [], []
    try:
        for line in open(path):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            x, y, w, h = map(float, p[:4])
            score, cls = int(p[4]), int(p[5])
            if score == 0 or w <= 0 or h <= 0:
                continue
            box = [x, y, x + w, y + h]
            if cls in PERSON:
                persons.append({"box": box, "area": w * h})
            elif cls in TWO_WHEEL:
                bikes.append({"box": box, "area": w * h, "cls": TWO_WHEEL[cls]})
    except FileNotFoundError:
        pass
    return persons, bikes


def pair(persons, bikes):
    cands = []
    for pi, p in enumerate(persons):
        pcx = (p["box"][0] + p["box"][2]) / 2
        pcy = (p["box"][1] + p["box"][3]) / 2
        ph = max(1.0, p["box"][3] - p["box"][1])
        for bi, b in enumerate(bikes):
            bcx = (b["box"][0] + b["box"][2]) / 2
            bcy = (b["box"][1] + b["box"][3]) / 2
            bw = max(1.0, b["box"][2] - b["box"][0])
            bh = max(1.0, b["box"][3] - b["box"][1])
            dx = abs(pcx - bcx) / bw
            dy = abs(pcy - bcy) / (ph + bh)
            ix = min(p["box"][2], b["box"][2]) - max(p["box"][0], b["box"][0])
            iy = min(p["box"][3], b["box"][3]) - max(p["box"][1], b["box"][1])
            inter = max(0.0, ix) * max(0.0, iy)
            if inter > 0.15 * min(p["area"], b["area"]):
                score = 0.1
            elif dx > 2.5 or dy > 0.9:
                continue
            else:
                score = dx + dy
            cands.append((pi, bi, score))
    cands.sort(key=lambda x: x[2])
    used_p = set()
    out = []
    for pi, bi, sc in cands:
        if pi in used_p:
            continue
        used_p.add(pi)
        out.append((pi, bi, sc))
        if len(out) >= 4:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-root", default="/mnt/e/vlm-data/raw/visdrone/VisDrone2019-DET-train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.det_root) / "images"
    ann_dir = Path(args.det_root) / "annotations"
    imgs = sorted(glob.glob(str(img_dir) + "/*.jpg"))
    if args.limit > 0:
        imgs = imgs[: args.limit]

    done = set()
    try:
        for l in open(args.out):
            done.add(json.loads(l)["item_id"])
    except FileNotFoundError:
        pass

    fout = open(args.out, "a", encoding="utf-8")
    n_item = 0
    n_pair_frame = 0
    for fi, fp in enumerate(imgs, 1):
        img_id = Path(fp).stem
        persons, bikes = load_ann(str(ann_dir / (img_id + ".txt")))
        if not bikes or not persons:
            continue
        pairs = pair(persons, bikes)
        if not pairs:
            continue
        n_pair_frame += 1
        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]
        made = 0
        for pi, bi, sc in pairs:
            item_id = f"vd_{img_id}__i{made}"
            if item_id in done:
                made += 1
                continue
            pb, bb = persons[pi]["box"], bikes[bi]["box"]
            x1 = min(pb[0], bb[0]); y1 = min(pb[1], bb[1])
            x2 = max(pb[2], bb[2]); y2 = max(pb[3], bb[3])
            w = x2 - x1; h = y2 - y1
            mx, my = w * 0.3, h * 0.3
            cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
            cx2 = min(W, int(x2 + mx)); cy2 = min(H, int(y2 + my))
            person_h = pb[3] - pb[1]
            if person_h < 80:
                made += 1
                continue
            cw, ch = cx2 - cx1, cy2 - cy1
            scale = min(1.0, 1280.0 / max(cw, ch))
            crop = img[cy1:cy2, cx1:cx2]
            if scale < 1.0:
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
            cv2.imwrite(str(out_dir / f"{item_id}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            fout.write(json.dumps({
                "item_id": item_id, "frame_id": img_id, "source": "visdrone",
                "crop_box": [cx1, cy1, cx2, cy2], "person_box": pb,
                "bike_box": bb, "bike_cls": bikes[bi]["cls"],
                "person_h": int(person_h), "pair_score": round(sc, 3),
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_item += 1
            made += 1
        if fi % 500 == 0:
            print(f"{fi}/{len(imgs)}，含人车对 {n_pair_frame} 帧，累计 {n_item} items", flush=True)

    fout.close()
    print(f"done. {n_pair_frame} 帧含人车对，共 {n_item} items -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
