#!/usr/bin/env python3
"""item 级抠图：YOLO 人车配对 → 1人1车裁剪 → item 图 + 清单。

解决图片级标注的致命缺陷：一帧多对人车时，「这张图标推行」说不清是谁在推行。
item = 恰好一对 (人, 两轮车) 的裁剪区域，作为标注/训练的最小单元。

配对规则（保守优先，宁缺毋滥）：
- 每辆车找「最近的人」：人框中心与车框中心的归一化距离 < 阈值，或人框与车框有交叠
- 一个人只配一辆车（最近的）；一辆车可出多个 item（多人推车场景）
- 每帧最多出 4 个 item，防密集场景爆炸

裁剪规则：
- 人车并集 + 30% 边距；人物在裁剪中高度 >= 80px（太小判不了骑/推）
- 单边最大 1280px（省显存）

输出：
- <out-dir>/<frame_id>__i<k>.jpg
- <out>/items.jsonl：{item_id, frame_id, crop_box, person_box, bike_box, bike_cls, person_h}

用法：
  python pair_items.py --frames-dir DIR --out-dir DIR2 --out items.jsonl [--only f1,f2] [--limit N]
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

_model = None


def det(img_bgr):
    global _model
    if _model is None:
        _model = YOLO("/tmp/yolov8n.pt")
    res = _model.predict(img_bgr, verbose=False, device="cpu", conf=0.3,
                         classes=[0, 1, 3])[0]
    persons, bikes = [], []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        cls = int(b.cls[0])
        area = (x2 - x1) * (y2 - y1)
        if cls == 0:
            persons.append({"box": [x1, y1, x2, y2], "area": area})
        else:
            bikes.append({"box": [x1, y1, x2, y2], "area": area,
                          "cls": "bicycle" if cls == 1 else "motorcycle"})
    return persons, bikes


def pair(persons, bikes):
    """返回 [(person_idx, bike_idx, score)]，score 越小越相关。"""
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
            dx = abs(pcx - bcx) / bw          # 水平偏离（几倍车宽）
            dy = abs(pcy - bcy) / (ph + bh)   # 垂直偏离
            # 交叠直接算强相关
            ix = min(p["box"][2], b["box"][2]) - max(p["box"][0], b["box"][0])
            iy = min(p["box"][3], b["box"][3]) - max(p["box"][1], b["box"][1])
            inter = max(0.0, ix) * max(0.0, iy)
            if inter > 0.15 * min(p["area"], b["area"]):
                score = 0.1
            elif dx > 2.5 or dy > 0.9:
                continue  # 明显无关
            else:
                score = dx + dy
            cands.append((pi, bi, score))
    cands.sort(key=lambda x: x[2])
    used_p, used_pair = set(), set()
    out = []
    for pi, bi, sc in cands:
        if pi in used_p:
            continue
        if (pi, bi) in used_pair:
            continue
        used_p.add(pi)
        used_pair.add((pi, bi))
        out.append((pi, bi, sc))
        if len(out) >= 4:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="逗号分隔的 frame_id 白名单")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.frames_dir + "/*.jpg"))
    if args.only:
        allow = set(args.only.split(","))
        files = [f for f in files if Path(f).stem in allow]
    if args.limit > 0:
        files = files[: args.limit]

    done = set()
    try:
        for l in open(args.out):
            done.add(json.loads(l)["item_id"])
    except FileNotFoundError:
        pass

    fout = open(args.out, "a", encoding="utf-8")
    n_item = 0
    n_frame_with = 0
    for fi, fp in enumerate(files, 1):
        fid = Path(fp).stem
        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]
        persons, bikes = det(img)
        if not bikes:
            continue
        pairs = pair(persons, bikes)
        if not pairs:
            continue
        n_frame_with += 1
        made_here = 0
        for pi, bi, sc in pairs:
            pb, bb = persons[pi]["box"], bikes[bi]["box"]
            item_id = f"{fid}__i{made_here}"
            if item_id in done:
                made_here += 1
                continue
            # 裁剪框：并集 + 30% 边距
            x1 = min(pb[0], bb[0]); y1 = min(pb[1], bb[1])
            x2 = max(pb[2], bb[2]); y2 = max(pb[3], bb[3])
            w = x2 - x1; h = y2 - y1
            mx, my = w * 0.3, h * 0.3
            cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
            cx2 = min(W, int(x2 + mx)); cy2 = min(H, int(y2 + my))
            person_h = pb[3] - pb[1]
            if person_h < 80:
                made_here += 1
                continue  # 人太小，判不了骑/推
            # 裁剪图单边上限 1280
            cw, ch = cx2 - cx1, cy2 - cy1
            scale = min(1.0, 1280.0 / max(cw, ch))
            crop = img[cy1:cy2, cx1:cx2]
            if scale < 1.0:
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
            cv2.imwrite(str(out_dir / f"{item_id}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            fout.write(json.dumps({
                "item_id": item_id, "frame_id": fid,
                "crop_box": [cx1, cy1, cx2, cy2], "person_box": pb,
                "bike_box": bb, "bike_cls": bikes[bi]["cls"],
                "person_h": int(person_h), "pair_score": round(sc, 3),
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_item += 1
            made_here += 1
        if fi % 200 == 0:
            print(f"{fi}/{len(files)} 帧，累计 {n_item} items", flush=True)

    fout.close()
    print(f"done. {n_frame_with} 帧含人车对，共 {n_item} items -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
