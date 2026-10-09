#!/usr/bin/env python3
"""预筛：从抽出的帧里挑出「人 + 自行车/电瓶车同框」的画面。

为什么要这道筛子：抽帧出来的画面大多数是普通街景，根本没有人和车。
直接让人去标，等于浪费 lawson 的时间翻无关图。
YOLO 检测器（CPU 能跑）先找画面里的 人 / 自行车 / 摩托车，
再判断人和车的位置贴不贴（车框往外扩一圈还能碰到人框，才算同框），
合格的写进 pool_manifest.jsonl，标注页面只加载这份清单。

用法：
    python filter_bike_frames.py --frames-dir /mnt/d/vlm-active/frames/trafficqa \
        --weights /tmp/yolov8n.pt
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

PERSON = 0
BICYCLE = 1
MOTORCYCLE = 3
WANTED = {PERSON, BICYCLE, MOTORCYCLE}


def expand(box, ratio, w, h):
    """把框往外扩一定比例（车旁边走着的人，框是不重叠的，要放宽判定）。"""
    x1, y1, x2, y2 = box
    dw = (x2 - x1) * ratio
    dh = (y2 - y1) * ratio
    return max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)


def intersects(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--weights", default="/tmp/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--expand-ratio", type=float, default=0.6, help="车框外扩比例，判定人车是否同框")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    imgs = sorted(frames_dir.glob("*.jpg"))
    if args.limit > 0:
        imgs = imgs[: args.limit]
    print(f"共 {len(imgs)} 帧待检测")

    model = YOLO(args.weights)
    pool = []
    total = len(imgs)

    for start in range(0, total, args.batch):
        chunk = imgs[start: start + args.batch]
        results = model.predict(
            [str(p) for p in chunk], verbose=False, device="cpu",
            conf=args.conf, classes=sorted(WANTED), imgsz=640,
        )
        for img_path, res in zip(chunk, results):
            persons, bikes = [], []
            for box in res.boxes:
                cls = int(box.cls)
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                conf = round(float(box.conf), 3)
                if cls == PERSON:
                    persons.append({"box": xyxy, "conf": conf})
                else:
                    bikes.append({"box": xyxy, "conf": conf, "type": "bicycle" if cls == BICYCLE else "motorcycle"})
            if not persons or not bikes:
                continue
            h, w = res.orig_shape
            hit = False
            for b in bikes:
                eb = expand(b["box"], args.expand_ratio, w, h)
                for p in persons:
                    if intersects(eb, p["box"]):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                pool.append({
                    "frame_id": img_path.stem,
                    "width": int(w),
                    "height": int(h),
                    "persons": persons,
                    "bikes": bikes,
                })
        done = start + len(chunk)
        if done % 400 < args.batch:
            print(f"  进度 {done}/{total}，命中 {len(pool)}", flush=True)

    out = frames_dir / "pool_manifest.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"完成：{len(pool)}/{total} 帧人车同框 -> {out}")


if __name__ == "__main__":
    main()
