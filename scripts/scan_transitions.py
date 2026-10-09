#!/usr/bin/env python3
"""E9-v2b：从「推行主题」视频里扫「骑行→推行」过渡对。

人工标注的推行帧全是静态图（sg_），没有视频帧——但推行主题的视频本身就是监督：
采样视频帧，用几何启发式给每帧打分（低=骑行像，高=推行像），
把「骑行帧→紧接着的推行帧」的过渡存成天然对比对。

用法：
    python scan_transitions.py --video-dir /path --videos BVxx BVyy \
        --out-dir /path/natr --out pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import cv2

from ultralytics import YOLO

_model = None


def det(img_bgr):
    global _model
    if _model is None:
        _model = YOLO("/tmp/yolov8n.pt")
    res = _model.predict(img_bgr, verbose=False, device="cpu", conf=0.2,
                         classes=[0, 1, 3])[0]
    persons = [[round(v, 1) for v in b.xyxy[0].tolist()] for b in res.boxes if int(b.cls) == 0]
    bikes = [[round(v, 1) for v in b.xyxy[0].tolist()] for b in res.boxes if int(b.cls) in (1, 3)]
    return persons, bikes


def ride_score(persons, bikes):
    best = None
    for p in persons:
        for b in bikes:
            pw, ph = p[2] - p[0], p[3] - p[1]
            bw, bh = b[2] - b[0], b[3] - b[1]
            dx = abs((p[0] + p[2]) / 2 - (b[0] + b[2]) / 2) / max(bw, 1)
            ix = max(0, min(p[2], b[2]) - max(p[0], b[0]))
            iy = max(0, min(p[3], b[3]) - max(p[1], b[1]))
            iou = ix * iy / max(pw * ph, 1)
            foot_rel = (p[3] - b[3]) / max(bh, 1)
            score = dx * 2 + (1 - min(iou, 1)) * 1.5 + max(foot_rel, 0) * 0.8
            if best is None or score < best:
                best = score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=1.0, help="采样间隔秒")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    made = 0

    for vid in args.videos:
        vpath = Path(args.video_dir) / vid
        if not vpath.exists():
            print(f"[跳过] 找不到 {vpath}")
            continue
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        dur = total / fps
        broken_meta = dur <= 0
        if broken_meta:
            # 有些下载的视频元数据坏了（时长0），改成顺序读帧自己计时
            fps, dur, total = 25, 10 ** 6, 0
        print(f"[扫描] {vid} 时长{'未知(顺序读)' if broken_meta else f'{dur:.0f}s'}", flush=True)

        samples = []
        t = 0.0
        step_frames = max(1, int(args.step * fps))
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % step_frames == 0:
                persons, bikes = det(frame)
                if persons and bikes:
                    sc = ride_score(persons, bikes)
                    if sc is not None:
                        samples.append((frame_no / fps, sc, frame))
            frame_no += 1
        cap.release()

        # 同视频内：任意 ride-like 样本 × 任意 push-like 样本 配成对比对
        rides = [(t, sc, fr) for t, sc, fr in samples if sc < 1.7]
        pushes = [(t, sc, fr) for t, sc, fr in samples if sc > 2.3]
        for (t0, s0, f0) in rides[:2]:
            for (t1, s1, f1) in pushes[:2]:
                rpath = out_dir / f"natr_{made:04d}_ride.jpg"
                ppath = out_dir / f"natr_{made:04d}_push.jpg"
                cv2.imwrite(str(rpath), f0, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(str(ppath), f1, [cv2.IMWRITE_JPEG_QUALITY, 90])
                fout.write(json.dumps({
                    "pair_id": f"natr_{made:04d}",
                    "video": vid,
                    "t_ride": round(t0, 2), "t_push": round(t1, 2),
                    "ride_score": round(s0, 2), "push_score": round(s1, 2),
                    "auto_label": True,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                made += 1
        print(f"  {vid}: 采样{len(samples)}帧，累计{made}对", flush=True)

    fout.close()
    print(f"完成：{made} 对 -> {args.out}")


if __name__ == "__main__":
    main()
