#!/usr/bin/env python3
"""E9-v2：视频天然对比对挖掘。

思路：对每张人工标注为「推行」的 B 站帧，回到它的源视频，取更早几秒的画面
（同一个人同一辆车，通常正坐着骑过来）——同场景自然过渡的「最小视觉对比」，
零合成痕迹（比扩散 inpainting 干净得多，S-VCO 的理想数据）。

判定候选帧是「骑行」用几何启发式：人和车框重叠大、水平中心对齐、
脚底不低于车底太多（坐着踩踏板）；打分低的（像推行）弃掉。

产出：
- pairs_natural.jsonl：天然对清单
- natr_frames/：骑行候选帧图片（命名 natr_xxx）
- 清单里标 auto_label，建议抽样人工复核

用法：
    python mine_natural_pairs.py --annotations a.jsonl \
        --bili-manifest /home/lawson/vlm-active/frames/bili/manifest.jsonl \
        --video-dir /home/lawson/vlm-data/raw/bili_videos \
        --out-dir /home/lawson/vlm-active/frames/natr --out pairs_natural.jsonl
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
    res = _model.predict(img_bgr, verbose=False, device="cpu", conf=0.3,
                         classes=[0, 1, 3])[0]
    persons, bikes = [], []
    for b in res.boxes:
        cls = int(b.cls)
        box = [round(v, 1) for v in b.xyxy[0].tolist()]
        if cls == 0:
            persons.append(box)
        else:
            bikes.append(box)
    return persons, bikes


def ride_score(persons, bikes):
    """低分=像骑行，高分=像推行。返回最优人车对的分数。"""
    best = None
    for p in persons:
        for b in bikes:
            pw, ph = p[2] - p[0], p[3] - p[1]
            bw, bh = b[2] - b[0], b[3] - b[1]
            pcx, bcx = (p[0] + p[2]) / 2, (b[0] + b[2]) / 2
            dx = abs(pcx - bcx) / max(bw, 1)
            ix = max(0, min(p[2], b[2]) - max(p[0], b[0]))
            iy = max(0, min(p[3], b[3]) - max(p[1], b[1]))
            iou = ix * iy / max(pw * ph, 1)
            foot_rel = (p[3] - b[3]) / max(bh, 1)  # 推行时脚明显低于车底
            score = dx * 2 + (1 - min(iou, 1)) * 1.5 + max(foot_rel, 0) * 0.8
            if best is None or score < best:
                best = score
    return best


def grab(video_path, t_sec, out_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--bili-manifest", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--back-range", type=float, default=14.0, help="往前找几秒")
    ap.add_argument("--back-min", type=float, default=3.0)
    args = ap.parse_args()

    truth = {}
    for l in open(args.annotations, encoding="utf-8"):
        a = json.loads(l)
        if a.get("label") == "push":
            truth[a["frame_id"]] = True

    meta = {}
    for l in open(args.bili_manifest, encoding="utf-8"):
        m = json.loads(l)
        if m["frame_id"] in truth:
            meta[m["frame_id"]] = m

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = open(args.out, "w", encoding="utf-8")
    made = 0
    for fid, m in sorted(meta.items()):
        vpath = Path(args.video_dir) / m["source_video"]
        if not vpath.exists():
            continue
        t_push = m["t_sec"]
        got = 0
        for back in [args.back_min + i * 2.0 for i in range(int((args.back_range - args.back_min) / 2) + 1)]:
            if got >= 1:
                break
            t_cand = max(0.0, t_push - back)
            cand_path = out_dir / f"natr_{made:04d}.jpg"
            frame = grab(vpath, t_cand, cand_path)
            if frame is None:
                continue
            persons, bikes = det(frame)
            if not persons or not bikes:
                continue
            sc = ride_score(persons, bikes)
            if sc is None or sc > 1.8:  # 太像推行，弃
                cand_path.unlink(missing_ok=True)
                continue
            pairs.write(json.dumps({
                "pair_id": f"natr_{made:04d}",
                "ride_frame": f"natr_{made:04d}",
                "push_frame": fid,
                "source_video": m["source_video"],
                "t_ride": round(t_cand, 2),
                "t_push": round(t_push, 2),
                "ride_score": round(sc, 3),
                "auto_label": True,
            }, ensure_ascii=False) + "\n")
            pairs.flush()
            made += 1
            got += 1
        if made % 5 == 0:
            print(f"  已挖 {made} 对", flush=True)
    pairs.close()
    print(f"完成：{made} 对天然对比 -> {args.out}")


if __name__ == "__main__":
    main()
