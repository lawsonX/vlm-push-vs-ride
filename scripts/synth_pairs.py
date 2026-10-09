#!/usr/bin/env python3
"""E9 合成数据：copy-paste 姿势互换，生成「最小视觉对比」推/骑对。

原理（来自 S-VCO/CF-VLM 调研）：
- 从骑行图抠出「坐着的人」，贴到推行图上（替换原站立的人）→ 合成「骑行版推行场景」
- 反向同理：推行图的人贴到骑行图上 → 合成「推行版骑行场景」
- 两张图只在「人的姿态」一个语义细节上不同 = 最小视觉对比对

流程：
1. rembg 抠人（U2Net，CPU 可跑）
2. 按目标位置/尺度缩放粘贴（用 YOLO 框对齐人的宽高）
3. 原位置用 surrounding pixels 简单修补（降低 ghost 痕迹）
4. 输出成对样本 + 清单（source 标记 synth，合成图过人工复核后再进训练集）

用法：
    python synth_pairs.py --annotations a.jsonl --frames-dir /path \
        --out-dir /path/synth --out-manifest synth.jsonl --n-pairs 20
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

_SEG = None


def seg_model():
    """懒加载 YOLO 分割模型（6.5MB，CPU 可跑，自动下载）。"""
    global _SEG
    if _SEG is None:
        from ultralytics import YOLO
        _SEG = YOLO("yolov8n-seg.pt")
    return _SEG


def cutout_person(img_bgr, box=None, pad=0.1):
    """YOLO-seg 在整张图上找人并抠出最大 person 的 RGBA 图。"""
    from ultralytics.engine.results import Results
    res = seg_model().predict(img_bgr, verbose=False, device="cpu", conf=0.3, classes=[0])[0]
    best, best_area = None, 0
    if res.masks is None:
        return None
    for i, cls in enumerate(res.boxes.cls.tolist()):
        if int(cls) != 0:
            continue
        m = res.masks.data[i].cpu().numpy()
        ys, xs = np.where(m > 0.5)
        if len(xs) < 500:
            continue
        area = len(xs)
        if area > best_area:
            best_area = area
            best = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, m)
    if best is None:
        return None
    x1, y1, x2, y2, m = best
    # 质量门 1：mask 覆盖率（防「整块背景当人像」的白块事故）
    coverage = (m > 0.5).sum() / max((x2 - x1) * (y2 - y1), 1)
    if coverage < 0.25:
        return None
    # 质量门 2：裁剪区颜色方差（纯白/纯背景区域拒绝）
    crop_chk = img_bgr[y1:y2, x1:x2]
    if crop_chk.std() < 18:
        return None
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1 - int((x2 - x1) * pad)), max(0, y1 - int((y2 - y1) * pad))
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_bgr[y1:y2, x1:x2].astype(np.float32)
    mask = (m[y1:y2, x1:x2] > 0.5).astype(np.float32)
    # 简单边缘羽化
    mask = cv2.GaussianBlur(mask, (5, 5), 0)[..., None]
    rgba = np.concatenate([crop[:, :, ::-1], mask * 255], axis=2).astype(np.uint8)
    return rgba


def paste_person(canvas_bgr, person_rgba, target_box):
    """把人贴到 canvas 的目标框位置（按高度缩放），返回新图。"""
    th = int(target_box[3] - target_box[1])
    ph, pw = person_rgba.shape[:2]
    scale = th / ph
    nw, nh = max(1, int(pw * scale)), th
    person = cv2.resize(person_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    cx = int((target_box[0] + target_box[2]) / 2)
    x1 = max(0, cx - nw // 2)
    y1 = int(target_box[3]) - nh  # 脚底对齐目标框底
    H, W = canvas_bgr.shape[:2]
    if y1 < 0 or x1 + nw > W:
        return None

    roi = canvas_bgr[y1:y1 + nh, x1:x1 + nw]
    if roi.shape[:2] != person.shape[:2]:
        person = person[:roi.shape[0], :roi.shape[1]]
        nh, nw = person.shape[:2]
    alpha = (person[:, :, 3:4] / 255.0).astype(np.float32)
    rgb = person[:, :, :3][:, :, ::-1].astype(np.float32)  # RGB->BGR
    blended = (alpha * rgb + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    out = canvas_bgr.copy()
    out[y1:y1 + nh, x1:x1 + nw] = blurred = blended
    return out


def load_boxes(pool_row):
    persons = pool_row.get("persons") or []
    bikes = pool_row.get("bikes") or []
    p = max(persons, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]), default=None)
    b = max(bikes, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]), default=None)
    return (p["box"] if p else None), (b["box"] if b else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--pool", required=True, help="pool_dedup.jsonl（含 persons/bikes 框）")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = {}
    for l in open(args.annotations, encoding="utf-8"):
        a = json.loads(l)
        if a.get("label") in ("push", "ride"):
            truth[a["frame_id"]] = a["label"]

    boxes = {}
    for l in open(args.pool, encoding="utf-8"):
        r = json.loads(l)
        if r["frame_id"] in truth:
            boxes[r["frame_id"]] = r

    frames_dir = Path(args.frames_dir)
    push_ids = [f for f, t in truth.items() if t == "push" and f in boxes]
    ride_ids = [f for f, t in truth.items() if t == "ride" and f in boxes]
    random.shuffle(push_ids)
    random.shuffle(ride_ids)

    made = 0
    manifest = open(args.out_manifest, "w", encoding="utf-8")
    # 双向各做 n_pairs/2
    jobs = [("push", push_ids, ride_ids), ("ride", ride_ids, push_ids)]
    for src_label, src_ids, donor_ids in jobs:
        for i in range(min(args.n_pairs // 2, len(src_ids), len(donor_ids))):
            sid, did = src_ids[i], donor_ids[i]
            srow, drow = boxes[sid], boxes[did]
            s_img = cv2.imread(str(frames_dir / f"{sid}.jpg"))
            d_img = cv2.imread(str(frames_dir / f"{did}.jpg"))
            if s_img is None or d_img is None:
                continue
            s_person_box, s_bike_box = load_boxes(srow)
            if not all([s_person_box, s_bike_box]):
                continue
            # 质量门 3：目标人太小贴出来看不清，跳过
            if s_person_box[3] - s_person_box[1] < 100:
                continue
            person_rgba = cutout_person(d_img)
            if person_rgba is None:
                print(f"抠不到人 {did}", flush=True)
                continue
            # 目标框：保持原人的位置，但用捐赠者的姿态
            out = paste_person(s_img, person_rgba, s_person_box)
            if out is None:
                continue
            fid = f"synth_{src_label}_{made:03d}"
            cv2.imwrite(str(out_dir / f"{fid}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            # 合成图的标签 = 捐赠者的类别（人是什么姿态，就是什么类）
            new_label = truth[did]
            manifest.write(json.dumps({
                "frame_id": fid, "source": f"synth_from_{did}",
                "base_frame": sid, "donor_frame": did,
                "label_hint": new_label,  # 只是提示，最终以人工复核为准
                "question": "图中这个人是在推行还是骑行这辆车？",
                "answer": {"push": "推行", "ride": "骑行"}[new_label],
                "image": str(out_dir / f"{fid}.jpg"),
                "auto_evidence": True,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            made += 1
            if made % 5 == 0:
                print(f"  已合成 {made}", flush=True)

    manifest.close()
    print(f"完成：{made} 张合成图 -> {out_dir}")


if __name__ == "__main__":
    main()
