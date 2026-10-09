#!/usr/bin/env python3
"""抽帧去重 pipeline：把视频切成有代表性的画面帧。

做什么：
1. 按时间间隔抽帧（默认每秒抽 1 帧）
2. 相邻帧去重：两帧画面的「感知哈希」太接近就扔掉，只留有变化的
3. 太模糊的帧扔掉（监控画面糊成一团没法标注也没法训练）
4. 每段视频最多留 N 帧，防止某段视频刷屏
5. 结果写两张东西：帧图片（jpg）+ 清单文件（jsonl，记录每张帧来自哪个视频第几秒）

为什么不直接用 ffmpeg 一把梭：后续要在清单里记时间戳和哈希，自己写更可控，也方便后面按格子筛选时追溯来源。

用法：
    python extract_frames.py --src /home/lawson/vlm-data/raw/TrafficQA/videos \
        --out /home/lawson/vlm-active/frames/trafficqa --max-per-video 3
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def ahash(frame_bgr) -> int:
    """平均哈希：把帧缩到 16x16 灰度，算每个像素比平均值亮还是暗，压成 256 位整数。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def blur_score(frame_bgr) -> float:
    """拉普拉斯方差，越小越糊。低于阈值的帧丢弃。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_video(video_path: Path, out_dir: Path, args, manifest):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 打不开: {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / args.fps)))  # 每隔多少帧抽一帧
    kept_hashes = []
    kept = 0
    idx = 0
    frame_no = 0

    while kept < args.max_per_video:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step != 0:
            frame_no += 1
            continue
        frame_no += 1

        if blur_score(frame) < args.min_blur:
            continue

        h = ahash(frame)
        # 和本视频已保留的帧比，太像就扔
        if any(hamming(h, k) < args.min_hamming for k in kept_hashes):
            continue

        kept_hashes.append(h)
        ts = frame_no / fps
        frame_id = f"{video_path.stem}_f{idx:03d}"
        out_path = out_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append({
            "frame_id": frame_id,
            "source_video": video_path.name,
            "t_sec": round(ts, 2),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "ahash": format(h, "064x"),
        })
        kept += 1
        idx += 1

    cap.release()


def _worker(task):
    """多进程入口：处理一段视频，返回它产出的清单行。"""
    video_path, out_dir, args_dict = task
    from argparse import Namespace

    args = Namespace(**args_dict)
    manifest = []
    process_video(Path(video_path), Path(out_dir), args, manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="视频目录")
    ap.add_argument("--out", required=True, help="帧输出目录")
    ap.add_argument("--fps", type=float, default=1.0, help="每秒抽几帧（默认 1）")
    ap.add_argument("--max-per-video", type=int, default=3, help="每段视频最多留几帧")
    ap.add_argument("--min-hamming", type=int, default=12, help="相邻帧哈希至少差多少位才保留（越小越严格）")
    ap.add_argument("--min-blur", type=float, default=30.0, help="模糊度下限，低于则丢弃")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个视频（调试用，0=全部）")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(src.glob("*.mp4"))
    if args.limit > 0:
        videos = videos[: args.limit]
    print(f"共 {len(videos)} 段视频待处理")

    # 多进程：每个进程处理一段视频，结果汇总（8 核机器用 6 核，给系统留余量）
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_worker, [(str(v), str(out_dir), vars(args)) for v in videos]))

    manifest = [row for part in results for row in part]

    mf_path = out_dir / "manifest.jsonl"
    with open(mf_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"完成：{len(manifest)} 帧 -> {out_dir}")
    print(f"清单 -> {mf_path}")


if __name__ == "__main__":
    main()
