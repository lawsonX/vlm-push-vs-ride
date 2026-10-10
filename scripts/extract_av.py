#!/usr/bin/env python3
"""PyAV 版抽帧：支持 AV1/HEVC（opencv 自带的 ffmpeg 解不了 AV1）。

逻辑同 extract_frames.py：每秒 1 帧 → 感知哈希去重 → 模糊帧丢弃 → 每视频上限。
用法：python extract_av.py --src <视频目录> --out <帧目录> --max-per-video 5
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


def ahash(frame_bgr) -> int:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def blur_var(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-video", type=int, default=5)
    ap.add_argument("--step", type=float, default=1.0, help="抽帧间隔秒")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(Path(args.src).glob("*.mp4"))
    fout = open(out_dir / "manifest.jsonl", "a", encoding="utf-8")
    n_total = 0
    for vpath in vids:
        vid = vpath.stem
        try:
            container = av.open(str(vpath))
        except Exception as e:
            print(f"[跳过] 打不开 {vid}: {e}", flush=True)
            continue
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        step_frames = max(1, int(args.step * fps))
        kept, last_hash = 0, None
        for fi, frame in enumerate(container.decode(stream)):
            if fi % step_frames != 0:
                continue
            if kept >= args.max_per_video:
                break
            img = frame.to_ndarray(format="bgr24")
            h = ahash(img)
            if last_hash is not None and hamming(h, last_hash) < 5:
                last_hash = h
                continue
            if blur_var(img) < 60:
                last_hash = h
                continue
            fid = f"{vid}.f{fi:06d}"
            cv2.imwrite(str(out_dir / f"{fid}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            fout.write(json.dumps({"frame_id": fid, "video": vid,
                                   "t": round(fi / fps, 2)}) + "\n")
            fout.flush()
            kept += 1
            n_total += 1
            last_hash = h
        container.close()
        print(f"{vid}: {kept} 帧", flush=True)
    fout.close()
    print(f"完成：{n_total} 帧 -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
