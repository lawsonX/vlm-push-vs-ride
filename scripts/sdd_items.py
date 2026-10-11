# -*- coding: utf-8 -*-
"""SDD（Stanford Drone Dataset）item 裁剪：标注驱动，不用 YOLO。

SDD 标注格式（annotations/<scene>/video<N>/annotations.txt）：
  track_id xmin ymin xmax ymax frame lost occluded generated "Label"
Label=Biker 的框就是「人+自行车」整体（俯视）。

做法：每个视频按 1fps 取帧号，解码视频到这些帧，
裁出框（外扩 25%），框高 >= 60px 才要，每个视频最多 max_per_video 个。

用法：
  python sdd_items.py --sdd /mnt/e/vlm-data/raw/sdd/extracted \
      --out /mnt/e/vlm-data/items_sdd [--max-per-video 40] [--stride 30]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdd", default="/mnt/e/vlm-data/raw/sdd/sdd_root")
    ap.add_argument("--out", default="/mnt/e/vlm-data/items_sdd")
    ap.add_argument("--max-per-video", type=int, default=40)
    ap.add_argument("--stride", type=int, default=30, help="帧间隔（30=1fps）")
    ap.add_argument("--min-h", type=int, default=60, help="框最小高度像素")
    args = ap.parse_args()

    sdd = Path(args.sdd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 收集标注：video_key -> {frame -> [boxes]}
    ann_files = sorted(sdd.glob("annotations/*/video*/annotations.txt"))
    print(f"标注文件 {len(ann_files)} 个")
    videos = {}
    for af in ann_files:
        parts = af.parts
        scene = parts[-3]
        vdir = parts[-2]
        key = f"{scene}_{vdir}"
        frames = defaultdict(list)
        for line in af.read_text().splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            try:
                tid, x1, y1, x2, y2, fr, lost, occ, gen = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6]), int(p[7]), int(p[8])
            except ValueError:
                continue
            label = p[9].strip('"')
            if label != "Biker" or lost == 1 or gen == 1:
                continue
            if fr % args.stride != 0:
                continue
            h = y2 - y1
            if h < args.min_h:
                continue
            frames[fr].append((tid, x1, y1, x2, y2))
        if frames:
            videos[key] = (scene, vdir, frames)

    # 找视频文件
    items = []
    n_videos_found = 0
    for key, (scene, vdir, frames) in sorted(videos.items()):
        vid_dir = sdd / "videos" / scene / vdir
        if not vid_dir.exists():
            vid_dir = sdd / scene / "videos" / vdir
        vfiles = list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*.mov")) + list(vid_dir.glob("*.avi")) if vid_dir.exists() else []
        if not vfiles:
            continue
        n_videos_found += 1
        target_frames = sorted(frames.keys())
        got = 0
        try:
            cap = av.open(str(vfiles[0]))
        except Exception as e:
            print(f"[skip] {key} 打不开: {e}", flush=True)
            continue
        stream = cap.streams.video[0]
        wanted = set(target_frames)
        saved_this_video = 0
        try:
            fn = -1
            for frame in cap.decode(stream):
                fn += 1
                if fn in wanted:
                    img = frame.to_image()
                    arr = np.asarray(img)
                    H, W = arr.shape[:2]
                    for k, (tid, x1, y1, x2, y2) in enumerate(frames[fn]):
                        bw, bh = x2 - x1, y2 - y1
                        mx, my = bw * 0.25, bh * 0.25
                        cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
                        cx2 = min(W, int(x2 + mx)); cy2 = min(H, int(y2 + my))
                        if cx2 - cx1 < 40 or cy2 - cy1 < 40:
                            continue
                        item_id = f"sdd_{key}_f{fn:06d}__t{tid}"
                        fp = out / f"{item_id}.jpg"
                        Image.fromarray(arr[cy1:cy2, cx1:cx2]).save(fp, quality=90)
                        items.append({"item_id": item_id, "frame_id": f"sdd_{key}_f{fn:06d}",
                                      "crop_box": [cx1, cy1, cx2, cy2],
                                      "person_box": [x1, y1, x2, y2], "bike_cls": "biker(sdd)",
                                      "person_h": int(bh), "scene": scene, "video": vdir, "t_sec": round(fn / 30, 1)})
                        saved_this_video += 1
                        if saved_this_video >= args.max_per_video:
                            break
                    wanted.discard(fn)
                    if saved_this_video >= args.max_per_video or not wanted:
                        break
        except Exception as e:
            print(f"[warn] {key} 解码中断(可能截断): {e}; 已存 {saved_this_video}", flush=True)
        cap.close()
        if n_videos_found % 10 == 0:
            print(f"{n_videos_found} 视频处理中，累计 {len(items)} items", flush=True)

    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"done. {n_videos_found} 视频，共 {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
