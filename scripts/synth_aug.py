#!/usr/bin/env python3
"""经典图像增强（非生成式）：只改光影色调，不改几何结构，标签按构造继承。

背景：生成式编辑（qwen-image-edit）试点发现两个问题——①绿网对真实人像的编辑
请求按图随机拒绝（deepfake 防护），无法自动化；②编辑可能改变人的姿势，
"标签继承"前提不成立，每张都要人工复核，失去零成本意义。

经典增强无这两个问题：只调亮度/对比度/色温/加噪，人和车的像素位置一个不动，
推行还是骑行的判据（脚、坐姿）完全保留。覆盖的场景鲁棒性目标：
夜晚/ dusk 暗光、阴天灰蒙蒙、雨天噪点+朦胧。

用法：
    python synth_aug.py --in-dir /path/frames --id-list ids.txt --out-dir /path/out
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def rng_for(name: str) -> np.random.RandomState:
    return np.random.RandomState(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))


def aug_dusk(img: Image.Image, rng) -> Image.Image:
    """夜晚/暗光：整体压暗、偏蓝、对比拉高、加噪（高ISO感）"""
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.45, 0.6)))
    img = ImageEnhance.Color(img).enhance(0.7)
    arr = np.asarray(img).astype(np.float32)
    arr[..., 2] *= 1.12  # 偏蓝
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(1.15)
    noise = rng.normal(0, 9, (img.size[1], img.size[0], 1))
    arr = np.clip(np.asarray(img).astype(np.float32) + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def aug_overcast(img: Image.Image, rng) -> Image.Image:
    """阴天：轻度压暗、去饱和、灰蒙蒙"""
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.8, 0.9)))
    img = ImageEnhance.Color(img).enhance(0.75)
    return ImageEnhance.Contrast(img).enhance(0.88)


def aug_rain(img: Image.Image, rng) -> Image.Image:
    """雨天：轻度模糊（雨幕）、降对比、加细噪点、偏灰绿"""
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img = ImageEnhance.Contrast(img).enhance(0.85)
    img = ImageEnhance.Color(img).enhance(0.7)
    arr = np.asarray(img).astype(np.float32)
    arr[..., 1] *= 1.05
    noise = rng.normal(0, 6, (img.size[1], img.size[0], 1))
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


AUGS = {"dusk": aug_dusk, "overcast": aug_overcast, "rain": aug_rain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--id-list", required=True, help="frame_id 清单，每行一个")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--augs", default="dusk,overcast,rain")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(args.id_list) if l.strip()]
    names = [n.strip() for n in args.augs.split(",")]

    n = 0
    for fid in ids:
        src = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = in_dir / f"{fid}{ext}"
            if p.exists():
                src = p
                break
        if src is None:
            continue
        img = Image.open(src).convert("RGB")
        for name in names:
            rng = rng_for(f"{fid}_{name}")
            out = AUGS[name](img, rng)
            out.save(out_dir / f"{fid}__aug_{name}.jpg", quality=90)
            n += 1
    print(f"增强完成：{len(ids)} 张图 × {len(names)} 种 = {n} 张，输出 {out_dir}")


if __name__ == "__main__":
    main()
