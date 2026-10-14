#!/usr/bin/env python3
"""「监控化」后处理：把文生图产的干净照片变成监控录像风格。

目的：文生图产出的是干净漂亮的街拍照片，和真实监控差距大（太清晰、色彩太艳、
无压缩痕迹）。真实城市治安监控的特征：分辨率不高、有噪点、色彩偏灰、
边缘模糊、有视频压缩块、四角略暗。这个脚本把这些特征加上。

用法：
    python synth_cctv_style.py --in-dir /mnt/e/vlm-data/synth/iter3 \
        --out-dir /mnt/e/vlm-data/synth/iter3_cctv

参数按文件名哈希随机微调，保证每张图的"监控感"不完全一样。
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def rng_for(name: str) -> np.random.RandomState:
    return np.random.RandomState(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))


def cctv_style(img: Image.Image, rng: np.random.RandomState) -> Image.Image:
    # 1. 缩到监控常见分辨率（1280x720），丢掉生成图的高清细节
    img = img.resize((1280, 720), Image.LANCZOS)

    # 2. 轻微模糊（镜头+压缩的共同效果）
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 0.9))))

    # 3. 降饱和度 + 轻微偏绿灰（监控白平衡特点）
    gray = img.convert("L")
    sat = float(rng.uniform(0.45, 0.65))
    img = Image.blend(Image.merge("RGB", (gray, gray, gray)), img, sat)
    arr = np.asarray(img).astype(np.float32)
    arr[..., 1] *= float(rng.uniform(0.95, 1.05))   # 绿色通道微调
    arr[..., 0] *= float(rng.uniform(0.92, 0.98))   # 红通道压一点
    arr = np.clip(arr, 0, 255)

    # 4. 加高斯噪点
    noise = rng.normal(0, float(rng.uniform(6, 12)), arr.shape[:2])[..., None]
    arr = np.clip(arr + noise, 0, 255)

    # 5. 对比度略压平（监控动态范围窄）
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.82, 0.92)))

    # 6. 四角暗角
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    vignette = np.clip(1.0 - 0.35 * (dist ** 2.2), 0.55, 1.0)
    arr = np.asarray(img).astype(np.float32) * vignette[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quality", type=int, default=70, help="JPEG 压缩质量（越低压缩块越明显）")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    for f in files:
        rng = rng_for(f.stem)
        img = Image.open(f).convert("RGB")
        styled = cctv_style(img, rng)
        # 7. JPEG 压缩出压缩痕迹
        out = out_dir / f"{f.stem}.jpg"
        styled.save(out, quality=args.quality)
        print(f"{f.name} -> {out.name}")
    print(f"共处理 {len(files)} 张")


if __name__ == "__main__":
    main()
