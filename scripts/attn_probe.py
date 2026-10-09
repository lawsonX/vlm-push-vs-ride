#!/usr/bin/env python3
"""M5：注意力热图——模型「看」推行图时，注意力落在哪里？

Qwen2.5-VL 是合并式架构（图像 token 直接进语言模型做自注意力），
所以「模型看哪」= 问题/答案 token 对图像 token 位置的注意力权重。

做法：
1. 对每张图，取问题里「推行」「骑行」所在 token 作为查询位置
2. 取最后几层注意力，平均多头，看这些查询位置对图像 token 区域的注意力分布
3. 把图像 token 的注意力按 ViT 网格还原成热力图，叠到原图上
4. 重点看：难例（模型答错的推行图）上，注意力有没有落在人脚/车的区域

用法：
    python attn_probe.py --model /path --frames-dir /path \
        --frame-ids sg_02_019 sg_05_003 b_xx --out /path/attn_out
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "图中这个人是在推行还是骑行这辆车？"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--frame-ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, -3, -6])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager",  # eager 才输出注意力权重
    ).to(device).eval()

    from PIL import Image
    for fid in args.frame_ids:
        img_path = Path(args.frames_dir) / f"{fid}.jpg"
        if not img_path.exists():
            print(f"[跳过] 找不到 {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "url": str(img_path)},
            {"type": "text", "text": QUESTION},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)

        n_img = inputs.get("image_grid_thw")
        with torch.no_grad():
            out_m = model(**inputs, output_attentions=True)
        atts = out_m.attentions  # tuple: n_layers × [1, heads, T, T]

        # 图像 token 数 = grid_t*grid_h*grid_w（Qwen2.5-VL 单图 grid_t=1）
        if n_img is not None:
            g = n_img[0].tolist()
            n_img_tokens = g[0] * g[1] * g[2]
            gh, gw = g[1], g[2]
        else:
            n_img_tokens, gh, gw = 0, 0, 0

        # 查询位置：「推行」「骑行」token（在词表里的 id）
        ids = inputs["input_ids"][0].tolist()
        # 动态找「推行」「骑行」的 token id（硬编码会错），找不到再退到最后 8 个
        tok = processor.tokenizer
        ids_push = set(tok.encode("推行", add_special_tokens=False))
        ids_ride = set(tok.encode("骑行", add_special_tokens=False))
        want = ids_push | ids_ride
        q_pos = [i for i, t in enumerate(ids) if t in want]
        if not q_pos:
            q_pos = list(range(max(0, len(ids) - 8), len(ids)))

        # 平均指定层的注意力：查询位置 → 图像 token 区域
        att_sum = None
        for li in args.layers:
            a = atts[li][0].float().mean(0)  # [T, T] 平均多头
            seg = a[q_pos, :n_img_tokens]     # [n_q, n_img_tokens]
            seg = seg.mean(0)
            att_sum = seg if att_sum is None else att_sum + seg
        att_map = (att_sum / len(args.layers)).cpu().numpy()

        # 精确映射：Qwen2.5-VL 的 ViT 做 2x2 patch 合并后进语言模型，
        # 图像内容 token = (gh//2)*(gw//2)，多余的是末尾附属 token，直接丢掉
        if gh and gw:
            nh, nw = gh // 2, gw // 2
            core = nh * nw
            if att_map.size >= core:
                heat = att_map[:core].reshape(nh, nw)
            else:
                nh = max(1, round((att_map.size * gh / gw) ** 0.5))
                nw = int(np.ceil(att_map.size / nh))
                padded = np.zeros(nh * nw, dtype=np.float32)
                padded[:att_map.size] = att_map
                heat = padded.reshape(nh, nw)
        else:
            side = int(np.ceil(att_map.size ** 0.5))
            padded = np.zeros(side * side, dtype=np.float32)
            padded[:att_map.size] = att_map
            heat = padded.reshape(side, side)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        heat_img = cv2.resize((heat * 255).astype(np.uint8), (img.width, img.height))
        heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        base = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)
        cv2.imwrite(str(out / f"{fid}_attn.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[完成] {fid}：图像token={n_img_tokens} ({gh}x{gw})，查询位置={q_pos}", flush=True)


if __name__ == "__main__":
    main()
