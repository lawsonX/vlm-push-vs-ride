#!/usr/bin/env python3
"""全池 VLM 质检：对每一帧回答四个问题，筛掉广告/宣传/教学/低质图。

模型策略：3B fp16 主力（快）+ fp32 副本兜底 NaN 帧（18.6GB 显存内双模型常驻）。
四问（单轮一次生成）：
1. 画面类型：真实监控/街面实拍 vs 广告/宣传/教学/摆拍/动漫
2. 有无自行车/电动车/摩托车
3. 人+两轮车关系：骑/推/站/无
4. 清晰度 + 人物占画面高度百分比

输出：qc_all.jsonl（可断点续跑）；字段 frame/type/twowheel/activity/clarity/person_pct/raw
用法：python vlm_qc.py [--frames-dir DIR] [--out FILE] [--limit N]
"""
import argparse
import glob
import json
import queue
import re
import threading

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "/mnt/e/vlm-data/models/Qwen2.5-VL-3B-Instruct"
FRAMES = "/mnt/d/vlm-active/frames/trafficqa"
OUT = "/mnt/d/vlm-active/results/qc_all.jsonl"

QC_PROMPT = (
    "请观察这张图片并依次回答四个问题，每行一个答案，格式「编号.结论」：\n"
    "1. 画面类型：是真实监控录像或街面实拍，还是广告、宣传片、教学视频、摆拍写真或动漫？\n"
    "2. 画面中是否有自行车、电动车或摩托车？\n"
    "3. 是否有人与两轮车同框？如果有，人是骑在车上、在车旁推行、还是站在车旁没有操控？\n"
    "4. 画面清晰度和人物大小：清晰度分高/中/低；画面中主要人物大约占画面高度的百分之几？"
)


def parse_qc(text):
    """从四问回答里解析结构化字段。"""
    t = text or ""

    def grab(n):
        m = re.search(rf"{n}\s*[.、)]\s*(.+)", t)
        return m.group(1).strip() if m else ""

    q1 = grab(1)
    bad_kw = ("广告", "宣传", "教学", "摆拍", "写真", "动漫", "海报", "漫画", "艺术照")
    good_kw = ("监控", "实拍", "抓拍", "街景", "录像", "现场", "路口", "街道", "道路")
    if any(k in q1 for k in bad_kw) and not any(k in q1 for k in good_kw):
        ftype = "bad"
    elif any(k in q1 for k in good_kw):
        ftype = "good"
    else:
        ftype = "unknown"

    q2 = grab(2)
    if "没有" in q2 or "无" in q2[:6]:
        twowheel = "no"
    elif any(k in q2 for k in ("自行车", "电动车", "摩托车", "单车", "电瓶车", "有")):
        twowheel = "yes"
    else:
        twowheel = "unknown"

    q3 = grab(3)
    if "推" in q3:
        activity = "push"
    elif "骑" in q3:
        activity = "ride"
    elif "站" in q3 or "没有操控" in q3:
        activity = "stand"
    elif "没有" in q3 or "无人" in q3 or "无" in q3[:6]:
        activity = "none"
    else:
        activity = "unknown"

    q4 = grab(4)
    if "高" in q4[:4]:
        clarity = "high"
    elif "中" in q4[:4]:
        clarity = "mid"
    elif "低" in q4[:4]:
        clarity = "low"
    else:
        clarity = "unknown"
    pct = None
    m = re.search(r"(\d+)\s*[%％]", q4)
    if m:
        pct = int(m.group(1))

    return {"type": ftype, "twowheel": twowheel, "activity": activity,
            "clarity": clarity, "person_pct": pct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=FRAMES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL)
    model16 = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    model32 = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    print("双精度模型就绪（fp16 主力 + fp32 兜底）", flush=True)

    files = sorted(glob.glob(args.frames_dir + "/*.jpg"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"总帧数: {len(files)}", flush=True)

    done = set()
    try:
        for l in open(args.out):
            done.add(json.loads(l)["frame"])
    except FileNotFoundError:
        pass
    files = [f for f in files if f.split("/")[-1] not in done]
    print(f"已完成 {len(done)}，本轮 {len(files)}", flush=True)

    q = queue.Queue(maxsize=8)
    it_lock = threading.Lock()
    it = iter(files)

    def prep(p):
        img = Image.open(p).convert("RGB")
        if img.width * img.height > 1_200_000:
            r = (1_200_000 / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * r), int(img.height * r)))
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "url": p},
                {"type": "text", "text": QC_PROMPT},
            ]}], tokenize=False, add_generation_prompt=True)
        return p, processor(text=[prompt], images=[img], return_tensors="pt")

    def producer():
        while True:
            with it_lock:
                p = next(it, None)
            if p is None:
                break
            try:
                q.put(prep(p))
            except Exception as e:
                print(f"[READ-FAIL] {p}: {e}", flush=True)
        for _ in range(2):
            q.put(None)

    for _ in range(2):
        threading.Thread(target=producer, daemon=True).start()

    fout = open(args.out, "a", encoding="utf-8")
    n_nan = 0
    i = 0
    while True:
        item = q.get()
        if item is None:
            i_none = getattr(main, "_none_seen", 0) + 1
            main._none_seen = i_none
            if i_none >= 2:
                break
            continue
        p, inputs = item
        batch = {k: v.to(device) for k, v in inputs.items()}
        use32 = False
        with torch.no_grad():
            out = model16.generate(**batch, max_new_tokens=150, do_sample=False)
        text = processor.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        rec = parse_qc(text)
        # 三路全 unknown = 大概率 fp16 NaN 乱码，fp32 重跑
        if rec["type"] == rec["twowheel"] == rec["activity"] == "unknown":
            use32 = True
            n_nan += 1
            with torch.no_grad():
                out = model32.generate(**batch, max_new_tokens=150, do_sample=False)
            text = processor.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
            rec = parse_qc(text)
        rec.update({"frame": p.split("/")[-1], "raw": text[:500], "used_fp32": use32})
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        i += 1
        if i % 200 == 0:
            print(f"{i}/{len(files)} (fp32兜底 {n_nan} 次)", flush=True)

    fout.close()
    print(f"done. 本轮 {i} 帧，NaN 兜底 {n_nan} 次 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
