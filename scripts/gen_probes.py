#!/usr/bin/env python3
"""云端验证「是非探针」答案，防标签推导与图像事实矛盾。

背景（2026-10-15 实测发现）：训练集里一张骑行金标的证据写着「臀部悬空、未接触车座」
（山地车手站立俯冲是标准骑行姿势），而纯按标签推导的探针答案「有没有坐在车座上 → 有」
与图矛盾。探针答案必须逐图验证：

1. 对每张金标图，把两个探针问题直接问云端 qwen3-vl-plus
2. 云端答案与「标签推导答案」比对，一致才写入 probes.jsonl
3. 不一致或解析失败 → 该图该探针不产出样本（宁缺毋滥）

用法：
    python gen_probes.py --annotations .../annotations_evidence.jsonl --frames-dir ... \
        --out .../probes.jsonl
"""

import argparse
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Q_SEAT = "图中这个人有没有坐在车座上？只回答「有」或「没有」。"
Q_FEET = "图中这个人的双脚是在地上走路，还是踩在车的踏板上？只回答「地上」或「踏板」。"
LABEL_WORD = {"push": "推行", "ride": "骑行"}


def load_key():
    for p in [Path(__file__).parent / "api_key.txt",
              Path("/root/vlm-lab/api_testing/api_key.txt")]:
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("api_key.txt not found")


def parse_seat(text):
    if not text:
        return None
    if "没有" in text:
        return "没有"
    m = re.search(r"有", text)
    return "有" if m else None


def parse_feet(text):
    if not text:
        return None
    if "踏板" in text:
        return "踏板"
    if "地上" in text:
        return "地上"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    targets = {a["frame_id"]: a for a in anns if a.get("label") in LABEL_WORD}
    print(f"金标图: {len(targets)} 张")

    client = OpenAI(api_key=load_key(), base_url=BASE_URL, timeout=60)

    def find_img(fid):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = Path(args.frames_dir) / f"{fid}{ext}"
            if p.exists():
                return p
        return None

    def ask(img_b64, q):
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": q},
            ]}],
            max_tokens=10, temperature=0.0)
        return r.choices[0].message.content or ""

    def work(fid_a):
        fid, a = fid_a
        img = find_img(fid)
        if img is None:
            return fid, None
        b64 = base64.b64encode(img.read_bytes()).decode()
        expect_seat = "有" if a["label"] == "ride" else "没有"
        expect_feet = "踏板" if a["label"] == "ride" else "地上"
        out = {}
        for key, q, parse, expect in (("seat", Q_SEAT, parse_seat, expect_seat),
                                      ("feet", Q_FEET, parse_feet, expect_feet)):
            for attempt in range(4):
                try:
                    ans = parse(ask(b64, q))
                    if ans == expect:
                        out[key] = ans
                    break
                except Exception:
                    if attempt == 3:
                        break
                    time.sleep(2 * (attempt + 1))
        return fid, (out if out else None)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (fid, out) in enumerate(ex.map(work, targets.items())):
            if out:
                results[fid] = out
            if (i + 1) % 100 == 0:
                print(f"  进度 {i+1}/{len(targets)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for fid, out in results.items():
            f.write(json.dumps({"frame_id": fid, **out}, ensure_ascii=False) + "\n")
    n_seat = sum(1 for r in results.values() if "seat" in r)
    n_feet = sum(1 for r in results.values() if "feet" in r)
    print(f"完成：{len(results)} 张图至少一个探针通过验证（seat {n_seat}，feet {n_feet}）")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
