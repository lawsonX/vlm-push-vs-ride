#!/usr/bin/env python3
"""给金标训练图自动生成「判断依据」（evidence），供 chain/guide 训练格式用。

思路（重要，防脏数据）：
1. 用云端 qwen3-vl-plus 按「先描述手/脚/身体位置，再给结论」的格式描述每张图
2. 解析它给的最终结论，和人工金标比对——一致的才保留，不一致的丢弃
   （云端模型在这任务上本身有错，人工标签才是裁判）
3. 只处理训练集图片，测试集不动；只处理 push/ride 金标，invalid/unsure 不碰
4. 产出 annotations_evidence.jsonl = 原标注文件的副本 + 填好 evidence 字段，
   直接作为 make_sft_data_v2.py 的输入

用法：
    python gen_evidence.py --annotations /mnt/e/vlm-data/label_pool/annotations.jsonl \
        --frames-dir /mnt/e/vlm-data/label_pool \
        --train-jsonl /mnt/e/vlm-data/sft_v2/train.jsonl \
        --out /mnt/e/vlm-data/label_pool/annotations_evidence.jsonl
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

# 和 run_matrix.py 保持一致的提问方式
DESCRIBE_Q = (
    "请先描述图中这个人的手放在车的什么位置、脚在哪里、身体是坐在车上还是站在车旁，"
    "然后再回答：他是在推行还是骑行这辆车？最后一行只写「推行」或「骑行」。"
)


def load_key():
    key_path = Path(__file__).parent / "api_key.txt"
    if not key_path.exists():
        key_path = Path("/root/vlm-lab/api_testing/api_key.txt")
    return key_path.read_text().strip()


def parse_label(text):
    text = text or ""
    last_push = text.rfind("推行")
    last_ride = text.rfind("骑行")
    if last_push == -1 and last_ride == -1:
        return "none"
    return "push" if last_push > last_ride else "ride"


def strip_conclusion(text):
    """把回答末尾的结论行去掉，只留证据描述部分。"""
    lines = [l for l in text.strip().splitlines()
             if not re.match(r"^\s*(结论|答案|所以)?\s*[:：]?\s*(推行|骑行)", l)]
    ev = "\n".join(lines).strip()
    # 再去掉最后一行孤零零的「推行/骑行」
    if ev.splitlines() and ev.splitlines()[-1].strip() in ("推行", "骑行"):
        ev = "\n".join(ev.splitlines()[:-1]).strip()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--train-jsonl", required=True, help="make_sft_data_v2 产出的 train.jsonl，用来限定处理范围")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    train_ids = set()
    for l in open(args.train_jsonl, encoding="utf-8"):
        train_ids.add(json.loads(l)["frame_id"])
    targets = [a for a in anns
               if a["frame_id"] in train_ids and a.get("label") in ("push", "ride")]
    # 每张图只处理一次（train.jsonl 里同一张图可能有多行）
    # 已有证据的跳过（增量：标注人写的或前一轮生成且通过校验的，不重复花钱）
    seen = set()
    uniq = []
    n_skip = 0
    for a in targets:
        if a["frame_id"] in seen:
            continue
        seen.add(a["frame_id"])
        if (a.get("evidence") or "").strip():
            n_skip += 1
            continue
        uniq.append(a)
    print(f"需要生成证据的训练图：{len(uniq)} 张（跳过已有证据 {n_skip} 张，模型 {args.model}）")

    client = OpenAI(api_key=load_key(), base_url=BASE_URL)
    results = {}

    def work(a):
        fid = a["frame_id"]
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = Path(args.frames_dir) / f"{fid}{ext}"
            if p.exists():
                img = p
                break
        if img is None:
            return fid, None, "no_image"
        b64 = base64.b64encode(img.read_bytes()).decode()
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": DESCRIBE_Q},
                    ]}],
                    max_tokens=400, temperature=0.0)
                text = r.choices[0].message.content or ""
                pred = parse_label(text)
                if pred != a["label"]:
                    return fid, None, f"mismatch(pred={pred})"
                ev = strip_conclusion(text)
                if not ev:
                    return fid, None, "empty_evidence"
                return fid, ev, "ok"
            except Exception as e:
                if attempt == 4:
                    return fid, None, f"error:{e}"
                time.sleep(2 * (attempt + 1))
        return fid, None, "unreachable"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (fid, ev, status) in enumerate(ex.map(work, uniq)):
            if ev:
                results[fid] = ev
            if (i + 1) % 25 == 0 or i + 1 == len(uniq):
                print(f"  进度 {i+1}/{len(uniq)}，已确认 {len(results)}")

    n_ok = len(results)
    n_mismatch = sum(1 for f in seen if f not in results)  # 粗略，含报错
    # 写输出：原标注副本 + evidence 字段
    with open(args.out, "w", encoding="utf-8") as f:
        for a in anns:
            b = dict(a)
            if a["frame_id"] in results:
                b["evidence"] = results[a["frame_id"]]
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"完成：{n_ok}/{len(uniq)} 张通过人工标签校验（其余被丢弃）")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
