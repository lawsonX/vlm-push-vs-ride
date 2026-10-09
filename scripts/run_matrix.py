#!/usr/bin/env python3
"""云端模型测试矩阵：同一批图，多个模型 × 三种问法，结果落盘。

三种问法：
1. direct      直接问：推行还是骑行？
2. describe    先描述证据再给结论（单轮版引导）
3. guide       多轮引导：先问手/脚/身体位置，再问结论（复刻 lawson 的发现）

结果写 jsonl：每张图每个模型每种问法一行，含原始回答、解析出的标签、耗时、token 数。

用法：
    python run_matrix.py --pool /home/lawson/vlm-active/frames/trafficqa/pool_manifest.jsonl \
        --frames-dir /home/lawson/vlm-active/frames/trafficqa \
        --models qwen3-vl-flash,qwen3-vl-plus --styles direct,describe,guide \
        --out /home/lawson/vlm-active/results/pilot.jsonl --limit 10
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

DIRECT_Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
DESCRIBE_Q = (
    "请先描述图中这个人的手放在车的什么位置、脚在哪里、身体是坐在车上还是站在车旁，"
    "然后再回答：他是在推行还是骑行这辆车？最后一行只写「推行」或「骑行」。"
)
GUIDE_Q1 = "观察图中这个人和这辆车：他的手放在车的什么位置？脚在哪里？身体是坐在车上还是站在车旁？请详细描述。"
GUIDE_Q2 = "根据你上面的描述，这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"


def load_key():
    key_path = Path(__file__).parent / "api_key.txt"
    return key_path.read_text().strip()


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def parse_label(text: str) -> str:
    """从回答里解析标签。优先看最后一个「推行/骑行」出现的位置。"""
    text = text or ""
    last_push = text.rfind("推行")
    last_ride = text.rfind("骑行")
    if last_push == -1 and last_ride == -1:
        return "none"
    return "push" if last_push > last_ride else "ride"


def ask(client, model, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=400, temperature=0.0
            )
            return {
                "answer": resp.choices[0].message.content,
                "latency": round(time.time() - t0, 2),
                "tokens": resp.usage.total_tokens if resp.usage else None,
            }
        except Exception as e:
            err = str(e)
            wait = 2 ** attempt
            if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                return {"answer": f"ERROR: {err}", "latency": None, "tokens": None}
            time.sleep(wait)
    return {"answer": "ERROR: retries exhausted", "latency": None, "tokens": None}


def build_messages(style, img_b64):
    img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
    if style == "direct":
        return [[{"role": "user", "content": [img, {"type": "text", "text": DIRECT_Q}]}]]
    if style == "describe":
        return [[{"role": "user", "content": [img, {"type": "text", "text": DESCRIBE_Q}]}]]
    if style == "guide":
        return [
            [{"role": "user", "content": [img, {"type": "text", "text": GUIDE_Q1}]}],
            [
                {"role": "user", "content": [img, {"type": "text", "text": GUIDE_Q1}]},
                {"role": "assistant", "content": "PLACEHOLDER_OBSERVATION"},
                {"role": "user", "content": [{"type": "text", "text": GUIDE_Q2}]},
            ],
        ]
    raise ValueError(f"unknown style: {style}")


def run_guide(client, model, img_b64):
    """多轮引导：第一轮观察，把观察结果塞进历史，再问结论。"""
    r1 = ask(client, model, build_messages("guide", img_b64)[0])
    obs = r1["answer"]
    if obs.startswith("ERROR"):
        return {"answer": obs, "latency": r1["latency"], "tokens": r1["tokens"], "observation": None}
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": GUIDE_Q1},
        ]},
        {"role": "assistant", "content": obs},
        {"role": "user", "content": [{"type": "text", "text": GUIDE_Q2}]},
    ]
    r2 = ask(client, model, msgs)
    return {"answer": r2["answer"], "latency": (r1["latency"] or 0) + (r2["latency"] or 0),
            "tokens": (r1["tokens"] or 0) + (r2["tokens"] or 0), "observation": obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--models", required=True, help="逗号分隔")
    ap.add_argument("--styles", default="direct,describe,guide")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        pool = pool[: args.limit]
    models = [m.strip() for m in args.models.split(",")]
    styles = [s.strip() for s in args.styles.split(",")]

    client = OpenAI(api_key=load_key(), base_url=BASE_URL, timeout=120)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "a", encoding="utf-8")
    lock_path = out_path.with_suffix(".done")

    tasks = []
    for row in pool:
        img_path = frames_dir / f"{row['frame_id']}.jpg"
        if not img_path.exists():
            continue
        tasks.append((row, b64_image(img_path)))

    total = len(tasks) * len(models) * len(styles)
    print(f"任务数：{len(tasks)} 张图 × {len(models)} 个模型 × {len(styles)} 种问法 = {total} 次调用")

    def work(task):
        row, img_b64 = task
        for model in models:
            for style in styles:
                if style == "guide":
                    r = run_guide(client, model, img_b64)
                else:
                    r = ask(client, model, build_messages(style, img_b64)[0])
                rec = {
                    "frame_id": row["frame_id"],
                    "model": model,
                    "style": style,
                    "label_pred": parse_label(r["answer"]),
                    "answer": r["answer"][:500],
                    "observation": (r.get("observation") or "")[:500],
                    "latency": r["latency"],
                    "tokens": r["tokens"],
                    "ts": time.strftime("%H:%M:%S"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for _ in ex.map(work, tasks):
            done += 1
            if done % 10 == 0:
                print(f"  图片进度 {done}/{len(tasks)}")

    fout.close()
    lock_path.touch()
    print(f"完成 -> {out_path}")


if __name__ == "__main__":
    main()
