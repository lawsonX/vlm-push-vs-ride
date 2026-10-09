#!/usr/bin/env python3
"""推行帧预筛：云端 qwen3-vl-flash 给未标注帧打「推行疑似」标签。

目的：lawson 标注时间最宝贵，而推行帧在池里很稀。让云端模型先粗筛一遍，
把「疑似推行」的帧排到标注 UI 最前面，lawson 的标注产出率能翻几倍。

输出：
- /mnt/d/vlm-active/results/prescreen.jsonl   每帧一行（可断点续跑）
- /mnt/d/vlm-active/frames/trafficqa/pool_ranked.jsonl  重排后的标注池（疑似推行在前）

用法：python screen_push.py [--limit N] [--workers 6]
"""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
POOL = "/mnt/d/vlm-active/frames/trafficqa/pool_dedup.jsonl"
ANN = "/mnt/d/vlm-active/frames/trafficqa/annotations.jsonl"
FRAMES = "/mnt/d/vlm-active/frames/trafficqa"
OUT = "/mnt/d/vlm-active/results/prescreen.jsonl"
RANKED = "/mnt/d/vlm-active/frames/trafficqa/pool_ranked.jsonl"


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    key = Path("/root/vlm-lab/api_testing/api_key.txt").read_text().strip()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    pool = [json.loads(l) for l in open(POOL)]
    labeled = {json.loads(l)["frame_id"]: json.loads(l).get("label")
               for l in open(ANN)}
    todo = [p for p in pool if labeled.get(p["frame_id"]) not in ("push", "ride")]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"待筛: {len(todo)} 帧", flush=True)

    done = {}
    if Path(OUT).exists():
        for l in open(OUT):
            r = json.loads(l)
            done[r["frame_id"]] = r
    todo = [p for p in todo if p["frame_id"] not in done]
    print(f"已完成 {len(done)}，本轮新筛 {len(todo)}", flush=True)

    fout = open(OUT, "a", encoding="utf-8")
    lock = __import__("threading").Lock()

    def work(p):
        fid = p["frame_id"]
        img_path = Path(FRAMES) / f"{fid}.jpg"
        if not img_path.exists():
            return
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": Q},
        ]}]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="qwen3-vl-flash", messages=msgs,
                    max_tokens=10, temperature=0.0)
                ans = resp.choices[0].message.content
                rec = {"frame_id": fid, "answer": ans,
                       "label_pred": parse_label(ans),
                       "ts": round(time.time())}
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                return
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    time.sleep(2 ** attempt)
                    continue
                if attempt == 3:
                    with lock:
                        fout.write(json.dumps({"frame_id": fid, "answer": f"ERROR: {err}",
                                               "label_pred": "none", "ts": round(time.time())},
                                              ensure_ascii=False) + "\n")
                        fout.flush()
                time.sleep(2)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"筛完 {len(todo)} 帧，耗时 {time.time()-t0:.0f}s", flush=True)

    # 重排标注池：疑似推行 > 不确定 > 骑行 > 已标注置底
    all_res = {}
    for l in open(OUT):
        r = json.loads(l)
        all_res[r["frame_id"]] = r
    order = {"push": 0, "none": 1, "ride": 2}
    def sort_key(p):
        fid = p["frame_id"]
        if labeled.get(fid) in ("push", "ride"):
            return (3, fid)
        return (order.get(all_res.get(fid, {}).get("label_pred", "none"), 1), fid)
    ranked = sorted(pool, key=sort_key)
    with open(RANKED, "w", encoding="utf-8") as f:
        for p in ranked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_push = sum(1 for r in all_res.values() if r["label_pred"] == "push")
    print(f"预筛结果: 疑似推行 {n_push} 帧；重排池 -> {RANKED}", flush=True)


if __name__ == "__main__":
    main()
