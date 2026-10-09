#!/usr/bin/env python3
"""第二道预筛：qwen3.8-omni-flash 复核 flash 判「推行」的帧。

flash 单模型假推行率 74%（它在这池子上过度报推行）；omni-flash 偏置方向
不同，两模型一致的「推行」可信度高得多。

输出：/mnt/d/vlm-active/results/prescreen2.jsonl，并重排 pool_ranked.jsonl
（两道都判推行 > 仅 flash > 其他）。
"""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
FRAMES = "/mnt/d/vlm-active/frames/trafficqa"
ANN = "/mnt/d/vlm-active/frames/trafficqa/annotations.jsonl"
POOL = "/mnt/d/vlm-active/frames/trafficqa/pool_dedup.jsonl"
OUT1 = "/mnt/d/vlm-active/results/prescreen.jsonl"
OUT2 = "/mnt/d/vlm-active/results/prescreen2.jsonl"
RANKED = "/mnt/d/vlm-active/frames/trafficqa/pool_ranked.jsonl"


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def main():
    key = Path("/root/vlm-lab/api_testing/api_key.txt").read_text().strip()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    r1 = {json.loads(l)["frame_id"]: json.loads(l) for l in open(OUT1)}
    candidates = [fid for fid, r in r1.items() if r["label_pred"] == "push"]
    print(f"flash 判推行的候选: {len(candidates)}", flush=True)

    done = set()
    if Path(OUT2).exists():
        done = {json.loads(l)["frame_id"] for l in open(OUT2)}
    todo = [f for f in candidates if f not in done]
    print(f"已完成 {len(done)}，本轮新筛 {len(todo)}", flush=True)

    fout = open(OUT2, "a", encoding="utf-8")
    import threading
    lock = threading.Lock()

    def work(fid):
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
                    model="qwen3.8-omni-flash", messages=msgs,
                    max_tokens=10, temperature=0.0)
                ans = resp.choices[0].message.content
                rec = {"frame_id": fid, "answer": ans,
                       "label_pred": parse_label(ans), "ts": round(time.time())}
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
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(work, todo))
    print(f"筛完，耗时 {time.time()-t0:.0f}s", flush=True)

    # 重排：两道一致推行 > 仅 flash 推行 > 不确定 > 骑行
    r2 = {json.loads(l)["frame_id"]: json.loads(l) for l in open(OUT2)}
    labeled = {json.loads(l)["frame_id"]: json.loads(l).get("label")
               for l in open(ANN)}
    pool = [json.loads(l) for l in open(POOL)]

    def key(p):
        fid = p["frame_id"]
        if labeled.get(fid) in ("push", "ride"):
            return (4, fid)
        f1 = r1.get(fid, {}).get("label_pred", "none")
        f2 = r2.get(fid, {}).get("label_pred", "none")
        if f1 == "push" and f2 == "push":
            return (0, fid)
        if f1 == "push":
            return (1, fid)
        if f2 == "push":
            return (2, fid)
        return (3, fid)

    ranked = sorted(pool, key=key)
    with open(RANKED, "w", encoding="utf-8") as f:
        for p in ranked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    both = sum(1 for fid in candidates if r2.get(fid, {}).get("label_pred") == "push")
    print(f"两道一致推行: {both}/{len(candidates)}；重排 -> {RANKED}", flush=True)


if __name__ == "__main__":
    main()
