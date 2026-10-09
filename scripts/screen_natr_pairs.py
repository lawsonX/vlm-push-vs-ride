#!/usr/bin/env python3
"""E9 天然对云端质检：flash 分别问每对的骑/推帧，两边都答对才留。

几何启发式的「ride 端」常把站车旁的人误判为骑行（natr_0020 即此）。
用云端模型把 60 对预筛成高置信子集，lawson 终审。

输出：/home/lawson/vlm-active/frames/natr2/pairs_screened.jsonl
"""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
DIR = Path("/home/lawson/vlm-active/frames/natr2")
OUT = DIR / "pairs_screened.jsonl"


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def main():
    key = Path(__file__).parent / "api_key.txt".read_text().strip()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    pairs = [json.loads(l) for l in open(DIR / "pairs_transition2.jsonl")]
    print(f"待质检: {len(pairs)} 对", flush=True)

    def ask(img_path):
        b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": Q},
        ]}]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="qwen3-vl-flash", messages=msgs,
                    max_tokens=10, temperature=0.0)
                return parse_label(resp.choices[0].message.content)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    time.sleep(2 ** attempt)
                    continue
                return "none"
        return "none"

    def work(p):
        pid = p["pair_id"]
        lab_ride = ask(DIR / f"{pid}_ride.jpg")
        lab_push = ask(DIR / f"{pid}_push.jpg")
        ok = (lab_ride == "ride") and (lab_push == "push")
        rec = dict(p)
        rec["screen_ride"] = lab_ride
        rec["screen_push"] = lab_push
        rec["screen_pass"] = ok
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(work, pairs))
    print(f"质检耗时 {time.time()-t0:.0f}s", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    passed = [r for r in results if r["screen_pass"]]
    print(f"通过: {len(passed)}/{len(results)} -> {OUT}", flush=True)
    from collections import Counter
    print("ride端判断分布:", dict(Counter(r["screen_ride"] for r in results)))
    print("push端判断分布:", dict(Counter(r["screen_push"] for r in results)))


if __name__ == "__main__":
    main()
