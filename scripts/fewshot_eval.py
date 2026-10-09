#!/usr/bin/env python3
"""E10：少样本提示（few-shot）对照实验——不调训练，纯改提示。

微调矩阵全军覆没（只动先验）。一个没试过的方向：4-shot 多模态提示
（2 推行 + 2 骑行示例图）能否不动权重就提升？对 3 个云端模型跑 48 帧测试集，
对比它们零样本的旧数字（4.4 节）。

输出：/home/lawson/vlm-active/results/fewshot_3models.jsonl
"""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
FRAMES = "/home/lawson/vlm-active/frames/trafficqa"
OUT = "/home/lawson/vlm-active/results/fewshot_3models.jsonl"
MODELS = ["qwen3.5-omni-flash"]

# 4-shot 示例：2 推行用测试集中的明确例（sg_05_000/sg_10_005，E5 全对的最 clear 案例），
# 评测时这两帧剔除（算 support set）；2 骑行用测试集外已标注帧
SHOTS = [
    ("sg_05_000", "push"),
    ("sg_10_005", "push"),
    ("b_11E41127kV_clip_008_f001", "ride"),
    ("b_11E41127kV_clip_020_f002", "ride"),
]
SHOT_EXCLUDE = {"sg_05_000", "sg_10_005"}


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    key = Path(__file__).parent / "api_key.txt".read_text().strip()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    test = [json.loads(l) for l in open("/tmp/test_balanced.jsonl")]
    test = [t for t in test if t["label"] in ("push", "ride")
            and t["frame_id"] not in SHOT_EXCLUDE]
    print(f"评测帧数(剔除shot帧): {len(test)}", flush=True)

    # 过滤存在的 shot
    shots = [(f, a) for f, a in SHOTS if Path(f"{FRAMES}/{f}.jpg").exists()][:4]
    print(f"shots: {[(f, a) for f, a in shots]}", flush=True)
    if len(shots) < 4:
        print("⚠️ shot 不足 4 个仍继续", flush=True)

    def build_msgs(fid):
        content = []
        for sf, sa in shots:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64(f'{FRAMES}/{sf}.jpg')}"}})
            content.append({"type": "text", "text": Q + "\n" + ("推行" if sa == "push" else "骑行")})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64(f'{FRAMES}/{fid}.jpg')}"}})
        content.append({"type": "text", "text": Q})
        return [{"role": "user", "content": content}]

    fout = open(OUT, "a", encoding="utf-8")
    import threading
    lock = threading.Lock()
    done = set()
    if Path(OUT).exists():
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["model"], r["frame_id"]))
    tasks = [(m, t) for m in MODELS for t in test if (m, t["frame_id"]) not in done]
    print(f"总任务 {len(MODELS)*len(test)}，已完成 {len(MODELS)*len(test)-len(tasks)}，本轮 {len(tasks)}", flush=True)

    def work(mt):
        m, t = mt
        msgs = build_msgs(t["frame_id"])
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=m, messages=msgs, max_tokens=10, temperature=0.0)
                ans = resp.choices[0].message.content
                rec = {"model": m, "frame_id": t["frame_id"],
                       "label_pred": parse_label(ans), "answer": ans[:60],
                       "truth": t["label"], "style": "fewshot4"}
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
                        fout.write(json.dumps({"model": m, "frame_id": t["frame_id"],
                                               "label_pred": "none", "answer": f"ERROR:{err}"[:60],
                                               "truth": t["label"], "style": "fewshot4"},
                                              ensure_ascii=False) + "\n")
                        fout.flush()
                time.sleep(2)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(work, tasks))
    print(f"完成，耗时 {time.time()-t0:.0f}s", flush=True)

    # 汇总
    from collections import Counter, defaultdict
    stat = defaultdict(Counter)
    for l in open(OUT):
        r = json.loads(l)
        stat[(r["model"], r["truth"])]["hit" if r["label_pred"] == r["truth"] else "miss"] += 1
    for m in MODELS:
        line = [m]
        for truth in ("push", "ride"):
            c = stat[(m, truth)]
            n = c["hit"] + c["miss"]
            line.append(f"{truth} {c['hit']}/{n}")
        print("  " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
