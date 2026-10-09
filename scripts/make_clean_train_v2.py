#!/usr/bin/env python3
"""lawson 标注完成后的干净训练集构建 v2。

与 make_clean_train.py 的区别：把测试集外新标注的「推行」帧纳入训练
（这是标注的目的），骑行帧从测试集外的非 NaN 池中配平。

输出：/home/lawson/vlm-active/sft/train_clean_v2.jsonl
"""
import json
import random

ANN = "/home/lawson/vlm-active/frames/trafficqa/annotations.jsonl"
TEST = "/tmp/test_balanced.jsonl"
NAN = "/home/lawson/vlm-active/results/nan_frames.json"
OUT = "/home/lawson/vlm-active/sft/train_clean_v2.jsonl"
Q = "图中这个人是在推行还是骑行这辆车？"

ann = [json.loads(l) for l in open(ANN)]
ann = [a for a in ann if a.get("label") in ("push", "ride")]
test = {json.loads(l)["frame_id"] for l in open(TEST)}
nan = {b["frame"][:-4] for b in json.load(open(NAN))}

push = [a for a in ann if a["label"] == "push" and a["frame_id"] not in test
        and a["frame_id"] not in nan]
ride = [a for a in ann if a["label"] == "ride" and a["frame_id"] not in test
        and a["frame_id"] not in nan]
rng = random.Random(7)
rng.shuffle(ride)

# 骑行与推行 1:1 配平（推行是瓶颈，骑行跟着推行的量走）
n = min(len(push), len(ride))
sel = push + ride[:max(n, 48)]  # 至少 48 条骑行，保证训练量
rng.shuffle(sel)

with open(OUT, "w", encoding="utf-8") as f:
    for a in sel:
        rec = {
            "frame_id": a["frame_id"],
            "image": "/home/lawson/vlm-active/frames/trafficqa/" + a["frame_id"] + ".jpg",
            "question": Q,
            "answer": "推行" if a["label"] == "push" else "骑行",
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

from collections import Counter
print("训练集:", len(sel), dict(Counter(a["label"] for a in sel)))
print("其中推行:", len(push), "（测试集外全部可用推行帧）")
if len(push) == 0:
    print("⚠️ 测试集外仍无推行帧——标注还没覆盖到，先去标 pool_ranked 前 130 张")
