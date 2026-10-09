#!/usr/bin/env python3
"""构建无污染训练集：纯骑行 48 条，与 48 帧测试集零重叠、剔除 NaN 帧。"""
import json
import random

ann = [json.loads(l) for l in open("/home/lawson/vlm-active/frames/trafficqa/annotations.jsonl")]
ann = [a for a in ann if a.get("label") == "ride"]
test = {json.loads(l)["frame_id"] for l in open("/tmp/test_balanced.jsonl")}
nan = {b["frame"][:-4] for b in json.load(open("/home/lawson/vlm-active/results/nan_frames.json"))}
usable = [a for a in ann if a["frame_id"] not in test and a["frame_id"] not in nan]
rng = random.Random(7)
rng.shuffle(usable)
sel = usable[:48]
Q = "图中这个人是在推行还是骑行这辆车？"
with open("/home/lawson/vlm-active/sft/train_clean_ride48.jsonl", "w") as f:
    for a in sel:
        rec = {
            "frame_id": a["frame_id"],
            "image": "/home/lawson/vlm-active/frames/trafficqa/" + a["frame_id"] + ".jpg",
            "question": Q,
            "answer": "骑行",
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("干净训练集(纯骑行, 与测试零重叠):", len(sel))
