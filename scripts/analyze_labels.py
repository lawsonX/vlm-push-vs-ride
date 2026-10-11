# -*- coding: utf-8 -*-
"""分析 label_pool 人工标注 vs 云端预筛的交叉结果。"""
import json
from collections import Counter

man = {json.loads(l)["frame_id"]: json.loads(l) for l in open("/mnt/e/vlm-data/label_pool/manifest.jsonl", encoding="utf-8") if l.strip()}
anns = [json.loads(l) for l in open("/mnt/e/vlm-data/label_pool/annotations.jsonl", encoding="utf-8") if l.strip()]

ct = Counter()
for a in anns:
    m = man.get(a["frame_id"], {})
    ct[(m.get("activity_qc"), a["label"])] += 1

print("预筛\\人工  push ride unsure invalid")
for pred in ["push", "ride", "stand"]:
    row = [ct.get((pred, h), 0) for h in ["push", "ride", "unsure", "invalid"]]
    print(f"{pred:8s} {row[0]:5d} {row[1]:4d} {row[2]:6d} {row[3]:7d}")

notes = [a.get("note", "") for a in anns if a["label"] == "invalid" and a.get("note")]
print(f"\ninvalid 中有备注的: {len(notes)}/{sum(1 for a in anns if a['label']=='invalid')}")
kw = ["广告", "重复", "背", "模糊", "采访", "摆拍", "看不到", "遮挡", "多人", "宣传", "文字", "水印", "小", "远"]
nc = Counter()
for n in notes:
    for k in kw:
        if k in n:
            nc[k] += 1
print("备注关键词:", dict(nc.most_common(12)))
print("\n--- invalid 备注样例（15 条）---")
shown = 0
for a in anns:
    if a["label"] == "invalid" and a.get("note") and shown < 15:
        print("  ", a["note"][:70])
        shown += 1
