#!/usr/bin/env python3
"""按「推行可能性」给池子里的帧排序，让标注优先看疑似推行的图。

为什么这么干：监控/车载画面里骑行是压倒多数，推行极少。
如果按随机顺序标注，可能标几百张都遇不到一张推行，测试集根本测不出
「推行被误判骑行」。这个脚本用 YOLO 框的几何关系做启发式打分：

- 骑行：人框和车框重叠大（人坐在车上），人几乎在车框正上方/正中间
- 推行：人框偏在车的一侧（手扶车把走路），框重叠小

打分后按「疑似推行优先」排序输出，标注页面加载这份清单即可。

只是优先级排序，不是真标签——最终标签永远以人工标注为准。

用法：
    python push_prioritize.py --pool pool_manifest.jsonl --out pool_ranked.jsonl
"""

import argparse
import json


def pair_features(p_box, b_box):
    """计算人和车框的几何关系特征。"""
    px1, py1, px2, py2 = p_box
    bx1, by1, bx2, by2 = b_box
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2

    pw, ph = px2 - px1, py2 - py1
    bw, bh = bx2 - bx1, by2 - y1 if False else by2 - by1

    # 水平偏移（占车宽比例）：推行时人偏一侧
    dx = abs(pcx - bcx) / max(bw, 1)
    # 框重叠面积占人框比例
    ix = max(0, min(px2, bx2) - max(px1, bx1))
    iy = max(0, min(py2, by2) - max(py1, by1))
    inter = ix * iy
    iou_person = inter / max(pw * ph, 1)
    # 人的脚底相对车底：推行走路时人脚底通常不低于车底太多
    foot_rel = (py2 - by2) / max(bh, 1)

    return {"dx": dx, "iou": iou_person, "foot_rel": foot_rel}


def push_score(f):
    """启发式：偏移大、重叠小 → 更像推行。返回越大越像推行。"""
    score = 0.0
    score += min(f["dx"], 2.0) * 2.0          # 横向偏移是最强信号
    score += (1.0 - min(f["iou"], 1.0)) * 1.5  # 重叠小 → 推行
    score -= abs(f["foot_rel"]) * 0.3          # 脚悬空（骑行）扣分
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    scored = []
    for row in rows:
        best = None
        for p in row.get("persons", []):
            for b in row.get("bikes", []):
                f = pair_features(p["box"], b["box"])
                s = push_score(f)
                if best is None or s > best[0]:
                    best = (s, f)
        if best is None:
            continue
        scored.append({
            "frame_id": row["frame_id"],
            "width": row["width"],
            "height": row["height"],
            "persons": row["persons"],
            "bikes": row["bikes"],
            "push_score": round(best[0], 3),
            "features": {k: round(v, 3) for k, v in best[1].items()},
        })

    # 分档：top 30% 标为 push_suspect，其余 normal，排序输出
    scored.sort(key=lambda r: -r["push_score"])
    cut = int(len(scored) * 0.3)
    for i, r in enumerate(scored):
        r["priority"] = "push_suspect" if i < cut else "normal"
        r.pop("push_score", None)
        r.pop("features", None)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(scored)
    print(f"完成: {n} 帧，前 {cut} 帧标记为 push_suspect（疑似推行优先标注）")


if __name__ == "__main__":
    main()
