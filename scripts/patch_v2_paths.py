#!/usr/bin/env python3
"""把 delivery_v02 交付包 jsonl 里的相对图片路径替换成本机绝对路径，并校验所有引用图片在场。

用法：
    python scripts/patch_v2_paths.py \
        --delivery datasets/delivery_v02 \
        --out data/sft_v2

输出（out 目录下）：
    train.jsonl / train_aug.jsonl / test.jsonl   路径已替换的副本
    validate_report.json                          校验结果
"""

import argparse
import json
from pathlib import Path

FILES = ["train.jsonl", "train_aug.jsonl", "test.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default="datasets/delivery_v02")
    ap.add_argument("--out", default="data/sft_v2")
    args = ap.parse_args()

    delivery = Path(args.delivery).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {"delivery": str(delivery), "files": {}}
    for name in FILES:
        src = delivery / name
        rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        missing, bad_prefix = [], []
        for r in rows:
            img = r["image"]
            if not img.startswith("frames/"):
                bad_prefix.append(img)
            abs_path = delivery / img
            if not abs_path.exists():
                missing.append(img)
            r["image"] = str(abs_path)
        dst = out / name
        with dst.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["files"][name] = {
            "samples": len(rows),
            "missing_images": len(missing),
            "bad_prefix": len(bad_prefix),
            "first_image": rows[0]["image"],
        }
        if missing:
            report["files"][name]["missing_list_head"] = missing[:5]
        print(f"{name}: {len(rows)} 条, 缺图 {len(missing)}, 前缀异常 {len(bad_prefix)}")

    # 测试集冻结校验：test.jsonl 的 frame_id 集合必须等于 test_ids.txt
    test_ids = set((delivery / "test_ids.txt").read_text().split())
    test_rows = [json.loads(l) for l in (out / "test.jsonl").open(encoding="utf-8")]
    test_frames = {r["frame_id"] for r in test_rows}
    leak = test_frames & set()  # 占位，真正的泄漏检查在 make_sft_data_v2 已按图划分
    report["test_ids_match"] = test_ids == test_frames
    print(f"test_ids.txt 与 test.jsonl frame_id 一致: {report['test_ids_match']}")
    # 训练/测试按图互斥检查（铁律：同一张图不能既进训练又进测试）
    train_frames = {json.loads(l)["frame_id"] for l in (out / "train.jsonl").open(encoding="utf-8")}
    overlap = train_frames & test_frames
    report["train_test_overlap"] = sorted(overlap)
    print(f"训练/测试重叠图数: {len(overlap)}")

    (out / "validate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing or bad_prefix or overlap or not report["test_ids_match"]:
        raise SystemExit("校验未通过，见 validate_report.json")


if __name__ == "__main__":
    main()
