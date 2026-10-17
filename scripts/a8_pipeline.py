#!/usr/bin/env python3
"""A8 自动管线：v0.7 数据落地后一键完成 清洗校验→多种子训练→评测→汇总。

用法（数据侧出 v0.7 后）：
    python scripts/a8_pipeline.py --src data/sft_v7 \
        --frames-dir datasets/delivery_v02/frames \
        --out-root ckpt/a8_v7 --seeds 42 7 99

步驟：
1. 路径替换 + 三重校验（缺图 / test_ids 冻结名单 / 训练测试互斥）
2. 每个种子训练（可 --skip-train 复用已有 ckpt）
3. 每种子在冻结测试集评测（256 token 固定口径，可 --skip-eval）
4. 汇总均值±极差表（markdown + json）——方差定案后的强制纪律

只测汇总逻辑（复用现有结果）：
    python scripts/a8_pipeline.py --aggregate-only \
        --summaries results/a1_baseline_preds.summary.json results/a1_replica_preds.summary.json results/a1_v04_seed7_preds.summary.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = "/hdd2/xiaolirui/conda_envs/vlm-lab/bin/python"
MODEL = "models/Qwen2.5-VL-3B-Instruct"


def patch_and_validate(src, frames_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    rows_out = {}
    for name in ("train.jsonl", "test.jsonl"):
        rows = [json.loads(l) for l in (Path(src) / name).open(encoding="utf-8") if l.strip()]
        keep, missing_frames = [], set()
        for r in rows:
            rel = r["image"].replace("frames/", "", 1)
            p = Path(frames_dir) / rel
            if p.exists():
                r["image"] = str(p.resolve())
                keep.append(r)
            else:
                missing_frames.add(r["frame_id"])
        dst = out / name
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                       encoding="utf-8")
        rows_out[name] = keep
        report[name] = {"原样本": len(rows), "保留": len(keep),
                        "缺图剔除": len(rows) - len(keep)}
        if missing_frames:
            report[name]["缺图帧"] = sorted(missing_frames)
    ids_path = Path(src) / "test_ids.txt"
    if ids_path.exists():
        ids = set(ids_path.read_text().split())
        tf = {r["frame_id"] for r in rows_out["test.jsonl"]}
        report["test_ids一致"] = ids == tf
        if ids != tf:
            report["test_ids多余"] = sorted(ids - tf)
            report["test_ids缺失"] = sorted(tf - ids)
    overlap = {r["frame_id"] for r in rows_out["train.jsonl"]} & {r["frame_id"] for r in rows_out["test.jsonl"]}
    report["训练测试重叠"] = sorted(overlap)
    (out / "validate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if overlap:
        raise SystemExit("训练/测试重叠，中止")
    return out / "train.jsonl", out / "test.jsonl"


def run_train(train, out, seed, epochs, gpu):
    cmd = [PY, "scripts/train_sft_v2.py", "--model", MODEL, "--data", str(train),
           "--out", str(out), "--epochs", str(epochs), "--seed", str(seed)]
    env = f"CUDA_VISIBLE_DEVICES={gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    print(f"[train seed={seed}] {' '.join(cmd)}")
    subprocess.run(f"{env} " + " ".join(cmd), shell=True, check=True)


def run_eval(ckpt, test, out, gpu):
    cmd = [PY, "scripts/eval_v2.py", "--ckpt", str(ckpt), "--test", str(test),
           "--out", str(out)]
    env = f"CUDA_VISIBLE_DEVICES={gpu}"
    print(f"[eval] {' '.join(cmd)}")
    subprocess.run(f"{env} " + " ".join(cmd), shell=True, check=True)


def aggregate(summaries, out_md=None):
    per_style = {}
    overalls = []
    for sp in summaries:
        s = json.loads(Path(sp).read_text(encoding="utf-8"))
        overalls.append(s["overall"])
        for style, c in s["per_style"].items():
            n = c.get("hit", 0) + c.get("miss", 0)
            acc = c.get("hit", 0) / n if n else None
            per_style.setdefault(style, []).append((acc, n, c.get("none", 0), c.get("unclear", 0)))
    def fmt(vals):
        vals = [v for v, *_ in vals if v is not None]
        if not vals:
            return "-"
        m = sum(vals) / len(vals)
        return f"{100*m:.1f}%±{100*(max(vals)-min(vals))/2:.1f} (n={len(vals)})"
    lines = ["| 问法 | 均值±半极差 |", "|---|---|"]
    for style in sorted(per_style):
        lines.append(f"| {style} | {fmt(per_style[style])} |")
    # overall 按计分样本聚合
    accs = []
    for o in overalls:
        n = o.get("hit", 0) + o.get("miss", 0)
        accs.append(o.get("hit", 0) / n if n else None)
    accs = [a for a in accs if a is not None]
    lines.append(f"| **overall** | **{100*sum(accs)/len(accs):.1f}%±{100*(max(accs)-min(accs))/2:.1f} (n={len(accs)})** |")
    md = "\n".join(lines)
    print("\n" + md)
    if out_md:
        Path(out_md).write_text(md + "\n", encoding="utf-8")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="数据侧交付目录（含 train/test jsonl + test_ids）")
    ap.add_argument("--frames-dir", default="datasets/delivery_v02/frames")
    ap.add_argument("--work-dir", default="data/sft_v7_abs")
    ap.add_argument("--out-root", default="ckpt/a8_v7")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--gpus", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--summaries", nargs="+", default=[])
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(args.summaries, "results/a8_aggregate.md")
        return

    train, test = patch_and_validate(args.src, args.frames_dir, args.work_dir)
    summaries = []
    for i, seed in enumerate(args.seeds):
        gpu = args.gpus[i % len(args.gpus)]
        ckpt = Path(args.out_root) / f"seed{seed}"
        if not args.skip_train:
            run_train(train, ckpt, seed, args.epochs, gpu)
        pred = Path("results") / f"{ckpt.name}_v7_preds.jsonl"
        if not args.skip_eval:
            run_eval(ckpt / "final", test, pred, gpu)
        summaries.append(pred.with_suffix(".summary.json"))
    aggregate(summaries, Path(args.out_root) / "a8_summary.md")
    print("done ->", args.out_root)


if __name__ == "__main__":
    main()
