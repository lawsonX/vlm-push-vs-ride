#!/usr/bin/env python3
"""按 docs/13 的规范，把人工标注结果转成 SFT 训练数据（v2）。

和 v1（make_sft_data.py）的区别：
- 一张金标图配多种问法：direct（随机同义问句）/ chain（证据→结论）/ guide（两轮引导）
- chain/guide 的证据描述来自标注人写的 evidence 字段，只做句式整理，不编造内容
- 按图划分训练/测试集（同一张图的所有问法样本不拆开），frame_id 哈希确定性抽样
- 测试集分层配平（推行/骑行各 N 张，默认 24），永远不参与训练
- 输出标准多轮对话 messages 格式，loss 只算 assistant 部分（训练侧负责）

用法：
    python make_sft_data_v2.py --annotations /mnt/e/vlm-data/label_pool/annotations.jsonl \
        --frames-dir /mnt/e/vlm-data/label_pool --out-dir /mnt/e/vlm-data/sft_v2

    # 换设备后重跑，只换路径前缀即可（图片不进 git，靠路径替换对接）
    python make_sft_data_v2.py --strip-prefix /mnt/e/vlm-data --add-prefix /new/path ...
"""

import argparse
import hashlib
import json
from pathlib import Path

DIRECT_QUESTIONS = [
    "图中这个人是在推行还是骑行这辆车？只回答一个词。",
    "观察图中的人和他的车：他是在推行还是骑行？只回答一个词。",
    "这张监控图里，这个人正在推行还是骑行这辆车？只回答一个词。",
    "图中这个人当前是在推车走还是骑车走？只回答一个词。",
    "请判断：图中的人是推行还是骑行这辆车？只回答一个词。",
]
LABEL_WORD = {"push": "推行", "ride": "骑行"}
CHAIN_QUESTION = ("请观察图中这个人的手和脚的位置、身体是坐在车上还是站在车旁，"
                  "先描述你看到的证据，最后再给出结论。")
GUIDE_Q1 = ("观察图中这个人和这辆车：他的手放在车的什么位置？脚在哪里？"
            "身体是坐在车上还是站在车旁？请详细描述。")
GUIDE_Q2 = "根据你上面的描述，这个人是在推行还是骑行这辆车？只回答一个词。"
IMG_PLACEHOLDER = "<图片占位>"


def h(frame_id: str) -> str:
    """确定性哈希，任何机器上重跑结果一致。"""
    return hashlib.md5(frame_id.encode("utf-8")).hexdigest()


def find_image(frames_dir: Path, frame_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = frames_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


def clean_evidence(ev: str) -> str:
    """句式整理：去首尾空白和尾部标点。不做内容改写、不编造。"""
    ev = ev.strip().rstrip("。.;；,，")
    return ev


def chain_answer(ev: str, label: str) -> str:
    return f"证据：{ev}。结论：{LABEL_WORD[label]}。"


def guide_first_answer(ev: str) -> str:
    # 标注人写的判断依据通常是"看脚…""腿在…"这种观察句式，直接采用
    return f"图中可以看到：{ev}。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-per-class", type=int, default=24,
                    help="测试集每个类别抽多少张（默认 24，两类共 48）")
    ap.add_argument("--balance", action="store_true",
                    help="训练集 push/ride 1:1 配平（推行是少数类）")
    # 2026-10-15 lawson 拍板：unsure/invalid 一律舍弃，永不进训练（含负样本）。
    # 原 --include-invalid 功能已删除。
    ap.add_argument("--strip-prefix", default=None, help="去掉 image 路径的旧前缀")
    ap.add_argument("--add-prefix", default=None, help="加上 image 路径的新前缀")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    anns = [json.loads(l) for l in open(args.annotations, encoding="utf-8") if l.strip()]
    by_id = {a["frame_id"]: a for a in anns}

    gold = {fid: a for fid, a in by_id.items() if a.get("label") in LABEL_WORD}
    invalid = {fid: a for fid, a in by_id.items() if a.get("label") == "invalid"}
    print(f"标注总条数 {len(anns)}：金标(push/ride) {len(gold)}，无效 {len(invalid)}，"
          f"其余(不确定/未标) {len(anns) - len(gold) - len(invalid)}")

    # 只保留图文件确实存在的
    missing = [fid for fid in list(gold) if find_image(frames_dir, fid) is None]
    for fid in missing:
        del gold[fid]
    if missing:
        print(f"warning: {len(missing)} 张金标找不到图文件，跳过")

    # ---- 按图划分训练/测试（分层、确定性）----
    def take_test(ids):
        ids = sorted(ids, key=h)  # 哈希排序 = 确定性随机
        return set(ids[:args.test_per_class])

    push_ids = [f for f, a in gold.items() if a["label"] == "push"]
    ride_ids = [f for f, a in gold.items() if a["label"] == "ride"]
    test_ids = take_test(push_ids) | take_test(ride_ids)
    train_ids = [f for f in gold if f not in test_ids]
    print(f"测试集: push={len(test_ids & set(push_ids))}, ride={len(test_ids & set(ride_ids))}"
          f"（共 {len(test_ids)} 张图）；训练集候选 {len(train_ids)} 张图")

    # ---- 训练集配平（可选）----
    if args.balance:
        tr_push = [f for f in train_ids if gold[f]["label"] == "push"]
        tr_ride = [f for f in train_ids if gold[f]["label"] == "ride"]
        n = min(len(tr_push), len(tr_ride))
        tr_push = sorted(tr_push, key=h)[:n]
        tr_ride = sorted(tr_ride, key=h)[:n]
        train_ids = tr_push + tr_ride
        print(f"配平后训练集: push={len(tr_push)}, ride={len(tr_ride)}")

    # ---- 生成样本 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ftrain = open(out_dir / "train.jsonl", "w", encoding="utf-8")
    ftest = open(out_dir / "test.jsonl", "w", encoding="utf-8")
    report = {"train": {}, "test": {}}

    def img_path(fid):
        p = str(find_image(frames_dir, fid))
        if args.strip_prefix and p.startswith(args.strip_prefix):
            p = p[len(args.strip_prefix):]
        if args.add_prefix:
            p = args.add_prefix + p
        return p

    def emit(fid, split):
        a = gold[fid]
        label = a["label"]
        ev = clean_evidence(a.get("evidence") or "")
        msgs_direct = [
            {"role": "user", "content": IMG_PLACEHOLDER + DIRECT_QUESTIONS[int(h(fid), 16) % len(DIRECT_QUESTIONS)]},
            {"role": "assistant", "content": LABEL_WORD[label]},
        ]
        samples = [("direct", msgs_direct)]
        if ev:
            samples.append(("chain", [
                {"role": "user", "content": IMG_PLACEHOLDER + CHAIN_QUESTION},
                {"role": "assistant", "content": chain_answer(ev, label)},
            ]))
            samples.append(("guide", [
                {"role": "user", "content": IMG_PLACEHOLDER + GUIDE_Q1},
                {"role": "assistant", "content": guide_first_answer(ev)},
                {"role": "user", "content": GUIDE_Q2},
                {"role": "assistant", "content": LABEL_WORD[label]},
            ]))
        fh = ftrain if split == "train" else ftest
        for style, msgs in samples:
            fh.write(json.dumps({
                "frame_id": fid, "image": img_path(fid), "style": style,
                "split": split, "label": label, "messages": msgs,
            }, ensure_ascii=False) + "\n")
            report[split][style] = report[split].get(style, 0) + 1

    for fid in sorted(train_ids, key=h):
        emit(fid, "train")
    for fid in sorted(test_ids, key=h):
        emit(fid, "test")

    ftrain.close()
    ftest.close()
    (out_dir / "report.json").write_text(json.dumps({
        "annotations": len(anns), "gold": len(gold),
        "test_images": len(test_ids),
        "train_images": len(train_ids),
        "samples": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {out_dir}/train.jsonl, test.jsonl, report.json")
    print(f"样本数: train={sum(report['train'].values())}, test={sum(report['test'].values())}")


if __name__ == "__main__":
    main()
