#!/usr/bin/env python3
"""难例挖掘：用云端快模型扫未标注的 items_* 图，挑出值得 lawson 标注的批次。

逻辑：
1. 扫描 items_* 目录，剔除已在 label_pool 的图
2. qwen3-vl-flash 直接问「推行还是骑行」，记录预测（便宜，约几厘钱一张）
3. 选批次：预测为「推行」的全部（推行稀缺，误报也由 lawson 标注时纠偏）
   + 预测为「骑行」的随机抽样（凑够标注批次规模）
4. 把选中的图复制进 label_pool、追加 manifest，供标注 UI 使用

用法：
    python mine_unlabeled.py --out-jsonl /mnt/e/vlm-data/mining_preds.jsonl
    python mine_unlabeled.py --select-only ...   # 已有预测结果时直接选批次
"""

import argparse
import base64
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DIRECT_Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"
ITEMS_DIRS = ["items_bili", "items_bili3", "items_bili4", "items_bili5",
              "items_mot17", "items_sdd", "items_visdrone"]
LABEL_POOL = Path("/mnt/e/vlm-data/label_pool")


def load_key():
    for p in [Path(__file__).parent / "api_key.txt",
              Path("/root/vlm-lab/api_testing/api_key.txt")]:
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("api_key.txt not found")


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def collect_candidates():
    """返回 [(source, frame_id, path)]，剔除已在 label_pool 的。"""
    existing = set()
    for line in open(LABEL_POOL / "manifest.jsonl", encoding="utf-8"):
        if line.strip():
            existing.add(json.loads(line)["frame_id"])
    cands = []
    for d in ITEMS_DIRS:
        pdir = Path("/mnt/e/vlm-data") / d
        if not pdir.exists():
            continue
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            fid = f"{d}__{f.stem}"
            if fid in existing:
                continue
            cands.append((d, fid, f))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="/mnt/e/vlm-data/mining_preds.jsonl")
    ap.add_argument("--model", default="qwen3-vl-flash")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--push-cap", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="本轮最多处理多少张（0=全部），配合增量输出分轮跑")
    ap.add_argument("--min-size", type=int, default=480, help="选图时最小边长像素，太小的不要（lawson 要求从严）")
    ap.add_argument("--exclude-sources", default="items_sdd,items_mot17",
                    help="选图时排除的来源目录（逗号分隔；sdd 小图已被 lawson 全部判无效）")
    ap.add_argument("--select-only", action="store_true",
                    help="跳过云端调用，直接从已有 out-jsonl 选批次")
    args = ap.parse_args()

    if not args.select_only:
        cands_all = collect_candidates()
        # 断点续跑：跳过已有预测结果的图
        done = set()
        out_path = Path(args.out_jsonl)
        if out_path.exists():
            for l in open(out_path, encoding="utf-8"):
                if l.strip():
                    done.add(json.loads(l)["frame_id"])
            print(f"已完成 {len(done)} 张，续跑")
        cands = [c for c in cands_all if c[1] not in done]
        if args.limit:
            cands = cands[:args.limit]
        print(f"本轮候选: {len(cands)} 张（模型 {args.model}）")
        client = OpenAI(api_key=load_key(), base_url=BASE_URL, timeout=60)

        def work(c):
            source, fid, path = c
            b64 = base64.b64encode(path.read_bytes()).decode()
            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": DIRECT_Q},
                        ]}],
                        max_tokens=20, temperature=0.0)
                    text = r.choices[0].message.content or ""
                    return {"frame_id": fid, "source": source, "path": str(path),
                            "pred": parse_label(text)}
                except Exception:
                    if attempt == 3:
                        return {"frame_id": fid, "source": source, "path": str(path),
                                "pred": "error"}
                    time.sleep(2 * (attempt + 1))
            return None

        with open(args.out_jsonl, "a", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(work, cands)):
                if res:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  进度 {i+1}/{len(cands)}", flush=True)
        print(f"本轮预测完成: {args.out_jsonl}")

    # ---- 选批次 ----
    preds = [json.loads(l) for l in open(args.out_jsonl, encoding="utf-8") if l.strip()]
    from collections import Counter
    print("预测分布:", Counter(p["pred"] for p in preds))

    # 过滤：排除低质量来源 + 太小的图 + 与池中已标图重复的（dHash 汉明距离 ≤ 6）
    from PIL import Image
    excluded = set(args.exclude_sources.split(","))
    cache_path = LABEL_POOL / "phash_cache.json"
    phash_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    labeled_hash = [phash_cache[fid] for fid in phash_cache if fid in
                    {json.loads(l)["frame_id"] for l in open(LABEL_POOL / "annotations.jsonl", encoding="utf-8") if l.strip()}]

    def _hamming(a, b):
        return bin(a ^ b).count("1")

    def _dhash(img, hash_size=8):
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = list(img.getdata())
        bits = 0
        for row in range(hash_size):
            for col in range(hash_size):
                bits = (bits << 1) | (1 if px[row * (hash_size + 1) + col] > px[row * (hash_size + 1) + col + 1] else 0)
        return bits

    kept, n_dup = [], 0
    sel_hashes = []
    for p in preds:
        if p["source"] in excluded or p["pred"] in ("error", "none"):
            continue
        try:
            with Image.open(p["path"]) as im:
                if min(im.size) < args.min_size:
                    continue
                h = _dhash(im)
        except Exception:
            continue
        if any(_hamming(h, lh) <= 6 for lh in labeled_hash + sel_hashes):
            n_dup += 1
            continue
        sel_hashes.append(h)
        kept.append(p)
    print(f"过滤后可用: {len(kept)} 张（排除 {len(preds)-len(kept)-n_dup} 张低质/错误，{n_dup} 张重复）")

    rng = random.Random(42)
    push = [p for p in kept if p["pred"] == "push"]
    ride = [p for p in kept if p["pred"] == "ride"]
    rng.shuffle(push), rng.shuffle(ride)
    sel = push[:args.push_cap] + ride[:max(0, args.batch_size - len(push[:args.push_cap]))]
    rng.shuffle(sel)

    # 复制进 label_pool + 追加 manifest
    man = open(LABEL_POOL / "manifest.jsonl", "a", encoding="utf-8")
    n = 0
    for p in sel:
        dst = LABEL_POOL / f"{p['frame_id']}.jpg"
        if dst.exists():
            continue
        shutil.copy(p["path"], dst)
        src_video = p["frame_id"].split("__")[1].rsplit("__", 1)[0] if "__" in p["frame_id"] else p["frame_id"]
        man.write(json.dumps({
            "frame_id": p["frame_id"], "pool": "mine1", "source": p["source"],
            "activity_qc": p["pred"], "clarity": "unknown", "view": None,
            "facing": None, "source_video": src_video,
        }, ensure_ascii=False) + "\n")
        # 新图哈希入缓存，后续轮次据此去重
        try:
            with Image.open(dst) as im:
                phash_cache[p["frame_id"]] = _dhash(im)
        except Exception:
            pass
        n += 1
    man.close()
    cache_path.write_text(json.dumps(phash_cache))
    print(f"已加入标注池: {n} 张（预测push {min(len(push), args.push_cap)} + 骑行抽样）")
    print("请重启标注 UI 后使用。")


if __name__ == "__main__":
    main()
