#!/usr/bin/env python3
"""云端 VLM 测试合成图：验证合成图能不能复现/暴露幻觉，评估训练价值。

对 iter3 的图（人工目检过的）分别测原图和监控化版本。
关键指标：
- 合成推行图被答成「骑行」的比例（复现真实世界幻觉 → 说明合成图抓对了难点）
- 合成骑行图的答对率
- 监控化处理后两个指标的变化

用法：
    python test_synth_vlm.py --qc /mnt/e/vlm-data/synth/iter3/qc.jsonl \
        --raw-dir /mnt/e/vlm-data/synth/iter3 --cctv-dir /mnt/e/vlm-data/synth/iter3_cctv \
        --out /mnt/e/vlm-data/synth/iter3/vlm_result.jsonl
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DIRECT_Q = "图中这个人是在推行还是骑行这辆车？只回答「推行」或「骑行」。"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True, help="目检标签 jsonl: file/label")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--cctv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    qc = [json.loads(l) for l in open(args.qc, encoding="utf-8") if l.strip()]
    client = OpenAI(api_key=load_key(), base_url=BASE_URL)
    jobs = []
    for q in qc:
        for variant, d in (("raw", args.raw_dir), ("cctv", args.cctv_dir)):
            for ext in (".png", ".jpg"):
                p = Path(d) / f"{q['file']}{ext}"
                if p.exists():
                    jobs.append((q["file"], q["label"], variant, p))
                    break
    print(f"测试任务: {len(jobs)} 张图（模型 {args.model}）")

    def work(job):
        fid, label, variant, path = job
        b64 = base64.b64encode(path.read_bytes()).decode()
        for attempt in range(5):
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
                return {"file": fid, "gold": label, "variant": variant,
                        "pred": parse_label(text), "raw_text": text[:100]}
            except Exception:
                if attempt == 4:
                    return {"file": fid, "gold": label, "variant": variant,
                            "pred": "error", "raw_text": ""}
                time.sleep(2 * (attempt + 1))
        return None

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(work, jobs)):
            if res:
                results.append(res)
            if (i + 1) % 20 == 0:
                print(f"  进度 {i+1}/{len(jobs)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总：按 variant × gold 统计
    from collections import defaultdict
    stat = defaultdict(lambda: [0, 0])
    for r in results:
        key = (r["variant"], r["gold"])
        stat[key][1] += 1
        if r["pred"] == r["gold"]:
            stat[key][0] += 1
    print("\n=== 准确率汇总（variant × 金标）===")
    for (variant, gold), (ok, n) in sorted(stat.items()):
        print(f"  {variant:5s}  gold={gold:4s}: {ok}/{n} = {ok/n*100:.0f}%")
    # 推行图的错误方向
    print("\n=== 推行图的错误方向 ===")
    for r in results:
        if r["gold"] == "push" and r["pred"] != "push":
            print(f"  {r['variant']:5s} {r['file']}: gold=push pred={r['pred']}")
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
