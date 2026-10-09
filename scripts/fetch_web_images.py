#!/usr/bin/env python3
"""从必应图片搜索抓「推行/骑行」真实图片，扩充标注池。

背景：车载事故视频数据里推行场景几乎为零（lawson 标注 100+ 张 0 推行后的判断）。
改用 web 图片搜索，按「推行高发场景」关键词抓图：
盗窃、违停整治、没电推行、进电梯、共享单车运维、过马路推行等。

流程：搜多组关键词 → 解析页面内嵌的图片直链（murl）→ 下载 → 尺寸/大小过滤 →
内容哈希去重 → 写清单（记录来源 query 和页面，可追溯）。

用法：
    python fetch_web_images.py --out /mnt/d/vlm-active/frames/web_push
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image
from io import BytesIO

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

QUERIES = [
    # 推行高发场景
    "电瓶车 推行 监控",
    "推行 电动车 监控画面",
    "偷电瓶车 监控 推行",
    "盗窃 电动车 监控截图 推行",
    "自行车 推行 监控",
    "共享单车 调度 推行",
    "运维人员 推行 共享单车",
    "电动车 推入电梯 监控",
    "电瓶车 进电梯 监控画面",
    "外卖员 推行 电动车 没电",
    "摩托车 推行",
    "推行 自行车 过马路 斑马线",
    "违停 电动车 推移 城管",
    "老人 推行 自行车",
    "推行 电瓶车 上坡",
    # 少量骑行对照（同一 web 域，平衡场景）
    "电瓶车 骑行 监控",
    "电动车 违章 骑行 监控截图",
    "骑自行车 监控画面",
]


def fetch_murls(query, n=30):
    """必应图片搜索结果页里嵌着 murl（原图直链），用正则批量抠出来。"""
    url = f"https://cn.bing.com/images/search?q={quote(query)}&form=HDRSC2"
    try:
        import html as htmlmod
        html = htmlmod.unescape(requests.get(url, headers=UA, timeout=30).text)
    except Exception:
        return []
    murls = re.findall(r'"murl"\s*:\s*"(https?://[^"]+)"', html)
    # 去重保序
    seen, out = set(), []
    for m in murls:
        m = m.replace("\\u002f", "/").replace("\\/", "/")
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:n]


def ok_image(content):
    """过滤：必须是真图片，最小 400x300，文件 20KB~8MB。"""
    if not (20_000 < len(content) < 8_000_000):
        return None
    try:
        im = Image.open(BytesIO(content))
        im.load()
        if im.width < 400 or im.height < 300:
            return None
        return im.size
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-query", type=int, default=25)
    ap.add_argument("--start", type=int, default=0, help="跳过前 N 个关键词（续跑用）")
    ap.add_argument("--append", action="store_true", help="追加模式，不清空旧清单")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = open(out / "web_manifest.jsonl", "a" if args.append else "w", encoding="utf-8")

    hashes = set()
    total = kept = 0
    queries = QUERIES[args.start:]
    for qi, q in enumerate(queries, args.start + 1):
        murls = fetch_murls(q)
        got = 0
        for u in murls:
            if got >= args.per_query:
                break
            try:
                r = requests.get(u, headers=UA, timeout=20)
            except Exception:
                continue
            size = ok_image(r.content)
            if not size:
                continue
            h = hashlib.md5(r.content).hexdigest()
            if h in hashes:
                continue
            hashes.add(h)
            fid = f"web_{qi:02d}_{got:03d}"
            (out / f"{fid}.jpg").write_bytes(r.content)
            manifest.write(json.dumps({
                "frame_id": fid, "source_video": f"web:{q}", "width": size[0], "height": size[1],
                "url": u, "query": q,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            got += 1
            kept += 1
            time.sleep(0.3)
        total += len(murls)
        print(f"[{qi}/{len(QUERIES)}] {q}: 抓到 {got} 张", flush=True)

    manifest.close()
    print(f"完成：候选 {total}，入库 {kept} -> {out}")


if __name__ == "__main__":
    main()
