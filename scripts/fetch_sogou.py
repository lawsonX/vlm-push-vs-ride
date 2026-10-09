#!/usr/bin/env python3
"""搜狗图片搜索抓图（直链在 oriPicUrl 字段里，126 代理的要解出原始 URL）。

用法：
    python fetch_sogou.py --out /mnt/d/vlm-active/frames/web_sogou \
        --queries "电瓶车 推行 监控" "推电动车进电梯 监控" --per-query 20
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, parse_qs

import requests
from PIL import Image
from io import BytesIO

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

ORI_RE = re.compile(r'"oriPicUrl"\s*:\s*"([^"]+)"')


def unwrap(u):
    """126 代理链接解出原始图片地址。"""
    u = u.replace("\\u002F", "/")
    if "nimg.ws.126.net" in u:
        try:
            inner = parse_qs(urlparse(u).query)["url"][0]
            return unquote(inner)
        except Exception:
            return None
    return u


def ok_image(content):
    if not (25_000 < len(content) < 10_000_000):
        return None
    try:
        im = Image.open(BytesIO(content))
        im.load()
        if im.width < 450 or im.height < 340:
            return None
        return im.size
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--per-query", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = open(out / "sogou_manifest.jsonl", "a", encoding="utf-8")

    hash_set = set()
    kept = 0
    for qi, q in enumerate(args.queries, 1):
        url = f"https://pic.sogou.com/pics?query={quote(q)}"
        try:
            html = requests.get(url, headers=UA, timeout=25).text.replace("\\u002F", "/")
        except Exception:
            print(f"[{qi}] {q}: 搜索页打不开", flush=True)
            continue
        cand = []
        for raw in ORI_RE.findall(html):
            u = unwrap(raw)
            if u and u.startswith("http") and u not in cand:
                cand.append(u)
        got = 0
        for u in cand:
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
            if h in hash_set:
                continue
            hash_set.add(h)
            fid = f"sg_{qi:02d}_{got:03d}"
            (out / f"{fid}.jpg").write_bytes(r.content)
            manifest.write(json.dumps({
                "frame_id": fid, "source_video": f"sogou:{q}", "width": size[0], "height": size[1],
                "url": u,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            got += 1
            kept += 1
            time.sleep(0.25)
        print(f"[{qi}/{len(args.queries)}] {q}: {got} 张", flush=True)

    manifest.close()
    print(f"完成，共 {kept} 张 -> {out}")


if __name__ == "__main__":
    main()
