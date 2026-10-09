#!/usr/bin/env python3
"""从新闻页面抓取内嵌图片（监控画面截图）。

背景：必应图片搜索的开放结果杂质太多（ stock 图、动漫壁纸）。
新闻页面里的配图才是真实监控截图：偷车推行、推电动车进电梯、
物业测试阻车系统等场景，正是「推行」最高频的画面。

做法：抓页面 HTML → 抠 <img> 的 src/data-src → 按尺寸过滤（去掉 logo/图标）
→ 内容哈希去重 → 存盘并记录来源页面（可追溯、可人工复核）。

用法：
    python fetch_page_images.py --out /mnt/d/vlm-active/frames/web_news --urls-file urls.txt
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image
from io import BytesIO

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

IMG_TAG = re.compile(r'<img[^>]+(?:src|data-src)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
EXT_OK = (".jpg", ".jpeg", ".png", ".webp")


def fetch_page_imgs(url, max_imgs=12):
    try:
        html = requests.get(url, headers=UA, timeout=25).text
    except Exception as e:
        print(f"  页面打不开 {url}: {str(e)[:60]}", flush=True)
        return []
    raw = IMG_TAG.findall(html)
    out, seen = [], set()
    for u in raw:
        u = u.strip()
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            u = urljoin(url, u)
        if not u.startswith("http") or not u.lower().split("?")[0].endswith(EXT_OK):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_imgs * 2:  # 多抠一点，下载失败有冗余
            break
    return out


def ok_image(content):
    if not (25_000 < len(content) < 10_000_000):
        return None
    try:
        im = Image.open(BytesIO(content))
        im.load()
        if im.width < 500 or im.height < 375:
            return None
        return im.size
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--urls-file", required=True, help="每行一个文章 URL")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    urls = [l.strip() for l in open(args.urls_file, encoding="utf-8") if l.strip()]

    manifest = open(out / "news_manifest.jsonl", "a", encoding="utf-8")
    hashes = {f.stem: None for f in out.glob("*.jpg")}
    hash_set = set()
    kept = 0

    for pi, page in enumerate(urls, 1):
        img_urls = fetch_page_imgs(page)
        got = 0
        for u in img_urls:
            if got >= 12:
                break
            try:
                r = requests.get(u, headers={**UA, "Referer": page}, timeout=20)
            except Exception:
                continue
            size = ok_image(r.content)
            if not size:
                continue
            h = hashlib.md5(r.content).hexdigest()
            if h in hash_set:
                continue
            hash_set.add(h)
            fid = f"news_{pi:02d}_{got:02d}"
            (out / f"{fid}.jpg").write_bytes(r.content)
            manifest.write(json.dumps({
                "frame_id": fid, "source_video": f"news:{page}", "width": size[0], "height": size[1],
                "url": u,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            got += 1
            kept += 1
        print(f"[{pi}/{len(urls)}] {page[:60]}: {got} 张", flush=True)

    manifest.close()
    print(f"完成，共 {kept} 张 -> {out}")


if __name__ == "__main__":
    main()
