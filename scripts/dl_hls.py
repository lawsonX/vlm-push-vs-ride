#!/usr/bin/env python3
"""下载 HLS(m3u8) 新闻视频并合并成 ts，供抽帧用。"""
import sys
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.163.com/"}

url = [l.strip() for l in open("/root/news_videos.txt") if l.strip()][0]
out = "/mnt/e/vlm-data/raw/news_videos/elevator_push.ts"

r = requests.get(url, headers=UA, timeout=30)
r.raise_for_status()
lines = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
print("分段数:", len(lines))

with open(out, "wb") as f:
    for seg in lines:
        u = urljoin(url, seg)
        resp = requests.get(u, headers=UA, timeout=30)
        f.write(resp.content)
print("saved:", out)
