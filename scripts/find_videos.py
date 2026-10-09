#!/usr/bin/env python3
"""从新闻页面提取内嵌视频地址（监控实录多是 mp4/m3u8）。"""
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0"}
urls = [l.strip() for l in open("/home/lawson/vlm-data/raw/_meta/news_urls.txt", encoding="utf-8") if l.strip()]
found = []
for u in urls:
    try:
        html = requests.get(u, headers=UA, timeout=20).text
    except Exception:
        continue
    vids = set(re.findall(r'https?://[^"\'\s<>]+\.mp4[^"\'\s<>]*', html))
    vids |= set(re.findall(r'https?://[^"\'\s<>]+\.m3u8[^"\'\s<>]*', html))
    vids = {v.replace("\\u002F", "/") for v in vids}
    if vids:
        print(u[:55], "->", len(vids))
        for v in sorted(vids)[:3]:
            print("   ", v[:130])
        found += sorted(vids)
print("总数:", len(found))
open("/home/lawson/vlm-data/raw/_meta/news_videos.txt", "w").write("\n".join(found))
