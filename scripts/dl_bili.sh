#!/bin/bash
# 按 BV 号列表下载 B 站视频（480p 以内，省带宽）
mkdir -p /home/lawson/vlm-data/raw/bili_videos
while read -r bv; do
    [ -z "$bv" ] && continue
    if ls /home/lawson/vlm-data/raw/bili_videos/"$bv".* >/dev/null 2>&1; then
        echo "skip $bv (已存在)"
        continue
    fi
    /home/lawson/miniforge3/envs/vlm-lab/bin/yt-dlp \
        --cookies /home/lawson/vlm-data/bili_cookie_netscape.txt \
        -f "bv*[height<=480]+ba/b[height<=480]/worst" \
        --merge-output-format mp4 \
        -o "/home/lawson/vlm-data/raw/bili_videos/%(id)s.%(ext)s" \
        --no-playlist \
        "https://www.bilibili.com/video/$bv" >> /home/lawson/vlm-active/logs/bili_dl.log 2>&1
    echo "done $bv" >> /home/lawson/vlm-active/logs/bili_dl.log
done < /home/lawson/vlm-data/raw/_meta/bili_bvs.txt
echo ALL_DONE >> /home/lawson/vlm-active/logs/bili_dl.log
