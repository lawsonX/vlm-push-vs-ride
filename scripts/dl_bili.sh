#!/bin/bash
# 按 BV 号列表下载 B 站视频（480p 以内，省带宽）
mkdir -p /mnt/e/vlm-data/raw/bili_videos
while read -r bv; do
    [ -z "$bv" ] && continue
    if ls /mnt/e/vlm-data/raw/bili_videos/"$bv".* >/dev/null 2>&1; then
        echo "skip $bv (已存在)"
        continue
    fi
    /root/miniconda3/envs/vlm-lab/bin/yt-dlp \
        --cookies /root/bili_cookie_netscape.txt \
        -f "bv*[height<=480]+ba/b[height<=480]/worst" \
        --merge-output-format mp4 \
        -o "/mnt/e/vlm-data/raw/bili_videos/%(id)s.%(ext)s" \
        --no-playlist \
        "https://www.bilibili.com/video/$bv" >> /root/bili_dl.log 2>&1
    echo "done $bv" >> /root/bili_dl.log
done < /root/bili_bvs.txt
echo ALL_DONE >> /root/bili_dl.log
