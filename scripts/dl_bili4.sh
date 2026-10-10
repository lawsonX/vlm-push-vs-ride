#!/bin/bash
# 第四轮 B 站批量下载：261 个 BV（自然场景关键词：路口监控/过马路/等红灯/电梯/天桥/地铁口等）
while read -r bv; do
    [ -z "$bv" ] && continue
    if ls /mnt/e/vlm-data/raw/bili_videos/"$bv".* >/dev/null 2>&1; then
        echo "skip $bv"
        continue
    fi
    /root/miniconda3/envs/vlm-lab/bin/yt-dlp \
        --cookies /root/bili_cookie_netscape.txt \
        --merge-output-format mp4 --remux-video mp4 \
        -o "/mnt/e/vlm-data/raw/bili_videos/%(id)s.%(ext)s" \
        --no-playlist \
        "https://www.bilibili.com/video/$bv" >> /root/bili_dl4.log 2>&1
    echo "done $bv" >> /root/bili_dl4.log
done < /root/bili_bvs4.txt
echo ALL_DONE >> /root/bili_dl4.log
