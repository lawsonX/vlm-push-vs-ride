#!/bin/bash
# 第三轮 B 站批量下载：310 个 BV（推行/骑行/被盗/执法等关键词）
# 480p 以内省带宽；已存在的自动跳过
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
        "https://www.bilibili.com/video/$bv" >> /root/bili_dl3.log 2>&1
    echo "done $bv" >> /root/bili_dl3.log
done < /root/bili_bvs3.txt
echo ALL_DONE >> /root/bili_dl3.log
