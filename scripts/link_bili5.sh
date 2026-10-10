#!/bin/bash
mkdir -p /mnt/e/vlm-data/raw/bili_videos_5
while read -r bv; do
    [ -z "$bv" ] && continue
    for f in /mnt/e/vlm-data/raw/bili_videos/"$bv".*.mp4; do
        [ -e "$f" ] && ln -sf "$f" /mnt/e/vlm-data/raw/bili_videos_5/
    done
done < /root/bili_bvs5.txt
ls /mnt/e/vlm-data/raw/bili_videos_5/ | wc -l
