#!/bin/bash
# E9 扩充 v2：用完整文件主名（含 .fXXXXX 后缀）重扫
VIDS=$(ls /home/lawson/vlm-data/raw/bili_videos/*.mp4 | sed 's|.*/||')
echo "扫描: $(echo $VIDS | wc -w) 个视频"
mkdir -p /home/lawson/vlm-active/frames/natr2
/hdd2/xiaolirui/conda_envs/vlm-lab/bin/python -u scripts/scan_transitions.py \
  --video-dir /home/lawson/vlm-data/raw/bili_videos \
  --videos $VIDS \
  --out-dir /home/lawson/vlm-active/frames/natr2 \
  --out /home/lawson/vlm-active/frames/natr2/pairs_transition2.jsonl \
  --step 1.0
echo SCAN_ALL_DONE
