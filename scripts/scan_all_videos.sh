#!/bin/bash
# E9 扩充 v2：用完整文件主名（含 .fXXXXX 后缀）重扫
VIDS=$(ls /mnt/e/vlm-data/raw/bili_videos/*.mp4 | sed 's|.*/||')
echo "扫描: $(echo $VIDS | wc -w) 个视频"
mkdir -p /mnt/d/vlm-active/frames/natr2
/root/miniconda3/envs/vlm-lab/bin/python -u /root/vlm-lab/data_pipeline/scan_transitions.py \
  --video-dir /mnt/e/vlm-data/raw/bili_videos \
  --videos $VIDS \
  --out-dir /mnt/d/vlm-active/frames/natr2 \
  --out /mnt/d/vlm-active/frames/natr2/pairs_transition2.jsonl \
  --step 1.0
echo SCAN_ALL_DONE
