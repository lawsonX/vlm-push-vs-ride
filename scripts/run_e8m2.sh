#!/bin/bash
# E8-v2 DPO 训练启动器 v2：用 setsid 脱离会话，防 WSL 会话回收
cd /root
setsid nohup /hdd2/xiaolirui/conda_envs/vlm-lab/bin/python -u scripts/train_dpo_manual_v2.py \
  --model /home/lawson/vlm-active/ckpt/e2_q25vl_v1/final_fixed2 \
  --pairs /home/lawson/vlm-active/sft/dpo_pairs_v2.jsonl \
  --out /home/lawson/vlm-active/ckpt/e8_manual_v2 \
  --lr 5e-5 --epochs 2 \
  > /home/lawson/vlm-active/logs/train_e8m2.log 2>&1 < /dev/null &
echo "PID=$!"
disown
