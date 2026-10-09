#!/bin/bash
# 等 Qwen2.5-VL 训练结束后，自动依次跑基线/微调评测
PY=/root/miniconda3/envs/vlm-lab/bin/python
while ps aux | grep -q "[t]rain_sft.py --model /mnt/e/vlm-data/models/Qwen2.5-VL"; do
    sleep 30
done
sleep 10
echo "=== training done, start evals ==="
$PY /root/vlm-lab/eval/eval_local.py \
    --ckpt /mnt/e/vlm-data/models/Qwen2.5-VL-3B-Instruct \
    --annotations /tmp/test_balanced.jsonl \
    --frames-dir /mnt/d/vlm-active/frames/trafficqa \
    --out /mnt/d/vlm-active/results/local_q25_base.jsonl
$PY /root/vlm-lab/eval/eval_local.py \
    --ckpt /mnt/d/vlm-active/ckpt/e2_q25vl_v1/final \
    --annotations /tmp/test_balanced.jsonl \
    --frames-dir /mnt/d/vlm-active/frames/trafficqa \
    --out /mnt/d/vlm-active/results/local_q25_e2lora.jsonl
echo "=== all evals done ==="
