#!/bin/bash
# 等 E7 训练结束后，自动评测（对比基线已有：push 50% / ride 75%）
PY=/home/lawson/miniforge3/envs/vlm-lab/bin/python
while ps aux | grep -q "[t]rain_sft.py --model /home/lawson/vlm-active/models/Qwen2.5-VL.*e7_chain"; do
    sleep 30
done
sleep 10
echo "=== e7 training done, start eval ==="
$PY scripts/eval_local.py \
    --ckpt /home/lawson/vlm-active/ckpt/e7_chain_v1/final \
    --annotations /tmp/test_balanced.jsonl \
    --frames-dir /home/lawson/vlm-active/frames/trafficqa \
    --out /home/lawson/vlm-active/results/local_q25_e7chain.jsonl
echo "=== e7 eval done ==="
