#!/bin/bash
# GPU 看门队列：发现显存空闲（邻居任务结束）自动开跑排队实验。
# 当前队列：E8 DPO（分钟级）→ 写标志文件通知后续 M5/E9。
# 注意：只在本项目的 vlm-lab 环境跑，不碰邻居任何东西。

PY=/home/lawson/miniforge3/envs/vlm-lab/bin/python
FREE_MB=5000   # 显存占用低于 5GB 视为空闲（2080Ti 共 22GB）
LOG=/home/lawson/vlm-active/logs/gpu_queue.log

echo "$(date) watcher started" >> $LOG
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$USED" -lt "$FREE_MB" ]; then
        echo "$(date) GPU 空闲（占用 ${USED}MB），开始 E8 DPO" >> $LOG
        $PY scripts/train_dpo.py \
            --model /home/lawson/vlm-active/ckpt/e2_q25vl_v1/final_fixed2 \
            --pairs /home/lawson/vlm-active/sft/dpo_pairs_v1.jsonl \
            --out /home/lawson/vlm-active/ckpt/e8_dpo_v1 >> $LOG 2>&1
        echo "$(date) E8 DPO 结束，标志文件已写" >> $LOG
        touch /home/lawson/vlm-active/GPU_FREE_SIGNAL
        break
    fi
    sleep 300
done
