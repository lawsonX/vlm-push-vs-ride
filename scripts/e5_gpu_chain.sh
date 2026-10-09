#!/bin/bash
# E5 GPU 版：数据量/分布曲线的训练+评测全自动链
PY=/home/lawson/miniforge3/envs/vlm-lab/bin/python
TRAIN=scripts/train_sft.py
EVAL=scripts/eval_local.py
BASE=/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct
SFT=/home/lawson/vlm-active/sft
CK=/home/lawson/vlm-active/ckpt
RES=/home/lawson/vlm-active/results
LOG=/home/lawson/vlm-active/logs/e5_gpu.log

run_one () {  # $1=数据文件 $2=实验名 $3=epochs
    echo "=== $2 start $(date +%H:%M) ===" >> $LOG
    $PY $TRAIN --model $BASE --data $SFT/$1 --mode llm_lora \
        --out $CK/$2 --epochs $3 --bs 1 --grad-accum 1 --lr 5e-5 >> $LOG 2>&1
    $PY $EVAL --ckpt $CK/$2/final --annotations /tmp/test_balanced.jsonl \
        --frames-dir /home/lawson/vlm-active/frames/trafficqa \
        --out $RES/local_q25_$2.jsonl >> $LOG 2>&1
    grep -E "召回|准确率" $LOG | tail -3 >> $LOG
    echo "=== $2 done $(date +%H:%M) ===" >> $LOG
}

run_one train_v3_mix72.jsonl e5_mix72 1
run_one train_v3_mixfull.jsonl e5_mixfull 1
echo "=== ALL E5 DONE ===" >> $LOG
