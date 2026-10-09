#!/bin/bash
# 批量重评：用 NaN 安全版评测脚本重跑历史全部检查点
set -x
PY=/hdd2/xiaolirui/conda_envs/vlm-lab/bin/python
EV=scripts/eval_local_nan.py
ANN=/tmp/test_balanced.jsonl
FR=/home/lawson/vlm-active/frames/trafficqa
RES=/home/lawson/vlm-active/results
CK=/home/lawson/vlm-active/ckpt

run() {  # run <ckpt> <tag>
  echo "===== $2 ====="
  $PY -u $EV --ckpt "$1" --annotations $ANN --frames-dir $FR \
      --out $RES/local2_$2.jsonl --all-fp32 2>&1 | grep -Ev "Loading|it/s"
}

run /home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct q25_base
run $CK/e2_q25vl_v1/final q25_e2lora
run $CK/e2_q25vl_v1/final_fixed2 q25_e2fixed
run $CK/e5_mix72/final q25_e5mix72
run $CK/e5_mixfull/final q25_e5mixfull
run $CK/e7_chain_v1/final_fixed q25_e7fixed
run $CK/e8_manual/final q25_e8dpo
run /home/lawson/vlm-active/models/Qwen3.5-0.8B q35_base
run $CK/e2_q35_gpu/final q35_e2v3
echo ALL_DONE
