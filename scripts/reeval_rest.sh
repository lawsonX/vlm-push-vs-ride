#!/bin/bash
# 批量重评（续）：跳过已完成的 base 和已确认损坏的 e2lora(final)
set -x
PY=/root/miniconda3/envs/vlm-lab/bin/python
EV=/root/vlm-lab/eval/eval_local_nan.py
ANN=/tmp/test_balanced.jsonl
FR=/mnt/d/vlm-active/frames/trafficqa
RES=/mnt/d/vlm-active/results
CK=/mnt/d/vlm-active/ckpt

run() {  # run <ckpt> <tag>
  echo "===== $2 ====="
  $PY -u $EV --ckpt "$1" --annotations $ANN --frames-dir $FR \
      --out $RES/local2_$2.jsonl --all-fp32 2>&1 | grep -Ev "Loading|it/s"
  sleep 10  # 让显存/内存回落，防连锁 OOM
}

run $CK/e2_q25vl_v1/final_fixed2 q25_e2fixed
run $CK/e5_mix72/final q25_e5mix72
run $CK/e5_mixfull/final q25_e5mixfull
run $CK/e7_chain_v1/final_fixed q25_e7fixed
run $CK/e8_manual/final q25_e8dpo
run /mnt/d/vlm-active/models/Qwen3.5-0.8B q35_base
run $CK/e2_q35_gpu/final q35_e2v3
echo ALL_DONE
