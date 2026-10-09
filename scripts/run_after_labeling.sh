#!/bin/bash
# lawson 标注完成后的一键复测流程（在 WSL 里跑）
# 前置：annotations.jsonl 里测试集外已出现新的 push 标签
set -e
PY=/root/miniconda3/envs/vlm-lab/bin/python
CK=/mnt/d/vlm-active/ckpt
RES=/mnt/d/vlm-active/results
SFT=/mnt/d/vlm-active/sft

echo "== 1. 构建干净训练集 v2（含新标注推行帧） =="
$PY /root/make_clean_train_v2.py || exit 1
N_PUSH=$(grep -c '"answer": "推行"' $SFT/train_clean_v2.jsonl)
if [ "$N_PUSH" -lt 5 ]; then
  echo "推行帧不足 5 张（当前 $N_PUSH），建议继续标注后再跑"
  exit 1
fi

echo "== 2. E2-clean-v2（LLM LoRA，Qwen2.5-VL-3B） =="
$PY -u /root/vlm-lab/finetune/train_sft.py --model /mnt/d/vlm-active/models/Qwen2.5-VL-3B-Instruct \
  --data $SFT/train_clean_v2.jsonl --mode llm_lora --out $CK/e2_clean_v2 \
  --lr 5e-5 --epochs 2 --bs 1 --grad-accum 8 2>&1 | tail -2

echo "== 3. E4b-clean-v2（merger） =="
$PY -u /root/vlm-lab/finetune/train_sft.py --model /mnt/d/vlm-active/models/Qwen2.5-VL-3B-Instruct \
  --data $SFT/train_clean_v2.jsonl --mode projector_only --out $CK/e4b_clean_v2 \
  --lr 5e-5 --epochs 2 --bs 1 --grad-accum 8 2>&1 | tail -2
# 注意：merger 是 fp32 训练，若 final 保存卡住，用 checkpoint-N 重建（见 rebuild_e4bclean.py）

echo "== 4. 评测（fp32） =="
for tag in e2_clean_v2 e4b_clean_v2; do
  $PY -u /root/vlm-lab/eval/eval_local_nan.py --ckpt $CK/$tag/final \
    --annotations /tmp/test_balanced.jsonl --frames-dir /mnt/d/vlm-active/frames/trafficqa \
    --out $RES/local2_q25_${tag}.jsonl --all-fp32 2>&1 | grep -E "召回|准确率"
done

echo "== 5. 若 lawson 写了 evidence，追加 E7-v2 =="
N_EV=$(python3 -c "
import json
rows=[json.loads(l) for l in open('/mnt/d/vlm-active/frames/trafficqa/annotations.jsonl')]
print(sum(1 for r in rows if r.get('evidence','').strip()))")
if [ "$N_EV" -gt 0 ]; then
  $PY /root/vlm-lab/finetune/make_chain_v2.py \
    --annotations /mnt/d/vlm-active/frames/trafficqa/annotations.jsonl \
    --styles $RES/styles_2models.jsonl $RES/web_styles_2models.jsonl $RES/web2_styles_2models.jsonl \
            $RES/direct_5models.jsonl $RES/web2_direct_5models.jsonl \
    --frames-dir /mnt/d/vlm-active/frames/trafficqa \
    --out $SFT/chain_v2.jsonl --nan-list $RES/nan_frames.json
  echo "E7-v2 数据就绪 -> $SFT/chain_v2.jsonl（训练命令另发）"
fi

echo ALL_STEPS_DONE
