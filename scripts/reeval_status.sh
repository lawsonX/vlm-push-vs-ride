#!/bin/bash
# 输出重评状态到 D 盘文件（Windows 侧直接可读）
OUT=/mnt/d/vlm-active/results/reeval_status.txt
{
  echo "time: $(date)"
  echo "--- 每个输出文件行数 ---"
  for f in /mnt/d/vlm-active/results/local2_*.jsonl; do
    echo "$(basename $f): $(wc -l < $f 2>/dev/null || echo missing)"
  done
  echo "--- 日志关键行 ---"
  grep -aE "=====|召回|准确率|Traceback|Error" /root/reeval3.log 2>/dev/null | tail -15
} > $OUT 2>&1
