#!/bin/bash
for f in local_q25_base local_q25_e2fixed local_q25_e2lora local_q25_e5_mix72 local_q25_e7fixed local_q25_e8dpo; do
  echo "== $f"
  grep "sg_03_016" "/mnt/d/vlm-active/results/$f.jsonl" | head -c 300
  echo
done
