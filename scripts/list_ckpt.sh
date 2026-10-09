#!/bin/bash
for d in /mnt/d/vlm-active/ckpt/*/; do
  echo "== $d"
  ls "$d" 2>/dev/null | head -6
done
