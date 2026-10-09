#!/bin/bash
for d in /home/lawson/vlm-active/ckpt/*/; do
  echo "== $d"
  ls "$d" 2>/dev/null | head -6
done
