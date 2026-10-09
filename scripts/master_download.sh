#!/bin/bash
# 总下载器：依次下 TrafficQA -> BDD100K -> Cityscapes，循环补漏直到下完。
# 每阶段把进度写进 /home/lawson/vlm-data/raw/_meta/download_status.txt，方便随时查看。
META=/home/lawson/vlm-data/raw/_meta
RAW=/home/lawson/vlm-data/raw
LOG=~/download_master.log

dl_list() {  # $1=列表文件 $2=目标根目录 $3=基础URL $4=阶段名
    local list="$1" dest="$2" base="$3" name="$4"
    while true; do
        local total=$(wc -l < "$list")
        local done=$(find "$dest" -type f -size +10k 2>/dev/null | wc -l)
        echo "$(date +%H:%M:%S) $name: $done/$total" >> "$META/download_status.txt"
        [ "$done" -ge "$total" ] && return 0
        cat "$list" | xargs -P 8 -I{} bash -c '
            f="{}"; out="'"$dest"'/$f"
            # 跳过条件：大小够 且 不是网页（防重定向页/半截文件被当已完成）
            if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 10240 ]; then
                head -c 200 "$out" | grep -qi "redirect\|<html" || exit 0
            fi
            curl -sSL --retry 5 --retry-all-errors -C - --create-dirs -m 300 -o "$out" "'"$base"'/$f" 2>>'"$LOG"'
            head -c 200 "$out" 2>/dev/null | grep -qi "redirect\|<html" && rm -f "$out"
        '
    done
}

echo "$(date) master start" >> "$LOG"

# 阶段 1：TrafficQA
dl_list "$META/tqa_list.txt" "$RAW/TrafficQA" "https://hf-mirror.com/datasets/fcxfcx/TrafficQA/resolve/main" "TrafficQA"

# 阶段 2：BDD100K（先拉清单）
if [ ! -s "$META/bdd_list.txt" ]; then
    for i in 1 2 3 4 5; do
        curl -sS -m 60 --retry 3 --retry-all-errors "https://hf-mirror.com/api/datasets/dgural/bdd100k" -o "$META/bdd_repo.json" && break
        sleep 3
    done
    python3 - "$META/bdd_repo.json" "$META/bdd_list.txt" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
files = [s["rfilename"] for s in d.get("siblings", [])]
jpgs = sorted(f for f in files if f.startswith("data/") and f.endswith(".jpg"))
open(sys.argv[2], "w").write("\n".join(jpgs) + "\n")
print("bdd files:", len(jpgs))
PYEOF
fi
dl_list "$META/bdd_list.txt" "$RAW/BDD100K" "https://hf-mirror.com/datasets/dgural/bdd100k/resolve/main" "BDD100K"

# 阶段 3：Cityscapes parquet（镜像 Chris1/cityscapes）
if [ ! -s "$META/cs_list.txt" ]; then
    for i in 1 2 3 4 5; do
        curl -sS -m 60 --retry 3 --retry-all-errors "https://hf-mirror.com/api/datasets/Chris1/cityscapes" -o "$META/cs_repo.json" && break
        sleep 3
    done
    python3 - "$META/cs_repo.json" "$META/cs_list.txt" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
files = [s["rfilename"] for s in d.get("siblings", [])]
pqs = sorted(f for f in files if f.endswith(".parquet"))
open(sys.argv[2], "w").write("\n".join(pqs) + "\n")
print("cityscapes files:", len(pqs))
PYEOF
fi
dl_list "$META/cs_list.txt" "$RAW/Cityscapes" "https://hf-mirror.com/datasets/Chris1/cityscapes/resolve/main" "Cityscapes"

echo "$(date) ALL DONE" >> "$LOG"
echo "$(date +%H:%M:%S) ALL DONE" >> "$META/download_status.txt"
