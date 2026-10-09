#!/bin/bash
# 通用 HuggingFace 镜像模型下载器：按仓库名下载全部文件（跳过 .gitattributes）。
# 用法：dl_hf_model.sh <仓库名> <目标目录>
set -u
REPO="$1"
DEST="$2"
BASE="https://hf-mirror.com/$REPO/resolve/main"
API="https://hf-mirror.com/api/models/$REPO"
mkdir -p "$DEST"

for i in 1 2 3 4 5 6 7 8; do
    curl -sS -m 600 --retry 3 --retry-all-errors "$API" -o "$DEST/.repo.json" && break
    echo "$(date +%H:%M:%S) api fetch retry $i" >> "$DEST/.dl.log"
    sleep 5
done

python3 - "$DEST/.repo.json" "$DEST/.files.txt" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
files = [s["rfilename"] for s in d.get("siblings", []) if s["rfilename"] != ".gitattributes"]
open(sys.argv[2], "w").write("\n".join(files) + "\n")
print("files:", len(files))
PYEOF

total=$(wc -l < "$DEST/.files.txt")
echo "$(date +%H:%M:%S) $REPO 共 $total 个文件" >> "$DEST/.dl.log"

cat "$DEST/.files.txt" | xargs -P 4 -I{} bash -c '
    f="{}"; out="'"$DEST"'/$f"
    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 100 ]; then
        head -c 200 "$out" | grep -qi "redirect\|<html" || exit 0
    fi
    curl -sSL --retry 8 --retry-all-errors -C - --create-dirs -m 3600 -o "$out" "'"$BASE"'/$f" 2>>"'"$DEST"'/.dl.log"
    head -c 200 "$out" 2>/dev/null | grep -qi "redirect\|<html" && rm -f "$out"
'
echo "$(date +%H:%M:%S) $REPO done" >> "$DEST/.dl.log"
