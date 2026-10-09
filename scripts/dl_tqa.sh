#!/bin/bash
# 并发下载 TrafficQA 视频。可重复运行，已存在的文件会跳过（大小校验）。
LIST="${TQA_LIST:-/home/lawson/vlm-data/raw/_meta/tqa_list.txt}"
BASE="https://hf-mirror.com/datasets/fcxfcx/TrafficQA/resolve/main"
DEST="${TQA_DEST:-/home/lawson/vlm-data/raw/TrafficQA}"

download_one() {
    f="$1"
    out="$DEST/$f"
    # 已存在且大于 10KB 认为下完了
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 10240 ]; then
        return 0
    fi
    curl -sSL --retry 5 -C - --create-dirs -m 180 -o "$out" "$BASE/$f"
    # 校验真身：mp4 文件开头是 ftyp，下成网页重定向页就删掉重来
    if [ -f "$out" ] && ! head -c 12 "$out" | grep -q "ftyp"; then
        rm -f "$out"
        return 1
    fi
}
export -f download_one
export DEST BASE

cat "$LIST" | xargs -P 8 -I{} bash -c 'download_one "$@"' _ {}
echo "DONE-RUN"
