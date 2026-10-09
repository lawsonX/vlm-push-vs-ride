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
    curl -sSL --retry 5 --retry-delay 3 -C - --create-dirs -m 300 -o "$out" "$BASE/$f"
    # 校验真身：mp4 文件开头是 ftyp，下成网页重定向页就删掉重来
    if [ -f "$out" ] && ! head -c 12 "$out" | grep -q "ftyp"; then
        rm -f "$out"
        return 1
    fi
}
export -f download_one
export DEST BASE

# 外层循环：xargs 一轮跑完后重扫缺失文件，直到全部下完或超过最大轮数。
# hf-mirror 对并发视频下载会 reset 多余连接（curl 52 空响应），并发调低 + 多轮补洞。
for round in $(seq 1 30); do
    missing=$(cat "$LIST" | xargs -P 3 -I{} bash -c 'download_one "$@"' _ {} | wc -l)
    # xargs 不统计失败数，改用文件存在性统计
    n_missing=0
    while read -r f; do
        [ -z "$f" ] && continue
        out="$DEST/$f"
        [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 10240 ] || n_missing=$((n_missing+1))
    done < "$LIST"
    echo "round $round 完成，仍缺 $n_missing 个"
    [ "$n_missing" -eq 0 ] && break
    sleep 5
done
echo "DONE-RUN"
