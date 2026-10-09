#!/bin/bash
# 下载 YOLO 分割模型（断点续传 + 无限重试，慢速网络专用）
cd /tmp
for i in $(seq 1 100); do
    curl -sSL -C - --retry 10 --retry-all-errors -m 1800 \
        -o yolov8n-seg.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt" \
        && break
    sleep 3
done
ls -la /tmp/yolov8n-seg.pt >> /root/segdl.log 2>&1
echo DOWNLOAD_FINISHED >> /root/segdl.log
