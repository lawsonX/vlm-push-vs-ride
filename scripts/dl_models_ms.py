#!/usr/bin/env python3
"""用 modelscope 按优先级下载模型权重。

hf-mirror 对本机 IP 限流后改走 modelscope（Qwen 官方维护镜像，国内速度快）。
modelscope 的仓库命名与 HF 一致。

用法：python dl_models_ms.py
"""
from modelscope import snapshot_download

MODELS = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "/home/lawson/vlm-active/models/Qwen2.5-VL-7B-Instruct"),
    ("Qwen/Qwen3.5-0.8B", "/home/lawson/vlm-active/models/Qwen3.5-0.8B"),
    ("Qwen/Qwen3.5-2B", "/home/lawson/vlm-active/models/Qwen3.5-2B"),
    ("AI-ModelScope/clip-vit-base-patch32", "/home/lawson/vlm-active/models/clip-vit-base-patch32"),
]

for repo, dest in MODELS:
    print(f"=== {repo} -> {dest}", flush=True)
    try:
        snapshot_download(repo, local_dir=dest)
        print(f"=== {repo} 完成", flush=True)
    except Exception as e:
        print(f"=== {repo} 失败: {e}", flush=True)
print("ALL-MODELS-DONE")
