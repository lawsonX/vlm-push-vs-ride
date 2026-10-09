#!/usr/bin/env python3
"""用 huggingface_hub + hf_transfer 按优先级下载模型权重。

hf_transfer 是 Rust 写的下载器：单文件内多线程分块，能跑满带宽，
且每块独立校验，比 curl 单连接稳定得多。走 hf-mirror 镜像。

用法：python dl_models_hf.py
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", "/home/lawson/vlm-active/models/Qwen2.5-VL-3B-Instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "/home/lawson/vlm-active/models/Qwen2.5-VL-7B-Instruct"),
    ("Qwen/Qwen3.5-0.8B", "/home/lawson/vlm-active/models/Qwen3.5-0.8B"),
    ("Qwen/Qwen3.5-2B", "/home/lawson/vlm-active/models/Qwen3.5-2B"),
    ("openai/clip-vit-base-patch32", "/home/lawson/vlm-active/models/clip-vit-base-patch32"),
]

for repo, dest in MODELS:
    print(f"=== {repo} -> {dest}", flush=True)
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            max_workers=4,
        )
        print(f"=== {repo} 完成", flush=True)
    except Exception as e:
        print(f"=== {repo} 失败: {e}", flush=True)
print("ALL-MODELS-DONE")
