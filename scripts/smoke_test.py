#!/usr/bin/env python3
"""dashscope 视觉模型冒烟测试：发一张图，验证视觉调用链路通不通。

用法：
    python smoke_test.py                # 用内置纯色图测试
    python smoke_test.py 某张图片.jpg    # 用指定图片测试

目的：确认 base_url、key、图片编码、模型名都配对，后面批量测试才能放心跑。
"""
import base64
import os
import sys
import tempfile

from openai import OpenAI

# 凭据从同目录的 api_key.txt 读（文件放在 WSL 里，不进 git、不写死在代码里）
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
if not os.path.exists(KEY_FILE):
    sys.exit(f"找不到 {KEY_FILE}，请把 dashscope 的 key 单独放这个文件里（只放一行 key 本身）")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-vl-flash"


def make_test_image() -> str:
    """画一张蓝底黄圆的简单图，存到临时文件，返回路径。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (40, 90, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 150, 350, 350), fill=(250, 210, 60))
    path = os.path.join(tempfile.gettempdir(), "smoke_test.png")
    img.save(path)
    return path


def main() -> None:
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()

    img_path = sys.argv[1] if len(sys.argv) > 1 else make_test_image()
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(api_key=key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "这张图里有什么？用一两句话回答。"},
                ],
            }
        ],
        max_tokens=100,
    )
    print("模型:", MODEL)
    print("回答:", resp.choices[0].message.content)
    print("用量:", resp.usage)


if __name__ == "__main__":
    main()
