#!/usr/bin/env python3
"""合成监控场景图批量生成器（百炼 bl CLI 驱动）。

用法：
    python gen_synth.py --out-dir /mnt/e/vlm-data/synth/iter3 --per-scenario 8

设计要点（都是踩坑换来的）：
- 活动描述放提示词最前面，风格描述放后面（先生成对的姿势，再套监控风）
- 用 subprocess 列表传参，不走 shell 字符串拼接（中文标点经 bash 变量传递会偶发出问题）
- 每张请求 n=4，请求之间 sleep（文生图 QPM 低）
- --watermark false 关闭水印（qwen-image 系列支持）
- 不把 --prompt-extend false 传给 qwen-image 同步模型（会得到无关回退图）
- 生成结果必须人工/视觉验收：本脚本只负责采样，标签以验收时看到的实际内容为准
"""

import argparse
import subprocess
import time
from pathlib import Path

STYLE = ("这是城市治安监控摄像头拍到的街景画面：镜头从路灯杆高处向下俯拍，广角畸变，"
         "人物只占画面的一小部分，画面有噪点颗粒感、色彩偏灰、边缘模糊、有视频压缩痕迹。"
         "背景是城市道路和斑马线。人物穿着日常、不看镜头、动作自然。")
ROLES = ["穿蓝色外套的中年男子", "穿深色运动服的年轻女子", "背双肩包的年轻学生",
         "穿黄色外卖服的骑手", "穿灰色夹克的老年男子", "拎塑料袋的中年女子"]

ACT = {
    "push_ebike": ("站在一辆电动车旁边，双手握住车把，身体完全站立，双脚交替在地上走路，"
                   "正在步行推行这辆电动车，胯部悬空、没有坐在车座上",
                   "坐在车上,骑车的姿势,脚踩踏板,摆拍,艺术写真"),
    "ride_ebike": ("端正地坐在电动车座垫上，双手握着车把，双脚踩在脚踏板上，"
                   "正在骑这辆电动车行驶",
                   "推车,站立走路,人在车旁步行,摆拍,艺术写真"),
    "push_bike": ("站在一辆自行车旁边，双手扶着车把，身体完全站立，双脚交替在地上走路，"
                  "正在步行推行这辆自行车，没有坐在车座上",
                  "骑车,坐在车座上,脚踩踏板,摆拍,艺术写真"),
    "ride_bike": ("坐在自行车座垫上，双手握着车把，双脚踩着脚踏板，正在骑自行车",
                  "推车,站立走路,摆拍,艺术写真"),
}


def build_prompt(activity: str, role: str) -> str:
    return f"画面中央，一名{role}{activity}。{STYLE}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-scenario", type=int, default=8, help="每个场景要多少张（n=4 每次）")
    ap.add_argument("--model", default="qwen-image-2.0")
    ap.add_argument("--sleep", type=float, default=8.0, help="请求间隔秒数（QPM 限制）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_calls = (args.per_scenario + 3) // 4
    role_idx = 0

    for scenario, (activity, negative) in ACT.items():
        for call in range(n_calls):
            role = ROLES[role_idx % len(ROLES)]
            role_idx += 1
            cmd = ["bl", "image", "generate", "--model", args.model,
                   "--size", "16:9", "--n", "4", "--watermark", "false",
                   "--negative-prompt", negative,
                   "--prompt", build_prompt(activity, role),
                   "--out-dir", str(out), "--out-prefix", f"{scenario}_r{call}"]
            print(f"[{scenario} 第{call+1}/{n_calls}次] 角色: {role}")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            tail = (r.stdout + r.stderr).strip().splitlines()
            print("  " + (tail[-1] if tail else "(no output)"))
            time.sleep(args.sleep)
    print(f"完成，输出目录: {out}")


if __name__ == "__main__":
    main()
