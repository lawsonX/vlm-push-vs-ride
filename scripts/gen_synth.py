#!/usr/bin/env python3
"""合成补充数据：用 qwen-image 生成监控视角的推行/骑行画面。

为什么：真实数据（车载视频）里推行场景几乎不存在（标注 98 张 0 推行）。
合成数据有 domain gap，但能把「推行」这个概念的画面补进池子，
对训练尤其有价值（测试集仍会以真实图为主，合成图单独标记来源）。

每张图记录 prompt 和来源标签（synth_push / synth_ride），
生成后同样过 YOLO 预筛确认人+车同框，再进标注池由人工最终确认。

用法：
    python gen_synth.py --out /mnt/d/vlm-active/frames/synth --n-push 100 --n-ride 50
"""

import argparse
import base64
import json
import time
from pathlib import Path

import requests

KEY = Path(__file__).parent.parent.joinpath("api_testing/api_key.txt").read_text().strip()
SUBMIT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
TASK = "https://dashscope.aliyuncs.com/api/v1/tasks/"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def gen_one(prompt, out_path, retries=3):
    """提交文生图任务并轮询结果，成功写文件返回 True。"""
    for a in range(retries):
        try:
            r = requests.post(SUBMIT, headers=HEADERS, json={
                "model": "qwen-image-2.0",
                "input": {"prompt": prompt},
                "parameters": {"size": "1440*1080", "n": 1},
            }, timeout=60)
            if r.status_code != 200:
                print("submit fail:", r.text[:150], flush=True)
                time.sleep(3)
                continue
            task_id = r.json()["output"]["task_id"]
            for _ in range(60):  # 最多等 3 分钟
                time.sleep(3)
                t = requests.get(TASK + task_id, headers=HEADERS, timeout=30).json()
                status = t["output"]["task_status"]
                if status == "SUCCEEDED":
                    url = t["output"]["results"][0]["url"]
                    img = requests.get(url, timeout=60).content
                    out_path.write_bytes(img)
                    return True
                if status in ("FAILED", "CANCELED"):
                    print("task fail:", str(t)[:150], flush=True)
                    break
        except Exception as e:
            print("err:", str(e)[:120], flush=True)
        time.sleep(3)
    return False

BASE = ("监控摄像头固定机位画面，{scene}，{time}，真实安防监控画质，"
        "略有色块噪点，画面角落有时间戳水印，{person}，")

PUSH_SCENES = [
    "一名中年男子在小区道路上推行一辆电动自行车，双手扶把，双脚着地走路，臀部不在车座上",
    "一名外卖员推行着没电的电动车走在人行道边，身体侧向一边，一步一步走",
    "一位老人推着一辆自行车慢慢走过路口，人完全站在车的一侧",
    "一名女子推行共享单车走在非机动车道边缘，双脚交替着地",
    "一名男子雨天推行电瓶车，一手扶把一手撑伞，车在缓慢移动",
    "两名行人其中一人推行着电动自行车另一人步行陪同，推行的人双脚走路",
    "一名男子在地下车库坡道推行电动车，身体直立走路，车倾斜着",
    "一名穿校服的男孩推着自行车走在放学路上，人完全没坐在车上",
]
RIDE_SCENES = [
    "一名男子骑着电动自行车行驶在非机动车道上，臀部坐在车座上，双脚踩在踏板上",
    "一名外卖员骑着电动车穿过路口，身体前倾坐着，脚踩踏板",
    "一名女子骑着共享单车在道路上骑行，坐姿正常，双脚在踩踏",
    "一名男子夜间骑着电瓶车，车灯亮着，人坐在车上",
    "一对骑电动车的男女，驾驶员坐着握把，后座载人，车在行驶",
    "一名老人骑着自行车沿路边骑行，屁股坐在车座上踩踏板",
]
TIMES = ["白天", "傍晚", "夜晚路灯下"]
SCENES_BG = ["小区门口道路", "城市路口", "人行道旁", "非机动车道", "商业街路边", "地下车库出入口"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-push", type=int, default=100)
    ap.add_argument("--n-ride", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "gen_log.jsonl", "w", encoding="utf-8")

    import itertools, random
    random.seed(0)
    combos = list(itertools.product(PUSH_SCENES, TIMES, SCENES_BG))
    random.shuffle(combos)

    def run(label, n, combos):
        made = 0
        for i, (action, t, bg) in enumerate(combos):
            if made >= n:
                break
            prompt = BASE.format(scene=bg, time=t, person=action)
            fid = f"synth_{label}_{made:04d}"
            ok = gen_one(prompt, out / f"{fid}.jpg")
            if ok:
                log.write(json.dumps({"frame_id": fid, "source": f"synth_{label}", "prompt": prompt},
                                     ensure_ascii=False) + "\n")
                log.flush()
                made += 1
                if made % 10 == 0:
                    print(f"{label}: {made}/{n}", flush=True)

    run("push", args.n_push, combos)
    run("ride", args.n_ride, [(a, t, b) for a in RIDE_SCENES for t in TIMES for b in SCENES_BG])
    log.close()
    print("done")


if __name__ == "__main__":
    main()
