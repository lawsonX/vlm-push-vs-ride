#!/usr/bin/env python3
"""item 云端质检：dashscope qwen3-vl-flash 四问筛选（本地 GPU 半残时的替代品）。

问题同 vlm_qc.py（画面类型/两轮车有无/人车关系/清晰度），解析逻辑复用。
输出 qc.jsonl 到 --frames-dir 下。可断点续跑。

用法：python vlm_qc_cloud.py --frames-dir <item目录> [--workers 6]
"""
import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

QC_PROMPT = (
    "请观察这张图片并依次回答六个问题，每行一个答案，格式「编号.结论」：\n"
    "1. 画面类型：是真实监控录像或街面实拍，还是广告、宣传片、教学视频、摆拍写真、新闻采访或动漫？\n"
    "2. 画面中是否有自行车、电动车或摩托车？\n"
    "3. 是否有人与两轮车同框？如果有，人是骑在车上、在车旁推行、还是站在车旁没有操控？\n"
    "4. 画面清晰度和人物大小：清晰度分高/中/低；画面中主要人物大约占画面高度的百分之几？\n"
    "5. 拍摄视角：固定安装的监控/航拍俯视画面，还是手持跟拍（包括从背后跟拍）、第一人称骑行视角、人物采访特写、或其他？\n"
    "6. 人物身体朝向：主要是正面、侧面、还是背面朝向镜头？看不清就答无法判断。"
)


def parse_qc(text):
    t = text or ""

    def grab(n):
        m = re.search(rf"{n}\s*[.、)]\s*(.+)", t)
        return m.group(1).strip() if m else ""

    q1 = grab(1)
    bad_kw = ("广告", "宣传", "教学", "摆拍", "写真", "动漫", "海报", "漫画", "艺术照")
    good_kw = ("监控", "实拍", "抓拍", "街景", "录像", "现场", "路口", "街道", "道路")
    if any(k in q1 for k in bad_kw) and not any(k in q1 for k in good_kw):
        ftype = "bad"
    elif any(k in q1 for k in good_kw):
        ftype = "good"
    else:
        ftype = "unknown"

    q2 = grab(2)
    if "没有" in q2 or "无" in q2[:6]:
        twowheel = "no"
    elif any(k in q2 for k in ("自行车", "电动车", "摩托车", "单车", "电瓶车", "有")):
        twowheel = "yes"
    else:
        twowheel = "unknown"

    q3 = grab(3)
    if "推" in q3:
        activity = "push"
    elif "骑" in q3:
        activity = "ride"
    elif "站" in q3 or "没有操控" in q3:
        activity = "stand"
    elif "没有" in q3 or "无人" in q3 or "无" in q3[:6]:
        activity = "none"
    else:
        activity = "unknown"

    q4 = grab(4)
    if "高" in q4[:4]:
        clarity = "high"
    elif "中" in q4[:4]:
        clarity = "mid"
    elif "低" in q4[:4]:
        clarity = "low"
    else:
        clarity = "unknown"
    pct = None
    m = re.search(r"(\d+)\s*[%％]", q4)
    if m:
        pct = int(m.group(1))

    return {"type": ftype, "twowheel": twowheel, "activity": activity,
            "clarity": clarity, "person_pct": pct,
            "view": parse_view(t), "facing": parse_facing(t)}


def parse_view(text):
    m = re.search(r"5\s*[.、)]\s*(.+)", text or "")
    q5 = m.group(1) if m else ""
    bad = ("跟拍", "手持", "第一人称", "POV", "采访", "自拍", "行车记录仪", "运动相机")
    good = ("固定", "监控", "航拍", "俯视", "俯瞰", "高空", "探头", "固定机位")
    if any(k in q5 for k in bad) and not any(k in q5 for k in good):
        return "other"
    if any(k in q5 for k in good):
        return "surveillance"
    return "unknown"


def parse_facing(text):
    m = re.search(r"6\s*[.、)]\s*(.+)", text or "")
    q6 = m.group(1) if m else ""
    if "背" in q6:
        return "back"
    if "侧" in q6:
        return "side"
    if "正" in q6:
        return "front"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    d = Path(args.frames_dir)
    out_path = d / "qc.jsonl"
    key = Path("/root/vlm-lab/api_testing/api_key.txt").read_text().strip()
    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    files = sorted(d.glob("*.jpg"))
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done.add(json.loads(l)["frame"])
            except Exception:
                pass
    todo = [f for f in files if f.name not in done]
    print(f"质检: {len(todo)}/{len(files)}", flush=True)

    fout = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()

    def work(fp):
        b64 = base64.b64encode(fp.read_bytes()).decode()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": QC_PROMPT},
        ]}]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="qwen3-vl-plus", messages=msgs,
                    max_tokens=220, temperature=0.0)
                text = resp.choices[0].message.content
                rec = {"frame": fp.name, "raw": text[:500]}
                rec.update(parse_qc(text))
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                return
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "timeout" in err.lower():
                    time.sleep(2 ** attempt)
                    continue
                if attempt == 3:
                    with lock:
                        fout.write(json.dumps({"frame": fp.name, "raw": f"ERROR:{err}"[:200],
                                               "type": "unknown", "twowheel": "unknown",
                                               "activity": "unknown", "clarity": "unknown",
                                               "person_pct": None}, ensure_ascii=False) + "\n")
                        fout.flush()
                time.sleep(2)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"done {len(todo)} in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
