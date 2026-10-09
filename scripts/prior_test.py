#!/usr/bin/env python3
"""纯文本先验测试：不看图，模型的先验是不是偏向「骑行」。

验证 H1 的语言先验部分：如果模型不看图都答骑行，说明「骑行」
这个说法在训练数据里占绝对多数，闭着眼睛都会猜骑行。

做法：同一道纯文本问题，每个模型采样 10 次（temperature 调高一点看分布），统计答案。

用法：
    python prior_test.py --models qwen3-vl-flash,qwen-vl-max --out /home/lawson/vlm-active/results/prior.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

QUESTIONS = [
    "监控画面里出现一个人和一辆电瓶车在一起，没有任何其他信息。这个人更可能是在推行还是在骑行？只回答「推行」或「骑行」。",
    "一张照片里有一个人和一辆自行车。不考虑画面细节，仅凭常识，这个人更可能是在推车还是在骑车？只回答「推行」或「骑行」。",
]


def parse_label(text):
    text = text or ""
    lp, lr = text.rfind("推行"), text.rfind("骑行")
    if lp == -1 and lr == -1:
        return "none"
    return "push" if lp > lr else "ride"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    key = (Path(__file__).parent / "api_key.txt").read_text().strip()
    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=60)
    models = [m.strip() for m in args.models.split(",")]

    fout = open(args.out, "w", encoding="utf-8")
    for model in models:
        for qi, q in enumerate(QUESTIONS):
            for s in range(args.samples):
                for attempt in range(4):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": q}],
                            max_tokens=20, temperature=0.8,
                        )
                        ans = resp.choices[0].message.content
                        break
                    except Exception as e:
                        if attempt == 3:
                            ans = f"ERROR: {e}"
                        else:
                            time.sleep(2 ** attempt)
                rec = {"model": model, "qid": qi, "sample": s,
                       "answer": (ans or "")[:100], "label": parse_label(ans)}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
        print(f"{model} done", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
