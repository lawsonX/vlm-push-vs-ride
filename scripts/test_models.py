# -*- coding: utf-8 -*-
import json, urllib.request
key = open("/root/vlm-lab/api_testing/api_key.txt").read().strip()
candidates = ["qwen3-vl-flash", "qwen3-vl-flash-latest", "qwen3-vl-plus",
              "qwen3-vl-plus-latest", "qwen2.5-vl-72b-instruct",
              "qwen2.5-vl-7b-instruct", "qwen-vl-max", "qwen-vl-plus"]
for m in candidates:
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps({"model": m, "messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 5}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            ok = bool(body.get("choices"))
            print(f"{m:30s} -> OK" if ok else f"{m:30s} -> 空响应 {str(body)[:100]}")
    except urllib.error.HTTPError as e:
        print(f"{m:30s} -> HTTP {e.code} {e.read()[:120]}")
    except Exception as e:
        print(f"{m:30s} -> {type(e).__name__} {e}")
