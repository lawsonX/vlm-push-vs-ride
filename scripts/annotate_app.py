#!/usr/bin/env python3
"""人工标注 Web UI。

干什么：把抽好的帧按清单展示出来，人逐张标「推行 / 骑行 / 不确定」，
每张可以写备注。标注自动保存，刷新页面不丢，下次打开接着标。

启动：
    python annotate_app.py --frames-dir /home/lawson/vlm-active/frames/trafficqa \
        --manifest /home/lawson/vlm-active/frames/trafficqa/manifest.jsonl --port 8321

打开浏览器访问 http://localhost:8321 即可。

操作方式：
    1 / 2 / 3     标为 推行 / 骑行 / 不确定
    ← / → 或 A / D   上一张 / 下一张
    备注框随时可写，失焦自动保存
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

FRAMES_DIR = None
MANIFEST = []          # 清单行列表，每项 {"frame_id": ..., "source_video": ..., ...}
ANNOTATIONS = {}       # frame_id -> {"label": ..., "note": ..., "evidence": ...}
ANN_PATH = None


def load_state():
    global MANIFEST, ANNOTATIONS
    with open(app.config["MANIFEST_PATH"], "r", encoding="utf-8") as f:
        MANIFEST = [json.loads(line) for line in f if line.strip()]
    if ANN_PATH.exists():
        with open(ANN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ANNOTATIONS[row["frame_id"]] = row


def save_annotation(frame_id, data):
    ANNOTATIONS[frame_id] = {"frame_id": frame_id, **data}
    # 全量重写，文件小，简单可靠
    with open(ANN_PATH, "w", encoding="utf-8") as f:
        for row in ANNOTATIONS.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.route("/")
def index():
    return PAGE


@app.route("/img/<frame_id>")
def img(frame_id):
    # 只允许访问清单里有的帧，防止目录穿越
    if not any(m["frame_id"] == frame_id for m in MANIFEST):
        return "not found", 404
    return send_from_directory(FRAMES_DIR, f"{frame_id}.jpg")


@app.route("/api/list")
def api_list():
    return jsonify({
        "total": len(MANIFEST),
        "done": sum(1 for m in MANIFEST if m["frame_id"] in ANNOTATIONS),
        "items": [
            {
                "frame_id": m["frame_id"],
                "labeled": m["frame_id"] in ANNOTATIONS,
                "meta": {k: v for k, v in m.items() if k != "ahash"},
            }
            for m in MANIFEST
        ],
    })


@app.route("/api/ann/<frame_id>")
def api_get(frame_id):
    row = ANNOTATIONS.get(frame_id, {})
    meta = next((m for m in MANIFEST if m["frame_id"] == frame_id), {})
    return jsonify({"ann": row, "meta": meta})


@app.route("/api/ann/<frame_id>", methods=["POST"])
def api_save(frame_id):
    data = request.get_json(force=True)
    save_annotation(frame_id, {
        "label": data.get("label", ""),
        "note": data.get("note", ""),
        "evidence": data.get("evidence", ""),
    })
    done = sum(1 for m in MANIFEST if m["frame_id"] in ANNOTATIONS)
    return jsonify({"ok": True, "done": done, "total": len(MANIFEST)})


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>推行 vs 骑行 标注</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #1e1e1e; color: #eee; }
  .wrap { display: flex; height: 100vh; }
  .left { flex: 1; display: flex; align-items: center; justify-content: center; background: #111; position: relative; }
  .left img { max-width: 96%; max-height: 92vh; object-fit: contain; }
  .right { width: 320px; padding: 16px; display: flex; flex-direction: column; gap: 12px; background: #252526; }
  .progress { font-size: 15px; }
  .bar { height: 8px; background: #444; border-radius: 4px; overflow: hidden; }
  .bar > div { height: 100%; background: #4caf50; width: 0%; transition: width .2s; }
  .labels { display: flex; gap: 8px; }
  .labels button { flex: 1; padding: 14px 0; font-size: 16px; border: 2px solid #555; border-radius: 8px; background: #333; color: #eee; cursor: pointer; }
  .labels button.sel-push { background: #2e7d32; border-color: #66bb6a; }
  .labels button.sel-ride { background: #1565c0; border-color: #42a5f5; }
  .labels button.sel-unsure { background: #b71c1c; border-color: #ef5350; }
  .labels button.sel-invalid { background: #455a64; border-color: #90a4ae; }
  textarea, input[type=text] { width: 100%; box-sizing: border-box; background: #333; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px; font-size: 14px; }
  .nav { display: flex; gap: 8px; }
  .nav button { flex: 1; padding: 10px 0; font-size: 15px; background: #3a3a3a; color: #eee; border: 1px solid #555; border-radius: 6px; cursor: pointer; }
  .guide { font-size: 13px; line-height: 1.7; background: #2d2d30; border-radius: 8px; padding: 10px 12px; color: #ccc; }
  .guide b { color: #ffd54f; }
  .meta { font-size: 12px; color: #999; }
  .savestate { font-size: 12px; color: #4caf50; min-height: 16px; }
  a { color: #64b5f6; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="left"><img id="pic" src="" alt="帧画面"></div>
  <div class="right">
    <div class="progress"><span id="pos">-</span> / <span id="total">-</span>（已标 <span id="done">-</span>）</div>
    <div class="bar"><div id="barfill"></div></div>
    <div class="labels">
      <button id="b-push" onclick="setLabel('push')">推行<br><small>按 1</small></button>
      <button id="b-ride" onclick="setLabel('ride')">骑行<br><small>按 2</small></button>
      <button id="b-unsure" onclick="setLabel('unsure')">不确定<br><small>按 3</small></button>
      <button id="b-invalid" onclick="setLabel('invalid')">无效/其他<br><small>按 4</small></button>
    </div>
    <div class="savestate" id="savestate"></div>
    <div>
      <div class="meta">备注（画面情况、可疑点）</div>
      <textarea id="note" rows="2"></textarea>
    </div>
    <div id="evblock" style="display:none">
      <div class="meta">证据描述（训练集专用：图中哪些细节支持你的判断）</div>
      <textarea id="evidence" rows="3"></textarea>
    </div>
    <div class="nav">
      <button onclick="nav(-1)">← 上一张 (A)</button>
      <button onclick="nav(1)">下一张 → (D)</button>
    </div>
    <div class="guide">
      <b>判断要领：</b><br>
      <b>看脚</b>：推行时双脚在地上走路；骑行时脚踩踏板。<br>
      <b>看屁股</b>：推行时人不在车座上；骑行时坐在车座上。<br>
      <b>看腿</b>：推行时腿在交替迈步；骑行时双腿弯曲基本不动。<br>
      车歪歪扭扭、人半跨在车上滑行 → 按「不确定」并在备注里写明。
    </div>
    <div class="meta" id="srcinfo"></div>
    <a id="orig" href="#" target="_blank">在新标签页打开原图</a>
  </div>
</div>
<script>
let items = [], idx = 0, dirty = false;

async function init() {
  const d = await (await fetch('/api/list')).json();
  items = d.items; document.getElementById('total').textContent = d.total;
  updateDone(d.done);
  // 默认跳到第一张没标的
  const first = items.findIndex(it => !it.labeled);
  idx = first === -1 ? 0 : first;
  load();
}
function updateDone(n) {
  document.getElementById('done').textContent = n;
  document.getElementById('barfill').style.width = (100 * n / items.length) + '%';
}
async function load() {
  const it = items[idx];
  document.getElementById('pos').textContent = idx + 1;
  document.getElementById('pic').src = '/img/' + it.frame_id;
  document.getElementById('orig').href = '/img/' + it.frame_id;
  document.getElementById('srcinfo').textContent =
    '来源: ' + (it.meta.source_video || '') + '  ' + (it.meta.t_sec || 0) + 's  ' +
    (it.meta.width || '?') + 'x' + (it.meta.height || '?');
  const d = await (await fetch('/api/ann/' + it.frame_id)).json();
  const a = d.ann || {};
  document.getElementById('note').value = a.note || '';
  document.getElementById('evidence').value = a.evidence || '';
  paintButtons(a.label || '');
  dirty = false;
}
function paintButtons(l) {
  document.getElementById('b-push').className = l === 'push' ? 'sel-push' : '';
  document.getElementById('b-ride').className = l === 'ride' ? 'sel-ride' : '';
  document.getElementById('b-unsure').className = l === 'unsure' ? 'sel-unsure' : '';
  document.getElementById('b-invalid').className = l === 'invalid' ? 'sel-invalid' : '';
}
async function setLabel(label) {
  const it = items[idx];
  paintButtons(label);
  await save({label});
  items[idx].labeled = true;
}
async function save(extra) {
  const it = items[idx];
  const body = {
    label: extra && extra.label !== undefined ? extra.label : currentLabel(),
    note: document.getElementById('note').value,
    evidence: document.getElementById('evidence').value,
  };
  const r = await (await fetch('/api/ann/' + it.frame_id, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  })).json();
  updateDone(r.done);
  const s = document.getElementById('savestate');
  s.textContent = '已保存 ' + new Date().toLocaleTimeString();
  setTimeout(() => s.textContent = '', 1500);
  dirty = false;
}
function currentLabel() {
  const b = document.getElementById('b-push');
  if (b.className === 'sel-push') return 'push';
  if (document.getElementById('b-ride').className === 'sel-ride') return 'ride';
  if (document.getElementById('b-unsure').className === 'sel-unsure') return 'unsure';
  if (document.getElementById('b-invalid').className === 'sel-invalid') return 'invalid';
  return '';
}
function nav(delta) {
  if (dirty) save({});
  idx = (idx + delta + items.length) % items.length;
  load();
}
document.getElementById('note').addEventListener('blur', () => save({}));
document.getElementById('evidence').addEventListener('blur', () => save({}));
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === '1') setLabel('push');
  else if (e.key === '2') setLabel('ride');
  else if (e.key === '3') setLabel('unsure');
  else if (e.key === '4') setLabel('invalid');
  else if (e.key === 'ArrowLeft' || e.key === 'a') nav(-1);
  else if (e.key === 'ArrowRight' || e.key === 'd') nav(1);
});
init();
</script>
</body>
</html>"""


def main():
    global FRAMES_DIR, ANN_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--evidence-field", action="store_true",
                    help="显示「证据描述」输入框（训练集标注用）")
    args = ap.parse_args()

    FRAMES_DIR = Path(args.frames_dir)
    ANN_PATH = FRAMES_DIR / "annotations.jsonl"
    app.config["MANIFEST_PATH"] = args.manifest
    if args.evidence_field:
        global PAGE
        PAGE = PAGE.replace('<div id="evblock" style="display:none">', '<div id="evblock">')
    load_state()
    print(f"标注池 {len(MANIFEST)} 张，已完成 {len(ANNOTATIONS)} 张")
    print(f"浏览器打开 http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
