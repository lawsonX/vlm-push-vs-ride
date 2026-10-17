#!/usr/bin/env python3
"""探针标注页面：给「看不清」第三类答案实验用。

任务（lawson 2026-10-17 批准）：对探针云端验证未通过的金标图（273 张），
人工回答两个探针问题，每个问题有三个选项，第三个是"看不清"：

  Q1 这个人有没有坐在车座上？   有 / 没有 / 看不清
  Q2 双脚是在地上还是踏板？     地上 / 踏板 / 看不清

产出 probe_labels.jsonl：{frame_id, seat, feet}，供 v0.6 训练数据使用。
已在 probes.jsonl 里通过云端验证的答案会预填并锁定（不用再答）。

用法：
    python probe_app.py --frames-dir /mnt/e/vlm-data/label_pool \
        --todo /mnt/e/vlm-data/label_pool/probe_label_todo.txt \
        --verified /mnt/e/vlm-data/label_pool/probes.jsonl --port 8322
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, request, Response

app = Flask(__name__)
FRAMES_DIR = Path(".")
TODO_PATH = Path("todo.txt")
VERIFIED = {}
LABELS = {}   # frame_id -> {seat, feet}
LABEL_PATH = Path("probe_labels.jsonl")
ANN = {}      # frame_id -> {labeled: bool, seat, feet}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>探针标注（坐没坐/脚在哪）</title>
<style>
body{font-family:sans-serif;background:#1e1e1e;color:#eee;margin:0;display:flex;height:100vh}
.left{flex:2;padding:10px;display:flex;align-items:center;justify-content:center}
.left img{max-width:100%;max-height:92vh}
.right{flex:1;padding:14px;display:flex;flex-direction:column;gap:10px;min-width:360px}
.q{background:#2a2a2a;border-radius:8px;padding:10px}
.q b{display:block;margin-bottom:8px}
.opts button{display:block;width:100%;margin:4px 0;padding:10px;font-size:15px;border:2px solid #555;border-radius:6px;background:#333;color:#eee;cursor:pointer}
.opts button.sel{background:#2e7d32;border-color:#66bb6a}
.opts button.locked{opacity:.55;cursor:default}
.bar{height:8px;background:#333;border-radius:4px;overflow:hidden}
#barfill{height:100%;background:#66bb6a;width:0%}
.nav button{padding:8px 14px;font-size:14px;background:#3a3a3a;color:#eee;border:1px solid #555;border-radius:6px;cursor:pointer}
.meta{color:#999;font-size:13px}
.skip{margin-top:auto;padding:10px;background:#7b1fa2;border:1px solid #ba68c8;border-radius:6px;color:#fff;cursor:pointer;font-size:15px}
</style></head><body>
<div class="left"><img id="pic"></div>
<div class="right">
  <div><span id="pos">-</span>/<span id="total">-</span> 已答 <span id="done">0</span> ｜ <span id="fid" class="meta"></span></div>
  <div class="bar"><div id="barfill"></div></div>
  <div class="q"><b>Q1：这个人有没有坐在车座上？</b><div class="opts" id="seat">
    <button onclick="set('seat','有')">有</button>
    <button onclick="set('seat','没有')">没有</button>
    <button onclick="set('seat','看不清')">看不清</button>
  </div></div>
  <div class="q"><b>Q2：这个人的双脚是在地上走路，还是踩在车的踏板上？</b><div class="opts" id="feet">
    <button onclick="set('feet','地上')">在地上走路</button>
    <button onclick="set('feet','踏板')">踩在踏板上</button>
    <button onclick="set('feet','看不清')">看不清</button>
  </div></div>
  <button class="skip" onclick="skipUnlabeled()">⤵ 跳到下一个未答的 (N)</button>
  <div class="nav">
    <button onclick="nav(-1)">← 上一张 (A)</button>
    <button onclick="nav(1)">下一张 → (D)</button>
  </div>
  <div class="meta">说明：发绿的选项是云端已验证的答案（锁定）；你需要答的是另一个。按 1/2/3 答 Q1，按 4/5/6 答 Q2。看不清就是看不清，猜的不算。</div>
</div>
<script>
let items=[], idx=0;
async function init(){
  const d = await (await fetch('/api/list')).json();
  items=d.items; document.getElementById('total').textContent=d.total;
  updateDone(d.done);
  const first = items.findIndex(it=>!it.labeled);
  idx = first===-1?0:first;
  load();
}
function updateDone(n){document.getElementById('done').textContent=n;
  document.getElementById('barfill').style.width=(100*n/items.length)+'%';}
async function load(){
  const it=items[idx];
  document.getElementById('pos').textContent=idx+1;
  document.getElementById('fid').textContent=it.frame_id;
  document.getElementById('pic').src='/img/'+it.frame_id;
  paint('seat', it.seat, it.seat_locked);
  paint('feet', it.feet, it.feet_locked);
}
function paint(q, val, locked){
  document.querySelectorAll('#'+q+' button').forEach(b=>{
    b.className = (b.textContent.includes('看不清')&&val==='看不清')?'sel'
      : ((q==='seat'&&(b.textContent==='有'&&val==='有'||b.textContent==='没有'&&val==='没有'))
      || (q==='feet'&&(b.textContent==='在地上走路'&&val==='地上'||b.textContent==='踩在踏板上'&&val==='踏板')))?'sel':'';
    if(locked) b.className += ' locked';
  });
}
async function set(q, val){
  const it=items[idx];
  if(it[q+'_locked']) return;
  it[q]=val;
  const r = await (await fetch('/api/ann/'+it.frame_id,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({q, val})})).json();
  it.labeled=r.labeled; updateDone(r.done);
  paint(q, val, false);
}
function nav(d){ idx=(idx+d+items.length)%items.length; load(); }
function skipUnlabeled(){
  for(let s=1;s<=items.length;s++){const i=(idx+s)%items.length;
    if(!items[i].labeled){idx=i;load();return;}}
  alert('全部答完了！');
}
document.addEventListener('keydown',e=>{
  if(['1','2','3'].includes(e.key)) set('seat',['有','没有','看不清'][+e.key-1]);
  else if(['4','5','6'].includes(e.key)) set('feet',['地上','踏板','看不清'][+e.key-4]);
  else if(e.key==='ArrowLeft'||e.key==='a') nav(-1);
  else if(e.key==='ArrowRight'||e.key==='d') nav(1);
  else if(e.key==='n'||e.key==='N') skipUnlabeled();
});
init();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/list")
def api_list():
    items = []
    for fid in ANN:
        a = ANN[fid]
        items.append({
            "frame_id": fid, "labeled": a["labeled"],
            "seat": a.get("seat"), "feet": a.get("feet"),
            "seat_locked": a.get("seat_locked", False),
            "feet_locked": a.get("feet_locked", False),
        })
    done = sum(1 for a in ANN.values() if a["labeled"])
    return jsonify({"total": len(items), "done": done, "items": items})


@app.route("/img/<frame_id>")
def img(frame_id):
    for ext in (".jpg", ".jpeg", ".png"):
        p = FRAMES_DIR / f"{frame_id}{ext}"
        if p.exists() and frame_id in ANN:
            return Response(p.read_bytes(), mimetype="image/jpeg")
    return "not found", 404


@app.route("/api/ann/<frame_id>", methods=["GET", "POST"])
def api_ann(frame_id):
    a = ANN[frame_id]
    if request.method == "POST":
        d = request.json
        q, val = d["q"], d["val"]
        if not a.get(q + "_locked"):
            a[q] = val
        a["labeled"] = bool(a.get("seat")) and bool(a.get("feet"))
        with open(LABEL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"frame_id": frame_id,
                                "seat": a.get("seat"), "feet": a.get("feet"),
                                "seat_locked": a.get("seat_locked", False),
                                "feet_locked": a.get("feet_locked", False)},
                               ensure_ascii=False) + "\n")
    return jsonify({"ann": a, "labeled": a["labeled"],
                    "done": sum(1 for x in ANN.values() if x["labeled"])})


def load_state():
    global ANN
    todo = [l.strip() for l in open(TODO_PATH, encoding="utf-8") if l.strip()]
    ANN = {}
    for fid in todo:
        v = VERIFIED.get(fid, {})
        seat, seat_locked = (v.get("seat"), True) if "seat" in v else (None, False)
        feet, feet_locked = (v.get("feet"), True) if "feet" in v else (None, False)
        ANN[fid] = {"seat": seat, "feet": feet,
                    "seat_locked": seat_locked, "feet_locked": feet_locked,
                    "labeled": bool(seat) and bool(feet)}
    if LABEL_PATH.exists():
        for l in open(LABEL_PATH, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            a = ANN.get(d["frame_id"])
            if a:
                for q in ("seat", "feet"):
                    if not a.get(q + "_locked") and d.get(q):
                        a[q] = d[q]
                a["labeled"] = bool(a.get("seat")) and bool(a.get("feet"))


def main():
    global FRAMES_DIR, TODO_PATH, LABEL_PATH, VERIFIED
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--todo", required=True)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8322)
    args = ap.parse_args()
    FRAMES_DIR = Path(args.frames_dir)
    TODO_PATH = Path(args.todo)
    LABEL_PATH = Path(args.out) if args.out else FRAMES_DIR / "probe_labels.jsonl"
    if args.verified:
        for l in open(args.verified, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                VERIFIED[d["frame_id"]] = d
    load_state()
    n_pre = sum(1 for a in ANN.values() if a["labeled"])
    print(f"探针标注池 {len(ANN)} 张（云端已答全 {n_pre} 张，待人工 {len(ANN)-n_pre} 张）")
    print(f"浏览器打开 http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
