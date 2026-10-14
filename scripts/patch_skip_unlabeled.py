# 给标注 UI 加「跳到未标注」功能（按钮 + N 键）
import io, sys

p = "/root/vlm-lab/annotation_ui/annotate_app.py"
src = io.open(p, encoding="utf-8").read()

def rep(old, new):
    global src
    assert src.count(old) == 1, "not unique: %r" % old[:40]
    src = src.replace(old, new)

# 1. 按钮区加按钮
rep("""      <button onclick="nav(1)">下一张 → (D)</button>
    </div>""",
"""      <button onclick="nav(1)">下一张 → (D)</button>
    </div>
    <div class="nav">
      <button onclick="skipUnlabeled()" style="background:#7b1fa2;border:1px solid #ba68c8">⤵ 跳到下一个未标注 (N)</button>
    </div>""")

# 2. 加 skipUnlabeled 函数（放在 nav 之后）
rep("""function nav(delta) {
  if (dirty) save({});
  idx = (idx + delta + items.length) % items.length;
  load();
}""",
"""function nav(delta) {
  if (dirty) save({});
  idx = (idx + delta + items.length) % items.length;
  load();
}
function skipUnlabeled() {
  for (let step = 1; step <= items.length; step++) {
    const i = (idx + step) % items.length;
    if (!items[i].labeled) { idx = i; load(); return; }
  }
  document.getElementById('savestate').textContent = '全部已标注完！';
}""")

# 3. 键盘监听加 N 键
rep("""  else if (e.key === 'ArrowRight' || e.key === 'd') nav(1);""",
"""  else if (e.key === 'ArrowRight' || e.key === 'd') nav(1);
  else if (e.key === 'n' || e.key === 'N') skipUnlabeled();""")

io.open(p, "w", encoding="utf-8").write(src)
print("patch ok")
