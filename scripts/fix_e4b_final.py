import os, shutil, signal

# 1. 杀掉卡死的训练进程
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        c = open("/proc/" + pid + "/cmdline", "rb").read().decode()
        if "train_sft" in c and "python -u" in c:
            os.kill(int(pid), signal.SIGKILL)
            print("killed", pid)
    except Exception:
        pass

# 2. 把写好的 tmp safetensors 转正
final = "/mnt/d/vlm-active/ckpt/e4b_projector/final"
tmps = [f for f in os.listdir(final) if f.startswith(".tmp")]
if tmps:
    src = os.path.join(final, tmps[0])
    dst = os.path.join(final, "model.safetensors")
    if not os.path.exists(dst):
        os.rename(src, dst)
        print("renamed", tmps[0], "-> model.safetensors")
    else:
        os.remove(src)
        print("removed duplicate tmp")

# 3. 补齐 tokenizer / preprocessor 文件（从基座复制）
base = "/mnt/d/vlm-active/models/Qwen2.5-VL-3B-Instruct"
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "preprocessor_config.json", "chat_template.json"]:
    s = os.path.join(base, fn)
    d = os.path.join(final, fn)
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copy(s, d)
        print("copied", fn)
print("final files:", sorted(os.listdir(final)))
