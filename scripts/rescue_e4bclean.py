import os, signal, shutil
from safetensors import safe_open
import torch

final = "/mnt/d/vlm-active/ckpt/e4b_clean/final"
tmp = os.path.join(final, [f for f in os.listdir(final) if f.startswith(".tmp")][0])

# 1. 验证 tmp 完整性：merger 必须非零、嵌入可读
with safe_open(tmp, framework="pt") as f:
    mk = next(k for k in f.keys() if "merger.mlp.0.weight" in k)
    m = f.get_tensor(mk).float().abs().max().item()
    ek = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
    e = f.get_tensor(ek).float().abs().max().item()
print("tmp merger abs max:", m)
print("tmp embed abs max:", e)
assert m > 0.001, "tmp 未写完（merger 全零），改从 checkpoint 重建"

# 2. 杀进程
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

# 3. 转正 + 补文件
os.rename(tmp, os.path.join(final, "model.safetensors"))
base = "/mnt/d/vlm-active/models/Qwen2.5-VL-3B-Instruct"
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "preprocessor_config.json", "chat_template.json"]:
    s = os.path.join(base, fn)
    if os.path.exists(s) and not os.path.exists(os.path.join(final, fn)):
        shutil.copy(s, os.path.join(final, fn))
        print("copied", fn)
print("rescued ->", sorted(os.listdir(final)))
