import os, signal
me = os.getpid()
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        c = open("/proc/" + pid + "/cmdline", "rb").read().decode()
        if "eval_local_nan" in c and "python -u" in c:
            os.kill(int(pid), signal.SIGKILL)
            print("killed", pid)
    except Exception:
        pass
print("scan done")
