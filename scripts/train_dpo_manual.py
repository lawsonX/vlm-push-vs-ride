#!/usr/bin/env python3
"""E8 极简手写版 DPO（7 对数据专用，绕开 trl 的显存问题）。

trl 的 DPOTrainer 在视觉模型上接连踩坑（fp16 缩放器、全量优化器、显存 33GB），
7 对数据的试点不值得继续纠缠。这个版本几十行，全透明：
- policy 和 ref 各前向两次（chosen/rejected 分开，batch=1），峰值显存 ~14GB
- DPO 损失：-logsigmoid(beta * ((pol_c-ref_c) - (pol_r-ref_r)))
- 答案部分的 logprob 求和（与 trl 口径一致）
- 保存时合并 LoRA + 恢复基座词嵌入（防污染，老教训）

用法：
    python train_dpo_manual.py --model /path/sft_fixed --pairs dpo.jsonl \
        --out /path/ckpt/e8_manual --beta 0.1 --lr 5e-6
"""
import argparse
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "图中这个人是在推行还是骑行这辆车？"


def seq_logprob(model, batch, answer_len, device):
    out = model(**batch)
    logits = out.logits[0]  # [T, V]
    labels = batch["input_ids"][0]
    # 答案部分是序列末尾的 answer_len 个 token：predict 位置 t 对应 label t+1
    tgt = labels[-answer_len:].to(device)
    pred = logits[-answer_len - 1:-1]
    lp = torch.log_softmax(pred.float(), dim=-1)
    return lp.gather(-1, tgt.unsqueeze(-1)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16).to(device)
    ref = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    print(f"偏好对: {len(pairs)}")

    from PIL import Image
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    for step, r in enumerate(pairs, 1):
        img = Image.open(r["image"]).convert("RGB")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "url": r["image"]},
                {"type": "text", "text": QUESTION},
            ]}], tokenize=False, add_generation_prompt=True)

        stats = []
        for text in (r["chosen"], r["rejected"]):
            full = prompt + text
            inputs = processor(text=[full], images=[img], return_tensors="pt")
            ans_ids = processor(text=[text], add_special_tokens=False)["input_ids"][0]
            batch = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                ref_lp = seq_logprob(ref, batch, len(ans_ids), device)
            pol_lp = seq_logprob(model, batch, len(ans_ids), device)
            stats.append((pol_lp, ref_lp))

        (pol_c, ref_c), (pol_r, ref_r) = stats
        margin = (pol_c - ref_c) - (pol_r - ref_r)
        loss = -F.logsigmoid(args.beta * margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step}/{len(pairs)} loss={float(loss):.4f} "
              f"margin={float(margin):.3f} (正=偏好方向学对了)", flush=True)

    # 保存：合并 LoRA + 恢复基座词嵌入
    print("合并并保存…")
    from safetensors import safe_open
    import os
    embed = None
    for fn in os.listdir(args.model):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(args.model, fn), framework="pt") as f:
                if "model.embed_tokens.weight" in f.keys():
                    embed = f.get_tensor("model.embed_tokens.weight")
                    break
    merged = model.merge_and_unload()
    if embed is not None:
        with torch.no_grad():
            for n, p in merged.named_parameters():
                if n.endswith("embed_tokens.weight"):
                    p.copy_(embed.to(p.dtype))
    merged.save_pretrained(args.out + "/final", safe_serialization=True)
    processor.save_pretrained(args.out + "/final")
    print("done ->", args.out + "/final")


if __name__ == "__main__":
    main()
