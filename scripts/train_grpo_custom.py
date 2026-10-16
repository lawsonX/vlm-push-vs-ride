#!/usr/bin/env python3
"""A5 自定义 GRPO（RLOO 风格）：TRL 的 VLM GRPO 在纯 transformers 生成路径上有
Qwen2.5-VL rope 索引的集成 bug，且装新版 vllm 有弄坏环境的风险，故用已有验证过的
组件手搓最小正确实现。

要点：
- 起点 A1 final，语言侧 rank 16 LoRA（docs/13 A5），视觉冻结
- 每步：B 个提示 × 每提示 G 个采样（temperature=1.0）
- 奖励：与评测同口径的判分（direct/probe_seat/probe_feet），对=1 错=0
- 优势 = 奖励 - 组内均值；loss = -(优势 × 完成token平均对数概率)
- 单步 on-policy 更新（无 PPO 裁剪，小学习率，pilot 定位）

用法：
    python train_grpo_custom.py --model ckpt/a1_baseline/final \
        --data data/sft_v2/train.jsonl --out ckpt/a5_grpo --steps 40
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from train_grpo_v2 import _judge, RL_STYLES  # 同一奖励口径

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def build_prompts(data_path):
    """提取 RL 提示：图片路径 + 文本 + 金标词 + 问法。"""
    from PIL import Image
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    recs = []
    for r in rows:
        if r["style"] not in RL_STYLES:
            continue
        user_msg = next(m for m in r["messages"] if m["role"] == "user")
        text = user_msg["content"].replace("<图片占位>", "", 1).strip()
        gold = next(m for m in r["messages"] if m["role"] == "assistant")["content"]
        recs.append({"image": r["image"], "text": text,
                     "gold": gold, "style": r["style"], "frame_id": r["frame_id"]})
    return recs


def encode_prompt(processor, image_path, text):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": text}]}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt_text], images=[img], return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/sft_v2/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts-per-step", type=int, default=4, help="B")
    ap.add_argument("--num-generations", type=int, default=8, help="G")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="采样温度；1.0 下答案非全对即全错，组内无对比就没有梯度")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-only", action="store_true",
                    help="先扫描提示池，只保留模型答错过至少一次的提示做 RL（全对的提示优势恒为0，无梯度）")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    for name, p in model.named_parameters():
        if "visual" in name:
            p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    recs = build_prompts(args.data)
    if args.hard_only:
        print("扫描提示池，找模型答错的提示…")
        model.eval()
        hard, acc_sum = [], 0.0
        with torch.no_grad():
            for j, rec in enumerate(recs, 1):
                enc = encode_prompt(processor, rec["image"], rec["text"])
                enc = {k: v.to(device) for k, v in enc.items()}
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=True, temperature=1.0,
                                     num_return_sequences=args.num_generations,
                                     pad_token_id=processor.tokenizer.pad_token_id)
                pl = enc["input_ids"].shape[1]
                hits = sum(_judge(rec["style"], processor.decode(g[pl:], skip_special_tokens=True),
                                  rec["gold"])[1] for g in gen)
                acc_sum += hits / args.num_generations
                if hits < args.num_generations:
                    hard.append(rec)
                if j % 50 == 0:
                    print(f"  {j}/{len(recs)}  running_acc={acc_sum/j:.3f} hard={len(hard)}", flush=True)
        print(f"扫描完成: 池准确率 {acc_sum/len(recs):.3f}, 难提示 {len(hard)}/{len(recs)}")
        recs = hard if hard else recs
    random.shuffle(recs)
    print(f"RL 提示池: {len(recs)} 条")
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "rl_log.jsonl", "w", encoding="utf-8")
    cursor = 0
    history = []
    for step in range(1, args.steps + 1):
        batch = [recs[(cursor + i) % len(recs)] for i in range(args.prompts_per_step)]
        cursor += args.prompts_per_step
        model.eval()
        all_prompts, all_completions, all_rewards = [], [], []
        with torch.no_grad():
            for rec in batch:
                enc = encode_prompt(processor, rec["image"], rec["text"])
                enc = {k: v.to(device) for k, v in enc.items()}
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temperature, num_return_sequences=args.num_generations,
                                     pad_token_id=pad_id)
                prompt_len = enc["input_ids"].shape[1]
                for g in gen:
                    comp = g[prompt_len:]
                    text = processor.decode(comp, skip_special_tokens=True)
                    _, ok = _judge(rec["style"], text, rec["gold"])
                    all_prompts.append((enc, prompt_len))
                    all_completions.append(comp)
                    all_rewards.append(1.0 if ok else 0.0)
        # 组内基线优势
        advs = []
        for i in range(len(batch)):
            grp = all_rewards[i * args.num_generations:(i + 1) * args.num_generations]
            m = sum(grp) / len(grp)
            advs += [r - m for r in grp]

        model.train()
        losses, logs = [], []
        for (enc, prompt_len), comp, adv, rec in zip(all_prompts, all_completions, advs, batch * args.num_generations):
            if adv == 0.0:
                continue  # 组内无差异的样本对梯度无贡献
            full = torch.cat([enc["input_ids"][0].cpu(), comp.cpu()])
            labels = full.clone()
            labels[:prompt_len] = -100
            att = torch.ones_like(full)
            inputs = {"input_ids": full.unsqueeze(0).to(device),
                      "attention_mask": att.unsqueeze(0).to(device),
                      "pixel_values": enc["pixel_values"].to(device),
                      "image_grid_thw": enc["image_grid_thw"].to(device)}
            # 注意：不传 mm_token_type_ids（长度随手动拼接对不上，模型会从 input_ids 自行推导）
            out_logits = model(**inputs).logits[0].float()
            tgt = inputs["input_ids"][0][1:]
            mask = (labels[1:] != -100).to(device)
            if mask.sum() == 0:
                continue
            lp = torch.log_softmax(out_logits[:-1], dim=-1)
            tok_lp = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            seq_lp = (tok_lp * mask).sum() / mask.sum()
            losses.append(-adv * seq_lp)
            logs.append({"frame_id": rec["frame_id"], "style": rec["style"],
                         "adv": adv, "comp": processor.decode(comp, skip_special_tokens=True)[:40]})
        if losses:
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)
        reward_mean = sum(all_rewards) / len(all_rewards)
        history.append(reward_mean)
        print(f"step {step}/{args.steps} loss={loss.item():.4f} reward={reward_mean:.3f} "
              f"({sum(all_rewards):.0f}/{len(all_rewards)})", flush=True)
        log_f.write(json.dumps({"step": step, "loss": loss.item(), "reward": reward_mean,
                                "samples": logs[:8]}, ensure_ascii=False) + "\n")
        log_f.flush()
        if step % args.save_every == 0:
            (out / f"adapter_step{step}").mkdir(exist_ok=True)
            model.save_pretrained(out / f"adapter_step{step}")

    model = model.merge_and_unload()
    model.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")
    print(f"done -> {out/'final'}  奖励曲线: {[round(h,3) for h in history]}")


if __name__ == "__main__":
    main()
