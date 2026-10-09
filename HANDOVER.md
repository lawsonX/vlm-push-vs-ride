# HANDOVER：新设备（多卡 GPU Linux 服务器）接手指南

> 写给接手本项目的 agent。原设备（Windows + WSL + 单卡 2080 Ti）因 GPU 驱动频繁崩溃退役。
> 本文件是唯一权威交接文档；docs/ 系列是过程记录，09 是主报告。

## 0. 一句话现状

科学问题已闭环（根因=数据饥荒，所有现有修复手段只动先验）；**剩余工作全部是数据驱动的实验**，需要 lawson 在新设备上继续标注 + GPU 训练。接手后第一件事读 `docs/09-实验报告-草稿.md` 的摘要和第 8 节。

## 1. 环境搭建（conda）

```bash
conda create -n vlm-lab python=3.11 -y
conda activate vlm-lab
pip install torch==2.6 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.x peft trl==1.14.1 accelerate safetensors \
            ultralytics opencv-python-headless pillow flask openai
# YOLO 权重（数据管线需要）：
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-seg.pt')"
```

参考基准：原环境 torch 2.6+cu124、transformers 5.x、单卡 22GB 可训 3B LoRA / 0.8B 全量 / fp32 merger；
**新设备多卡注意**：本项目全部脚本按单卡写，多卡请先跑通单卡再考虑并行（每个实验独立进程一张卡即可，无需分布式）。

## 2. 需要重新下载的东西（都不在仓库里）

### 2.1 模型（HuggingFace，国内用镜像 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`）

| 模型 | 用途 | 备注 |
|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | 主力实验模型 | 最重要 |
| Qwen/Qwen2.5-VL-7B-Instruct | 规模对照（基线偏推行） | fp16 推理约 15GB |
| Qwen/Qwen3.5-0.8B、Qwen3.5-2B | 原生多模态对照 | 词嵌入有 tie_word_embeddings 坑，见报告 6.1 |
| openai/clip-vit-base-patch32 | 纯视觉对照 | 小 |

### 2.2 图像帧（约 11229 张、~5GB，重建流程）

帧是多个来源抽取的，**清单都在仓库里、图片不在**。最小可用重建（只做剩余实验只需前 3 行）：

1. **必须**：把 `data/annotations.jsonl` 与 `data/test_balanced.jsonl` 里 frame_id 对应的图片找回。frame_id 命名规则 = 文件名去 `.jpg`。来源分布：sg_* 来自 TrafficQA 视频抽帧、j_*/y_*/c_*/news_* 来自搜狗图片/新闻抓取、b_*/BV* 来自 B 站视频抽帧。
2. **推荐**： TrafficQA 数据集（原项目从它的 4110 段视频抽帧，配合 `scripts/extract_frames.py` + `scripts/filter_bike_frames.py` 重跑可还原 sg_* 帧）。
3. B 站视频（b_*/BV* 帧的来源）：原视频列表可从 data/pools/pool_manifest.jsonl 的 frame_id 反推（BV 号即视频号），用 yt-dlp 下载后 `scripts/scan_transitions.py`/`extract_frames.py` 重抽。B 站有反爬，原项目用浏览器 cookie + yt-dlp，必要时 lawson 提供新 cookie。
4. 如果暂时找不回全部帧：**优先保证 48 张测试集 + 334 张标注帧在场**，其余帧只影响数据扩充实验。

### 2.3 密钥（lawson 提供，不入库）

- 阿里云 dashscope API key（云端模型测试/预筛/质检用）：放 `api_testing/api_key.txt`（与 scripts/ 里 api 脚本同目录约定），原版在 lawson 的 Windows 桌面 `dashscope_api.txt`。
- GitHub：lawson 已登录 gh cli。

## 3. 目录落位建议

```
~/vlm-lab/           ← 代码（本仓库 scripts/ 内容按子目录放：finetune/ eval/ data_pipeline/ annotation_ui/ api_testing/）
~/vlm-active/        ← 运行数据（对应仓库 data/ 展开 + frames/ + models/ + ckpt/ + results/）
```

脚本里写死的路径主要是 `/mnt/d/vlm-active/...`（原 D 盘挂载点），新设备上用 sed 批量替换为自己的路径，或保持 `~/vlm-active` 结构后替换 `/mnt/d` → `$HOME` 即可。

## 4. 接手后的待办（按优先级）

1. **lawson 标注**（最高优先级）：标注 UI `python annotation_ui/annotate_app.py --frames-dir <帧目录> --manifest data/pools/pool_ranked.jsonl --port 8321`。前 130 帧是双模型预筛的疑似推行（`data/results/prescreen*.jsonl`）。每标一张推行帧顺手写 `evidence` 字段（一句「你看到的证据」，如「双脚着地、臀部悬空」）。
2. **标注量达标后跑一键复测**：`scripts/run_after_labeling.sh`（构建干净训练集 → E2/E4b 干净版训练 → fp32 评测 → 有 evidence 则生成 E7-v2 数据）。注意脚本里路径需按第 3 节替换。
3. **E7-v2 人工证据链训练**（数据就绪后）：用 `scripts/make_chain_v2.py` 生成证据链数据，`scripts/train_sft.py --mode llm_lora` 训练。
4. **E9 天然对比对**：24 对清单在 `data/natr/`（pairs_screened.jsonl 是 14 对高置信 + pairs_transition.jsonl 10 对），图片需从源视频重抽（frame 名含视频与时间点）。lawson 过目后做对比训练。
5. **可拖动性探针**（可选补完）：`scripts/malleability_probe.py`，原设备 GPU 崩溃前未完成 3B/7B 部分；2B 已有结果（|Δlogit| 均值 0.003，远小于翻答案所需）。

## 5. 关键坑位备忘（血泪经验，都写在报告 6.0d~6.1 节）

- **fp16 NaN 帧**：部分图像让 Qwen-VL 视觉编码器输出 NaN、生成乱码。黑名单 `data/results/nan_frames.json`（1943 张，按 3B 扫的，**不能跨模型复用**——7B 全正常）。评测必须带乱码兜底（eval_local_nan.py 已内置）。
- **训练/测试重叠**：本项目曾因 48 测试帧中 25 张混进训练集得出假「突破」。所有新实验必须用 `scripts/make_clean_train_v2.py`（内置重叠检查）。
- **9p/网络文件系统保存大模型会卡死**：save_pretrained 到 NFS/SMB 可能产出 merger 全零的半 written 文件。只信 Trainer 的 checkpoint 目录，最终产物从 checkpoint 重建（参照 scripts/rebuild_e2_7b.py 的 PeftModel.from_pretrained 路线，勿手拼 state_dict）。
- **词嵌入污染**：Qwen3.5 系（tie_word_embeddings）训练后保存必须恢复基座词嵌入（train_sft.py 已内置；此前「CPU 训练不可靠」系误诊）。
- **DPO 偏好对方向**：云端模型错误并集天然一边倒（35:5），用前必须配平（scripts/train_dpo_manual_v4.py 已内置 2:1 配平 + NaN 预扫）。
- **transformers 5.x 命名**：ViT 注意力是融合 qkv（不是 q/k/v_proj）；视觉→语言连接层叫 visual.merger（不是 projector）。

## 6. 联系

决策类问题问 lawson；实验细节先查 docs/09 报告，找不到再翻 docs/01~08 进度日志。
