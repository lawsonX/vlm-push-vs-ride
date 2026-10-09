# VLM 推行 vs 骑行：一个场景的全链路调查

监控/街面画面里「人推行两轮车」被多个视觉语言模型误判为「骑行」。本项目以这个具体场景为例，完成从**问题定义 → 假设 → 数据工程 → 多模型测试 → 归因 → 修复实验 → 报告**的完整闭环。

## 核心结论（详见 docs/09-实验报告-草稿.md）

1. **根因是数据饥荒**：55.8 万条主流训练标注中真「推行」描述约 0~2 条，模型没有这个概念
2. **八类修复手段全部只动先验**：SFT 各配方、DPO、ViT LoRA、merger、全量微调、少样本提示——没有任何方法教会模型「看证据」
3. 完整错误链：画面模糊 → 先验直答 → 编造证据 → 注意力没看人车区域
4. 两个工程警告：fp16 下部分图像致视觉编码器 NaN（全库 17% 帧中招，黑名单在 data/results/nan_frames.json）；训练/测试重叠会把记忆伪装成泛化

## 目录结构

```
docs/       实验方案、进度日志、归因报告、最终报告（09 是主报告）
scripts/    全部代码：数据管线 / 标注 UI / 训练 / 评测 / DPO / 预筛 / 天然对挖掘
data/       核心数据资产（小而精）：
  annotations.jsonl   334 张人工标注（推行24/骑行210/无效，含 evidence 字段）
  test_balanced.jsonl 48 帧配平测试集（24 推行 + 24 骑行）——所有主指标的分母
  sft/                各版本训练数据（含去 NaN 版、干净版）
  results/            全部评测结果、NaN 黑名单、预筛与质检产物
  pools/              标注池清单（YOLO 几何信息，标注 UI 的输入）
  natr/               E9 天然对比对清单（24 对，图片需按 HANDOVER 重建）
reference/  前期调研文档（VLM 后训练数据与对齐调研）
HANDOVER.md 新设备接手指南（环境、数据下载、复现步骤、待办）
```

## 快速复现（单条评测示例）

```bash
python scripts/eval_local_nan.py --ckpt <模型路径> \
  --annotations data/test_balanced.jsonl --frames-dir <帧目录> \
  --out /tmp/preds.jsonl --all-fp32
```

## 状态

主报告 v0.5 完成。剩余实验（E7-v2 人工证据链、E9 天然对训练、干净版 SFT）因原设备 GPU 故障暂停，待新设备接续，见 HANDOVER.md。
