# SFT 数据集 v0.7（2026-10-18）

在 v0.6 基础上吸收 lawson 现场复核（docs/lawson-v07现场操作单.md）：

1. **标注规范定义 A 生效**（lawson 拍板）：脚踩车身任何承脚部位（自行车脚蹬、摩托脚踏、电动车搁脚板）都算「踏板」；只有脚直接接触地面才算「地上」。
2. **4 帧整图作废**（lawson 按 0）：items_bili3__BV1e3411w7WL.f30080.f000210__i2、bili3__BV1x9ypBDEQX.f100026.f000000__i1、items_bili3__BV1bY9hBzEqa.f100026.f000150__i0、visdrone__vd_9999999_00420_d_0000185__i2（末者在测试集，测试 43→42 张）。共移除 22 条样本。
3. **10 条探针答案修正**（盲标 diff）：9 条 feet 踏板→地上（8 条属跨坐：坐座+双脚着地）、1 条 feet 踏板→看不清。
4. **17 帧跨坐确认**（lawson 按 7）：其中 9 帧原探针 gold 为「踏板」改「地上」，7 条 caption 补跨坐描述；**1 帧标签改正**：visdrone__vd_0000166_01558_d_0000123__i3 由 push→ride（跨坐按规范计骑行，该帧正是此前发现的「金标自相矛盾」帧）。
5. 复核背景：Qwen3.5-2B 与金标在 85 帧上冲突，lawson 盲标裁决后 **152/170 条与原金标一致**——v0.6 金标质量获确认，2B 的 seat 偏差是主要假信号。

规模：train 748 条 / 130 图；test 216 条 / 42 图（含 chain/guide 各 30）；test_ids.txt 已同步。
盲标原始记录：`datasets/delivery_v02/frames/probe_labels.jsonl`。

训练侧重训：`python scripts/a8_pipeline.py --src data/sft_v7 --out-root ckpt/a8_v7 --seeds 42 7 99`（已执行，结果见 docs/31）。
