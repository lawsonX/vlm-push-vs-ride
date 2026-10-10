# -*- coding: utf-8 -*-
"""第五轮 BV 清单过滤：去重、减去已下载、标题黑名单，输出 /root/bili_bvs5.txt"""
from pathlib import Path

TSV = "/root/bili_bvs5_candidates.tsv"
OLD_LISTS = ["/root/bili_bvs3.txt", "/root/bili_bvs4.txt"]
VID_DIR = Path("/mnt/e/vlm-data/raw/bili_videos")
OUT = "/root/bili_bvs5.txt"

BLOCK = ["测评", "横评", "推荐", "开箱", "选购", "指南", "攻略", "教程", "教学",
         "英语", "听力", "口语", "FSD", "智驾", "辅助驾驶", "自动驾驶",
         "特斯拉", "蔚来", "萤火虫", "尊界", "问界", "雷军", "华为", "余承东",
         "海康", "职场内幕", "充电桩", "充电费用", "充电卡", "快充", "换电",
         "改装", "罚款", "申诉", "交管", "罚单", "新规", "限速", "征信",
         "锁车", "纪录片", "白噪音", "网红打卡", "促销", "诗人", "大白菜",
         "进口肉", "猪肉", "车棚", "车位", "门禁", "UFO", "梗指南", "播客",
         "WandB", "TensorBoard", "调参", "vLLM", "折叠", "滑板车", "代驾",
         "练习", "学车", "驾校", "科目", "考驾照", "行车记录仪", "记录仪",
         "安装", "拆解", "试玩", "科普", "缅北", "电诈", "哨兵", "维修",
         "刹车", "底盘", "续航", "电池", "电机", "怎么选", "值得买", "性价比",
         "入门", "新手", "峰哥", "网红", "女骑", "跑山", "偷肉", "骗局", "隐私",
         "冲顶", "起火", "爆燃", "燃爆", "自燃", "爆炸", "烧成", "火灾", "燃烧",
         "充电", "灵异", "未解之谜", "科学家", "老外", "外国", "日本", "秋叶原",
         "打瞌睡", "玩手机", "低头", "遮阳伞", "判决书", "律师", "说案"]

old_bvs = set()
for f in OLD_LISTS:
    p = Path(f)
    if p.exists():
        old_bvs |= {l.strip() for l in p.read_text().splitlines() if l.strip().startswith("BV")}
old_bvs |= {p.name.split(".")[0] for p in VID_DIR.glob("*.mp4")}

seen, keep, drop_kw, drop_old = set(), [], 0, 0
for line in Path(TSV).read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) < 2:
        continue
    bv, title = parts[0].strip(), parts[1]
    if bv in seen:
        continue
    seen.add(bv)
    if bv in old_bvs:
        drop_old += 1
        continue
    if any(k in title for k in BLOCK):
        drop_kw += 1
        continue
    keep.append(bv)

Path(OUT).write_text("\n".join(keep) + "\n")
print(f"候选 {len(seen)}，已下载跳过 {drop_old}，黑名单 {drop_kw}，保留 {len(keep)} -> {OUT}")
