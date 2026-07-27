# 2026072700_rpillar_a0_a0 训练报告（resbag skeleton，LLM 填充主观段）

> **文档定位**：<一句话定位>
> **数据来源**：`output/train_log/vod/2026072700_rpillar_a0_a0/`
> **评估口径**：moderate_R40（VoD EAA）

## 摘要

- 模型 / tag：`rpillar_a0` / `stage1_Astar_a0`
- best epoch：73
- 末 epoch：80
- map_r40：{'car': 37.69, 'pedestrian': 31.7, 'cyclist': 66.92, 'mean': 45.44}
- 参数量 / 计算量：0.1778 M / 68.607 GFLOPs
- commit：fe5fe64

## 结论（LLM 填）

<best ckpt 当前复测 mAP，与对照的 gap 归因>

## 已知偏差（LLM 填）

<结构/数据/评估口径偏差>

## 复现指引（LLM 填）

```bash
python .claude/skills/resbag/resbag.py make \
  --output_root output/train_log/vod/2026072700_rpillar_a0_a0 \
  --dataset vod --tag stage1_Astar_a0 --model rpillar_a0 \
  --cfg_file <path> --batch_size <N>
```
