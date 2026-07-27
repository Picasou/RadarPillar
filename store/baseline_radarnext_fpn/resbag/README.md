# 2026072222_radarnext_fpn_0722_paper 训练报告（resbag skeleton，LLM 填充主观段）

> **文档定位**：<一句话定位>
> **数据来源**：`output/train_log/vod/2026072222_radarnext_fpn_0722_paper/`
> **评估口径**：moderate_R40（VoD EAA）

## 摘要

- 模型 / tag：`radarnext_fpn` / `0722_paper`
- best epoch：61
- 末 epoch：80
- map_r40：{'car': 30.96, 'pedestrian': 34.23, 'cyclist': 69.05, 'mean': 44.75}
- 参数量 / 计算量：1.1057 M / 83.659 GFLOPs
- commit：c2201e2

## 结论（LLM 填）

<best ckpt 当前复测 mAP，与对照的 gap 归因>

## 已知偏差（LLM 填）

<结构/数据/评估口径偏差>

## 复现指引（LLM 填）

```bash
python .claude/skills/resbag/resbag.py make \
  --output_root output/train_log/vod/2026072222_radarnext_fpn_0722_paper \
  --dataset vod --tag 0722_paper --model radarnext_fpn \
  --cfg_file <path> --batch_size <N>
```
