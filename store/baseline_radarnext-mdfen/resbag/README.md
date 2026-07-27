# 2026072222_radarnext_mdfen_0722_paper 训练报告（resbag skeleton，LLM 填充主观段）

> **文档定位**：<一句话定位>
> **数据来源**：`output/train_log/vod/2026072222_radarnext_mdfen_0722_paper/`
> **评估口径**：moderate_R40（VoD EAA）

## 摘要

- 模型 / tag：`radarnext_mdfen` / `0722_paper`
- best epoch：80
- 末 epoch：80
- map_r40：{'car': 32.55, 'pedestrian': 35.24, 'cyclist': 68.58, 'mean': 45.46}
- 参数量 / 计算量：1.6401 M / 113.222 GFLOPs
- commit：c2201e2

## 结论（LLM 填）

<best ckpt 当前复测 mAP，与对照的 gap 归因>

## 已知偏差（LLM 填）

<结构/数据/评估口径偏差>

## 复现指引（LLM 填）

```bash
python .claude/skills/resbag/resbag.py make \
  --output_root output/train_log/vod/2026072222_radarnext_mdfen_0722_paper \
  --dataset vod --tag 0722_paper --model radarnext_mdfen \
  --cfg_file <path> --batch_size <N>
```
