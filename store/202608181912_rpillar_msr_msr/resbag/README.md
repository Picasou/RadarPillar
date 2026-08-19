# 202608181912_rpillar_msr_msr 训练报告（resbag skeleton，LLM 填充主观段）

> **文档定位**：<一句话定位>
> **数据来源**：`output/train_log/msr/202608181912_rpillar_msr_msr/`
> **评估口径**：moderate_R40（VoD EAA）

## 摘要

- 模型 / tag：`rpillar_msr` / `msr`
- best epoch：80
- 末 epoch：80
- map_r40：{'car': None, 'pedestrian': None, 'cyclist': None, 'mean': None}
- 参数量 / 计算量：0.1873 M / 5.597 GFLOPs
- commit：a3c395f

## 结论（LLM 填）

<best ckpt 当前复测 mAP，与对照的 gap 归因>

## 已知偏差（LLM 填）

<结构/数据/评估口径偏差>

## 复现指引（LLM 填）

```bash
python .claude/skills/resbag/resbag.py make \
  --output_root output/train_log/msr/202608181912_rpillar_msr_msr \
  --dataset msr --tag msr --model rpillar_msr \
  --cfg_file <path> --batch_size <N>
```
