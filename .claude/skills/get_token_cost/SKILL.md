---
name: get_token_cost
description: Use when the user asks about token usage / cost ("用了多少 token", "token 消耗", "成本统计"), or wants to visualize Claude Code token usage over time. Scans all project transcripts, aggregates usage by model × time bucket, and renders an interactive HTML bar chart (stacked by model, switchable metric: total/input/cache_creation/cache_read/output) with date-range and granularity (30min/1h/4h/1day) controls.
---

# Token 成本可视化

## 何时用

用户询问 token 用量、消耗统计、按时间/模型分布，或要求"可视化 token 成本"时。

## 产出

单文件交互式 HTML（Chart.js via CDN，内嵌全部数据，可分享）：

- **柱状图**：X = 时段，Y = token，每柱**按模型分段着色**
- **指标切换**：Total / Input / CacheCreation / CacheRead / Output
- **日期范围**：起始/终止日期选择（默认拉满可用区间）
- **时间颗粒度**：30min / 1h / 4h / 1day
- **KPI 条**：五项总量，点击即作为指标切换入口
- **图例**：各模型在该指标下的占比

## 执行

确定性逻辑全部在脚本里，本 skill 只负责调用：

```bash
python3 .claude/skills/get_token_cost/collect_cost.py
```

可选参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--root` | `~/.claude/projects` | 扫描根目录 |
| `--out` | `./.claude/skills/get_token_cost/token_cost.html` | 输出路径（默认与脚本同目录，每次覆盖） |
| `--start` / `--end` / `--bucket` | 全区间 / 1h | 仅作 UI 默认提示（过滤在前端动态完成，无需重跑脚本） |

执行后把生成的 HTML 路径告诉用户即可。**日期/颗粒度/指标的过滤都在前端完成，无需重跑脚本。**

## 数据口径

- 来源：`~/.claude/projects/*/*.jsonl` 中 assistant 消息的 `usage` 字段
- 四个指标：`input_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens` / `output_tokens`
- 模型取 `message.model`；缺失归为 `unknown`；`<synthetic>` 保留
- 时区 `Asia/Shanghai`

## 模型调色板（固定）

`glm-5.2`=`#4EA8DE` · `MiniMax-M3`=`#F0883E` · `k3`=`#A371F7` · `<synthetic>`=`#6E7681` · `unknown`=`#444C56`

新增模型沿用 `collect_cost.py` 的 `MODEL_COLORS` 顺序，未知模型自动着灰色。
