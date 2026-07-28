---
name: model-train
description: Use when the user asks to train a RadarPillar model ("训练 X", "跑 X"). Triggers .claude/skills/model-train/scripts/unified_rpillar_pipeline.sh — one shell does train → eval ×N → viz → pickbest → resbag, with cron progress reporting via .claude/skills/model-train/scripts/brief.sh.
---

# model-train

训练 RadarPillar 模型——**两脚本即够用**：

## 入口：`unified_rpillar_pipeline.sh`

```bash
# 一行启动（前台；可 nohup 后台跑）
bash .claude/skills/model-train/scripts/unified_rpillar_pipeline.sh

# 自定义（env var 覆盖）
MODEL=rpillar_a4_rezero \
CFG_FILE=experiments/YAML/a4_rezero.yaml \
LAST_N_EVAL=10 \
  bash .claude/skills/model-train/scripts/unified_rpillar_pipeline.sh

# 干跑一遍：只看参数不启动训练
SHOW_ARGS=1 bash .claude/skills/model-train/scripts/unified_rpillar_pipeline.sh
```

**流程**（同 shell 内串行，`set -euo pipefail`）：

```
step 1  参数配置 (env vars; :="${VAR:=default}")
step 2  环境激活 (PYTHONNOUSERSITE=1 + conda angle + GPU)
step 3  train (前端训练, --skip_eval, ~1h10m)
step 4  eval × N (末 LAST_N_EVAL=10 个 ckpt; 每 ~2.5 min GPU)
step 5  pickbest (Car_3d/moderate_R40 max → best.pth; inline)
step 6  resbag (artifact 落袋)
```

## 参数一览

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | `rpillar_a4_lnpost` | 模型 slug |
| `CFG_FILE` | `experiments/YAML/a4_lnpost.yaml` | 训练 cfg |
| `EPOCHS` | 80 | 总 epoch |
| `BATCH_SIZE` | 16 | 单 GPU |
| `WORKERS` | 2 | dataloader |
| `GPU` | 0 | CUDA_VISIBLE_DEVICES |
| `OUTPUT_ROOT` | 自动 `output/train_log/vod/<date>_<model>_<tag>/` | |
| `LAST_N_EVAL` | 10 | 末 N ckpt 跑 eval |
| `RUN_VIZ` | true | eval 后可视化 |
| `RUN_PICKBEST` | true | 按 Car R40 挑 best.pth |
| `RUN_RESBAG` | true | resbag 落袋 |
| `SHOW_ARGS` | 0 | 设 1 只显示参数 |

## cron：`brief.sh`（进度汇报）

每 10 min cron 跑一次解析训练 log：

```cron
*/10 * * * * /path/to/.claude/skills/model-train/scripts/brief.sh /path/to/train.log /path/to/output_root >> /path/to/brief.out 2>&1
```

输出格式（单行）：
```
[<time> CST] epN/M | loss=X.X lr=Y.Y | ETA=2h38m ≈完成 07-28 21:34 | nan=0 oom=0
```

## 推导：cfg → shell？

如果用户给 yaml：unified pipeline 自己 `train.py --cfg_file a4_lnpost.yaml` 起训。
如果用户给 shell：unified pipeline 仍调 `train.py --cfg_file $CFG_FILE`（cfg 永远在那），shell 是训练配置的统一壳。

**不要再生成 eval_rpillar_*.sh**——eval 在 unified pipeline step 4 里 inline 调 `tools/test.py` + 可选 `tools/visualize_eval.py`，不需要独立外壳。

## 监控/异常

- 训练期：crontab brief 10 min 自动拉进度
- 异常：unified pipeline `set -euo pipefail` 任一步非 0 退出 → 看 `${OUTPUT_ROOT}/logs/step*.log` 定位
- 跑完：unified 退出 0 → resbag 已落袋

## 任务用完

```
crontab -l | grep skill_id=rpillar  # 看当前 A/B 的 brief 条目
crontab -l | grep -v skill_id=rpillar_<model>_<rand> | crontab -  # 撤某条
rm -rf .tmp/<date>/<slug>/           # 删临时进度
```
