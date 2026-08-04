---
name: model-train
description: Use when the user asks to train models ("训练 X", "跑 X"). Generates a per-task workflow script with retry + endpoint verification, spawned in tmux (parent=/init, survives Claude session sleep). Generic — works for any training command + done-marker.
---

# model-train

泛化训练任务编排 skill。用户给任务列表 → 生成 workflow 脚本 → 在 tmux 内跑 → 4 类保护 + 5 类硬化自动生效。

## 入口

```bash
# 命令行 (单/多任务)
bash scripts/generate_workflow.sh \
    --tasks "tag1|cmd1|marker1,tag2|cmd2|marker2" \
    --max-retry 3

# 文件 (每行: tag<TAB>cmd<TAB>marker)
bash scripts/generate_workflow.sh --tasks-file tasks.tsv
```

参数:
- `--tasks` / `--tasks-file` 必填
- `--max-retry` 每任务重试次数 (默认 3)
- `--gpu-limit` MiB 阈值, brief 超此值告警 (默认 7000)
- `--no-gpu` 关 GPU 监听
- `--task` 自定义任务名

输出: `workflow_<tags>.sh` (skill 默认放项目根)

## skill 调用方 (agent) 的下一步

```bash
SKILL=/path/to/.claude/skills/model-train
WORKFLOW=/path/to/workflow_<task>.sh

# 1. tmux 启动 (parent=/init)
bash $SKILL/helpers/tmux_spawn.sh rpillar_<TASK> /path/to/project \
    "bash $WORKFLOW 2>&1 | tee /tmp/<TASK>.log"

# 2. cron 装 brief + watchdog (错开 30s)
echo '*/10 * * * * bash $SKILL/scripts/brief.sh <TASK>' | crontab -
echo '*/10 * * * * sleep 30 && bash $SKILL/helpers/watchdog.sh <TASK> <WORKFLOW>' >> crontab -l
```

## 4 类保护机制

| 机制 | 文件 | 作用 |
|---|---|---|
| **tmux_spawn** | `helpers/tmux_spawn.sh` | 进程挂 /init, 跨 Claude session 存活 |
| **watchdog** | `helpers/watchdog.sh` | driver 死了 → 10min 内 tmux 自动重启 |
| **brief** | `scripts/brief.sh` | cron 10min 写 ep/loss/ETA + DRIVER 健康 + GPU 显存 |
| **done_notifier** | `helpers/done_notifier.sh` | 监听完成 + 校验所有 marker + 触发 post-task hook |

## 5 类硬化 (H1-H5)

| # | 硬化 | 文件 | 堵盲区 |
|---|---|---|---|
| H1 | pipeline.sh 内置 retry N 次 | `scripts/pipeline.sh` | NaN/OOM 偶发, 单任务失败 |
| H2 | watchdog 启动检查 cron, 死了自动启 | `helpers/watchdog.sh` | cron 守护进程死 |
| H3 | brief 扫 nvidia-smi, 显存 >7G 告警 | `scripts/brief.sh` | GPU OOM 无痕 |
| H4 | done_notifier 校验所有 marker 齐才标 complete | `helpers/done_notifier.sh` | 部分任务空洞 |
| H5 | generate_workflow.sh 末尾 `bash -n` 语法检查 | `scripts/generate_workflow.sh` | 生成脚本有语法 bug |

## 任务 spec 格式

每项 `tag|cmd|marker`:
- **tag**: 标识符 (用于命名输出 + 日志)
- **cmd**: 完整 shell 命令 (含 cd / python 调用)
- **marker**: 成功标志文件 (workflow 跳过 + done_notifier 校验)

示例:
```bash
# RadarPillar
"b1|python tools/train.py --cfg experiments/YAML/b1.yaml --bs 8|output/b1/model_store.yaml"

# 任意
"exp1|python train.py --data data.h5|output/exp1/done.json"
```

## 手动操作

```bash
tmux attach -t rpillar_<TASK>            # 看实时 (Ctrl-b d 退出)
tmux kill-session -t rpillar_<TASK>      # 杀
tail -f /tmp/<TASK>.brief.out             # 简报
bash helpers/watchdog.sh <TASK>           # 手动重启
```

## 任务用完清理

```bash
crontab -l | grep -v "<TASK>" | crontab -       # 撤 cron
tmux kill-session -t rpillar_<TASK>             # 杀 tmux
rm -f /tmp/<TASK>.* /tmp/<TASK>.brief.out       # 清临时
rm -f workflow_<task>.sh                         # 清 workflow
# 训练产物 (output/) 保留, 是用户资产
```

## 不能保证的 (诚实声明)

- 训练**收敛**到目标指标 (cfg + 数据 + seed 决定)
- driver **永不死** (10min 内救)
- 训练**永不 OOM** (H3 告警 + H1 retry 覆盖大部分, 偶发仍可能)

## Legacy

- `scripts/run_all_repdwc.sh`: stage2 b5-b9 专用 driver (RadarPillar 项目专用)
- `实验报告模板.md`: 训练报告写作模板