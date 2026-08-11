---
name: model-train
description: Use when the user asks to train models ("训练 X", "跑 X"). User gives tag list only; skill auto-finds each model's train script + cfg, generates a per-model _full.sh (train+eval+pickbest[max+median]+resbag), runs them serially in tmux (parent=/init, survives Claude session sleep) with skip+retry, then emits a cross-experiment report. 4 protections + hardenings.
---

# model-train

训练任务编排 skill。**用户只说要训练哪些模型（tag 列表）**，skill 自动找每个模型的训练脚本 + cfg，从模板生成 `_full.sh`（train+eval+pickbest[max+median]+resbag 一条龙），串行调度（marker 跳过 + retry），全跑完出跨实验综合报告。4 类保护 + 硬化自动生效。

## 完整工作流

```
① 生成  你说 --tasks "n2,n3"
         skill 找 train_rpillar_<tag>.sh + <tag>.yaml
         从 templates/full_chain.template.sh 生成每个 _full.sh
         生成 workflow 调度器, tmux 启动

② 运行  (tmux, 全程保护: tmux保活 / watchdog救场 / brief汇报)
         每模型串行跑 _full.sh:
           a. train → b. eval 末 N ckpt → c. pickbest(max+median双落) → d. resbag
         落 marker; 挂了重试 N 次; 已完成跳过

③ 收尾  全模型跑完 → make_conclusion.py 出数据表 + 套报告模板 → 综合报告
```

## 入口

```bash
# 你只给 tag 列表
bash scripts/generate_workflow.sh --tasks "n2,n3" --max-retry 3
# 或文件 (每行一个 tag)
bash scripts/generate_workflow.sh --tasks-file tags.txt
```

参数:
- `--tasks` / `--tasks-file` 必填（纯 tag，逗号分隔或每行一个）
- `--max-retry` 整链重试次数 (默认 3)
- `--epochs` / `--bs` 训练超参 (默认 80 / 8)
- `--train-dir` / `--yaml-dir` 覆盖路径约定 (默认 `experiments/SH` / `experiments/YAML`)
- `--dataset` 数据集 (默认 vod)
- `--task` 自定义任务名

约定：tag=n2 → `experiments/SH/train_rpillar_n2.sh` + `experiments/YAML/n2.yaml`。

## 中间脚本隔离规则（硬性）

**生成的 workflow、`_full.sh` 等中间脚本一律落 `.tmp/model-train/<task>/`，绝不进正式目录**（项目根 / `experiments/` / `tools/` 等）。正式目录只放源码与用户资产。任务用完后 `.tmp/` 随手清。

- `workflow_<task>.sh` → `.tmp/model-train/<task>/`
- `train_<tag>_full.sh` → `.tmp/model-train/<task>/`
- 综合报告（`_report_<task>.md`）→ `output/train_log/<dataset>/`（这是结果产物，不是中间脚本，正常落 output）

## skill 调用方 (agent) 的下一步

```bash
SKILL=/path/to/.claude/skills/model-train
WORKFLOW=/path/to/workflow_<task>.sh

# 1. tmux 启动 (parent=/init)
bash $SKILL/helpers/tmux_spawn.sh rpillar_<TASK> /path/to/project \
    "bash $WORKFLOW 2>&1 | tee /tmp/<TASK>.log"

# 2. cron 装 brief + watchdog (错开 30s)
( crontab -l 2>/dev/null
  echo "*/10 * * * * bash $SKILL/scripts/brief.sh <TASK> >> /tmp/<TASK>.brief.out 2>&1"
  echo "*/10 * * * * sleep 30 && bash $SKILL/helpers/watchdog.sh <TASK> <WORKFLOW>"
) | crontab -
```

> **禁止用 CronCreate 加第二层汇报。** brief.sh 已每 10min 写进度到 `/tmp/<TASK>.brief.out`，用户用 `tail -f` 查看。agent 不需要再用 Claude session cron 读文件再汇报——那是重复且无法自动清理。

## 4 类保护机制

| 机制 | 文件 | 作用 |
|---|---|---|
| **tmux_spawn** | `helpers/tmux_spawn.sh` | 进程挂 /init, 跨 Claude session 存活 |
| **watchdog** | `helpers/watchdog.sh` | driver 死了 → 10min 内 tmux 自动重启 |
| **brief** | `scripts/brief.sh` | cron 10min 写 ep/loss/ETA + DRIVER 健康 + GPU 显存 |
| **done_notifier** | `helpers/done_notifier.sh` | 监听完成 + 校验所有 marker + 触发 post-task hook |

## 硬化 (H2-H5)

| # | 硬化 | 文件 | 堵盲区 |
|---|---|---|---|
| H1 | workflow 循环 marker 跳过 + 整链 retry N 次 | 生成的 `workflow_<task>.sh` (`run_model`) | 单模型偶发失败 |
| H2 | watchdog 启动检查 cron, 死了自动启 | `helpers/watchdog.sh` | cron 守护进程死 |
| H3 | brief 扫 nvidia-smi, 显存 >7G 告警 | `scripts/brief.sh` | GPU OOM 无痕 |
| H4 | done_notifier 校验所有 marker 齐才标 complete | `helpers/done_notifier.sh` | 部分任务空洞 |
| H5 | generate_workflow.sh 末尾 `bash -n` 语法检查 | `scripts/generate_workflow.sh` | 生成脚本有语法 bug |

## 结构

| 层 | 谁 | 干啥 |
|---|---|---|
| 模板 | `templates/full_chain.template.sh` | 链逻辑只写一份；按 tag 渲染出各 `_full.sh` |
| 链 | 生成的 `train_<tag>_full.sh` | train（`--skip_eval`）+ eval 末 N ckpt + pickbest（**max + median 双落**：`best.pth` + `best_median.pth`）+ resbag + 落 marker |
| 调度 | 生成的 `workflow_<task>.sh`（`run_model`） | marker 跳过 + 整链 retry N 次，串行调 `_full.sh` |
| 报告 | `make_conclusion.py` + `实验报告模板.md` | 数据表（喂模板核心结果段）+ 模板骨架 |

marker = `output/<TAG>.done`（`_full.sh` 末尾落）。pickbest 口径 = `Car_3d/moderate_R40`。

## 任务 spec 格式

每项一个 `_full.sh` 路径（TAG 从文件名推，marker 自动 = `output/<TAG>.done`）：

```bash
--tasks "experiments/SH/train_n2_full.sh,experiments/SH/train_n3_full.sh"
```

全模型跑完后，workflow 自动调 `make_conclusion.py` 聚合所有 model_store.yaml，出 `output/train_log/<dataset>/_conclusion_<task>.md`（对比表 + 最优 + 异常）。

## 手动操作

```bash
tmux attach -t rpillar_<TASK>            # 看实时 (Ctrl-b d 退出)
tmux kill-session -t rpillar_<TASK>      # 杀
tail -f /tmp/<TASK>.brief.out             # 简报
bash helpers/watchdog.sh <TASK>           # 手动重启
```

## 任务用完清理（workflow done 后必须执行）

```bash
crontab -l | grep -v "<TASK>" | crontab -       # 撤 OS cron
tmux kill-session -t rpillar_<TASK>             # 杀 tmux
rm -f /tmp/<TASK>.* /tmp/<TASK>.brief.out       # 清临时
rm -f workflow_<task>.sh                         # 清 workflow
# 训练产物 (output/) 保留, 是用户资产
```

> **如果用 CronCreate 设过 session 级汇报 cron，必须用 CronDelete 逐个撤掉。** 用 CronList 查残留。不撤干净会导致任务结束后无限空转汇报。

## 不能保证的 (诚实声明)

- 训练**收敛**到目标指标 (cfg + 数据 + seed 决定)
- driver **永不死** (10min 内救)
- 训练**永不 OOM** (H3 告警 + H1 retry 覆盖大部分, 偶发仍可能)

## 报告体系

- **resbag README**（单模型）：落袋时 resbag.py 自动写，存证 + 该模型数据。
- **综合报告**（多实验，全跑完自动生成）：`make_conclusion.py` 聚合所有 model_store.yaml 出数据表（mAP/参数/算力对比 + 最优 + 异常），附 `实验报告模板.md` 骨架。判断段（结论/gap 归因）由 LLM/人填。落 `output/train_log/<dataset>/_report_<task>.md`。