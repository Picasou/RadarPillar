---
name: model-train
description: Use when the user asks to train a RadarPillar model ("训练 MDFEN", "跑 xxx 模型", "开始训练"), or to run model training end-to-end. Handles the full training pipeline: generate train_<model>.sh from template, launch via WSL, register 10-min cron log briefings, post-training val on last 20 epochs + pick best.pth, and write experiment record. 训练场景覆盖 long-term-task-plan 的通用简报规则。在用户要求训练雷达检测模型时使用。
---

# 模型训练全链路

## 概述

RadarPillar 模型训练的**端到端 0 介入流水线**：一次性入参 → 调脚本生成 `.sh` → 启动 → cron 汇报 → 末 20 epoch val → best.pth → 实验记录。

**核心原则：能用脚本完成的，绝不依赖大模型。** 确定性/可解析/可模板化的任务（脚本渲染、自检、日志解析、best 挑选、记录生成）全部固化在 [train_pipeline.py](train_pipeline.py)；大模型只负责**解析用户自然语言入参 → 调脚本 → 处理异常**。

本 skill 是 long-term-task-plan 的**领域层**：训练场景下覆盖 long-term 的通用规则（10min 周期一致；嵌套时两份简报合并）。

## 何时使用

用户要求训练 RadarPillar 模型时（"训练 X"、"跑 X"、"开始训练 X"）。

**不使用**：纯 eval/推理（用 eval_radarpillar.sh）；tracker 仿真（非训练）。

## 一次性入参（收集后全程 0 介入）

| 入参 | 用途 | 必填 |
|---|---|---|
| 模型名 | `train_<模型>.sh` 文件名 + 日志目录标识 | 必填 |
| cfg 路径 | 训练 cfg；缺时自动在 `tools/cfgs/` 模糊匹配 | 必填 |
| 已有 `.sh` 模板路径 | 跳过自动造壳；缺则 skill 自动从 `train_radarpillar.sh` 模板造 | 可选 |
| 数据集 | OUTPUT_ROOT 数据集段 + cfg 命名一致性提示 | 必填 |
| 超参 | batch_size / workers / epochs / GPU | 默认 16/2/80/0 |
| 备注 tag | OUTPUT_ROOT 目录尾段 + `.tmp/<日期>/<slug>/` 的 slug 段 | 必填 |
| 是否可视化 | 写入 `.sh`（对齐 temp.md 第 6 点） | 默认 False |

> 缺失项用合理默认填，不反问——除非关键项（cfg/模型名）完全无法推断。

> **slug 命名建议**：调 `init_lt_task.py` 时 `--slug train-<模型>-<tag>`（例：`train-rp-base`），与目录约定对齐。`tag` 简短可读，不要含空格、日期（日期由 OUTPUT_ROOT 时间戳段承担）、特殊符号；`tag` 决定 `OUTPUT_ROOT` 尾段与 `.tmp/<日期>/<slug>/` 的 slug 命名。
>
> **模型名校验**：模型名走 `[a-z0-9_-]+`，非法字符（空格、`/`、`.`、中文等）会在 make_shell/gen 报错，避免路径注入。

## 端到端流程（每步都是脚本调用）

```dot
digraph mt {
  rankdir=LR;
  node [shape=box];
  "1 preflight" -> "2 gen";
  "2 gen" -> "3 WSL 启动";
  "3 WSL 启动" -> "4 挂 cron brief+autofinish";
}
```
> 步骤 4 注册两条训练机 cron：brief（每 10min 训练汇报）+ autofinish（每小时检查，训完自动收尾）。两路都由训练机本地驱动，不依赖 Claude 会话。

### 执行约定

所有训练机命令统一经 WSL 下发（训练在 WSL2 Linux，Claude 在 Windows）：
```bash
wsl bash -lc 'cd <工程根> && python .claude/skills/model-train/train_pipeline.py <子命令> ...'
```
**工程根与 conda env 名由各 `.sh` 内既有逻辑自探测**（`cd $(dirname)/../..` + conda 自激活），本 skill 不绑定具体路径/env，换机可用。

### 1. 启动前自检（preflight，护城河，不可跳过）

```bash
wsl bash -lc 'cd <工程根> && python .claude/skills/model-train/train_pipeline.py preflight \
  [--model <模型> | --cfg_file <cfg>] --dataset <数据集> --batch_size <bs>'
```
- 未给 `--cfg_file` 时传 `--model`，脚本自动在 `tools/cfgs/` 定位 cfg（不靠 LLM glob）。
- 脚本自动核验：cfg 存在 / detector 类已注册（模型落地）/ 数据集可定位 / 列既有 OUTPUT_ROOT 命名风格 / batch 显存提示。**任一失败 → 退出码非 0，LLM 不得裸启动。**

> 这一步取代 LLM 手动 grep/Glob 核验——确定性检查就该交给脚本。

### 2. 生成 `train_<模型>.sh`（gen + 自动造壳）

```bash
wsl bash -lc 'cd <工程根> && python .claude/skills/model-train/train_pipeline.py gen \
  --model <模型> --dataset <数据集> --cfg_file <cfg> \
  --batch_size <bs> --workers <w> --epochs <ep> --gpu <g> \
  --tag <备注> [--visualize]'
```

**自动造壳路由**（按 `train_<模型>.sh` 是否已存在）：

| 场景 | 动作 |
|---|---|
| `train_<模型>.sh` 已存在 | gen 仅渲染顶部 7 变量（CFG_FILE / BATCH_SIZE / WORKERS / EPOCHS / GPU / EXTRA_TAG / OUTPUT_ROOT），**不动壳内部** |
| `train_<模型>.sh` 缺 | gen 自动调 `make_shell`：复制 `train_radarpillar.sh` 模板 → 把 `CFG_FILE` 默认值行替换成 `--cfg_file` → 落盘 `train_<模型>.sh` |
| 想强制覆盖已存在壳 | 显式调 `make_shell --force` 或在 gen 前 `--no_auto_make_shell` 后手工 make |

仅造壳（不渲染顶部变量）：
```bash
wsl bash -lc 'cd <工程根> && python .claude/skills/model-train/train_pipeline.py make_shell \
  --model <模型> --dataset <数据集> --cfg_file <cfg> [--force]'
```

**OUTPUT_ROOT 自动**：`output/train_log/<数据集>/<YYYYMMDDHH>_<模型>_<备注>/`

**壳内部默认（skill 规约，**不动**）**：
- `SKIP_EVAL=True`
- `SET_CFGS=("OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False")`
- `RUN_MODE=background`
- conda env 自探测 `angle`

> 这一步取代 LLM 手动写 `.sh`——模板替换 + 自动造壳是纯确定性，不该靠大模型手写。

### 3. WSL 启动

```bash
wsl bash -lc 'cd <工程根> && bash .claude/skills/model-train/train_<模型>.sh'
```
记录 PID / LOG 路径 / OUTPUT_ROOT 到 `.tmp/<YYYY-MM-DD>/<slug>/<slug>.md`（slug 建议 `train-<模型>-<tag>`，与 init_lt_task.py 同目录约定）。骨架复用 [init_lt_task.py](../../long-term-task-plan/init_lt_task.py)，沿用 long-term schema；LLM 在该文件里追加 PID/LOG/OUTPUT_ROOT 三个字段值。

### 4. 挂 cron：训练汇报 + 收尾（训练机本地驱动）

**关键：所有定时任务都在训练机本地 crontab 驱动，Claude 不参与定时循环。** Claude 会话一旦关闭，会话内 CronCreate 就失效；只有训练机本地 crontab 可靠。

注册两条 crontab：

**(a) 每 10min 简报**（brief）：
```bash
wsl bash -lc '(crontab -l 2>/dev/null; echo "*/10 * * * * cd <工程根> && python .claude/skills/model-train/train_pipeline.py brief --model <模型> --log <LOG> --output_root <OUTPUT_ROOT> >> <OUTPUT_ROOT>/brief.log") | crontab -'
```

**(b) 每小时收尾检查**（autofinish，见步骤 5）。

brief 子命令自动解析 LOG（epoch/loss/lr/ETA/NaN/OOM）+ 按 temp.md 第 5 点模板格式化输出。**日志解析是确定性任务，绝不靠大模型 tail+理解。** brief 输出追加到 `OUTPUT_ROOT/brief.log`，是后续 `record` 子命令的简报历史数据源。**不要**用 `.tmp/<日期>/<slug>/` 替代——产物（`OUTPUT_ROOT/`）与临时进度文件（`.tmp/`）分开存放。

### 5. 收尾链（autofinish，训练机 cron 驱动，不依赖会话）

**关键：收尾（val→pickbest→record）也由训练机 cron 驱动，不依赖会话内 LLM poll。** 训练结束的判定与收尾串跑全部在脚本里完成——会话即使关闭，收尾也会自动触发。

注册一条额外的训练机 crontab（比 brief 低频，如每小时检查一次是否训完）：
```bash
wsl bash -lc '(crontab -l 2>/dev/null; echo "0 * * * * cd <工程根> && python .claude/skills/model-train/train_pipeline.py autofinish \
  --model <模型> --dataset <数据集> --cfg_file <cfg> \
  --batch_size <bs> --workers <w> --epochs <ep> --gpu <g> \
  --tag <备注> --output_root <OUTPUT_ROOT> --log <LOG>") | crontab -'
```
autofinish 子命令自动：
1. 判定训练结束（目标 ckpt `checkpoint_epoch_<EPOCHS>.pth` 已生成）→ 未结束则跳过
2. 末 20 epoch val（复用 eval_radarpillar.sh 的 `all` 模式，`START_EPOCH=EPOCHS-20`）
3. pickbest：按 Car 3D AP 挑 best.pth 落到 `<OUTPUT_ROOT>/best.pth`
4. record：聚合配置/metric/best.pth/简报历史，生成 `experiments/<YYYYMMDDHH>_<模型>_<备注>.md`（参考 RPiN.md 风格）

> 这三步串跑是确定性的编排，不该靠会话内 LLM poll 触发。脚本判定训练结束即自动收尾。

收尾完成后，清理 `.tmp/<YYYY-MM-DD>/<slug>/` 整棵目录（用 `rm -rf`）+ 对应的 brief/autofinish 两条 crontab 条目。

### 6. 训练报告（autofinish 完成后 LLM 撰写）

脚本只产**原料**（experiments/<...>.md + best.pth + OUTPUT_ROOT/eval + 可视化），**报告文本由 LLM 手动撰写**，对齐 [`实验报告模板.md`](实验报告模板.md)（同目录下）的六章结构（实验设置 / 核心结果 / 关键排查 / 评价 / 后续 debug / 产物清单）。模板已留好 `<占位符>`，LLM copy 骨架填充即可。

**触发时机**：`autofinish` 跑完 `val → pickbest → record` 后，**LLM 一次性手写报告**；不靠 cron，不靠会话内 poll 触发。

**报告落点**：
- 建议位置：`note/<模型名>/<YYYYMMDD>_<模型>_<tag>_报告.md`
- 示例参照：`note/radarpillar复现结论.md`（已按这约定写好）

**图片引用约定**（**不拷贝**）：

- 用 markdown 相对链接，**源图留在原地**——LLM 写报告时把路径抄进 `![alt](asset/<模型>/xxx.png)`
- 资产目录约定 `note/asset/<模型>/` 下放：
  - `loss_curve.png`（训练 loss 曲线）
  - `tb_loss_curves.png`（tensorboard 截屏）
  - `<模型>_frames/frame_NNNNN.png`（BEV 帧采样 8 张）
  - 其它调试对比图
- 已有的 `note/asset/radarpillar/` 是参照样例

**报告输入原料**（LLM 从以下源头取数据，不用 grep 瞎找）：

- 配置 + metric + best 路径：`experiments/<时间戳>_<模型>_<tag>.md`（`record` 子命令产物）
- 简报历史：`OUTPUT_ROOT/brief.log`
- 评测原始数据：`OUTPUT_ROOT/eval/epoch_*/val/*/result.*`
- 可视化产物：见「图片引用约定」

> 报告含主观判断（gap 归因、后续 debug 优先级），**LLM 写，脚本不替写**——脚本替写会把判断偷渡成代码。

## 自愈（对齐 long-term 0 介入）

- **NaN/OOM**：brief 检测到 → 记 `.tmp/` 阻塞段 → 降 batch / 续 ckpt 自愈；自愈失败才升级人工。
- **进程中断**：依赖 ckpt 续跑，`.tmp/` 记中断点。
- **cron 未触发**：`.tmp/` + LOG 仍是 ground truth，可手动恢复。

## Rationalization 表（堵漏洞，源自 baseline 实测）

| 借口 | 现实 |
|---|---|
| 「脚本难记，我直接手写 `.sh`」 | 模板替换是确定性的，大模型手写易出错。必须用 `gen` 子命令。 |
| 「自检我自己 grep 一下就行」 | 确定性检查交给 `preflight` 脚本，别用大模型手动核验。 |
| 「简报我自己 tail 日志读一下」 | 日志解析是确定性的。必须用 `brief` 子命令 + cron 驱动。 |
| 「best 我自己看下哪个好」 | metric 比较是确定性的。必须用 `pickbest`（autofinish 内置）。 |
| 「训完我 poll 到进程退出再触发 val/record」 | 收尾靠会话内 LLM poll 会断流。必须用 `autofinish` + 训练机 cron，训练结束自动收尾。 |
| 「SKIP_EVAL 关掉，边训边 eval 更稳」 | 违反 temp.md 第 2 点。默认 SKIP_EVAL=True。 |
| 「末 20 epoch 太多，评 5 个就行」 | 违反 temp.md 第 3 点。必须末 20。 |
| 「cron 麻烦，我用 CronCreate 自己定时」 | 会话关闭就失效。必须训练机本地 crontab 驱动。 |
| 「模型/cfg 先猜个路径试试」 | preflight 会拦。未落地模型不裸启动。 |
| 「可视化我等会再配」 | 违反 temp.md 第 6 点。`gen` 时就要定 `--visualize`。 |
| 「`.tmp/` 文件随便放顶层也行」 | 违反 `.tmp/<日期>/<slug>/` 目录约定。进度文件按任务独占子目录，便于按日/按任务清理。 |
| 「脚本替我写报告省事」 | 报告含主观判断（gap 归因、后续 debug 优先级），LLM 写；脚本只产原料（配置 / metric / 路径）。 |
| 「图片我复制一份到 note」 | 用 markdown 相对链接，**不拷贝**。源图在 `note/asset/<模型>/`，删源图才需要重引。 |
| 「换模型我自己复制 `.sh` 就行」 | skill 现已支持自动造壳（`make_shell` + `gen --auto_make_shell`），用户不该手工复制。 |
| 「壳内部 cfg 默认值我自己改」 | `make_shell` 自动替换 CFG_FILE 默认值行；改其它壳内部行（conda / SKIP_EVAL 等）属于改规约，应当改 skill 模板而非手工改壳。 |

## Red Flags — 停下重来

- 跳过 `preflight` 直接 `gen`/启动
- 手写 `.sh` 而非用 `gen` 子命令
- 手动 tail 日志做简报而非 `brief` + cron
- 手动比较 metric 挑 best 而非 `pickbest`（autofinish 内置）
- 收尾（val/pickbest/record）靠会话内 LLM poll 而非 `autofinish` + 训练机 cron
- 跳过末 20 epoch val
- 简报或收尾由 Claude 会话内驱动而非训练机 cron
- 训练中途反问用户超参/可视化
- 未生成实验记录 md 就宣告完成

**以上任一 = 违反训练规范，立即修正。**
