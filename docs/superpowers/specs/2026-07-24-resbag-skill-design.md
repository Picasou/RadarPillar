# resbag skill 设计文档（训练结果落袋归档）

- **日期**：2026-07-24
- **状态**：设计已确认，待落地
- **作者**：iHoward_PC + Claude
- **关联**：model-train skill（`.claude/skills/model-train/`）、RadarNeXt 移植主线

## 1. 背景与目标

### 1.1 现状痛点

- 训练结果散落：报告在 `experiments/*.md`、`note/<模型>/`，图像在 `note/asset/<模型>/`，权重在 `output/train_log/<ds>/<name>/ckpt/`，cfg 在 `tools/cfgs/`。**git 历史会改 cfg，无法回溯训练时所用 cfg**。
- 无结构化总览：跨实验对比靠人工 grep / 读散落 md。mAP 仅 Car 单类（record 旧逻辑），缺三类 + R11 双口径。
- 无「阶段贪心接力」锚点：best.pth 路径不固定、无 commit/seed/ts 元数据，后续阶段无法稳定引用前一阶段最佳权重。

### 1.2 目标

一次性入参 → 把一次训练/评估的产物**自包含硬复制归档**到 `<OUTPUT_ROOT>/resbag/`，并在同一实验目录下产出 `model_store.yaml` 单实验总览（每实验一份，不上 dataset 级中央）。跨实验对比走 `resbag list` 运行时聚合（仅 view 层）。可被 model-train 自动调用，也可手动调用（外部训练复用）。

### 1.3 非目标

- 不做冷备份/异地容灾（硬复制在本机 OUTPUT_ROOT 内，源与副本同盘）。
- 不替代 record（record 仍是训练流水线的实时记录产物；resbag 是训练完成后的归档快照）。
- 不做跨实验自动 diff/报告（`list` 子命令只聚合展示，分析交人）。

## 2. 架构定位

新建独立 skill `.claude/skills/resbag/`，单一职责：**落袋 + 索引**。

```
model-train autofinish:  val → pickbest → record
                                            ↓ 调用
                                      resbag make   ←── 也可手动 python resbag.py make ...
```

`model-train` 与 `resbag` 解耦：前者管训练流水线，后者管归档。手动训练/外部训练可绕过 model-train 直接调 `resbag make`。

## 3. 目录结构

落点：`output/train_log/<dataset>/<训练名>/resbag/`

```
resbag/
├── index.yaml         # 单实验结构化记录（权威，机读）
├── README.md          # 人读（结论 / 已知偏差 / 复现指引，LLM 撰写主观段）
├── cfg.yaml           # 模型/数据/训练 cfg 副本（训练时 git HEAD 的 cfg 快照）
├── train.sh           # 实际执行命令 + env + bs/epochs/workers
├── best.pth           # pickbest 选的 best（硬复制，阶段贪心接力锚点）
├── last.pth           # 末 epoch 权重（resbag 从 ckpt/ 最大 epoch 号复制的派生产物）
├── train.log          # loss/lr/time（复制自 OUTPUT_ROOT/log_train_*.txt）
├── eval_results.json  # best ckpt 的 eval 原始 ret_dict（权威 mAP 源）
└── asset/             # 硬复制自 OUTPUT_ROOT/asset/，保留原原子目录名（含 <model>_frames/ 等）
    ├── loss_curve.png
    ├── ap_curve.png
    └── <model>_frames/frame_NNNNN.png
```

**关于 `last.pth`**：model-train 流水线本身**不**生成 last.pth（pickbest 只产 best.pth）。resbag make 负责从 `<OUTPUT_ROOT>/ckpt/checkpoint_epoch_<max>.pth` 复制生成 `<OUTPUT_ROOT>/resbag/last.pth`。**末 epoch = 实际 ckpt 最大号**，与 `cfg.NUM_EPOCHS` 无关——未训完时差异巨大（如 1 vs 80）。

**关于 `asset/`**：model-train 继续按现行规则落 `<OUTPUT_ROOT>/asset/`（不改）；resbag 硬复制整个 `OUTPUT_ROOT/asset/` 目录树到 `resbag/asset/`，保留原子目录名（如 `<model>_frames/`）。目录命名遵守源，不重写。

### 3.1 硬复制策略（已确认）

- **所有文件硬复制**（`shutil.copy2`，保元数据）。resbag 自包含、可独立迁移、不受 OUTPUT_ROOT 删除影响。
- 代价：权重占双倍空间（best+last 各 ~20MB × 实验数）。可接受——单实验 <50MB，归档价值 > 空间成本。
- 幂等：重复 `resbag make` 同一 OUTPUT_ROOT 时，按 index.yaml 已存 + ts 判断，覆盖更新而非累加。

### 3.2 单实验总览 `model_store.yaml`

落点：**`output/train_log/<dataset>/<训练名>/model_store.yaml`**（**每实验一份**，与 `resbag/` 同级；非 dataset 级中央文件）。

```yaml
version: 1
folder: 2026072222_radarnext_mdfen_0722_paper   # 与目录树完全匹配（即此文件所在目录名）
tag: 0722_paper
dataset: vod
map_r40: {car: 32.55, pedestrian: 35.24, cyclist: 68.58, mean: 45.46}
map_r11: {car: null, pedestrian: null, cyclist: null, mean: null}   # 恒 null：eval.py 注释了非 _R40 键（§5.1）
params_m: 1.6401
flops_g: 226.37
commit: c2201e2
ts: 2026-07-24
status: done             # done | blocked
note: best.pth=ep80
```

**架构变化**：从「单实验 index + dataset 级中央总览」改为 **「单实验总览 = 总览」(per-experiment model_store.yaml)**。优势：
- 每实验自包含：删 OUTPUT_ROOT 时 archive 仍完整（resbag/ + model_store.yaml 一个不少）。
- 无中央文件竞态：dataset 维度不必 fcntl.flock（同 dataset 并发写各自的实验目录，无共享写）。
- 跨实验对比：`resbag list` 仍支持运行时 glob 所有 `<dataset>/*/model_store.yaml` 聚合成总览（**只在 view 层**，不写中央盘）。

**写入原子性**（**强制要求**）：写 `model_store.yaml` 用 `temp-file + os.replace` 模式：

```python
tmp = model_store.with_suffix(f'.tmp.{pid}')
yaml.safe_dump(data, tmp.open('w'), sort_keys=False, allow_unicode=True)
os.fsync(tmp); os.replace(tmp, model_store)   # 同目录 rename 原子
```

**并发安全**：本文件的写盘无中央竞态（per-experiment 单写主），但**仍需** `/tmp/resbag_store_<dataset>_<name>.lock`（dataset+name 维度，fcntl.flock LOCK_EX）防同一实验多次 `make` 并发（幂等覆盖也应串行化，避免半成品）。与 autofinish 的 model 锁正交。

**`list -o` 边界**：list 默认只 glob 打印到 stdout；`-o` 仅写「另起的派生聚合文件」（如 `output/train_log/<dataset>/_index.yaml`），**禁止覆盖任何 model_store.yaml**——每实验 model_store.yaml 是 make 唯一写主，list 只读视图。

## 4. index.yaml schema（单实验）

```yaml
version: 1
folder: 2026072222_radarnext_mdfen_0722_paper   # 与目录树完全匹配
tag: 0722_paper                                  # OUTPUT_ROOT 尾段 = 训练名
dataset: vod
cfg: cfg.yaml                                    # 相对 resbag/
ckpt: best.pth
eval: eval_results.json
map_r11: {car: null, pedestrian: null, cyclist: null, mean: null}   # 恒 null：eval.py 注释了非 _R40 键（§5.1）
map_r40: {car: 32.55, pedestrian: 35.24, cyclist: 68.58, mean: 45.46}
params_m: 1.6401          # numel 实测，单位 M（含 trainable 标注）
flops_g: 226.37           # thop 实测 GFLOPs
metric_caliber:
  map: moderate            # easy/moderate/hard
  recall: R40              # R11/R40
  iou: {car: 0.5, ped_cyc: 0.25}
  filter: EAA              # 评估过滤口径
seed: false                # cfg OPTIMIZATION.FIX_RANDOM_SEED
optimizer: {name: adamw, lr: 0.003, wd: 0.01, decay: [35, 45]}
commit: c2201e2           # 训练时 git HEAD（short）
ts: 2026-07-24           # 落袋时间
status: done             # done | blocked（仅 make 写）
note: best.pth=ep80      # 自由文本备注
```

**status 枚举收敛**：仅 `done` / `blocked` 两个值。make 在训练完成后跑，永远只可能这两种结果；`pending`/`running` 由 model-train 训练启动阶段负责（不在 resbag 范围）。

## 5. 数据源与解析

### 5.1 mAP（读 eval_results.json，权威）

**results.json 定位**（**统一 rglob**，不写死路径层数）：

```
output_root/
└── eval/
    ├── epoch_<best>/val/<eval_extra_tag>/results.json      # 单点 eval（最常见）
    └── eval_all_default/default/epoch_<best>/val/.../      # 末 N epoch 全量 eval
        └── results.json
```

实际两种布局层数不同（FPN 用后者），**禁止**字面写 `eval/epoch_<best>/...`。resbag 用 `output_root.rglob('results.json')` 收集所有 candidates，再按内部路径中的 `epoch_<N>` 与 `best_epoch` 精确匹配取对应那份。

`best_epoch` 推导：从 `OUTPUT_ROOT/best.pth` 字节大小 → 匹配 `OUTPUT_ROOT/ckpt/checkpoint_epoch_*.pth` 找同字节的 ep（已实测：mdfen best.pth=20281290 == checkpoint_epoch_80.pth；fpn best.pth=13720574 == checkpoint_epoch_61.pth）。

**三类 R40 解析**：results.json `ret_dict` 直接读 `Car_3d/moderate_R40` / `Pedestrian_3d/moderate_R40` / `Cyclist_3d/moderate_R40`。mAP mean = 三类均值（缺类不计、留 null）。

**R11（**显式声明**：当前永远 `null`**）：
- eval.py 中所有非 `_R40` 的 `ret_dict` 赋值被注释掉（`pcdet/datasets/kitti/kitti_object_eval_python/eval.py:728-736`），results.json 实际只产 `_R40` 键。
- 因此 `map_r11` 全字段恒为 `null`，直至 eval.py 取消注释后重跑。
- **不要**写「有则填无则 null」的歧义措辞——这是结构性的，不是个例缺失。
- 若未来要 R11：选 (a) 取消 eval.py 注释重跑；或 (b) 从 `train.log` 正则提取 `3d AP:...` 文本（与 pickbest fallback 同款），不推荐。

**best 索引策略**：pickbest 不写任何 epoch sidecar，必须从 best.pth 字节反推（见上）。

### 5.2 params / flops（实测）

**resbag 独立实现**（**不** import model-train `_count_params_flops`）：
- 理由：`_count_params_flops(cfg_file, model, batch_size)` 的 `model` 形参是死参数（body 内未引用），且 import train_pipeline 会带 ROOT/SHELLS_DIR 等全局副作用 + 命令行入口污染。
- 实现：build 网络（用 `__file__` 推导 ROOT，`sys.path.insert(0, ROOT)` 自保）→ `sum(p.numel())` 得参数量（含 trainable 区分）→ thop.profile 得 MACs/FLOPs（喂正确形状 `(M, num_points, features)` voxel + int32 coords）。
- 全程 try 兜底：build/thop 失败 → params_m/flops_g=null + note 注明「build/thop 失败：<类名>」，不阻塞落袋。
- 口径标注：`flops_g` 在 `metric_caliber` 注明 thop + bs。

### 5.3 commit / seed / optimizer

- `commit`：`git rev-parse --short HEAD`（make 运行时取，反映落袋时 HEAD；与训完时 HEAD 可能略偏，note 可附训完时间）。
- `seed`：读 cfg `OPTIMIZATION.FIX_RANDOM_SEED`。
- `optimizer`：读 cfg `OPTIMIZATION.OPTIMIZER/LR/WEIGHT_DECAY/DECAY_STEP_LIST`。

### 5.4 数据源全清单（防 resbag 再去 grep）

| index.yaml 字段 | 数据源 | 解析方式 |
|---|---|---|
| cfg（MODEL/DATA_CONFIG/OPTIMIZATION 三段） | 同 cfg.yaml 副本 | PyYAML load |
| best.pth | OUTPUT_ROOT/best.pth | shutil.copy2 |
| last.pth | OUTPUT_ROOT/ckpt/checkpoint_epoch_<max>.pth | max(int(re.search(...))) |
| best_epoch | OUTPUT_ROOT/best.pth ↔ ckpt/checkpoint_epoch_*.pth 字节匹配 | os.path.getsize |
| map_r40 三类 | best_epoch 对应 results.json 的 ret_dict | key 直读 |
| map_r11 | null（结构性） | — |
| params_m / flops_g | cfg.yaml build 网络 + thop | 独立实现 |
| commit | git rev-parse --short HEAD | subprocess |
| seed | cfg OPTIMIZATION.FIX_RANDOM_SEED | load |
| optimizer | cfg OPTIMIZATION.{OPTIMIZER,LR,WD,DECAY_STEP_LIST} | load |
| train.log | OUTPUT_ROOT/log_train_<YYYYMMDD-HHMMSS>.txt（最新一份） | max by mtime |
| train.sh | tools/scripts/train_<model>.sh | shutil.copy2 |
| eval_results.json | best_epoch 对应 results.json | shutil.copy2 |
| model_store.yaml | 产出位置 = OUTPUT_ROOT/model_store.yaml（per-experiment） | 由 resbag make 写，原子 temp + os.replace |

### 5.5 解析边界（明确禁区）

- **不**解析 record md 产物：record 的 `.md` 是 LLM 撰写的人读报告，散落 seed/optimizer/params 易被人改写；resbag 一律从 cfg + 重算/重读，不反喂 record。
- **不**解析 train.log 文本（除非未来要 R11 走正则方案）：results.json 是权威结构化源。
- **不**复用 model-train 的 stray 兜底（`record` line 782 硬编码 `vod_radarnext_mdfen` 路径）：resbag 用 rglob 从根解决。

## 6. 子命令（resbag.py）

| 子命令 | 作用 | 用法 |
|---|---|---|
| `make` | 主入口：收集 OUTPUT_ROOT 产物 → 建 resbag/（硬复制 cfg/sh/log/eval/pth、产 index.yaml + README 骨架）→ 写 `model_store.yaml` 单实验总览（per-experiment） | `python resbag.py make --output_root <DIR> --dataset <DS> --tag <TAG> --model <MODEL> [--note <NOTE>]` |
| `list` | glob 所有 `<dataset>/*/model_store.yaml` 聚合成跨实验总览（默认 stdout；`-o` 写另起的派生文件如 `<dataset>/_index.yaml`，**禁止覆盖任何 model_store.yaml**） | `python resbag.py list [--dataset <ds>] [-o <OUT>]` |
| `show` | 打印单个实验 index.yaml | `python resbag.py show --folder <name>` |

`make` 必需入参：`--output_root --dataset --tag --model`（dataset/tag/model 缺一不可——dataset 决定 model_store.yaml 父目录（`output/train_log/<dataset>/<name>/`），tag = OUTPUT_ROOT 尾段（用于从 OUTPUT_ROOT 名解析 + filename_ref 字段），model 决定 train.sh 源路径（`tools/scripts/train_<model>.sh`）与锁名）。

### 6.1 make 的健壮性

- 缺 best.pth → status=blocked，记 `note: best.pth missing`，仍落袋其余文件。
- 缺 eval/results.json → map_r40 全 null，status=blocked。
- params/flops build 失败 → params_m/flops_g=null + note 注明 `build/thop 失败：<类名>`，不阻塞落袋。
- 权重硬复制失败（磁盘满）→ 报错退出，不产半成品 index.yaml。
- 检测 `<OUTPUT_ROOT>/FINISHED_PARTIAL`（train_pipeline.py:949 训练中途崩溃标记）→ `status=blocked` + `note: crash@ep{max_epoch}`，区分 partial 收尾与正常 done。
- **last.pth 不存在不视为 blocked**：resbag 找不到 `ckpt/checkpoint_epoch_*.pth` 时 last.pth 不生成（其余文件照落）。

## 7. model-train 集成

### 7.1 autofinish 调用点（**修正版**）

在 `train_pipeline.py` 的 `cmd_autofinish` 中，`record` 步骤成功后、`[OK]` 打印之前新增（具体行：record 子进程 rc=0 检查之后、line 1023 print 之前）：

```python
# record 之后调用 resbag make（落袋归档）—— 必须传完整入参
resbag_py = str(ROOT / '.claude' / 'skills' / 'resbag' / 'resbag.py')
subprocess.run(
    [sys.executable, resbag_py,
     'make',
     '--output_root', str(output_root),
     '--dataset', args.dataset,
     '--tag', args.tag,
     '--model', args.model],
    cwd=ROOT,            # 必须：pcdet import + 路径相对
    timeout=600,         # 权重硬复制给 10min 余量
    check=False,         # 归档失败不让 autofinish 整体失败
    env=dict(os.environ),  # 继承 env（conda base / CUDA）
)
```

**关键修正**（相对原 spec）：
- 用 `ROOT`（不是 `REPO`，原 spec 错）→ resbag.py 路径 = `ROOT/.claude/skills/resbag/resbag.py`。
- **必传** `--model`（决定 train.sh 源）+ `--cfg_file`/`--batch_size`（入参组原 spec 漏了，导致 params/flops/optimizer/seed 全 null）—— 修正版同步补上 `--cfg_file`/`--batch_size`：

```python
subprocess.run(
    [sys.executable, resbag_py, 'make',
     '--output_root', str(output_root),
     '--dataset', args.dataset,
     '--tag', args.tag,
     '--model', args.model,
     '--cfg_file', args.cfg_file,
     '--batch_size', str(args.batch_size)],
    cwd=ROOT, timeout=600, check=False, env=dict(os.environ),
)
```

`check=False`：resbag 失败不让 autofinish 整体失败（归档是 nice-to-have，训练已成功）；但 `timeout=600` 防止 resbag 卡死占用 autofinish 的 `/tmp/autofinish_{model}.lock` 阻塞同模型下一次 cron。

### 7.2 SKILL.md 第 6 节改写

- 「报告落点」改为指向 `<OUTPUT_ROOT>/resbag/README.md`。
- 图像落 `<OUTPUT_ROOT>/resbag/asset/`。
- 删除「报告放 note/」的旧约定。

### 7.3 旧报告迁移

`note/radarpillar复现结论.md`、已迁的 `output/.../asset/` 等历史报告，按对应 OUTPUT_ROOT 迁到 `resbag/README.md` + `resbag/asset/`（一次性人工/脚本迁移，不在 autofinish 自动做）。

## 8. 实现计划

1. 建 `.claude/skills/resbag/`（SKILL.md + resbag.py + 模板）。
2. 实现 `make`：cfg/sh/log/eval 复制 + pth 硬复制 + index.yaml + README 骨架 + params/flops。
3. 实现 `list` / `show`。
4. 改 model-train：autofinish 调 resbag + SKILL.md 第 6 节。
5. 迁移历史报告（note/ → resbag/）。
6. 验证：对 `radarnext_mdfen_0722_paper` 跑 `resbag make`，核对 index.yaml 各字段。

## 9. 验收标准

- [ ] `resbag make` 对已训练 OUTPUT_ROOT 产出完整 resbag/（8 类文件齐全，权重硬复制可独立加载；best.pth / last.pth / eval_results.json 字节相同于源）。
- [ ] index.yaml 的 map_r40/params/flops/commit/seed/optimizer 与实测一致（对照 45.46 / 1.64M / 226 GFLOPs / c2201e2 / False / adamw）。
- [ ] map_r11 全 null（结构性，eval.py 不写非 _R40 键）。
- [ ] last.pth 来自 `ckpt/checkpoint_epoch_<max>.pth`（非 NUM_EPOCHS）。
- [ ] eval_results.json 经 rglob 命中两种布局（单点 + eval_all_default），按 best_epoch 正确匹配。
- [ ] model_store.yaml 落点为 per-experiment（`OUTPUT_ROOT/model_store.yaml`），按 ts 字段幂等覆盖更新；并发写用 fcntl.flock 串行化（dataset+name 锁）。
- [ ] 写入用 `temp + os.replace` 原子，损坏不留下半成品。
- [ ] model-train autofinish 跑完调 resbag 不报错（check=False 兜底）；resbag 卡死超时（timeout=600）不阻塞同模型下一次 cron。
- [ ] `list -o` 写另起的派生文件，**不**覆盖任何 model_store.yaml（中央或实验级都禁）。
- [ ] 历史报告从 note/ 迁到 resbag/README.md + resbag/asset/，note/ 无残留。

## 10. 风险与权衡

- **空间**：硬复制权重双倍占用（best+last 各 ~20MB × 实验数）。可接受：单实验 <50MB，归档价值 > 空间成本。缓解：可加 `--no-last` 跳 last.pth。
- **model_store 并发写**：per-experiment 单写主，无中央竞态；同实验多次 make 用 dataset+name 维度 lockfile 串行化（`/tmp/resbag_store_<dataset>_<name>.lock`，fcntl.flock LOCK_EX 阻塞），与 autofinish 的 model 锁正交。
- **commit 滞后**：落袋时 git HEAD 可能已比训练时新（继续 commit）。缓解：note 字段附 best epoch，必要时从 train.log 时间戳反推训完时间。
- **R11 恒 null**：eval.py 注释了非 _R40 键，map_r11 永远是 null。诚实标注而非「有则填」歧义；若要 R11 需主动改 eval.py 重跑。
- **last.pth 与 pickbest 契约错位**：pickbest 只产 best.pth，resbag 自负 last.pth 派生（ckpt/最大 epoch）。last.pth 字节 = ckpt/最大 epoch 文件，与 best.pth 可能不同（best 可能在中间 epoch）。
