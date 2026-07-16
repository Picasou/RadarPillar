# RadarPillar 训练计划（VoD 数据集）

> 状态：审阅稿（未 commit）。批准后转入分步执行。
> 范围：训练 RadarPillar 两个变体（large 最佳 + base 基线对照），产出 loss 曲线与预测可视化，与原论文对标。

---

## 1. 目标与成功标准（总）

- **目标**：在 OpenPCDet fork 上，用 VoD 数据训练 RadarPillar（large + base），得到可用检测模型并完成论文级对比。
- **对比论文**：RadarPillars (Musiat et al., arXiv:2408.05020, 5-frame VoD val)。
- **评测口径**：VoD 官方 IoU（Car@0.5，Pedestrian/Cyclist@0.25），同时给 entire-area + driving-corridor。
- **成功判据**（"相差不能太多"）：
  - large 变体 Entire-area mAP 落在论文 **50.7 ± ~8 点**；各类 AP 与论文（Car 41.1 / Ped 38.6 / Cyc 72.6）偏离 ≤ ~10 点。
  - base 作为消融对照（预期 mAP ≤ large）。
  - 必须产出：① loss 曲线 PNG（TensorBoard 标量 + 训练 log 解析两版）；② ≥10 帧预测 PNG（BEV+相机双面板）；③ 论文对比表。
- **算力约束已知**：单卡 RTX 3070 Ti / 8 GB（论文用 RTX 4070 Ti、batch 8）。绝对 AP 可能因显存/算力偏低几到十几点，不影响 pipeline 正确性。
- **硬件实测结论（已验证）**：① 显存——RP-large bs=8 前向+反向峰值仅 **3.3 GB**，远低于 8GB，无 OOM 风险；② 数据 I/O——VoD 经 9p 挂载（`/mnt/d`），但 `workers=2` 多进程预读下每 batch 仅 83ms < GPU 计算 ~300ms，**9p 不构成训练瓶颈，无需迁移数据到本地盘**。

## 2. 环境与现状（总）

| 项 | 值 | 处置 |
|---|---|---|
| GPU | RTX 3070 Ti / 8 GB | batch 16→**8**（OOM 再降 4） |
| torch/spconv/pcdet | 2.4.1+cu124 / 2.3.8 / import OK | 无需改 |
| VoD 数据 | `data/VoD/.../radar_5frames/` infos+gt_db 已有 | 因 Cyclist 聚合需重生 |
| RadarPillar ckpt | 无 | 从零训 |

## 3. 已识别坑与预案（总）

| 坑 | 位置 | 预案 |
|---|---|---|
| 训练结束自动 eval 会 NameError | [train.py:6,267](../../../tools/train.py) `repeat_eval_ckpt` import 被注释但仍调用 | 强制 `--skip_eval`，eval 走独立 `tools/test.py` |
| 脚本路径失效 | [train_radarpillar.sh:55](../../../tools/scripts/train_radarpillar.sh)、[eval_vod.sh](../../../tools/scripts/eval_vod.sh) 指旧扁平路径 + 旧 dataroot | 适配新 `radarpillar/` 子目录与实际路径 |
| Cyclist GT 偏少 | [vod_dataset_radar.yaml](../../../tools/cfgs/dataset/vod_dataset_radar.yaml) `CLASS_MAPPINGS` 仅 no-op，两轮车被丢弃 | 聚合映射 + 重生 infos |

## 4. 配置选型（分）

| 变体 | yaml | 特征 | 用途 |
|---|---|---|---|
| **large（best）** | `radarpillar/vod_radarpillar_large.yaml` | 2D backbone 64/128/256、论文 anchor 尺寸(4.17/1.84/1.57)、旋转增强、NUM_FILTERS[64]、NMS 0.1、80ep 需手设 | 主结果，对标论文 |
| base（对照） | `radarpillar/vod_radarpillar.yaml` | PillarAttention 小 backbone、velocity decomposition、NUM_FILTERS[32]、NMS 0.15、自带 80ep | 消融对照 |

> 另有 `vod_radarpillar_rot.yaml`（base+旋转）、`vod_radarpillar-yedek.yaml`（实验性 backup）本次不用。

## 5. 执行步骤（分）

### 5.1 前置适配
1. **备份现有 infos**：`cp -r data/VoD/.../radar_5frames/*.pkl .tmp/infos_backup/`。
2. **Cyclist 聚合（按 VoD 官方协议）**：改 [vod_dataset_radar.yaml](../../../tools/cfgs/dataset/vod_dataset_radar.yaml) `CLASS_MAPPINGS`，加 `bicycle/rider/moped_scooter/motor → Cyclist`；`bicycle_rack/human_depiction/ride_other/ride_uncertain` 不计入（先核对本地 [eval.py](../../../pcdet/datasets/kitti/kitti_object_eval_python/eval.py) 类映射确保训练 GT 与评测口径一致）。
3. **重生 infos**：`python tools/scripts/create_vod_data.py`（覆写 `radar_5frames/` 下 pkl + gt_database）。
4. **修脚本** [train_radarpillar.sh](../../../tools/scripts/train_radarpillar.sh)：
   - `CFG_FILE=tools/cfgs/model/vod_models/radarpillar/vod_radarpillar_large.yaml`
   - `BATCH_SIZE=8`、`WORKERS=2`、`EPOCHS=80`
   - `EXTRA_TAG=rp_large_0716`（修 typo）
   - 保留 `SKIP_EVAL=True`；`SET_CFGS` 关 early_stop；`RUN_MODE=background`；`FIX_RANDOM_SEED=True`
5. **修脚本** [eval_vod.sh](../../../tools/scripts/eval_vod.sh)：
   - `CFG_FILE`/`DATAROOT`/`CKPT` 适配新路径与 epoch_80
   - `RUN_VIZ=True`、`N_VIZ_SAMPLES≥10`、`SCORE_THRESH=0.1`

### 5.2 large 训练（后台，跨轮监控）
```bash
bash tools/scripts/train_radarpillar.sh
```
- 输出：`output/cfgs/model/vod_models/vod_radarpillar/rp_large_0716/{ckpt,logs,tensorboard}/`
- 监控：周期 tail log，汇报 epoch/loss/lr、OOM/NaN、ETA；OOM→降 batch 重启，NaN→查 lr/grad clip。

### 5.3 large eval + 可视化
```bash
bash tools/scripts/eval_vod.sh   # EVAL_MODE=single, CKPT=checkpoint_epoch_80.pth
```
- 产出：per-class AP（entire+corridor）log、`tb_loss_curves.png`、`loss_curve.png`、`frame_*.png`。

### 5.4 base 基线对照
同 5.2–5.3，`CFG_FILE=.../vod_radarpillar.yaml`、`EXTRA_TAG=rp_base_0716`。

### 5.5 论文对比（large）
| 指标 | 论文(5-frame Entire) | 我们 | Δ |
|---|---|---|---|
| Car | 41.1 | _eval 后填_ | |
| Pedestrian | 38.6 | | |
| Cyclist | 72.6 | | |
| mAP | 50.7 | | |
| Driving mAP | 70.5 | | |

## 6. 收尾
- 汇总两变体 loss 曲线 + 预测 PNG + 对比表。
- 清理 `.tmp/`。
- 不 commit（除非用户要求）。

## 7. 训练时间预估（实测，bs=8, workers=2）

训练集 5139 样本 → bs=8 → 每 epoch 643 iter（完整 forward+backward+optim+数据加载实测）：

| 变体 | sec/iter | sec/epoch | epochs | 纯训练 ETA |
|---|---|---|---|---|
| RP-large | 0.21s | 134s | **80**（已定，对齐 decay list） | **3.0h** |
| RP-base | 0.15s | 97s | 80 | 2.1h |

- **串行纯训练**：large(80) + base(80) ≈ 5.1h；加 eval/可视化/系统余量 → **实际约 6–7h**。
- RadarPillar 两变体 + RadarNeXt 两变体合计 ≈ **23–28h（约 1 天）**，需跨多轮对话。

## 8. 结果偏离论文时的自动诊断 loop

训练 + eval 完成后，若**结果与论文差距较大**（large Entire-area mAP 偏离论文 50.7 超过容差，或某类 AP 偏离 > 10 点），**不直接交付**，而是自动进入诊断循环找原因：

```
eval 出结果 → 对比论文 → 超容差？
   ├─ 否 → 正常收尾，产出对比表 + 可视化
   └─ 是 → 进入诊断 loop（见下），直到定位原因或修复后重训
```

诊断 loop 的排查维度（逐一排查、定位后再决定是否重训）：
1. **数据正确性**：Cyclist 聚合是否生效（核对重生 infos 里 Cyclist GT 数量是否 ~翻倍）、FOV 过滤是否误丢样本、pkl infos 与标签一致性。
2. **评测口径**：IoU 阈值（Car@0.5/Ped@0.25/Cyc@0.25）、entire vs driving-corridor 取值、NMS 阈值/score 阈值是否合理、R40 计算。
3. **训练健康度**：loss 曲线是否收敛/震荡/发散、lr 调度是否匹配 epoch 数、是否 NaN、最佳 ckpt 选哪个 epoch。
4. **配置忠实度**：anchor 尺寸、特征编码、数据增强、与论文 recipe 的差异点。
5. **硬件/算力差异**：单卡 8GB batch 8 vs 论文 batch 8（算力差异导致的已知偏差，记录但不强行追平）。

> 该 loop 由调度机制（scheduled wakeup）驱动：训练在后台跑，每轮唤醒时检查进度；训练完成后做 eval + 对比，若超容差则按上述维度循环诊断，定位到可修复原因（如 Cyclist 未聚合、lr 调度错配）则修复并重训，属客观硬件差异则记录归档。**用户无需全程盯守**。

## 9. 待确认项（已收敛）
1. ✅ Cyclist 聚合：按 VoD 官方协议（`bicycle/rider/moped_scooter/motor→Cyclist`，其余不计入）。
2. ✅ batch=8（实测峰值 3.3GB，安全）。
3. ✅ **large epochs=80**（用户已定；改 yaml `NUM_EPOCHS: 40→80`，对齐 `DECAY_STEP_LIST: [35,45]` lr 调度）。
4. eval 是否加 `--save_to_file`（导出 KITTI 格式）？默认不加，可按需开。
