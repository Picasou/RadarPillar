# RadarNeXt 训练计划（VoD 数据集）

> 状态：审阅稿（未 commit）。批准后转入分步执行。
> 范围：训练 RadarNeXt（mdfen 最佳 + fpn 对照两个变体），产出 loss 曲线与预测可视化，与原论文对标。与 RadarPillar 同等对待。

---

## 1. 目标与成功标准（总）

- **目标**：在 OpenPCDet fork 上训练 RadarNeXt（mdfen + fpn），得到可用模型并完成论文级对比。
- **对比论文**：RadarNeXt (Jia et al., arXiv:2501.02314, 5-scan VoD val)。
- **评测口径**：VoD 官方 IoU（Car@0.5，Pedestrian/Cyclist@0.25）。RadarNeXt 论文只报 combined per-class AP/mAP，不分 entire/corridor。
- **成功判据**：
  - mdfen 变体 mAP 落在论文 **50.48 ± ~8 点**；各类 AP 与论文（Car 37.44 / Ped 41.83 / Cyc 72.16）偏离 ≤ ~10 点。
  - fpn 作为消融对照（论文 mAP 47.98）。
  - 必须产出：① loss 曲线 PNG；② ≥10 帧预测 PNG；③ 论文对比表。
- **算力约束已知**：单卡 RTX 3070 Ti / 8 GB（论文用 RTX A4000、batch 8）。
- **硬件实测结论（已验证）**：① 显存——RN-mdfen bs=8 前向+反向峰值 **5.96 GB**，8GB 可跑但偏紧，OOM 备降 bs=4；② 数据 I/O——VoD 经 9p 挂载，`workers=2` 预读下每 batch 83ms < GPU 计算，**9p 不构成训练瓶颈**，无需迁数据。

## 2. 环境与现状（总）

| 项 | 值 | 处置 |
|---|---|---|
| GPU | RTX 3070 Ti / 8 GB | batch 8（yaml 默认），OOM 降 4 |
| ported 模块 | 11 个齐全（mdfen/rep_dwc/rep_common/backbone_mdfen/backbone_fpn/second_fpn/mobileone/radarnext_center_head/losses/transfusion 等） | build 已通过 |
| DCNv3 | **纯 PyTorch `DCNv3_pytorch`**（mdfen 用此），无需编译 .so | 不阻塞训练 |
| 已有 ckpt | mdfen + fpn 各 ep11–15 | 可续训，见待确认项 |
| RadarNeXt train 脚本 | **无** | 需新建 `train_radarnext.sh` |

### 关键事实澄清（纠正 recon 误判）
- recon 曾报 `from pcdet.ops.dcnv3 import DCNv3` 失败 → 误判为阻塞。实测 [mdfen_neck.py:28](../../../pcdet/models/backbones_2d/mdfen_neck.py) 用的是 `DCNv3_pytorch`（纯 PyTorch），**不依赖编译 .so**，mdfen 能训。
- 参数量（build 验证）：mdfen **1.637M**（backbone_2d 1.132M + head 0.505M，论文 1.58M ✓）、fpn **1.102M**（论文 0.899M，接近）。

## 3. 已识别坑与预案（总）

| 坑 | 位置 | 预案 |
|---|---|---|
| 训练结束自动 eval NameError | [train.py:6,267](../../../tools/train.py) | 强制 `--skip_eval`，eval 走 `tools/test.py` |
| 无 train_radarnext.sh | tools/scripts/ 无 radarnext 脚本 | cp train_radarpillar.sh 改 CFG |
| BACKBONE_3D 为空 | yaml 中 `BACKBONE_3D:` 故意留空 | detector 模板有 guard，自动短路，无需处理 |
| Cyclist GT 偏少 | 与 RadarPillar 共用 vod_dataset_radar.yaml | 复用 RadarPillar 阶段的聚合 infos（一次重生两模型共用） |

## 4. 配置选型（分）

| 变体 | yaml | 特征 | 参数量 | 论文 mAP |
|---|---|---|---|---|
| **mdfen（best）** | `radarnext/vod_radarnext_mdfen.yaml` | anchor-free CenterPoint + RepDWC + MDFEN neck（DCNv3）+ RadarNeXtCenterHead（Focal+L1+IoU+dIoU） | 1.637M | **50.48** |
| fpn（对照） | `radarnext/vod_radarnext_fpn.yaml` | RepDWC + SecondFPN neck | 1.102M | 47.98 |

## 5. 执行步骤（分）

### 5.1 前置适配
1. **infos 复用**：若 RadarPillar 阶段已完成 Cyclist 聚合重生，直接共用；否则先做聚合重生（见 RadarPillar 计划 5.1）。
2. **新建 train 脚本**：`cp tools/scripts/train_radarpillar.sh tools/scripts/train_radarnext.sh`，改：
   - `CFG_FILE=tools/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen.yaml`
   - `EXTRA_TAG=rn_mdfen_0716`
   - `BATCH_SIZE=8`、`WORKERS=2`、`EPOCHS=80`
   - 保留 `SKIP_EVAL=True`；`RUN_MODE=background`
   - 续训开关：`CKPT=output/cfgs/model/vod_models/vod_radarnext_mdfen/task7_mdfen_short/ckpt/checkpoint_epoch_15.pth`（见待确认项）
3. **适配 eval_vod.sh**：`CFG_FILE`/`CKPT` 指向 radarnext yaml 与 epoch_80。

### 5.2 mdfen 训练（后台，跨轮监控）
```bash
bash tools/scripts/train_radarnext.sh
```
- 输出：`output/cfgs/model/vod_models/vod_radarnext_mdfen/rn_mdfen_0716/{ckpt,logs,tensorboard}/`
- 监控：同 RadarPillar（OOM/NaN/ETA）。

### 5.3 mdfen eval + 可视化
```bash
bash tools/scripts/eval_vod.sh
```
- 产出：per-class AP log、`tb_loss_curves.png`、`loss_curve.png`、`frame_*.png`。

### 5.4 fpn 对照
同 5.2–5.3，`CFG_FILE=.../vod_radarnext_fpn.yaml`、`EXTRA_TAG=rn_fpn_0716`。

### 5.5 论文对比（mdfen）
| 变体 | Car | Ped | Cyclist | mAP | params |
|---|---|---|---|---|---|
| MDFEN（论文） | 37.44 | 41.83 | 72.16 | **50.48** | 1.58M |
| 我们 | _eval 后填_ | | | | 1.637M |
| FPN（论文） | 37.96 | 38.28 | 67.69 | 47.98 | 0.899M |
| 我们 | | | | | 1.102M |

## 6. 收尾
- 汇总两变体 loss 曲线 + 预测 PNG + 对比表。
- 清理 `.tmp/`。
- 不 commit（除非用户要求）。

## 6.5 训练时间预估（实测，bs=8, workers=2）

训练集 5139 样本 → bs=8 → 每 epoch 643 iter（完整 forward+backward+optim 实测）：

| 变体 | sec/iter | sec/epoch | epochs | 纯训练 ETA |
|---|---|---|---|---|
| RN-mdfen | 0.57s | 368s | 80 | **8.2h**（最慢，DCNv3+MDFEN 计算重） |
| RN-fpn | 0.39s | 253s | 80 | **5.6h** |

- RadarNeXt 两变体纯训练 ≈ **13.8h**；加 eval/可视化/系统余量 → **实际约 18–22h**。
- RP + RN 四变体合计 ≈ **23–28h（约 1 天）**，需跨多轮对话，用后台启动 + 跨轮监控。

## 6.6 结果偏离论文时的自动诊断 loop

与 RadarPillar 同机制。mdfen eval 后若 mAP 偏离论文 50.48 超容差、或某类 AP 偏离 > 10 点，自动进入诊断循环：

诊断维度：
1. **数据正确性**：Cyclist 聚合是否生效、infos 一致性。
2. **评测口径**：IoU 阈值、NMS/score 阈值（注意 RadarNeXt head 自带 `NMS_THRESH: 0.2` 与 POST_PROCESSING 的 0.15 不同，需核对用哪个）、R40 计算。
3. **训练健康度**：loss 收敛/震荡、lr 调度（AdamW step decay 的 `DECAY_STEP_LIST` 是否匹配 80ep）、是否 NaN、best ckpt 选择。
4. **配置忠实度**：optimizer 是否真用了 AdamW（见 §8 方案 A）、DCNv3 配置、rectifier/corner/iou 辅助头、与论文 recipe 差异。
5. **参数量对齐**：mdfen 1.637M（论文 1.58M ✓）、fpn 1.102M（论文 0.899M，偏大需关注是否影响复现）。
6. **硬件/算力差异**：单卡 8GB（论文 A4000），记录客观偏差。

> 超容差则按上述维度循环排查；定位到可修复原因（如 optimizer 未生效 AdamW、NMS 用错、lr 调度错配）则修复并重训；属客观硬件差异则记录归档。由 scheduled wakeup 驱动，用户无需全程盯守。

## 7. 待确认项（已收敛）
1. ✅ **从 0 训练**（忠实论文 80ep，不复用 ep11–15 ckpt）。
2. ✅ batch=8（实测 5.96GB 可跑，OOM 备降 4）。
3. ✅ **优化器用 AdamW 方案 A**（改 `build_optimizer` 加 `adamw` 分支 + yaml 配 step decay）。
4. RadarNeXt 与 RadarPillar 串行（先 RP 后 RN），同卡。确认顺序。
5. eval 是否加 `--save_to_file`？默认不加。

## 8. 训练超参对齐论文
| | 论文 RadarNeXt | 我们 |
|---|---|---|
| GPU | RTX A4000 | RTX 3070 Ti/8GB |
| batch | 8 | 8（实测 5.96GB，OOM→4） |
| epochs | 80 | 80 |
| 优化器 | **AdamW, lr 3e-3, step decay** | 见下：方案 A/B 二选一 |

> **优化器决策（用户已定：用 AdamW 对齐论文）**。当前 yaml 用 `OPTIMIZER: adam_onecycle`（Adam+OneCycle 调度）。代码 [build_optimizer](../../../tools/utils/train_utils/optimization/__init__.py) 现有 `adam`/`sgd`/`adam_onecycle` 三分支，`adam` 分支用的是 `optim.Adam`（非 AdamW）。两个落地方案：
> - **方案 A（最忠实，需改代码）**：在 `build_optimizer` 加 `adamw` 分支（`optim.AdamW`）+ yaml 设 `OPTIMIZER: adamw` + 阶梯衰减调度（`DECAY_STEP_LIST`/`LR_DECAY`）。需改 `__init__.py`（新增分支，按 CLAUDE.md 须用户确认）。
> - **方案 B（不改代码，最接近）**：yaml 设 `OPTIMIZER: adam`（Adam + 阶梯衰减）。Adam vs AdamW 在 weight_decay=0.01 下差别极小，step decay 调度与论文一致。
>
> **已定方案 A（用户确认）**：在 [build_optimizer](../../../tools/utils/train_utils/optimization/__init__.py) 加 `adamw` 分支（`optim.AdamW`，betas 用默认 (0.9,0.999)）+ yaml 设 `OPTIMIZER: adamw` + 阶梯衰减调度（`DECAY_STEP_LIST: [35,45]` / `LR_DECAY: 0.1`）。改 `__init__.py` 单函数内新增分支，不破坏现有 `adam/sgd/adam_onecycle` 逻辑。
